"""Execution loop: генерация → проверки → security review → «коммит», все вызовы через шлюз."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from ..agent_constants import (
    LOOP_ARTIFACTS_DIR,
    LOOP_GATEWAY_MODE,
    LOOP_GEN_MODEL,
    LOOP_GEN_TEMPERATURE,
    LOOP_MAX_ATTEMPTS,
    LOOP_RATE_LIMIT_PER_MIN,
    LOOP_REPEAT_ESCALATION,
    LOOP_REVIEW_MODEL,
    LOOP_REVIEW_TEMPERATURE,
)
from ..gateway import RateLimiter, proxy_chat
from ..providers import AIProvider
from .prompts import generator_prompt, review_prompt
from .review import parse_verdict, security_feedback
from .sandbox import checks_feedback, extract_code, run_checks
from .schema import GatewayEvent, LoopAttempt, LoopRun, LoopTask, SecurityVerdict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# Свой лимитер оркестратора: пользовательские 10/мин рассчитаны на человека, а цикл делает
# два вызова на попытку и упёрся бы в лимит на второй задаче.
_LIMITER = RateLimiter(limit=LOOP_RATE_LIMIT_PER_MIN)


def _stuck_rule(run: LoopRun) -> str:
    """Правило, блокирующее последние LOOP_REPEAT_ESCALATION попыток подряд, или пустая строка."""
    tail = [a for a in run.attempts if a.security][-LOOP_REPEAT_ESCALATION:]
    if len(tail) < LOOP_REPEAT_ESCALATION:
        return ""
    common = set.intersection(
        *({f.rule for f in a.security.blocking} for a in tail if a.security)
    )
    return sorted(common)[0] if common else ""


def _event(stage: str, result) -> GatewayEvent:
    """Проход через шлюз → строка отчёта цикла: что guard сделал со входом и выходом."""
    return GatewayEvent(
        stage=stage,
        request_id=result.request_id,
        status=result.status,
        mode=result.mode,
        model=result.model,
        input_action=result.input.action,
        input_kinds=result.input.kinds(),
        output_action=result.output.action,
        output_kinds=result.output.kinds(),
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        cost_rub=result.cost_rub,
    )


async def _call_through_gateway(
    provider: AIProvider,
    stage: str,
    prompt: str,
    model: str,
    temperature: float,
    client: str,
) -> tuple[str, GatewayEvent, str | None]:
    """Один вызов модели через шлюз. Возвращает (текст, событие для отчёта, ошибка).

    `enforce_output=False` — потому что потребитель ответа не человек, а компилятор: маскирование
    на выходе подменяет в сгенерированном модуле ссылки и ключи плейсхолдерами, и код перестаёт
    работать. Находки при этом считаются и попадают в аудит, а решение о приёмке принимает
    security review, который видит код целиком.
    """
    result = await proxy_chat(
        provider,
        prompt,
        model=model,
        temperature=temperature,
        mode=LOOP_GATEWAY_MODE,
        client_ip=client,
        limiter=_LIMITER,
        enforce_output=False,
    )
    event = _event(stage, result)
    if result.status != "ok":
        return result.answer, event, f"{result.status}: {result.answer}"
    return result.answer, event, None


def save_artifact(task_id: str, run_id: str, code: str, status: str, warnings: list[str]) -> str:
    """«Коммит»: принятый код кладётся в артефакты с пометкой, чем закончилось ревью.

    Настоящий git-коммит цикл не делает намеренно — в проекте коммит только по команде человека.
    Артефакт с шапкой выполняет ту же роль для отчёта: видно, что именно приняли и с чем.
    """
    directory = _PROJECT_ROOT / LOOP_ARTIFACTS_DIR / task_id
    directory.mkdir(parents=True, exist_ok=True)
    header = [
        f"# Принято execution loop: {status}",
        f"# Прогон: {run_id}, {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
    ]
    header += [f"# Warning (не блокирует): {w}" for w in warnings]
    file = directory / f"{run_id}.py"
    file.write_text("\n".join(header) + "\n\n" + code.strip() + "\n", encoding="utf-8")
    return str(file.relative_to(_PROJECT_ROOT))


async def run_task(
    provider: AIProvider,
    task: LoopTask,
    *,
    gen_model: str = LOOP_GEN_MODEL,
    review_model: str = LOOP_REVIEW_MODEL,
    max_attempts: int = LOOP_MAX_ATTEMPTS,
    run_id: str = "",
) -> LoopRun:
    """Цикл по одной задаче: до `max_attempts` попыток, каждая — генерация, проверки, ревью."""
    run_id = run_id or uuid.uuid4().hex[:8]
    run = LoopRun(task_id=task.id, title=task.title, traps=task.traps)
    client = f"loop:{run_id}"
    feedback = ""

    for number in range(1, max(1, max_attempts) + 1):
        attempt = LoopAttempt(number=number, feedback_in=feedback)
        run.attempts.append(attempt)

        answer, event, error = await _call_through_gateway(
            provider, "generate", generator_prompt(task.prompt, feedback),
            gen_model, LOOP_GEN_TEMPERATURE, client,
        )
        attempt.gateway.append(event)
        if error:
            attempt.outcome = "gateway_blocked"
            attempt.error = error
            feedback = f"Предыдущая попытка не прошла шлюз: {error}. Верни модуль заново."
            continue

        attempt.code = extract_code(answer)
        if not attempt.code:
            attempt.outcome = "no_code"
            attempt.error = "в ответе нет блока с кодом"
            feedback = "В прошлом ответе не было блока ```python с кодом. Верни только модуль."
            continue

        attempt.checks = run_checks(attempt.code, task.tests)
        if not attempt.checks_ok:
            attempt.outcome = "checks_failed"
            feedback = checks_feedback(attempt.checks)
            continue

        review_answer, review_event, review_error = await _call_through_gateway(
            provider, "review", review_prompt(attempt.code),
            review_model, LOOP_REVIEW_TEMPERATURE, client,
        )
        attempt.gateway.append(review_event)
        if review_error:
            # Шлюз не пропустил ревью — код непроверен, а непроверенный код не коммитим.
            attempt.security = SecurityVerdict(error=review_error)
        else:
            attempt.security = parse_verdict(review_answer)

        if attempt.security.error or attempt.security.blocking:
            attempt.outcome = "security_blocked"
            feedback = security_feedback(attempt.security)
            if stuck := _stuck_rule(run):
                # Одно и то же правило блокирует раз за разом: дальше крутить цикл вредно.
                # Ревьюер на LLM под давлением повторов начинает принимать косметику за фикс.
                run.status = "escalated"
                run.escalation = (
                    f"правило `{stuck}` блокирует {LOOP_REPEAT_ESCALATION} попытки подряд — "
                    "задача уходит человеку, а не на ещё один круг"
                )
                return run
            continue

        warnings = [f"[{f.severity}] {f.title}" for f in attempt.security.warnings]
        attempt.outcome = "accepted_with_warnings" if warnings else "accepted"
        run.status = "committed_with_warnings" if warnings else "committed"
        run.artifact = save_artifact(task.id, run_id, attempt.code, run.status, warnings)
        return run

    run.status = "failed"
    return run


async def run_tasks(
    provider: AIProvider,
    tasks: list[LoopTask],
    *,
    gen_model: str = LOOP_GEN_MODEL,
    review_model: str = LOOP_REVIEW_MODEL,
    max_attempts: int = LOOP_MAX_ATTEMPTS,
) -> list[LoopRun]:
    """Прогон набора задач по очереди: порядок сохраняется, каждая задача — свой run_id."""
    return [
        await run_task(
            provider,
            task,
            gen_model=gen_model,
            review_model=review_model,
            max_attempts=max_attempts,
        )
        for task in tasks
    ]


def loop_summary(runs: list[LoopRun]) -> dict:
    """Свод по прогону: что поймал security step, что поймал шлюз, что прошло мимо обоих."""
    security: dict[str, int] = {}
    warned: dict[str, int] = {}
    gateway: dict[str, int] = {}
    missed: dict[str, int] = {}
    for run in runs:
        for rule in run.caught_by_security():
            security[rule] = security.get(rule, 0) + 1
        for rule in run.warned_by_security():
            warned[rule] = warned.get(rule, 0) + 1
        for kind in run.caught_by_gateway():
            gateway[kind] = gateway.get(kind, 0) + 1
        for trap in run.missed_traps():
            missed[trap] = missed.get(trap, 0) + 1
    return {
        "tasks": len(runs),
        "committed": sum(1 for r in runs if r.status == "committed"),
        "committed_with_warnings": sum(1 for r in runs if r.status == "committed_with_warnings"),
        "escalated": sum(1 for r in runs if r.status == "escalated"),
        "failed": sum(1 for r in runs if r.status == "failed"),
        "attempts": sum(len(r.attempts) for r in runs),
        "llm_calls": sum(r.llm_calls for r in runs),
        "cost_rub": round(sum(r.cost_rub for r in runs), 4),
        "caught_by_security": dict(sorted(security.items(), key=lambda kv: -kv[1])),
        "warned_by_security": dict(sorted(warned.items(), key=lambda kv: -kv[1])),
        "caught_by_gateway": dict(sorted(gateway.items(), key=lambda kv: -kv[1])),
        "missed_by_both": dict(sorted(missed.items(), key=lambda kv: -kv[1])),
    }
