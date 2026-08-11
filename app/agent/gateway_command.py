"""День 13 (advance): команда `/gateway` — прокси с guard'ами, прогон корпуса и просмотр аудита.

Работает как `/indirect`: перехватываем префикс в последней реплике пользователя. В чат уходит
карточка прохода через шлюз, таблица корпуса «поймали / пропустили» или сводка аудит-лога.
"""

from __future__ import annotations

from ..agent_constants import (
    GATEWAY_DEFAULT_MODE,
    GATEWAY_MAX_PROMPT_CHARS,
    GATEWAY_MODES,
    GATEWAY_MODEL,
    GATEWAY_RATE_LIMIT_PER_MIN,
)
from ..gateway import CorpusRun, GatewayResult, audit_stats, read_records

_PREFIX = "/gateway"
_SUBCOMMANDS = ("selftest", "audit")
_STATUS_TITLE = {
    "ok": "✅ ответ выдан",
    "blocked_input": "🛑 запрос заблокирован на входе",
    "blocked_output": "🛑 ответ заблокирован на выходе",
    "rate_limited": "⏳ сработал rate limit",
    "too_long": "📏 промпт слишком длинный",
    "error": "⚠️ ошибка вызова модели",
}
_MODE_TITLE = {
    "hybrid": "ключи блокируем, ПДн маскируем",
    "block": "блокируем любую находку",
    "mask": "маскируем всё и пропускаем",
    "off": "guard выключен",
}
_USAGE = (
    "### `/gateway` — LLM-шлюз с input/output guard\n\n"
    "Прокси между пользователем и моделью: ищет секреты во входе, проверяет ответ, "
    "держит лимит запросов и пишет аудит.\n\n"
    "```\n/gateway мой ключ sk-proj-abc123XYZ456def789 не работает, почему?\n```\n\n"
    f"- режим guard первым словом: `{'`, `'.join(GATEWAY_MODES)}` (по умолчанию `{GATEWAY_DEFAULT_MODE}`);\n"
    "- `/gateway selftest` — прогон корпуса кейсов по детекторам, без вызова модели;\n"
    "- `/gateway audit` — последние записи аудита и сводка по стоимости.\n\n"
    f"Лимит: {GATEWAY_RATE_LIMIT_PER_MIN} запросов в минуту с адреса, промпт до "
    f"{GATEWAY_MAX_PROMPT_CHARS} символов, модель `{GATEWAY_MODEL}`."
)
# Сколько записей аудита показываем в карточке /gateway audit.
_AUDIT_ROWS = 10


def detect_gateway_command(text: str) -> tuple[bool, str, str, str]:
    """Возвращает (is_gateway, подкоманда, режим guard, оставшийся промпт)."""
    s = (text or "").lstrip()
    if not s.lower().startswith(_PREFIX):
        return False, "", "", ""
    rest = s[len(_PREFIX):].lstrip(" :-—")
    sub = ""
    mode = GATEWAY_DEFAULT_MODE
    words = rest.split()
    while words:
        token = words[0].strip().lower()
        if token in _SUBCOMMANDS and not sub:
            sub = token
        elif token in GATEWAY_MODES:
            mode = token
        else:
            break
        words.pop(0)
    return True, sub, mode, " ".join(words).strip()


def usage_markdown() -> str:
    """Подсказка по `/gateway` — она же ответ на команду без аргументов."""
    return _USAGE


def _findings_line(result: GatewayResult) -> str:
    if not result.input.findings:
        return "секретов не найдено"
    parts = [
        f"`{f.kind}` ({f.variant}, `{f.digest}`)" for f in result.input.findings
    ]
    return ", ".join(parts)


def render_gateway_card(result: GatewayResult) -> str:
    """Карточка одного прохода через шлюз: вход, выход, деньги."""
    status = _STATUS_TITLE.get(result.status, result.status)
    lines = [
        "### LLM Gateway",
        "",
        f"**{status}** · режим `{result.mode}` ({_MODE_TITLE.get(result.mode, '')}) · "
        f"`{result.model}` · запрос `{result.request_id}`",
        "",
        "| Этап | Итог | Детали |",
        "|---|---|---|",
        f"| Rate limit | {result.rate.used}/{result.rate.limit} за минуту | "
        f"{'пропущен' if result.rate.allowed else f'ждать {result.rate.retry_after_sec} с'} |"
        if result.rate
        else "| Rate limit | — | не проверялся |",
        f"| Input guard | `{result.input.action}` | {_findings_line(result)} |",
        f"| Вызов модели | {'да' if result.llm_called else '**нет**'} | "
        f"{result.time_ms / 1000:.1f} с |",
        f"| Output guard | `{result.output.action}` | "
        f"{', '.join(result.output.kinds()) or 'находок нет'} |",
    ]
    tokens = result.prompt_tokens + result.completion_tokens
    estimated = " (оценка)" if result.tokens_estimated else ""
    lines += [
        "",
        f"**Стоимость:** {tokens} токенов{estimated} "
        f"({result.prompt_tokens} + {result.completion_tokens}) · {result.cost_rub} ₽",
        "",
    ]
    if result.input.action != "pass" and result.input.prompt:
        title = "Что ушло в модель" if result.llm_called else "Что ушло бы в модель"
        lines += [f"**{title}:**", "", "```", result.input.prompt, "```", ""]
    lines += ["---", "", result.answer or "_пусто_"]
    return "\n".join(lines)


def render_selftest_card(run: CorpusRun) -> str:
    """Таблица корпуса: что поймали, что пропустили, где действие разошлось с ожидаемым."""
    lines = [
        "### Корпус шлюза: что поймали, что пропустили",
        "",
        f"Кейсов: **{run.total}** · совпало с ожиданием: **{run.passed}** · "
        f"пропусков: {len(run.missed_cases)} · неверных действий: {len(run.wrong_action)}",
        "",
        "| Кейс | Режим | Ожидали | Нашли | Как нашли | Действие | Итог |",
        "|---|---|---|---|---|---|---|",
    ]
    for outcome in run.outcomes:
        expected = ", ".join(outcome.expect_kinds) or "—"
        found = ", ".join(outcome.found_kinds) or "—"
        variants = ", ".join(outcome.variants) or "—"
        action = (
            f"`{outcome.action}`"
            if outcome.action == outcome.expect_action
            else f"`{outcome.action}` вместо `{outcome.expect_action}`"
        )
        verdict = "✅" if outcome.ok else ("❌ пропуск" if outcome.missed else "⚠️ другое действие")
        lines.append(
            f"| `{outcome.case_id}` | `{outcome.mode}` | {expected} | {found} | {variants} | "
            f"{action} | {verdict} |"
        )
    missed = [o for o in run.outcomes if o.missed]
    if missed:
        lines += ["", "**Пропущенное — и почему:**", ""]
        lines += [f"- `{o.case_id}`: {o.note}" for o in missed]
    return "\n".join(lines)


def render_audit_card(limit: int = _AUDIT_ROWS) -> str:
    """Последние записи аудита и сводка: сколько заблокировано, что ловилось, во что обошлось."""
    records = read_records(limit)
    if not records:
        return "### Аудит шлюза\n\nЛог пуст: через шлюз ещё не проходило ни одного запроса."
    stats = audit_stats(records)
    lines = [
        "### Аудит шлюза",
        "",
        f"Записей в выборке: **{stats['requests']}** · дошло до модели: {stats['llm_calls']} · "
        f"вызовов сэкономлено: {stats['saved_calls']} · "
        f"{stats['prompt_tokens'] + stats['completion_tokens']} токенов · {stats['cost_rub']} ₽",
        "",
        "| Время | Запрос | Статус | Вход | Секреты | Выход | ₽ |",
        "|---|---|---|---|---|---|---|",
    ]
    for record in records:
        secrets = ", ".join(
            f"{f['kind']}/{f['digest']}" for f in record.get("input_findings") or []
        )
        output_kinds = ", ".join(
            sorted({f["kind"] for f in record.get("output_findings") or []})
        )
        lines.append(
            f"| {str(record.get('ts', ''))[11:19]} | `{record.get('request_id', '')}` | "
            f"{record.get('status', '')} | `{record.get('input_action', '')}` | "
            f"{secrets or '—'} | `{record.get('output_action', '')}`"
            f"{f' ({output_kinds})' if output_kinds else ''} | {record.get('cost_rub', 0)} |"
        )
    if stats["secrets_by_kind"]:
        found = ", ".join(f"`{k}` × {v}" for k, v in stats["secrets_by_kind"].items())
        lines += ["", f"**Перехвачено секретов:** {found}"]
    return "\n".join(lines)
