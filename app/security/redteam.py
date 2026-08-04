"""Прогон атак по версиям промпта: вызов модели, вердикт и сводка по корпусу."""

from __future__ import annotations

import asyncio

from ..agent_constants import SECURITY_MODEL, SECURITY_PROMPT_VERSIONS, SECURITY_TEMPERATURE
from ..confidence.inference import complete
from ..providers import AIProvider
from ..routing.pricing import cost_rub_model
from .schema import Attack, AttackVerdict, RedteamRun
from .verdict import judge
from .victims import victim_messages, victim_secret_text

# Больше — быстрее прогон корпуса, но чаще упираемся в rate limit провайдера.
_MAX_PARALLEL = 4


async def run_attack(
    provider: AIProvider,
    attack: Attack,
    version: str,
    *,
    model: str = SECURITY_MODEL,
    temperature: float = SECURITY_TEMPERATURE,
) -> AttackVerdict:
    """Одна атака против одной версии промпта."""
    if version not in SECURITY_PROMPT_VERSIONS:
        raise ValueError(
            f"Неизвестная версия промпта: {version!r}. Доступны: {', '.join(SECURITY_PROMPT_VERSIONS)}."
        )
    verdict = AttackVerdict(
        attack_id=attack.id,
        target=attack.target,
        version=version,
        vector=attack.vector,
        technique=attack.technique,
        model=model,
    )
    messages = victim_messages(attack, version)
    try:
        call = await complete(provider, model, messages, temperature)
    except (ValueError, LookupError, OSError) as exc:
        verdict.error = f"{type(exc).__name__}: {exc}"
        return verdict

    verdict.reply = call.text
    verdict.time_ms = call.time_ms
    verdict.prompt_tokens = call.prompt_tokens
    verdict.completion_tokens = call.completion_tokens
    verdict.cost_rub = cost_rub_model(model, call.prompt_tokens, call.completion_tokens)
    verdict.broken, verdict.signals, verdict.refused = judge(
        attack, call.text, victim_secret_text(attack.target)
    )
    return verdict


async def run_corpus(
    provider: AIProvider,
    attacks: list[Attack],
    version: str,
    *,
    model: str = SECURITY_MODEL,
    temperature: float = SECURITY_TEMPERATURE,
    max_parallel: int = _MAX_PARALLEL,
) -> RedteamRun:
    """Весь набор атак против одной версии промпта; порядок вердиктов = порядок корпуса."""
    semaphore = asyncio.Semaphore(max(1, max_parallel))

    async def guarded(attack: Attack) -> AttackVerdict:
        async with semaphore:
            return await run_attack(
                provider, attack, version, model=model, temperature=temperature
            )

    verdicts = await asyncio.gather(*(guarded(a) for a in attacks))
    return RedteamRun(version=version, verdicts=list(verdicts))


async def compare_versions(
    provider: AIProvider,
    attacks: list[Attack],
    *,
    versions: tuple[str, ...] = SECURITY_PROMPT_VERSIONS,
    model: str = SECURITY_MODEL,
    temperature: float = SECURITY_TEMPERATURE,
) -> dict[str, RedteamRun]:
    """Один и тот же корпус по каждой версии промпта — это и есть проверка «устоял ли»."""
    runs: dict[str, RedteamRun] = {}
    for version in versions:
        runs[version] = await run_corpus(
            provider, attacks, version, model=model, temperature=temperature
        )
    return runs


def fixed_by_hardening(runs: dict[str, RedteamRun]) -> dict[str, list[str]]:
    """Что изменилось между версиями: закрытые атаки, оставшиеся дыры и новые пробои."""
    if "v1" not in runs or "v2" not in runs:
        return {"fixed": [], "still_broken": [], "regressed": []}
    v1 = {v.attack_id: v.broken for v in runs["v1"].verdicts}
    v2 = {v.attack_id: v.broken for v in runs["v2"].verdicts}
    common = [aid for aid in v1 if aid in v2]
    return {
        "fixed": [aid for aid in common if v1[aid] and not v2[aid]],
        "still_broken": [aid for aid in common if v1[aid] and v2[aid]],
        "regressed": [aid for aid in common if not v1[aid] and v2[aid]],
    }
