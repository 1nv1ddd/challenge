"""Вариант B: разбор заявки по этапам — нормализация → решение → письмо; плюс выбор режима."""

from __future__ import annotations

import time
from datetime import date

from ..agent_constants import (
    INTAKE_MAX_REPAIRS,
    INTAKE_MODES,
    INTAKE_MONO_MODEL,
    INTAKE_REQUIRED_FIELDS,
    INTAKE_STAGE_MODELS,
    INTAKE_TEMPERATURE,
)
from ..providers import AIProvider
from .monolithic import run_monolithic
from .policy import decide_by_rules
from .schema import IntakeFields, IntakeResult, StageCall, StageDecision
from .stages import compose_stage, decide_stage, normalize_stage, stage_metrics


def _today_iso(today: str | None) -> str:
    """Дата обращения: от неё считаются все относительные сроки в письме."""
    if not today:
        return date.today().isoformat()
    try:
        return date.fromisoformat(today.strip()).isoformat()
    except ValueError as exc:
        raise ValueError("Дата обращения должна быть в формате YYYY-MM-DD.") from exc


def _result(
    mode: str,
    letter: str,
    today: str,
    fields: IntakeFields,
    decision: StageDecision,
    reply: tuple[str, str] | None,
    stages: list[StageCall],
    t_start: float,
) -> IntakeResult:
    subject, body = reply if reply else ("", "")
    return IntakeResult(
        mode=mode,
        letter=letter,
        today=today,
        fields=fields,
        decision=decision,
        reply_subject=subject,
        reply_body=body,
        stages=stages,
        metrics=stage_metrics(stages, round((time.monotonic() - t_start) * 1000)),
    )


async def run_staged(
    provider: AIProvider,
    letter: str,
    today: str,
    *,
    models: dict[str, str] | None = None,
    temperature: float = INTAKE_TEMPERATURE,
    max_repairs: int = INTAKE_MAX_REPAIRS,
    decide_with_rules: bool = False,
) -> IntakeResult:
    """Три коротких этапа вместо одного большого запроса; этап 2 можно заменить политикой в коде."""
    stage_models = {**INTAKE_STAGE_MODELS, **(models or {})}
    mode = "staged_rules" if decide_with_rules else "staged"
    t_start = time.monotonic()
    stages: list[StageCall] = []

    fields, normalize_log = await normalize_stage(
        provider,
        letter,
        today,
        model=stage_models["normalize"],
        temperature=temperature,
        max_repairs=max_repairs,
    )
    stages.append(normalize_log)
    if fields is None:
        # Дальше идти нельзя: решать и писать письмо не по чему. Заявка уходит на уточнение.
        decision = StageDecision(
            "clarify", "missing_fields", missing=list(INTAKE_REQUIRED_FIELDS), source="failed"
        )
        return _result(mode, letter, today, IntakeFields(), decision, None, stages, t_start)

    if decide_with_rules:
        decision = decide_by_rules(fields, date.fromisoformat(today))
    else:
        decided, decide_log = await decide_stage(
            provider,
            fields,
            today,
            model=stage_models["decide"],
            temperature=temperature,
            max_repairs=max_repairs,
        )
        stages.append(decide_log)
        # Поля уже нормализованы, поэтому у сорванного этапа 2 есть честный запасной путь —
        # применить ту же политику кодом. У монолитного варианта такого пути нет.
        decision = decided or decide_by_rules(fields, date.fromisoformat(today))
        if decided is None:
            decision.source = "rules_fallback"

    reply, compose_log = await compose_stage(
        provider,
        fields,
        decision,
        model=stage_models["compose"],
        temperature=temperature,
        max_repairs=max_repairs,
    )
    stages.append(compose_log)
    return _result(mode, letter, today, fields, decision, reply, stages, t_start)


async def run_intake(
    provider: AIProvider,
    letter: str,
    *,
    mode: str = "staged",
    today: str | None = None,
    mono_model: str = INTAKE_MONO_MODEL,
    models: dict[str, str] | None = None,
    temperature: float = INTAKE_TEMPERATURE,
    max_repairs: int = INTAKE_MAX_REPAIRS,
) -> IntakeResult:
    """Разбор письма выбранным способом: один большой запрос или цепочка коротких этапов."""
    text = (letter or "").strip()
    if not text:
        raise ValueError("Пустое письмо — нечего разбирать.")
    if mode not in INTAKE_MODES:
        raise ValueError(f"Неизвестный режим «{mode}»; доступны: {', '.join(INTAKE_MODES)}.")
    day = _today_iso(today)

    if mode == "mono":
        return await run_monolithic(
            provider, text, day, model=mono_model, temperature=temperature, max_repairs=max_repairs
        )
    return await run_staged(
        provider,
        text,
        day,
        models=models,
        temperature=temperature,
        max_repairs=max_repairs,
        decide_with_rules=mode == "staged_rules",
    )
