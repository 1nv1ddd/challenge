"""Вариант A: один большой запрос — поля, решение и письмо клиенту приходят одним ответом."""

from __future__ import annotations

import time

from ..agent_constants import INTAKE_MAX_REPAIRS, INTAKE_MONO_MODEL, INTAKE_TEMPERATURE
from ..providers import AIProvider
from .parse import parse_monolithic
from .prompts import monolithic_messages
from .schema import IntakeFields, IntakeResult, StageDecision
from .stages import call_stage, stage_metrics

# Разбор не состоялся целиком: в монолитном ответе нечего спасать по частям.
_FAILED = StageDecision("clarify", "missing_fields", source="failed")


async def run_monolithic(
    provider: AIProvider,
    letter: str,
    today: str,
    *,
    model: str = INTAKE_MONO_MODEL,
    temperature: float = INTAKE_TEMPERATURE,
    max_repairs: int = INTAKE_MAX_REPAIRS,
) -> IntakeResult:
    """Один вызов модели на всю задачу: извлечение, решение по политике и ответ клиенту."""
    t_start = time.monotonic()
    parsed, call_log = await call_stage(
        provider,
        model,
        "monolithic",
        monolithic_messages(letter, today),
        parse_monolithic,
        temperature=temperature,
        max_repairs=max_repairs,
    )
    wall_ms = round((time.monotonic() - t_start) * 1000)
    if parsed is None:
        fields, decision, subject, body = IntakeFields(), _FAILED, "", ""
    else:
        fields, decision, subject, body = parsed
    return IntakeResult(
        mode="mono",
        letter=letter,
        today=today,
        fields=fields,
        decision=decision,
        reply_subject=subject,
        reply_body=body,
        stages=[call_log],
        metrics=stage_metrics([call_log], wall_ms),
    )
