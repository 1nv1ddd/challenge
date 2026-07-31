"""Этапы инференса: один короткий вызов с ремонтом формата плюс метрики по цепочке этапов."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from ..agent_constants import INTAKE_MAX_REPAIRS, INTAKE_STAGE_MODELS, INTAKE_TEMPERATURE
from ..confidence.inference import complete
from ..providers import AIProvider, Message
from ..routing.pricing import cost_rub_model
from .parse import parse_decision, parse_fields, parse_reply
from .prompts import (
    repair_messages,
    stage_compose_messages,
    stage_decide_messages,
    stage_normalize_messages,
)
from .schema import IntakeFields, StageCall, StageDecision

_T = TypeVar("_T")


async def call_stage(
    provider: AIProvider,
    model: str,
    stage: str,
    messages: list[Message],
    parse: Callable[[str], _T],
    *,
    temperature: float = INTAKE_TEMPERATURE,
    max_repairs: int = INTAKE_MAX_REPAIRS,
) -> tuple[_T | None, StageCall]:
    """Вызов одного этапа: строгий парсер и до max_repairs повторов, если формат не сошёлся."""
    call_log = StageCall(stage=stage, model=model)
    current = messages
    attempt = 0
    while True:
        call = await complete(provider, model, current, temperature)
        call_log.calls += 1
        call_log.raw = call.text
        call_log.time_ms += call.time_ms
        call_log.prompt_tokens += call.prompt_tokens
        call_log.completion_tokens += call.completion_tokens
        call_log.cost_rub = cost_rub_model(
            model, call_log.prompt_tokens, call_log.completion_tokens
        )

        try:
            parsed = parse(call.text)
        except ValueError as exc:
            error = str(exc)
        else:
            call_log.error = None
            return parsed, call_log

        call_log.error = error
        if call_log.first_error is None:
            call_log.first_error = error
        if attempt >= max_repairs:
            return None, call_log
        attempt += 1
        call_log.repaired = True
        current = repair_messages(messages, call.text, error)


async def normalize_stage(
    provider: AIProvider,
    letter: str,
    today: str,
    *,
    model: str = INTAKE_STAGE_MODELS["normalize"],
    temperature: float = INTAKE_TEMPERATURE,
    max_repairs: int = INTAKE_MAX_REPAIRS,
) -> tuple[IntakeFields | None, StageCall]:
    """Этап 1 — анализ и нормализация входа: письмо превращается в 8 канонических полей."""
    return await call_stage(
        provider,
        model,
        "normalize",
        stage_normalize_messages(letter, today),
        parse_fields,
        temperature=temperature,
        max_repairs=max_repairs,
    )


async def decide_stage(
    provider: AIProvider,
    fields: IntakeFields,
    today: str,
    *,
    model: str = INTAKE_STAGE_MODELS["decide"],
    temperature: float = INTAKE_TEMPERATURE,
    max_repairs: int = INTAKE_MAX_REPAIRS,
) -> tuple[StageDecision | None, StageCall]:
    """Этап 2 — решение по политике: на входе только нормализованные поля, выход строго enum."""
    return await call_stage(
        provider,
        model,
        "decide",
        stage_decide_messages(fields, today),
        parse_decision,
        temperature=temperature,
        max_repairs=max_repairs,
    )


async def compose_stage(
    provider: AIProvider,
    fields: IntakeFields,
    decision: StageDecision,
    *,
    model: str = INTAKE_STAGE_MODELS["compose"],
    temperature: float = INTAKE_TEMPERATURE,
    max_repairs: int = INTAKE_MAX_REPAIRS,
) -> tuple[tuple[str, str] | None, StageCall]:
    """Этап 3 — формирование результата: письмо клиенту по уже принятому решению."""
    return await call_stage(
        provider,
        model,
        "compose",
        stage_compose_messages(fields, decision),
        parse_reply,
        temperature=temperature,
        max_repairs=max_repairs,
    )


def stage_metrics(stages: list[StageCall], wall_ms: int) -> dict:
    """Сводка по цепочке: сколько вызовов, из них ремонтных, во что обошлось."""
    return {
        "llm_calls": sum(stage.calls for stage in stages),
        "repair_calls": sum(max(0, stage.calls - 1) for stage in stages),
        "stages": len(stages),
        "time_ms": wall_ms,
        "prompt_tokens": sum(stage.prompt_tokens for stage in stages),
        "completion_tokens": sum(stage.completion_tokens for stage in stages),
        "cost_rub": round(sum(stage.cost_rub for stage in stages), 4),
        "models": [stage.model for stage in stages],
    }
