"""Уровень 2: вызов большой модели со строгим разбором ответа и ремонтом формата."""

from __future__ import annotations

import json

from ..agent_constants import (
    MICRO_LABELS,
    MICRO_LLM_MODEL,
    MICRO_MAX_REPAIRS,
    MICRO_REASON_MAX_LEN,
    MICRO_TEMPERATURE,
)
from ..confidence.constraints import extract_json_object
from ..providers import AIProvider
from ..staged.schema import StageCall
# Вызов с ремонтом формата уже написан для этапов Дня 9 и от предметной области не зависит.
from ..staged.stages import call_stage
from .prompts import classify_messages
from .schema import LabelAnswer


def parse_label(raw: str) -> LabelAnswer:
    """Проверка формата уровня 2: строгий JSON, метка из enum, confidence в [0, 1]."""
    candidate = extract_json_object(raw)
    if candidate is None:
        raise ValueError("формат: в ответе нет JSON-объекта")
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"формат: JSON не разбирается ({exc.msg})") from exc
    if not isinstance(data, dict):
        raise ValueError("формат: ожидался JSON-объект")

    label = str(data.get("label") or "").strip().lower()
    if label not in MICRO_LABELS:
        raise ValueError(f"значение: label={label!r} вне списка {list(MICRO_LABELS)}")
    raw_confidence = data.get("confidence")
    try:
        confidence = float(str(raw_confidence).replace(",", "."))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"формат: confidence={raw_confidence!r} не число") from exc
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"значение: confidence={confidence} вне отрезка [0, 1]")
    reason = str(data.get("reason") or "").strip()
    if len(reason) > MICRO_REASON_MAX_LEN:
        raise ValueError(f"формат: reason длиннее {MICRO_REASON_MAX_LEN} символов")

    return LabelAnswer(label=label, confidence=confidence, reason=reason)


async def classify_with_llm(
    provider: AIProvider,
    text: str,
    *,
    model: str = MICRO_LLM_MODEL,
    temperature: float = MICRO_TEMPERATURE,
    max_repairs: int = MICRO_MAX_REPAIRS,
) -> tuple[LabelAnswer | None, StageCall]:
    """Классификация большой моделью; None в ответе означает, что формат не сошёлся и после ремонта."""
    return await call_stage(
        provider,
        model,
        "llm_fallback",
        classify_messages(text),
        parse_label,
        temperature=temperature,
        max_repairs=max_repairs,
    )
