"""Один нестримовый вызов провайдера с метриками: время, токены, стоимость в рублях."""

from __future__ import annotations

import time
from dataclasses import dataclass

from ..agent_constants import INPUT_PRICE_RUB_PER_MILLION, OUTPUT_PRICE_RUB_PER_MILLION
from ..providers import AIProvider, Message


@dataclass
class CallResult:
    text: str
    time_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


async def complete(
    provider: AIProvider, model: str, messages: list[Message], temperature: float
) -> CallResult:
    """Собирает стрим провайдера в один ответ и вытаскивает usage из финального meta."""
    t_start = time.monotonic()
    parts: list[str] = []
    meta: dict = {}
    async for result in provider.stream_chat(messages, model, temperature):
        if result.text:
            parts.append(result.text)
        if result.meta:
            meta = result.meta
    elapsed_ms = round((time.monotonic() - t_start) * 1000)
    return CallResult(
        text="".join(parts).strip(),
        time_ms=int(meta.get("time_ms") or elapsed_ms),
        prompt_tokens=int(meta.get("prompt_tokens") or 0),
        completion_tokens=int(meta.get("completion_tokens") or 0),
    )


def cost_rub(prompt_tokens: int, completion_tokens: int) -> float:
    """Стоимость вызовов по прайсу из agent_constants (₽ за миллион токенов)."""
    rub = (
        prompt_tokens * INPUT_PRICE_RUB_PER_MILLION
        + completion_tokens * OUTPUT_PRICE_RUB_PER_MILLION
    ) / 1_000_000
    return round(rub, 4)
