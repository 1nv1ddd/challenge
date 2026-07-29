"""Стоимость вызова с учётом модели: тиры роутинга различаются по прайсу на порядок."""

from __future__ import annotations

from ..agent_constants import (
    INPUT_PRICE_RUB_PER_MILLION,
    OUTPUT_PRICE_RUB_PER_MILLION,
    ROUTING_MODEL_PRICES_RUB,
)


def model_prices(model: str) -> tuple[float, float]:
    """(₽ за миллион prompt-токенов, ₽ за миллион completion-токенов) для модели."""
    return ROUTING_MODEL_PRICES_RUB.get(
        model, (INPUT_PRICE_RUB_PER_MILLION, OUTPUT_PRICE_RUB_PER_MILLION)
    )


def cost_rub_model(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Цена одного вызова конкретной модели (в отличие от единого прайса `confidence.cost_rub`)."""
    in_price, out_price = model_prices(model)
    rub = (prompt_tokens * in_price + completion_tokens * out_price) / 1_000_000
    return round(rub, 4)
