"""Миксин агента: классификация интента через micro-model с fallback на LLM (День 10 advance)."""

from __future__ import annotations

from ..agent_constants import (
    MICRO_DEFAULT_STRATEGY,
    MICRO_LLM_MODEL,
    MICRO_STRATEGIES,
    MICRO_TEMPERATURE,
)
from ..micro import IntentResult, classify_intent


class AgentIntentMixin:
    async def classify_intent(
        self,
        provider_name: str,
        text: str,
        *,
        strategy: str = MICRO_DEFAULT_STRATEGY,
        llm_model: str = MICRO_LLM_MODEL,
        temperature: float = MICRO_TEMPERATURE,
    ) -> IntentResult:
        """Классифицирует обращение; модель уровня 2 валидируется только там, где может понадобиться."""
        if strategy not in MICRO_STRATEGIES:
            raise ValueError(
                f"Неизвестная стратегия «{strategy}»; доступны: {', '.join(MICRO_STRATEGIES)}."
            )
        provider = self._validate_provider(provider_name)
        _, allow_fallback = MICRO_STRATEGIES[strategy]
        if allow_fallback:
            await self._validate_model(provider, provider_name, llm_model)
        return await classify_intent(
            provider,
            text,
            strategy=strategy,
            llm_model=llm_model,
            temperature=self._normalize_temperature(temperature),
        )
