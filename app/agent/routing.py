"""Миксин агента: ответ на вопрос через каскад моделей (День 8 advance)."""

from __future__ import annotations

from ..agent_constants import ROUTING_LARGE_MODEL, ROUTING_SMALL_MODEL, ROUTING_TEMPERATURE
from ..routing import RouteResult, route_answer


class AgentRoutingMixin:
    async def route_question(
        self,
        provider_name: str,
        question: str,
        *,
        small_model: str = ROUTING_SMALL_MODEL,
        large_model: str = ROUTING_LARGE_MODEL,
        temperature: float = ROUTING_TEMPERATURE,
    ) -> RouteResult:
        """Маршрутизирует запрос между тирами; обе модели валидируются как в обычном чате."""
        provider = self._validate_provider(provider_name)
        await self._validate_model(provider, provider_name, small_model)
        await self._validate_model(provider, provider_name, large_model)
        return await route_answer(
            provider,
            question,
            small_model=small_model,
            large_model=large_model,
            temperature=self._normalize_temperature(temperature),
        )
