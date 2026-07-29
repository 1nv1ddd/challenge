"""Миксин агента: триаж обращения поддержки через гейт уверенности (День 7 advance)."""

from __future__ import annotations

from ..agent_constants import TRIAGE_MAX_REPAIRS, TRIAGE_SAMPLES, TRIAGE_TEMPERATURE
from ..confidence import Verdict, triage


class AgentTriageMixin:
    async def triage_message(
        self,
        provider_name: str,
        model: str,
        text: str,
        *,
        samples: int = TRIAGE_SAMPLES,
        temperature: float = TRIAGE_TEMPERATURE,
        max_repairs: int = TRIAGE_MAX_REPAIRS,
    ) -> Verdict:
        """Вердикт гейта по обращению; провайдер и модель валидируются как в обычном чате."""
        provider = self._validate_provider(provider_name)
        await self._validate_model(provider, provider_name, model)
        return await triage(
            provider,
            model,
            text,
            samples=samples,
            temperature=self._normalize_temperature(temperature),
            max_repairs=max_repairs,
        )
