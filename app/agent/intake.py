"""Миксин агента: разбор письма-заявки одним запросом или по этапам (День 9 advance)."""

from __future__ import annotations

from ..agent_constants import INTAKE_MONO_MODEL, INTAKE_STAGE_MODELS, INTAKE_TEMPERATURE
from ..staged import IntakeResult, run_intake


class AgentIntakeMixin:
    async def parse_intake(
        self,
        provider_name: str,
        letter: str,
        *,
        mode: str = "staged",
        today: str = "",
        mono_model: str = INTAKE_MONO_MODEL,
        stage_models: dict[str, str] | None = None,
        temperature: float = INTAKE_TEMPERATURE,
    ) -> IntakeResult:
        """Разбирает заявку выбранным режимом; модели всех задействованных этапов валидируются."""
        provider = self._validate_provider(provider_name)
        models = {**INTAKE_STAGE_MODELS, **(stage_models or {})}
        if mode == "mono":
            used = [mono_model]
        elif mode == "staged_rules":
            # Этап 2 считается кодом — модель для него не нужна.
            used = sorted({models["normalize"], models["compose"]})
        else:
            used = sorted({models["normalize"], models["decide"], models["compose"]})
        for model in used:
            await self._validate_model(provider, provider_name, model)
        return await run_intake(
            provider,
            letter,
            mode=mode,
            today=today or None,
            mono_model=mono_model,
            models=models,
            temperature=self._normalize_temperature(temperature),
        )
