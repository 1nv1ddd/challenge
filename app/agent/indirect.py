"""Миксин агента: прогон ловушек непрямой инъекции по наборам слоёв защиты (День 12 advance)."""

from __future__ import annotations

from ..agent_constants import INDIRECT_MODEL, INDIRECT_PRESETS, INDIRECT_TEMPERATURE
from ..indirect import LayerRun, compare_presets, load_cases, select_cases


class AgentIndirectMixin:
    async def run_indirect_cases(
        self,
        provider_name: str,
        *,
        presets: tuple[str, ...] = ("none", "all"),
        ids: tuple[str, ...] = (),
        scenario: str = "",
        source: str = "",
        hiding: str = "",
        model: str = INDIRECT_MODEL,
        temperature: float = INDIRECT_TEMPERATURE,
    ) -> dict[str, LayerRun]:
        """Гоняет отобранные ловушки по каждому пресету защиты."""
        unknown = [p for p in presets if p not in INDIRECT_PRESETS]
        if unknown:
            raise ValueError(
                f"Неизвестные пресеты защиты: {', '.join(unknown)}; "
                f"доступны: {', '.join(INDIRECT_PRESETS)}."
            )
        cases = select_cases(
            load_cases(), ids=ids, scenario=scenario, source=source, hiding=hiding
        )
        if not cases:
            raise ValueError("Под фильтр не попала ни одна ловушка из корпуса.")
        provider = self._validate_provider(provider_name)
        await self._validate_model(provider, provider_name, model)
        return await compare_presets(
            provider,
            cases,
            tuple(presets),
            model=model,
            temperature=self._normalize_temperature(temperature),
        )
