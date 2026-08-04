"""Миксин агента: ред-тим промптов — прогон корпуса инъекций по версиям (День 11 advance)."""

from __future__ import annotations

from ..agent_constants import SECURITY_MODEL, SECURITY_PROMPT_VERSIONS, SECURITY_TEMPERATURE
from ..security import RedteamRun, compare_versions, load_attacks, select_attacks


class AgentSecurityMixin:
    async def redteam_prompts(
        self,
        provider_name: str,
        *,
        versions: tuple[str, ...] = SECURITY_PROMPT_VERSIONS,
        ids: tuple[str, ...] = (),
        target: str = "",
        vector: str = "",
        technique: str = "",
        model: str = SECURITY_MODEL,
        temperature: float = SECURITY_TEMPERATURE,
    ) -> dict[str, RedteamRun]:
        """Гоняет отобранные атаки по каждой версии промпта-жертвы."""
        unknown = [v for v in versions if v not in SECURITY_PROMPT_VERSIONS]
        if unknown:
            raise ValueError(
                f"Неизвестные версии промпта: {', '.join(unknown)}; "
                f"доступны: {', '.join(SECURITY_PROMPT_VERSIONS)}."
            )
        attacks = select_attacks(
            load_attacks(), ids=ids, target=target, vector=vector, technique=technique
        )
        if not attacks:
            raise ValueError("Под фильтр не попала ни одна атака из корпуса.")
        provider = self._validate_provider(provider_name)
        await self._validate_model(provider, provider_name, model)
        return await compare_versions(
            provider,
            attacks,
            versions=tuple(versions),
            model=model,
            temperature=self._normalize_temperature(temperature),
        )
