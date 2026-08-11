"""Миксин агента: прогон execution loop с security step по задачам корпуса (День 14 advance)."""

from __future__ import annotations

from ..agent_constants import LOOP_GEN_MODEL, LOOP_MAX_ATTEMPTS, LOOP_REVIEW_MODEL
from ..loop import LoopRun, load_tasks, run_tasks, select_tasks


class AgentLoopMixin:
    async def run_execution_loop(
        self,
        provider_name: str,
        *,
        ids: tuple[str, ...] = (),
        gen_model: str = LOOP_GEN_MODEL,
        review_model: str = LOOP_REVIEW_MODEL,
        max_attempts: int = LOOP_MAX_ATTEMPTS,
    ) -> list[LoopRun]:
        """Гоняет отобранные задачи через цикл: генерация, проверки, security review, «коммит»."""
        tasks = select_tasks(load_tasks(), ids)
        if not tasks:
            raise ValueError("Под фильтр не попала ни одна задача корпуса.")
        provider = self._validate_provider(provider_name)
        for model in {gen_model, review_model}:
            await self._validate_model(provider, provider_name, model)
        return await run_tasks(
            provider,
            tasks,
            gen_model=gen_model,
            review_model=review_model,
            max_attempts=max_attempts,
        )
