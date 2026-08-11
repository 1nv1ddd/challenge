"""Двухуровневый инференс: micro-model решает сама, большую модель зовём только при UNSURE."""

from __future__ import annotations

import time
from pathlib import Path

from ..agent_constants import (
    MICRO_BANK_PATH,
    MICRO_DEFAULT_STRATEGY,
    MICRO_EMBED_MODEL,
    MICRO_FALLBACK_LABEL,
    MICRO_KNN_K,
    MICRO_LLM_MODEL,
    MICRO_MAX_REPAIRS,
    MICRO_STRATEGIES,
    MICRO_TEMPERATURE,
)
from ..providers import AIProvider
from ..staged.schema import StageCall
from .backends import get_backend
from .gate import judge
from .llm import classify_with_llm
from .schema import IntentResult, LabelAnswer, MicroVerdict


async def run_micro(
    text: str,
    backend: str,
    *,
    k: int = MICRO_KNN_K,
    bank_path: str | Path = MICRO_BANK_PATH,
    embed_model: str = MICRO_EMBED_MODEL,
) -> MicroVerdict:
    """Уровень 1 целиком: соседи из банка → метка со score и статусом OK/UNSURE."""
    t_start = time.monotonic()
    impl = await get_backend(backend, bank_path=bank_path, embed_model=embed_model)
    neighbors = await impl.neighbors(text, k)
    return judge(backend, text, neighbors, time_ms=round((time.monotonic() - t_start) * 1000))


def _metrics(
    strategy: str,
    micro: MicroVerdict | None,
    llm: StageCall | None,
    wall_ms: int,
) -> dict:
    """Сводка запроса: главное здесь — сколько вызовов большой модели он стоил (0 или 1+)."""
    return {
        "strategy": strategy,
        "backend": micro.backend if micro else None,
        "micro_calls": 1 if micro else 0,
        "micro_score": micro.score if micro else None,
        "micro_status": micro.status if micro else None,
        "escalate_reason": micro.escalate_reason if micro else None,
        "escalated": llm is not None,
        "llm_calls": llm.calls if llm else 0,
        "repair_calls": max(0, llm.calls - 1) if llm else 0,
        "llm_model": llm.model if llm else None,
        "micro_ms": micro.time_ms if micro else 0,
        "llm_ms": llm.time_ms if llm else 0,
        "time_ms": wall_ms,
        "prompt_tokens": llm.prompt_tokens if llm else 0,
        "completion_tokens": llm.completion_tokens if llm else 0,
        "cost_rub": round(llm.cost_rub if llm else 0.0, 4),
    }


async def classify_intent(
    provider: AIProvider | None,
    text: str,
    *,
    strategy: str = MICRO_DEFAULT_STRATEGY,
    llm_model: str = MICRO_LLM_MODEL,
    temperature: float = MICRO_TEMPERATURE,
    max_repairs: int = MICRO_MAX_REPAIRS,
    k: int = MICRO_KNN_K,
    bank_path: str | Path = MICRO_BANK_PATH,
    embed_model: str = MICRO_EMBED_MODEL,
) -> IntentResult:
    """Классификация обращения выбранной стратегией: micro-model, большая модель или связка."""
    message = (text or "").strip()
    if not message:
        raise ValueError("Пустое обращение — нечего классифицировать.")
    if strategy not in MICRO_STRATEGIES:
        raise ValueError(
            f"Неизвестная стратегия «{strategy}»; доступны: {', '.join(MICRO_STRATEGIES)}."
        )
    backend, allow_fallback = MICRO_STRATEGIES[strategy]
    t_start = time.monotonic()

    micro: MicroVerdict | None = None
    if backend is not None:
        micro = await run_micro(
            message, backend, k=k, bank_path=bank_path, embed_model=embed_model
        )
        if micro.ok or not allow_fallback:
            # Уровень 2 не нужен: либо micro-model уверена, либо ей запрещено эскалировать.
            return IntentResult(
                strategy=strategy,
                text=message,
                label=micro.label,
                source="micro",
                micro=micro,
                llm=None,
                llm_answer=None,
                metrics=_metrics(
                    strategy, micro, None, round((time.monotonic() - t_start) * 1000)
                ),
            )

    if provider is None:
        raise ValueError("Для стратегии с fallback нужен провайдер большой модели.")
    answer, call = await classify_with_llm(
        provider,
        message,
        model=llm_model,
        temperature=temperature,
        max_repairs=max_repairs,
    )
    label, source = _final_label(answer, micro)
    return IntentResult(
        strategy=strategy,
        text=message,
        label=label,
        source=source,
        micro=micro,
        llm=call,
        llm_answer=answer,
        metrics=_metrics(strategy, micro, call, round((time.monotonic() - t_start) * 1000)),
    )


def _final_label(answer: LabelAnswer | None, micro: MicroVerdict | None) -> tuple[str, str]:
    """Если уровень 2 не отдал разбираемый формат — берём метку micro-model, а не выдумываем."""
    if answer is not None:
        return answer.label, "llm"
    if micro is not None:
        return micro.label, "micro_after_llm_error"
    return MICRO_FALLBACK_LABEL, "failed"
