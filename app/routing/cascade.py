"""Каскад моделей: дешёвая отвечает первой, сильная подключается только при недостатке уверенности."""

from __future__ import annotations

import asyncio
import time

import httpx

from ..agent_constants import (
    ROUTING_CONSISTENCY_SAMPLES,
    ROUTING_CONSISTENCY_TEMPERATURE,
    ROUTING_LARGE_MODEL,
    ROUTING_SMALL_MODEL,
    ROUTING_TEMPERATURE,
)
from ..confidence.inference import complete  # один нестримовый вызов с usage — уже есть в проекте
from ..providers import AIProvider
from .preroute import classify_question
from .pricing import cost_rub_model
from .prompts import answer_messages
from .schema import RouteAttempt, RouteResult
from .signals import assess_answer, split_confidence


async def _call_tier(
    provider: AIProvider, model: str, tier: str, question: str, temperature: float
) -> RouteAttempt:
    """Вызов одного тира с ценой по прайсу именно этой модели."""
    call = await complete(provider, model, answer_messages(question), temperature)
    return RouteAttempt(
        tier=tier,
        model=model,
        answer=call.text,
        time_ms=call.time_ms,
        prompt_tokens=call.prompt_tokens,
        completion_tokens=call.completion_tokens,
        cost_rub=cost_rub_model(model, call.prompt_tokens, call.completion_tokens),
    )


def _metrics(attempts: list[RouteAttempt], wall_ms: int, escalated: bool) -> dict:
    prompt_tokens = sum(a.prompt_tokens for a in attempts)
    completion_tokens = sum(a.completion_tokens for a in attempts)
    # Вызов дешёвой модели, ответ которой не пригодился — честная цена промаха маршрутизации.
    wasted = sum(a.cost_rub for a in attempts[:-1]) if escalated and len(attempts) > 1 else 0.0
    return {
        "llm_calls": len(attempts),
        "time_ms": wall_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_rub": round(sum(a.cost_rub for a in attempts), 4),
        "wasted_rub": round(wasted, 4),
    }


async def route_answer(
    provider: AIProvider,
    question: str,
    *,
    small_model: str = ROUTING_SMALL_MODEL,
    large_model: str = ROUTING_LARGE_MODEL,
    temperature: float = ROUTING_TEMPERATURE,
    consistency_samples: int = ROUTING_CONSISTENCY_SAMPLES,
    consistency_temperature: float = ROUTING_CONSISTENCY_TEMPERATURE,
) -> RouteResult:
    """Ответ на вопрос с маршрутизацией: pre-routing по тексту, затем эскалация по уверенности."""
    text = (question or "").strip()
    if not text:
        raise ValueError("Пустой запрос — нечего маршрутизировать.")

    t_start = time.monotonic()
    preroute = classify_question(text)
    attempts: list[RouteAttempt] = []

    if preroute.tier == "large":
        # Запрос заведомо сложный: не тратим вызов дешёвой модели впустую.
        attempt = await _call_tier(provider, large_model, "large", text, temperature)
        attempts.append(attempt)
        body, _ = split_confidence(attempt.answer)
        reason = "pre-routing: " + ", ".join(preroute.reasons)
        wall_ms = round((time.monotonic() - t_start) * 1000)
        return RouteResult(
            question=text,
            answer=body,
            model=large_model,
            tier="large",
            escalated=True,
            escalation_reason=reason,
            preroute=preroute,
            attempts=attempts,
            metrics={**_metrics(attempts, wall_ms, escalated=False), "path": "large"},
        )

    # Дешёвая модель стоит десятки раз меньше сильной, поэтому спросить её дважды
    # и сравнить ответы дешевле, чем один раз эскалировать зря. Дубли берём горячее
    # основного ответа: на одной температуре выборки совпадают и сигнала не дают.
    samples = max(1, consistency_samples)
    temperatures = [temperature, *[consistency_temperature] * (samples - 1)]
    drafts = await asyncio.gather(
        *(_call_tier(provider, small_model, "small", text, t) for t in temperatures)
    )
    small = drafts[0]
    small.assessment = assess_answer(text, small.answer, peers=[d.answer for d in drafts[1:]])
    attempts.extend(drafts)
    small_body, _ = split_confidence(small.answer)

    if not small.assessment.escalate:
        wall_ms = round((time.monotonic() - t_start) * 1000)
        return RouteResult(
            question=text,
            answer=small_body,
            model=small_model,
            tier="small",
            escalated=False,
            escalation_reason=small.assessment.reason,
            preroute=preroute,
            attempts=attempts,
            metrics={**_metrics(attempts, wall_ms, escalated=False), "path": "small"},
        )

    try:
        large = await _call_tier(provider, large_model, "large", text, temperature)
    except httpx.HTTPError as exc:
        # Сильная модель недоступна — отдаём ответ дешёвой, но честно помечаем, что эскалация не удалась.
        attempts.append(
            RouteAttempt(tier="large", model=large_model, answer="", error=str(exc))
        )
        wall_ms = round((time.monotonic() - t_start) * 1000)
        return RouteResult(
            question=text,
            answer=small_body,
            model=small_model,
            tier="small",
            escalated=False,
            escalation_reason=f"{small.assessment.reason}; эскалация не удалась: {exc}",
            preroute=preroute,
            attempts=attempts,
            metrics={**_metrics(attempts, wall_ms, escalated=False), "path": "small (fallback)"},
        )

    attempts.append(large)
    large_body, _ = split_confidence(large.answer)
    wall_ms = round((time.monotonic() - t_start) * 1000)
    return RouteResult(
        question=text,
        answer=large_body,
        model=large_model,
        tier="large",
        escalated=True,
        escalation_reason=small.assessment.reason,
        preroute=preroute,
        attempts=attempts,
        metrics={**_metrics(attempts, wall_ms, escalated=True), "path": "small→large"},
    )
