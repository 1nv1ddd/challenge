"""Redundancy: несколько независимых прогонов одного запроса и голосование по ним."""

from __future__ import annotations

import asyncio
from collections import Counter

from ..agent_constants import (
    TRIAGE_ACTION_SAFETY_RANK,
    TRIAGE_MAX_REPAIRS,
    TRIAGE_SAMPLES,
    TRIAGE_TEMPERATURE,
)
from ..providers import AIProvider
from .constraints import check_invariants, parse_decision
from .inference import complete
from .prompts import repair_messages, triage_messages
from .schema import Decision, SampleResult


async def _one_sample(
    provider: AIProvider,
    model: str,
    text: str,
    index: int,
    *,
    temperature: float,
    max_repairs: int,
) -> SampleResult:
    """Прогон + до max_repairs повторных инференсов, если формат/инварианты не прошли."""
    messages = triage_messages(text)
    sample = SampleResult(index=index, decision=None, violations=[], raw="", calls=0)
    attempt = 0
    while True:
        call = await complete(provider, model, messages, temperature)
        sample.calls += 1
        sample.raw = call.text
        sample.time_ms += call.time_ms
        sample.prompt_tokens += call.prompt_tokens
        sample.completion_tokens += call.completion_tokens

        try:
            decision = parse_decision(call.text)
        except ValueError as exc:
            violations = [str(exc)]
            decision = None
        else:
            violations = check_invariants(decision, text)

        if not violations:
            sample.decision = decision
            sample.violations = []
            return sample

        sample.decision = None
        sample.violations = violations
        if not sample.first_violations:
            sample.first_violations = violations
        if attempt >= max_repairs:
            return sample
        attempt += 1
        sample.repaired = True
        messages = repair_messages(text, call.text, violations)


async def sample_decisions(
    provider: AIProvider,
    model: str,
    text: str,
    *,
    samples: int = TRIAGE_SAMPLES,
    temperature: float = TRIAGE_TEMPERATURE,
    max_repairs: int = TRIAGE_MAX_REPAIRS,
) -> list[SampleResult]:
    """N параллельных прогонов одного и того же обращения."""
    tasks = [
        _one_sample(
            provider, model, text, i, temperature=temperature, max_repairs=max_repairs
        )
        for i in range(samples)
    ]
    return list(await asyncio.gather(*tasks))


def consensus(samples: list[SampleResult]) -> tuple[Decision | None, float, dict[str, int]]:
    """(решение большинства, доля согласия, распределение голосов) по валидным сэмплам."""
    valid = [s for s in samples if s.decision is not None]
    if not valid:
        return None, 0.0, {}
    votes = Counter(s.decision.vote_key for s in valid)
    top = max(votes.values())
    # При равенстве голосов берём самый безопасный вариант — escalate раньше, чем close.
    tied = sorted(
        (key for key, count in votes.items() if count == top),
        key=lambda key: TRIAGE_ACTION_SAFETY_RANK.get(key.split("/", 1)[1], 99),
    )
    winner_key = tied[0]
    winner = next(s.decision for s in valid if s.decision.vote_key == winner_key)
    agreement = round(top / len(valid), 2)
    return winner, agreement, dict(votes)
