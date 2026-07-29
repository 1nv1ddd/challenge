"""Гейт триажа: redundancy + constraints + self-check → confidence и статус OK/UNSURE/FAIL."""

from __future__ import annotations

import time

from ..agent_constants import (
    TRIAGE_CONFIDENCE_OK,
    TRIAGE_CONFIDENCE_UNSURE,
    TRIAGE_MAX_REPAIRS,
    TRIAGE_RISKY_ACTIONS,
    TRIAGE_SAMPLES,
    TRIAGE_SELFCHECK_BELOW,
    TRIAGE_SELFCHECK_BONUS,
    TRIAGE_SELFCHECK_PENALTY,
    TRIAGE_TEMPERATURE,
)
from ..providers import AIProvider
from .constraints import check_invariants
from .inference import cost_rub
from .redundancy import consensus, sample_decisions
from .schema import SampleResult, SelfCheck, Verdict
from .selfcheck import run_selfcheck

# Единственное действие, которым система отвечает, когда автоматике доверять нельзя.
_HUMAN_FALLBACK = "escalate"


def _metrics(samples: list[SampleResult], check: SelfCheck | None, wall_ms: int) -> dict:
    llm_calls = sum(s.calls for s in samples) + (1 if check else 0)
    repair_calls = sum(max(0, s.calls - 1) for s in samples)
    prompt_tokens = sum(s.prompt_tokens for s in samples) + (check.prompt_tokens if check else 0)
    completion_tokens = sum(s.completion_tokens for s in samples) + (
        check.completion_tokens if check else 0
    )
    return {
        "llm_calls": llm_calls,
        "repair_calls": repair_calls,
        "selfcheck_calls": 1 if check else 0,
        "time_ms": wall_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_rub": cost_rub(prompt_tokens, completion_tokens),
    }


def _score(agreement: float, valid_ratio: float, check: SelfCheck | None) -> float:
    """Confidence = согласие сэмплов + доля прошедших проверки, поправленная self-check'ом."""
    base = 0.7 * agreement + 0.3 * valid_ratio
    if check is not None:
        if check.verdict == "confirm":
            base += TRIAGE_SELFCHECK_BONUS
        elif check.verdict == "reject":
            base -= TRIAGE_SELFCHECK_PENALTY
    return round(max(0.0, min(1.0, base)), 2)


def _status(confidence: float, violations: list[str]) -> str:
    if violations:
        return "FAIL"
    if confidence >= TRIAGE_CONFIDENCE_OK:
        return "OK"
    if confidence >= TRIAGE_CONFIDENCE_UNSURE:
        return "UNSURE"
    return "FAIL"


async def triage(
    provider: AIProvider,
    model: str,
    text: str,
    *,
    samples: int = TRIAGE_SAMPLES,
    temperature: float = TRIAGE_TEMPERATURE,
    max_repairs: int = TRIAGE_MAX_REPAIRS,
) -> Verdict:
    """Триаж обращения с явной оценкой уверенности; автоматика применяется только при OK."""
    t_start = time.monotonic()
    cleaned = (text or "").strip()
    if not cleaned:
        return Verdict(
            status="FAIL",
            confidence=0.0,
            decision=None,
            applied_action=_HUMAN_FALLBACK,
            agreement=0.0,
            votes={},
            violations=["формат: пустое обращение — нечего триажить"],
            selfcheck=None,
            samples=[],
            metrics=_metrics([], None, 0),
        )

    results = await sample_decisions(
        provider,
        model,
        cleaned,
        samples=samples,
        temperature=temperature,
        max_repairs=max_repairs,
    )
    decision, agreement, votes = consensus(results)
    valid_ratio = round(len([s for s in results if s.decision is not None]) / len(results), 2)

    if decision is None:
        # Ни один сэмпл не прошёл формат/инварианты даже после ремонта.
        violations = sorted({v for s in results for v in s.violations})
        wall_ms = round((time.monotonic() - t_start) * 1000)
        return Verdict(
            status="FAIL",
            confidence=0.0,
            decision=None,
            applied_action=_HUMAN_FALLBACK,
            agreement=0.0,
            votes={},
            violations=violations,
            selfcheck=None,
            samples=results,
            metrics=_metrics(results, None, wall_ms),
        )

    violations = check_invariants(decision, cleaned)
    pre_confidence = _score(agreement, valid_ratio, None)
    needs_selfcheck = (
        pre_confidence < TRIAGE_SELFCHECK_BELOW or decision.action in TRIAGE_RISKY_ACTIONS
    )
    check = (
        await run_selfcheck(provider, model, cleaned, decision) if needs_selfcheck else None
    )
    confidence = _score(agreement, valid_ratio, check)
    status = _status(confidence, violations)
    wall_ms = round((time.monotonic() - t_start) * 1000)
    return Verdict(
        status=status,
        confidence=confidence,
        decision=decision,
        applied_action=decision.action if status == "OK" else _HUMAN_FALLBACK,
        agreement=agreement,
        votes=votes,
        violations=violations,
        selfcheck=check,
        samples=results,
        metrics=_metrics(results, check, wall_ms),
    )
