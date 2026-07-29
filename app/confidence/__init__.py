"""День 7 (advance): триаж обращений поддержки с явной оценкой уверенности инференса."""

from __future__ import annotations

from .constraints import check_invariants, parse_decision
from .pipeline import triage
from .schema import Decision, SampleResult, SelfCheck, Verdict

__all__ = [
    "Decision",
    "SampleResult",
    "SelfCheck",
    "Verdict",
    "check_invariants",
    "parse_decision",
    "triage",
]
