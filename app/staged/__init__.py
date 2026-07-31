"""Декомпозиция инференса (День 9 advance): разбор заявки одним запросом или цепочкой этапов."""

from __future__ import annotations

from .monolithic import run_monolithic
from .pipeline import run_intake, run_staged
from .policy import decide_by_rules, policy_text
from .schema import IntakeFields, IntakeResult, StageCall, StageDecision

__all__ = [
    "IntakeFields",
    "IntakeResult",
    "StageCall",
    "StageDecision",
    "decide_by_rules",
    "policy_text",
    "run_intake",
    "run_monolithic",
    "run_staged",
]
