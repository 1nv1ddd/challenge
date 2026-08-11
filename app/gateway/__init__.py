"""День 13 (advance): LLM Gateway — прокси с input/output guard, rate limit, аудитом и ценой."""

from __future__ import annotations

from .audit import audit_path, audit_stats, build_record, log_request, read_records
from .corpus import load_cases, run_case, run_corpus, select_cases
from .cost import Usage, estimate_tokens, usage_of
from .detectors import guard_input, mask_text, scan_text
from .output_guard import guard_output, validate_output
from .pipeline import proxy_chat
from .prompts import gateway_messages, system_message
from .ratelimit import RateLimiter, limiter
from .schema import (
    CaseOutcome,
    CorpusRun,
    GatewayCase,
    GatewayResult,
    InputVerdict,
    OutputFinding,
    OutputVerdict,
    RateDecision,
    SecretFinding,
)

__all__ = [
    "CaseOutcome",
    "CorpusRun",
    "GatewayCase",
    "GatewayResult",
    "InputVerdict",
    "OutputFinding",
    "OutputVerdict",
    "RateDecision",
    "RateLimiter",
    "SecretFinding",
    "Usage",
    "audit_path",
    "audit_stats",
    "build_record",
    "estimate_tokens",
    "gateway_messages",
    "guard_input",
    "guard_output",
    "limiter",
    "load_cases",
    "log_request",
    "mask_text",
    "proxy_chat",
    "read_records",
    "run_case",
    "run_corpus",
    "scan_text",
    "select_cases",
    "system_message",
    "usage_of",
    "validate_output",
]
