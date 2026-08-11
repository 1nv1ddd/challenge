"""День 12 (advance): indirect prompt injection — ловушки во внешнем контенте и три слоя защиты."""

from __future__ import annotations

from .agents import agent_messages, system_message
from .corpus import corpus_stats, load_cases, select_cases
from .guard import apply_guard, find_urls, host_allowed, host_of, validate_output
from .hiding import decode_zero_width, hide_payload, visible_text
from .pipeline import (
    build_document,
    compare_presets,
    layer_effect,
    run_case,
    run_preset,
    visible_part,
)
from .sanitize import sanitize_document
from .schema import GuardFinding, IndirectCase, IndirectResult, LayerRun, SanitizeReport

__all__ = [
    "GuardFinding",
    "IndirectCase",
    "IndirectResult",
    "LayerRun",
    "SanitizeReport",
    "agent_messages",
    "apply_guard",
    "build_document",
    "compare_presets",
    "corpus_stats",
    "decode_zero_width",
    "find_urls",
    "hide_payload",
    "host_allowed",
    "host_of",
    "layer_effect",
    "load_cases",
    "run_case",
    "run_preset",
    "sanitize_document",
    "select_cases",
    "system_message",
    "validate_output",
    "visible_part",
    "visible_text",
]
