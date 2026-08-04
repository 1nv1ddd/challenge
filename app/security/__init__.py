"""День 11 (advance): prompt injection — корпус атак, промпты-жертвы и ред-тим двух версий."""

from __future__ import annotations

from .corpus import corpus_stats, load_attacks, select_attacks
from .prompts import boundary_rules, canary_line, sanitize_untrusted, wrap_document, wrap_user_input
from .redteam import compare_versions, fixed_by_hardening, run_attack, run_corpus
from .schema import Attack, AttackVerdict, RedteamRun
from .verdict import canary_leak, is_refusal, judge, prompt_leak
from .victims import bank_system_prompt, victim_messages, victim_secret_text, victim_system_message

__all__ = [
    "Attack",
    "AttackVerdict",
    "RedteamRun",
    "bank_system_prompt",
    "boundary_rules",
    "canary_leak",
    "canary_line",
    "compare_versions",
    "corpus_stats",
    "fixed_by_hardening",
    "is_refusal",
    "judge",
    "load_attacks",
    "prompt_leak",
    "run_attack",
    "run_corpus",
    "sanitize_untrusted",
    "select_attacks",
    "victim_messages",
    "victim_secret_text",
    "victim_system_message",
    "wrap_document",
    "wrap_user_input",
]
