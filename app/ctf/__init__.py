"""Пакет арены Дня 15: Оракул с секретным кодом за проходом шлюза, форма сдачи кода."""

from __future__ import annotations

from .oracle import load_secret, oracle_system, verify_code

__all__ = ["oracle_system", "load_secret", "verify_code"]
