"""Явные текстовые якоря из запроса пользователя для keyword-поиска по чанкам (гибридный RAG)."""

from __future__ import annotations

import re


def rag_keyword_needles(user_text: str, *, max_needles: int = 20) -> list[str]:
    """
    Извлечь подстроки для SQL LIKE по индексу. Дополняет семантический поиск:
    коды PL-*, NODE-Q-*, константы, имена файлов и устойчивые термины из корпуса.
    """
    if not (user_text or "").strip():
        return []
    t = user_text
    low = t.lower()
    needles: set[str] = set()

    for m in re.finditer(r"(?i)PL-\d{3}-RET", t):
        needles.add(m.group(0))

    for m in re.finditer(r"(?i)NODE-Q-\d+", t):
        needles.add(m.group(0))

    if "_mcp_max_steps" in low:
        needles.add("_MCP_MAX_STEPS")

    if "22_eval" in low:
        needles.add("22_eval")
        # Вопросы «что в 22_eval…» часто не содержат сами строки REL-* / PolarEval — подтягиваем типичные якоря файла.
        needles.add("REL-DAY22")
        needles.add("PolarEval Kit")
    if "rel-day22" in low:
        needles.add("REL-DAY22")

    if "polareval" in low:
        needles.add("PolarEval")
        needles.add("PolarEval Kit")

    if "chunks.sqlite" in low:
        needles.add("chunks.sqlite")

    if "feature_flag:retention_v1" in low or (
        "retention_v1" in low and "feature" in low
    ):
        needles.add("feature_flag:retention_v1")

    if "shard_map" in low:
        needles.add("shard_map")

    if "audit_bus" in low:
        needles.add("audit_bus")

    if "oncall-vol" in low:
        for m in re.finditer(r"oncall-vol\d+@[^\s`]+", low):
            needles.add(m.group(0))

    if re.search(r"(?i)\bA\.42\b", t) or "a.42" in low.replace(" ", ""):
        needles.add("A.42")
        needles.add("NODE-Q-042")

    if ("расшифров" in low or "аббревиат" in low or "означа" in low) and "rag" in low:
        needles.add("Retrieval-Augmented Generation")

    cleaned: list[str] = []
    seen_lower: set[str] = set()
    for n in needles:
        s = (n or "").strip()
        if len(s) < 4:
            continue
        k = s.lower()
        if k in seen_lower:
            continue
        seen_lower.add(k)
        cleaned.append(s)
        if len(cleaned) >= max_needles:
            break
    return cleaned
