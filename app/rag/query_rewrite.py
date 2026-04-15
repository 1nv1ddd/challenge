"""День 23: детерминированное «переписывание» запроса перед эмбеддингом (без второго вызова LLM)."""

from __future__ import annotations

import re
from typing import Any

from .anchors import rag_keyword_needles

# Шумовые вежливые префиксы — убираем отдельной строкой, если после них есть суть.
_POLITE_LINE = re.compile(
    r"^(привет|здравствуй|добрый\s+(день|вечер|утро)|hi|hello)[!.,\s]*$",
    re.IGNORECASE,
)


def heuristic_query_rewrite(user_text: str) -> tuple[str, dict[str, Any]]:
    """
    Нормализует текст и добавляет компактную строку якорей (PL-*, NODE-Q-*, …),
    чтобы эмбеддинг и гибридный слой лучше цеплялись за корпус.
    """
    t = (user_text or "").strip()
    if not t:
        return "", {"query_rewrite": False}

    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if len(lines) >= 2 and _POLITE_LINE.match(lines[0]):
        lines = lines[1:]
        t = "\n".join(lines).strip()
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    if not t:
        return "", {"query_rewrite": False}

    needles = rag_keyword_needles(user_text)
    if not needles:
        changed = t != (user_text or "").strip()
        return t, {"query_rewrite": changed, "rewrite_needles_added": 0}

    extra = " ".join(sorted(needles)[:14])
    augmented = f"{t}\n\n[retrieval_terms] {extra}"
    return augmented, {"query_rewrite": True, "rewrite_needles_added": len(needles)}
