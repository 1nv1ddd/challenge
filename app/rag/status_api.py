"""Ответ GET /api/rag/status без циклических импортов с app.main."""

from __future__ import annotations

from typing import Any

from .index_meta import index_needs_build, rag_index_stats
from .pipeline import default_index_path


def build_rag_status_response() -> dict[str, Any]:
    p = default_index_path()
    if index_needs_build(p):
        return {
            "ok": True,
            "indexed": False,
            "path": str(p),
            "hint": (
                "Индекс пуст или отсутствует. При RAG_AUTO_BUILD=1 и ROUTERAI_API_KEY "
                "он соберётся при старте; иначе: python scripts/build_rag_index.py"
            ),
        }
    return {"ok": True, "indexed": True, "path": str(p), "stats": rag_index_stats(p)}
