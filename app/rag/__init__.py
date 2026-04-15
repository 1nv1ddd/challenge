"""RAG: индексация корпуса, эмбеддинги RouterAI, SQLite, поиск."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .chunking import chunk_fixed_size, chunk_structural
    from .retrieve import retrieve_for_query

__all__ = [
    "chunk_fixed_size",
    "chunk_structural",
    "retrieve_for_query",
]


def __getattr__(name: str) -> Any:
    if name == "chunk_fixed_size":
        from .chunking import chunk_fixed_size as fn

        return fn
    if name == "chunk_structural":
        from .chunking import chunk_structural as fn

        return fn
    if name == "retrieve_for_query":
        from .retrieve import retrieve_for_query as fn

        return fn
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
