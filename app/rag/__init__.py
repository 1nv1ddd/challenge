"""RAG: индексация корпуса, эмбеддинги RouterAI, SQLite, поиск."""

from .chunking import chunk_fixed_size, chunk_structural
from .retrieve import retrieve_for_query

__all__ = [
    "chunk_fixed_size",
    "chunk_structural",
    "retrieve_for_query",
]
