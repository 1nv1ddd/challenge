"""Поиск top-k по косинусной близости."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Awaitable, Callable

import numpy as np

from .chunking import ChunkRecord
from .store import load_matrix_for_strategy


def _norm_rows(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return m / n


def search_cosine(
    query_vec: list[float],
    meta: list[ChunkRecord],
    matrix: np.ndarray,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    if not meta or matrix.size == 0:
        return []
    q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
    qn = _norm_rows(q)
    mn = _norm_rows(matrix)
    sim = (mn @ qn.T).ravel()
    k = min(top_k, len(meta))
    idx = np.argpartition(-sim, kth=k - 1)[:k]
    idx = idx[np.argsort(-sim[idx])]
    out: list[dict[str, Any]] = []
    for i in idx:
        j = int(i)
        m = meta[j]
        out.append(
            {
                "score": float(sim[j]),
                "chunk_id": m["chunk_id"],
                "source": m["source"],
                "title": m["title"],
                "section": m["section"],
                "text": m["text"][:4000],
                "strategy": m["strategy"],
            }
        )
    return out


EmbedBatchFn = Callable[[list[str]], Awaitable[list[list[float]]]]


async def retrieve_for_query(
    index_path: Path,
    query: str,
    strategy: str,
    embed_fn: EmbedBatchFn,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    qv = (await embed_fn([query]))[0]
    meta, mat = load_matrix_for_strategy(index_path, strategy)
    return search_cosine(qv, meta, mat, top_k=top_k)
