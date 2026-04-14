"""Поиск top-k по косинусной близости."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Awaitable, Callable

import numpy as np

from .anchors import rag_keyword_needles
from .chunking import ChunkRecord
from .store import fetch_chunks_by_substrings, load_matrix_for_strategy

KEYWORD_HIT_SCORE = 0.987


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


def merge_hits_by_max_score(
    hit_lists: list[list[dict[str, Any]]],
    *,
    max_chunks: int,
) -> list[dict[str, Any]]:
    """Объединить результаты нескольких поисков: один chunk_id — лучший score."""
    best: dict[str, dict[str, Any]] = {}
    for hits in hit_lists:
        for h in hits:
            cid = str(h["chunk_id"])
            prev = best.get(cid)
            if prev is None or float(h["score"]) > float(prev["score"]):
                best[cid] = dict(h)
    merged = sorted(best.values(), key=lambda x: float(x["score"]), reverse=True)
    return merged[:max_chunks]


def multi_search_merge(
    query_vecs: list[list[float]],
    meta: list[ChunkRecord],
    matrix: np.ndarray,
    *,
    per_k: int,
    max_chunks: int,
) -> list[dict[str, Any]]:
    lists = [search_cosine(qv, meta, matrix, top_k=per_k) for qv in query_vecs]
    return merge_hits_by_max_score(lists, max_chunks=max_chunks)


def augment_hits_with_keyword_match(
    index_path: Path,
    strategy: str,
    user_text: str,
    semantic_hits: list[dict[str, Any]],
    *,
    max_total: int,
    per_needle_limit: int = 5,
    max_keyword_chunks: int = 12,
) -> list[dict[str, Any]]:
    """
    Гибридный RAG: чанки, где буквально встречаются коды PL-*, NODE-Q-*, константы и т.д.,
    поднимаются в топ даже при низкой косинусной близости общего запроса.
    """
    needles = rag_keyword_needles(user_text)
    if not needles:
        return semantic_hits[:max_total]
    rows = fetch_chunks_by_substrings(
        index_path,
        strategy,
        needles,
        per_needle_limit=per_needle_limit,
        max_total=max_keyword_chunks,
    )
    kw_hits: list[dict[str, Any]] = [{**r, "score": KEYWORD_HIT_SCORE} for r in rows]
    return merge_hits_by_max_score([semantic_hits, kw_hits], max_chunks=max_total)


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
