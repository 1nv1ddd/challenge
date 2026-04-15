"""День 23: порог по similarity, топ-K до/после отсечения, лексический реранк."""

from __future__ import annotations

import re
from typing import Any

from .retrieve import KEYWORD_HIT_SCORE

_TOKEN_RE = re.compile(r"[\w\-]+", re.UNICODE)


def is_keyword_boosted_hit(h: dict[str, Any]) -> bool:
    return float(h.get("score", 0)) >= KEYWORD_HIT_SCORE - 1e-6


def filter_by_min_similarity(
    hits: list[dict[str, Any]],
    min_score: float,
    *,
    keep_keyword_hits: bool = True,
) -> list[dict[str, Any]]:
    if min_score <= 0:
        return list(hits)
    out: list[dict[str, Any]] = []
    for h in hits:
        s = float(h["score"])
        if keep_keyword_hits and is_keyword_boosted_hit(h):
            out.append(h)
        elif s >= min_score:
            out.append(h)
    return out


def _tokens(s: str) -> set[str]:
    return {x.lower() for x in _TOKEN_RE.findall(s) if len(x) > 1}


def lexical_overlap_ratio(query: str, chunk_text: str) -> float:
    q = _tokens(query)
    c = _tokens(chunk_text)
    if not q or not c:
        return 0.0
    return len(q & c) / max(1, len(q))


def rerank_hits_lexical(
    hits: list[dict[str, Any]],
    query: str,
    *,
    alpha: float = 0.35,
) -> list[dict[str, Any]]:
    """score' = (1-α)·cosine + α·lexical_overlap; якорные keyword-хиты не принижаются."""
    if not hits:
        return []
    out: list[dict[str, Any]] = []
    for h in hits:
        base = float(h["score"])
        lex = lexical_overlap_ratio(query, str(h.get("text", "")))
        if is_keyword_boosted_hit(h):
            combined = max(base, 0.55 + 0.45 * min(1.0, lex * 1.2))
        else:
            combined = (1.0 - alpha) * base + alpha * min(1.0, lex * 1.15)
        hh = dict(h)
        hh["score"] = combined
        hh["rerank_lexical"] = round(lex, 4)
        out.append(hh)
    out.sort(key=lambda x: float(x["score"]), reverse=True)
    return out


def take_top_k(hits: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    if k <= 0:
        return []
    merged = sorted(hits, key=lambda x: float(x["score"]), reverse=True)
    return merged[:k]


def postprocess_hits(
    hits: list[dict[str, Any]],
    query_for_lexical: str,
    *,
    top_k_final: int,
    min_similarity: float,
    rerank_mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    n_prefilter = len(hits)
    h = list(hits)
    h = filter_by_min_similarity(h, min_similarity)
    n_after_cut = len(h)
    mode = (rerank_mode or "none").strip().lower()
    rerank_applied = False
    if mode == "lexical" and h:
        h = rerank_hits_lexical(h, query_for_lexical)
        rerank_applied = True
    h = take_top_k(h, top_k_final)
    return h, {
        "hits_prefilter": n_prefilter,
        "hits_after_score_filter": n_after_cut,
        "rerank_mode": mode if rerank_applied else "none",
        "rerank_applied": rerank_applied,
    }


def rag_enhancements_enabled(rag_cfg: dict[str, Any] | None) -> bool:
    if not rag_cfg:
        return False
    if rag_cfg.get("query_rewrite"):
        return True
    try:
        if float(rag_cfg.get("min_similarity") or 0) > 0:
            return True
    except (TypeError, ValueError):
        pass
    r = str(rag_cfg.get("rerank") or "none").strip().lower()
    if r not in ("", "none"):
        return True
    if rag_cfg.get("top_k_fetch") is not None:
        return True
    return False
