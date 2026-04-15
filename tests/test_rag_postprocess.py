"""День 23: фильтр по similarity и лексический реранк (без сети)."""

from __future__ import annotations

import unittest

from app.rag.post_retrieval import (
    filter_by_min_similarity,
    lexical_overlap_ratio,
    postprocess_hits,
    rerank_hits_lexical,
)
from app.rag.retrieve import KEYWORD_HIT_SCORE


class TestRagPostprocess(unittest.TestCase):
    def test_filter_drops_low_cosine_keeps_keyword(self) -> None:
        hits = [
            {"chunk_id": "a", "score": 0.2, "text": "foo", "source": "x.md"},
            {"chunk_id": "b", "score": KEYWORD_HIT_SCORE, "text": "PL-FOO bar", "source": "y.md"},
        ]
        out = filter_by_min_similarity(hits, 0.35)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["chunk_id"], "b")

    def test_lexical_overlap(self) -> None:
        r = lexical_overlap_ratio("hello world test", "hello world and more")
        self.assertGreater(r, 0.5)

    def test_rerank_changes_order(self) -> None:
        hits = [
            {"chunk_id": "1", "score": 0.9, "text": "unrelated zzz", "source": "a.md"},
            {"chunk_id": "2", "score": 0.5, "text": "PolarLine retention audit_bus", "source": "b.md"},
        ]
        out = rerank_hits_lexical(hits, "PolarLine audit_bus retention", alpha=0.5)
        self.assertEqual(out[0]["chunk_id"], "2")

    def test_postprocess_top_k(self) -> None:
        hits = [{"chunk_id": str(i), "score": 0.1 + i * 0.01, "text": "t", "source": "s"} for i in range(10)]
        final, meta = postprocess_hits(
            hits,
            "query",
            top_k_final=3,
            min_similarity=0.0,
            rerank_mode="none",
        )
        self.assertEqual(len(final), 3)
        self.assertEqual(meta["hits_prefilter"], 10)


if __name__ == "__main__":
    unittest.main()
