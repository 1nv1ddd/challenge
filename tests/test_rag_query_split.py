"""Multi-query разбиение и объединение хитов RAG."""

from __future__ import annotations

import unittest

from app.rag.query_split import rag_subqueries
from app.rag.retrieve import merge_hits_by_max_score


class TestRagQuerySplit(unittest.TestCase):
    def test_single_short_unchanged(self):
        self.assertEqual(rag_subqueries("Что такое PL-203-RET?"), ["Что такое PL-203-RET?"])

    def test_numbered_list_splits(self):
        t = """1. По справочнику PolarLine: PL-203-RET SLA?
2. Что с shard_map при PL-101-RET?
3. Код узла A.42?"""
        out = rag_subqueries(t)
        self.assertGreaterEqual(len(out), 3)
        self.assertTrue(any("PL-203" in x for x in out))
        self.assertTrue(any("shard_map" in x for x in out))

    def test_inline_numbered_same_line(self):
        t = (
            "1. По справочнику PolarLine: для PL-203-RET SLA и email? "
            "2. Что запрещено с shard_map при PL-101-RET?"
        )
        out = rag_subqueries(t)
        self.assertGreaterEqual(len(out), 2)

    def test_readme_triggers_expansion(self):
        out = rag_subqueries("Где по README этого репозитория лежит SQLite-индекс RAG?")
        self.assertTrue(any("chunks.sqlite" in x or "README" in x for x in out))

    def test_merge_prefers_higher_score(self):
        a = [{"chunk_id": "x", "score": 0.5, "source": "f", "text": "t"}]
        b = [{"chunk_id": "x", "score": 0.9, "source": "f", "text": "t"}]
        m = merge_hits_by_max_score([a, b], max_chunks=5)
        self.assertEqual(len(m), 1)
        self.assertAlmostEqual(m[0]["score"], 0.9)


if __name__ == "__main__":
    unittest.main()
