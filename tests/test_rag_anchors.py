"""Якоря для keyword-RAG."""

from __future__ import annotations

import unittest

from app.rag.anchors import rag_keyword_needles


class TestRagAnchors(unittest.TestCase):
    def test_pl_and_node(self):
        t = "PL-203-RET SLA и NODE-Q-042 в A.42?"
        n = rag_keyword_needles(t)
        self.assertTrue(any("PL-203-RET" in x or "pl-203-ret" in x.lower() for x in n))
        self.assertTrue(any("NODE-Q-042" in x for x in n))

    def test_mcp_and_eval(self):
        t = "_MCP_MAX_STEPS в agent.py и 22_eval_knowledge REL-DAY22"
        n = rag_keyword_needles(t)
        low = [x.lower() for x in n]
        self.assertIn("_mcp_max_steps", low)
        self.assertTrue(any("22_eval" in x for x in low))
        self.assertTrue(any("rel-day22" in x.lower() for x in n))
        self.assertTrue(any("polareval kit" in x.lower() for x in n))


if __name__ == "__main__":
    unittest.main()
