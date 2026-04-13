"""Chunking и метаданные RAG (без сети и эмбеддингов)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.rag.chunking import chunk_fixed_size, chunk_structural  # noqa: E402
from app.rag.pipeline import collect_document_paths  # noqa: E402


class TestRagChunking(unittest.TestCase):
    def test_fixed_has_metadata(self) -> None:
        rows = chunk_fixed_size(
            "a" * 5000,
            source="demo/x.md",
            title="x.md",
            chunk_chars=800,
            overlap=80,
        )
        self.assertGreaterEqual(len(rows), 5)
        for r in rows:
            self.assertEqual(r["strategy"], "fixed")
            self.assertEqual(r["source"], "demo/x.md")
            self.assertEqual(r["title"], "x.md")
            self.assertIsNone(r["section"])
            self.assertTrue(r["chunk_id"])
            self.assertTrue(r["text"])

    def test_structural_markdown_sections(self) -> None:
        md = """# Title\n\nintro\n\n## A\n\nbody a\n\n## B\n\nbody b long\n""" + ("x" * 3000)
        rows = chunk_structural(
            md,
            source="s.md",
            title="s.md",
            is_markdown=True,
            max_section_chars=500,
            subchunk_chars=400,
            subchunk_overlap=50,
        )
        self.assertGreater(len(rows), 2)
        sections = {r.get("section") for r in rows}
        self.assertTrue(any(s and "A" in s for s in sections if s))

    def test_corpus_paths_include_handbook(self) -> None:
        corpus = ROOT / "data" / "rag_corpus"
        if not corpus.is_dir():
            self.skipTest("data/rag_corpus missing")
        paths = collect_document_paths(corpus, extra_files=[])
        self.assertTrue(any("00_polarline_handbook" in p.name for p in paths))


if __name__ == "__main__":
    unittest.main()
