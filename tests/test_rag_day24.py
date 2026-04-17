"""День 24: порог «не знаю», обязательный формат в промпте."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from app.rag.day24 import (
    answer_min_score_from_cfg,
    build_day24_appendix_markdown,
    combined_insufficient_evidence,
    day24_output_format_block,
    explicit_anchors,
    filter_hits_by_anchors,
    insufficient_evidence,
    max_hit_score,
    merge_hits_for_appendix,
    refusal_system_message,
    unmatched_anchors,
)
from app.rag.retrieve import KEYWORD_HIT_SCORE


class TestDay24Grounding(unittest.TestCase):
    def test_insufficient_empty(self) -> None:
        self.assertTrue(insufficient_evidence([], 0.2))

    def test_insufficient_low_cosine(self) -> None:
        hits = [{"chunk_id": "a", "score": 0.1, "text": "x", "source": "f.md"}]
        self.assertTrue(insufficient_evidence(hits, 0.25))

    def test_sufficient_cosine(self) -> None:
        hits = [{"chunk_id": "a", "score": 0.5, "text": "x", "source": "f.md"}]
        self.assertFalse(insufficient_evidence(hits, 0.25))

    def test_keyword_bypasses_low_cosine(self) -> None:
        hits = [{"chunk_id": "a", "score": KEYWORD_HIT_SCORE, "text": "PL-1", "source": "f.md"}]
        self.assertFalse(insufficient_evidence(hits, 0.99))

    def test_compare_both_weak(self) -> None:
        a = [{"chunk_id": "1", "score": 0.05, "source": "x", "text": "t"}]
        b = [{"chunk_id": "2", "score": 0.06, "source": "y", "text": "t"}]
        self.assertTrue(combined_insufficient_evidence([a, b], 0.2))

    def test_compare_one_strong(self) -> None:
        a = [{"chunk_id": "1", "score": 0.05, "source": "x", "text": "t"}]
        b = [{"chunk_id": "2", "score": 0.8, "source": "y", "text": "t"}]
        self.assertFalse(combined_insufficient_evidence([a, b], 0.2))

    def test_refusal_message_structure(self) -> None:
        t = refusal_system_message(best_score=0.1, threshold=0.25, hit_count=3)
        self.assertIn("## Ответ", t)
        self.assertIn("## Источники", t)
        self.assertIn("## Цитаты", t)

    def test_format_block_has_markdown_headers(self) -> None:
        b = day24_output_format_block(compare_mode=False)
        self.assertIn("## Ответ", b)
        self.assertIn("Источники", b)
        self.assertIn("Цитаты", b)

    def test_appendix_markdown_headers(self) -> None:
        hits = [
            {
                "chunk_id": "c1",
                "source": "a.md",
                "section": "S",
                "score": 0.9,
                "text": "Hello world fragment here.",
            }
        ]
        md = build_day24_appendix_markdown(hits)
        self.assertIn("## Источники", md)
        self.assertIn("## Цитаты", md)
        self.assertIn("c1", md)
        self.assertIn("Hello world", md)

    def test_merge_hits_prefers_higher_score(self) -> None:
        a = [{"chunk_id": "x", "source": "f.md", "score": 0.1, "text": "a"}]
        b = [{"chunk_id": "x", "source": "f.md", "score": 0.8, "text": "b"}]
        m = merge_hits_for_appendix(a, b)
        self.assertEqual(len(m), 1)
        self.assertEqual(m[0]["text"], "b")

    def test_answer_min_from_cfg(self) -> None:
        self.assertEqual(answer_min_score_from_cfg({"answer_min_score": 0.4}), 0.4)
        self.assertEqual(answer_min_score_from_cfg({}), answer_min_score_from_cfg(None))

    def test_day24_questions_json(self) -> None:
        p = Path(__file__).resolve().parent.parent / "data" / "rag_eval" / "day24_questions.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        self.assertEqual(len(data), 10)
        for row in data:
            self.assertIn("id", row)
            self.assertIn("question", row)
            self.assertIn("checks", row)

    def test_max_hit_score(self) -> None:
        self.assertEqual(max_hit_score([]), 0.0)
        self.assertEqual(
            max_hit_score(
                [
                    {"score": 0.2},
                    {"score": 0.7},
                ]
            ),
            0.7,
        )

    def test_explicit_anchors_extraction(self) -> None:
        self.assertEqual(explicit_anchors("Что такое PL-999-RET?"), ["PL-999-RET"])
        self.assertEqual(
            explicit_anchors("сравни PL-101-RET и NODE-Q-042"),
            ["PL-101-RET", "NODE-Q-042"],
        )
        self.assertEqual(explicit_anchors("общий вопрос без кодов"), [])

    def test_unmatched_anchor_triggers_refusal(self) -> None:
        hits = [
            {"chunk_id": "a", "score": 0.99, "text": "PL-101-RET SLA 6 минут", "source": "h.md"},
            {"chunk_id": "b", "score": 0.98, "text": "PL-103-RET SLA 8 минут", "source": "h.md"},
        ]
        self.assertTrue(
            insufficient_evidence(hits, 0.25, user_text="Что такое PL-999-RET?")
        )
        self.assertEqual(
            unmatched_anchors("Что такое PL-999-RET?", hits), ["PL-999-RET"]
        )

    def test_matched_anchor_is_sufficient(self) -> None:
        hits = [
            {"chunk_id": "a", "score": 0.9, "text": "PL-101-RET SLA 6 минут", "source": "h.md"},
        ]
        self.assertFalse(
            insufficient_evidence(hits, 0.25, user_text="Что такое PL-101-RET?")
        )
        self.assertEqual(unmatched_anchors("Что такое PL-101-RET?", hits), [])

    def test_combined_refuses_when_anchor_missing_everywhere(self) -> None:
        a = [{"chunk_id": "1", "score": 0.99, "source": "x", "text": "PL-101-RET"}]
        b = [{"chunk_id": "2", "score": 0.99, "source": "y", "text": "PL-103-RET"}]
        self.assertTrue(
            combined_insufficient_evidence(
                [a, b], 0.25, user_text="Что такое PL-999-RET?"
            )
        )

    def test_filter_hits_keeps_only_matching_anchor(self) -> None:
        hits = [
            {"chunk_id": "1", "score": 0.9, "text": "PL-101-RET SLA 6 минут", "source": "h.md"},
            {"chunk_id": "2", "score": 0.9, "text": "PL-103-RET SLA 8 минут", "source": "h.md"},
            {"chunk_id": "3", "score": 0.9, "text": "PL-407-RET SLA 12 минут", "source": "h.md"},
        ]
        kept = filter_hits_by_anchors(hits, "Что такое PL-103-RET?")
        self.assertEqual([h["chunk_id"] for h in kept], ["2"])

    def test_filter_hits_no_anchor_in_query_keeps_all(self) -> None:
        hits = [
            {"chunk_id": "1", "score": 0.9, "text": "text-a", "source": "h.md"},
            {"chunk_id": "2", "score": 0.9, "text": "text-b", "source": "h.md"},
        ]
        kept = filter_hits_by_anchors(hits, "какие коды есть?")
        self.assertEqual(len(kept), 2)

    def test_filter_hits_fallback_when_nothing_matches(self) -> None:
        hits = [
            {"chunk_id": "1", "score": 0.9, "text": "PL-101-RET", "source": "h.md"},
        ]
        kept = filter_hits_by_anchors(hits, "Что такое PL-999-RET?")
        self.assertEqual(len(kept), 1)

    def test_refusal_message_mentions_missing_anchor(self) -> None:
        t = refusal_system_message(
            best_score=0.99,
            threshold=0.25,
            hit_count=5,
            missing_anchors=["PL-999-RET"],
        )
        self.assertIn("PL-999-RET", t)


if __name__ == "__main__":
    unittest.main()
