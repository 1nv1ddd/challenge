"""Эвристики роутинга: разбор строки CONFIDENCE, сигналы неуверенности, pre-routing."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent_constants import ROUTING_ESCALATE_BELOW, ROUTING_PREROUTE_HARD_SCORE  # noqa: E402
from app.routing.preroute import classify_question  # noqa: E402
from app.routing.pricing import cost_rub_model  # noqa: E402
from app.routing.signals import answers_agree, assess_answer, split_confidence  # noqa: E402

_LONG = "Столица Франции — Париж, это административный и культурный центр страны."


class TestSplitConfidence(unittest.TestCase):
    def test_plain_line_is_parsed_and_removed(self) -> None:
        body, value = split_confidence(f"{_LONG}\nCONFIDENCE: 0.95")
        self.assertEqual(value, 0.95)
        self.assertEqual(body, _LONG)

    def test_markdown_and_comma_forms(self) -> None:
        for raw, expected in (
            ("Ответ.\n**CONFIDENCE: 0.8**", 0.8),
            ("Ответ.\n`confidence = 0,72`", 0.72),
            ("Ответ.\n> CONFIDENCE: .5", 0.5),
        ):
            body, value = split_confidence(raw)
            self.assertEqual(value, expected, raw)
            self.assertEqual(body, "Ответ.", raw)

    def test_missing_line_returns_none(self) -> None:
        body, value = split_confidence("Просто ответ без служебной строки.")
        self.assertIsNone(value)
        self.assertEqual(body, "Просто ответ без служебной строки.")

    def test_last_occurrence_wins(self) -> None:
        _, value = split_confidence("Формат: CONFIDENCE: 0.10\nТекст.\nCONFIDENCE: 0.9")
        self.assertEqual(value, 0.9)

    def test_value_is_clamped(self) -> None:
        _, value = split_confidence("Ответ.\nCONFIDENCE: 1.0")
        self.assertEqual(value, 1.0)


class TestAssessAnswer(unittest.TestCase):
    def test_high_selfscore_clean_answer_stays_small(self) -> None:
        a = assess_answer("Столица Франции?", f"{_LONG}\nCONFIDENCE: 0.97")
        self.assertFalse(a.escalate)
        self.assertEqual(a.confidence, 0.97)
        self.assertEqual(a.signals, [])

    def test_low_selfscore_escalates(self) -> None:
        a = assess_answer("Сколько будет 17 × 23?", f"{_LONG}\nCONFIDENCE: 0.4")
        self.assertTrue(a.escalate)
        self.assertLess(a.confidence, ROUTING_ESCALATE_BELOW)

    def test_missing_confidence_line_lowers_base(self) -> None:
        a = assess_answer("Столица Франции?", _LONG)
        self.assertIsNone(a.self_reported)
        self.assertTrue(a.escalate)
        self.assertIn("нет строки CONFIDENCE", a.signals[0])

    def test_truncated_answer_penalised_only_without_contract_line(self) -> None:
        cut = assess_answer("Расскажи про Париж", "Париж — столица Франции и крупнейший гор")
        self.assertIn("оборван", " ".join(cut.signals))
        # Та же обрезка, но со строкой CONFIDENCE — значит модель дописала до конца.
        whole = assess_answer("Расскажи про Париж", f"{_LONG}\nCONFIDENCE: 0.9")
        self.assertNotIn("оборван", " ".join(whole.signals))

    def test_hedging_costs_confidence(self) -> None:
        a = assess_answer("Кто написал «Дюну»?", f"Скорее всего, это {_LONG}\nCONFIDENCE: 0.9")
        self.assertLess(a.confidence, 0.9)
        self.assertIn("неуверенности", " ".join(a.signals))

    def test_refusal_always_escalates_despite_high_selfscore(self) -> None:
        a = assess_answer(
            "Какой курс доллара сегодня?",
            "У меня недостаточно данных, чтобы ответить на этот вопрос.\nCONFIDENCE: 0.99",
        )
        self.assertTrue(a.escalate)
        self.assertIn("не знает", " ".join(a.signals))

    def test_short_answer_on_simple_question_is_not_penalised(self) -> None:
        a = assess_answer("Какой химический символ у золота?", "Au\nCONFIDENCE: 0.99")
        self.assertEqual(a.signals, [])
        self.assertFalse(a.escalate)

    def test_short_answer_on_hard_question_is_flagged(self) -> None:
        a = assess_answer(
            "Спроектируй схему кеширования и подробно обоснуй выбор",
            "Используй Redis.\nCONFIDENCE: 0.99",
        )
        self.assertIn("короткий", " ".join(a.signals))
        self.assertLess(a.confidence, 0.99)

    def test_empty_answer_escalates(self) -> None:
        a = assess_answer("Вопрос?", "")
        self.assertTrue(a.escalate)
        self.assertIn("пустой ответ", " ".join(a.signals))


class TestConsistency(unittest.TestCase):
    def test_same_conclusion_different_wording_agrees(self) -> None:
        agree, why = answers_agree(
            "17 × 23 = 391.",
            "Считаем: 17 × 20 = 340, 17 × 3 = 51. Итого 391.",
        )
        self.assertTrue(agree, why)

    def test_different_final_numbers_disagree(self) -> None:
        agree, why = answers_agree("В слове 4 буквы «о».", "В слове 5 букв «о».")
        self.assertFalse(agree)
        self.assertIn("итог разошёлся", why)

    def test_digit_grouping_is_normalised(self) -> None:
        agree, _ = answers_agree("Ответ: 12 300 секунд.", "Ответ: 12300 секунд.")
        self.assertTrue(agree)

    def test_unrelated_texts_disagree_by_words(self) -> None:
        agree, why = answers_agree("Столица Австралии — Канберра.", "Это зависит от контекста.")
        self.assertFalse(agree)
        self.assertIn("по составу", why)

    def test_divergence_overrides_high_selfscore(self) -> None:
        a = assess_answer(
            "Сколько букв «о» в слове «молокоотсос»?",
            "В слове «молокоотсос» 4 буквы «о».\nCONFIDENCE: 1.0",
            peers=["В слове «молокоотсос» 5 букв «о».\nCONFIDENCE: 1.0"],
        )
        self.assertEqual(a.self_reported, 1.0)
        self.assertTrue(a.escalate)
        self.assertIn("разошлись", " ".join(a.signals))


class TestPreRoute(unittest.TestCase):
    def test_simple_question_starts_small(self) -> None:
        pre = classify_question("Столица Австралии?")
        self.assertEqual(pre.tier, "small")
        self.assertEqual(pre.reasons, [])

    def test_single_marker_is_not_enough(self) -> None:
        pre = classify_question("Сравни HTTP и HTTPS")
        self.assertEqual(pre.score, 1)
        self.assertEqual(pre.tier, "small")

    def test_several_markers_preroute_to_large(self) -> None:
        pre = classify_question(
            "Спроектируй схему кеширования и подробно обоснуй выбор при условии 10k RPS"
        )
        self.assertGreaterEqual(pre.score, ROUTING_PREROUTE_HARD_SCORE)
        self.assertEqual(pre.tier, "large")

    def test_multi_step_math_counts(self) -> None:
        pre = classify_question(
            "Посчитай пошагово: 3 коробки по 12 ручек, из них продали 17 штук — сколько осталось?"
        )
        self.assertIn("расчёт в несколько действий", pre.reasons)
        self.assertEqual(pre.tier, "large")

    def test_two_questions_and_length_count(self) -> None:
        pre = classify_question("Что такое TCP? А чем он отличается от UDP? " + "детали " * 60)
        self.assertEqual(pre.tier, "large")
        self.assertIn("несколько вопросов в одном запросе", pre.reasons)


class TestPricing(unittest.TestCase):
    def test_tiers_differ_by_model(self) -> None:
        small = cost_rub_model("openai/gpt-4.1-nano", 1_000_000, 1_000_000)
        large = cost_rub_model("openai/gpt-4.1", 1_000_000, 1_000_000)
        self.assertAlmostEqual(small, 50.7, places=1)
        self.assertGreater(large / small, 15)

    def test_unknown_model_falls_back_to_project_price(self) -> None:
        self.assertAlmostEqual(cost_rub_model("who/knows", 1_000_000, 0), 15.0, places=1)


if __name__ == "__main__":
    unittest.main()
