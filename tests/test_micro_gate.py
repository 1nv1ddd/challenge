"""Уровень 1: банк примеров, tfidf-бэкенд, пороги гейта и строгий разбор ответа уровня 2."""

from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent_constants import MICRO_LABELS, MICRO_MIN_CHARS  # noqa: E402
from app.micro.backends import TfidfBackend, normalize_text  # noqa: E402
from app.micro.bank import BankItem, bank_digest, bank_labels, load_bank  # noqa: E402
from app.micro.gate import (  # noqa: E402
    ESCALATE_FALLBACK_LABEL,
    ESCALATE_LOW_MARGIN,
    ESCALATE_LOW_SCORE,
    ESCALATE_NO_NEIGHBOR,
    ESCALATE_SHORT,
    judge,
    thresholds,
)
from app.micro.llm import parse_label  # noqa: E402
from app.micro.schema import Neighbor  # noqa: E402

_BANK_PATH = ROOT / "data" / "micro_bank.jsonl"


def _neighbors(*pairs: tuple[str, float]) -> list[Neighbor]:
    return [Neighbor(label=label, similarity=sim, text=f"{label} пример") for label, sim in pairs]


class BankTests(unittest.TestCase):
    """Банк примеров — это модель уровня 1, поэтому он должен быть валиден сам по себе."""

    def test_project_bank_loads_and_covers_all_labels(self):
        items = load_bank(_BANK_PATH)
        counts = bank_labels(items)
        self.assertEqual(len(items), sum(counts.values()))
        for label in MICRO_LABELS:
            self.assertGreaterEqual(counts[label], 3, f"мало примеров метки {label}")

    def test_unknown_label_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bank.jsonl"
            path.write_text('{"label": "spam", "text": "текст"}\n', encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_bank(path)
        self.assertIn("вне списка", str(ctx.exception))

    def test_broken_json_names_the_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bank.jsonl"
            path.write_text(
                '{"label": "billing", "text": "ок"}\n{"label": "billing"\n', encoding="utf-8"
            )
            with self.assertRaises(ValueError) as ctx:
                load_bank(path)
        self.assertIn("строка 2", str(ctx.exception))

    def test_missing_file_is_a_value_error(self):
        with self.assertRaises(ValueError):
            load_bank(ROOT / "data" / "no_such_bank.jsonl")

    def test_digest_changes_with_content(self):
        one = [BankItem("billing", "счёт")]
        self.assertEqual(bank_digest(one), bank_digest([BankItem("billing", "счёт")]))
        self.assertNotEqual(bank_digest(one), bank_digest([BankItem("billing", "счета")]))


class TfidfBackendTests(unittest.TestCase):
    """Офлайн-бэкенд: без сети, детерминированный, устойчивый к регистру и пунктуации."""

    @classmethod
    def setUpClass(cls):
        cls.backend = TfidfBackend(load_bank(_BANK_PATH))

    def _neighbors(self, text: str, k: int = 5) -> list[Neighbor]:
        return asyncio.run(self.backend.neighbors(text, k))

    def test_normalize_text_folds_case_punctuation_and_yo(self):
        self.assertEqual(normalize_text("Счёт, пожалуйста!!!"), "счет пожалуйста")

    def test_neighbors_are_sorted_and_limited(self):
        found = self._neighbors("списали деньги дважды за месяц", k=3)
        self.assertEqual(len(found), 3)
        self.assertEqual([n.similarity for n in found], sorted((n.similarity for n in found), reverse=True))

    def test_payment_message_lands_on_billing(self):
        self.assertEqual(self._neighbors("Дважды списали оплату за подписку")[0].label, "billing")

    def test_login_message_lands_on_account(self):
        self.assertEqual(self._neighbors("Не могу войти в кабинет, пароль не подходит")[0].label, "account")

    def test_case_and_punctuation_do_not_change_result(self):
        plain = self._neighbors("пропали задачи за неделю")
        shouted = self._neighbors("ПРОПАЛИ ЗАДАЧИ ЗА НЕДЕЛЮ!!!")
        self.assertEqual([n.label for n in plain], [n.label for n in shouted])

    def test_unknown_alphabet_gives_zero_similarity(self):
        self.assertEqual(self._neighbors("[[[ ]]]")[0].similarity, 0.0)


class GateTests(unittest.TestCase):
    """Пороги гейта: каждая причина эскалации проверяется отдельно.

    Пороги задаются явно: логика гейта не должна ломаться от новой калибровки констант.
    """

    LIMITS = {"accept": 0.62, "sim_floor": 0.45, "margin_min": 0.03, "margin_full": 0.10}

    def judge(self, text: str, neighbors: list[Neighbor], backend: str = "embed"):
        return judge(backend, text, neighbors, limits=self.LIMITS)

    def test_project_thresholds_are_used_by_default(self):
        found = _neighbors(("billing", 0.86), ("billing", 0.78), ("technical", 0.41))
        self.assertEqual(
            judge("embed", "Списали деньги дважды за подписку", found).score,
            judge(
                "embed",
                "Списали деньги дважды за подписку",
                found,
                limits=thresholds("embed"),
            ).score,
        )

    def test_confident_case_is_ok(self):
        verdict = self.judge("Списали деньги дважды за подписку", _neighbors(
            ("billing", 0.86), ("billing", 0.78), ("technical", 0.41)
        ))
        self.assertEqual(verdict.status, "OK")
        self.assertEqual(verdict.label, "billing")
        self.assertIsNone(verdict.escalate_reason)
        self.assertTrue(verdict.ok)

    def test_short_input_escalates_before_any_other_check(self):
        verdict = self.judge("не грузит", _neighbors(("technical", 0.92), ("technical", 0.9)))
        self.assertEqual(verdict.escalate_reason, ESCALATE_SHORT)
        self.assertLess(len("не грузит"), MICRO_MIN_CHARS)

    def test_no_close_neighbor_escalates(self):
        verdict = self.judge("Обращение про что-то незнакомое", _neighbors(
            ("feedback", 0.31), ("other", 0.30)
        ))
        self.assertEqual(verdict.escalate_reason, ESCALATE_NO_NEIGHBOR)

    def test_fallback_label_always_escalates(self):
        verdict = self.judge("Предлагаем сотрудничество по продвижению", _neighbors(
            ("other", 0.88), ("other", 0.85), ("feedback", 0.44)
        ))
        self.assertEqual(verdict.label, "other")
        self.assertEqual(verdict.escalate_reason, ESCALATE_FALLBACK_LABEL)

    def test_two_close_classes_escalate_by_margin(self):
        verdict = self.judge("Оплатил, но тариф не переключился", _neighbors(
            ("billing", 0.71), ("technical", 0.70), ("billing", 0.62)
        ))
        self.assertEqual(verdict.escalate_reason, ESCALATE_LOW_MARGIN)

    def test_weak_but_separated_case_escalates_by_score(self):
        verdict = self.judge("Обращение средней понятности", _neighbors(
            ("technical", 0.50), ("technical", 0.49), ("feedback", 0.44)
        ))
        self.assertEqual(verdict.escalate_reason, ESCALATE_LOW_SCORE)

    def test_empty_neighbors_do_not_crash(self):
        verdict = self.judge("Любое обращение подлиннее", [])
        self.assertEqual(verdict.status, "UNSURE")
        self.assertEqual(verdict.escalate_reason, ESCALATE_NO_NEIGHBOR)

    def test_votes_are_weighted_by_similarity(self):
        verdict = self.judge("Пропали файлы из архива проекта", _neighbors(
            ("data_loss", 0.90), ("technical", 0.30), ("technical", 0.29)
        ))
        self.assertEqual(verdict.label, "data_loss")
        self.assertGreater(verdict.votes, 0.5)

    def test_unknown_backend_has_no_thresholds(self):
        with self.assertRaises(ValueError):
            judge("magic", "Обращение подлиннее порога", _neighbors(("billing", 0.9)))


class ParseLabelTests(unittest.TestCase):
    """Формат уровня 2: разбираем только строгий JSON с меткой из enum."""

    def test_plain_json(self):
        answer = parse_label('{"label": "billing", "confidence": 0.9, "reason": "про оплату"}')
        self.assertEqual((answer.label, answer.confidence), ("billing", 0.9))

    def test_json_in_fence_and_prose(self):
        raw = 'Вот ответ:\n```json\n{"label": "account", "confidence": 0.7, "reason": "вход"}\n```'
        self.assertEqual(parse_label(raw).label, "account")

    def test_label_case_is_normalized(self):
        self.assertEqual(parse_label('{"label": "Technical", "confidence": 1}').label, "technical")

    def test_comma_decimal_confidence(self):
        self.assertEqual(parse_label('{"label": "feedback", "confidence": "0,55"}').confidence, 0.55)

    def test_unknown_label_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_label('{"label": "sales", "confidence": 0.9}')

    def test_confidence_out_of_range_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_label('{"label": "billing", "confidence": 1.4}')

    def test_confidence_must_be_a_number(self):
        with self.assertRaises(ValueError):
            parse_label('{"label": "billing", "confidence": "высокая"}')

    def test_prose_without_json_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_label("Мне кажется, это про оплату.")


if __name__ == "__main__":
    unittest.main()
