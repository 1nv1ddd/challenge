"""Строгие форматы этапов и политика приёма заявки: разбор ответов и решение по правилам."""

from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.staged.parse import (  # noqa: E402
    parse_decision,
    parse_fields,
    parse_monolithic,
    parse_reply,
)
from app.staged.policy import decide_by_rules  # noqa: E402
from app.staged.schema import IntakeFields  # noqa: E402

_TODAY = date(2026, 7, 30)
_FIELDS_BLOCK = """company: ООО Ромашка
product: pipe_steel
qty_kg: 1500
budget_rub: 400000
deadline: 2026-08-20
region: ural
contact: +79991234567
payment: prepay"""
_DECISION_BLOCK = "DECISION: accept\nREASON: ok\nMISSING: -"
_REPLY_BLOCK = "SUBJECT: Заявка принята\nПодтверждаем приём заявки, менеджер свяжется с вами."


def _fields(**overrides) -> IntakeFields:
    """Полная заявка, проходящая политику; в тестах меняем по одному полю."""
    base = {
        "company": "ООО Ромашка",
        "product": "pipe_steel",
        "qty_kg": 1500,
        "budget_rub": 400000,
        "deadline": "2026-08-20",
        "region": "ural",
        "contact": "+79991234567",
        "payment": "prepay",
    }
    return IntakeFields(**{**base, **overrides})


class ParseFieldsTest(unittest.TestCase):
    def test_canonical_block(self):
        fields = parse_fields(_FIELDS_BLOCK)
        self.assertEqual(fields.company, "ООО Ромашка")
        self.assertEqual(fields.product, "pipe_steel")
        self.assertEqual(fields.qty_kg, 1500)
        self.assertEqual(fields.budget_rub, 400000)
        self.assertEqual(fields.deadline, "2026-08-20")
        self.assertEqual(fields.region, "ural")
        self.assertEqual(fields.contact, "+79991234567")
        self.assertEqual(fields.payment, "prepay")
        self.assertEqual(fields.missing(), [])

    def test_fences_and_chatter_ignored(self):
        raw = "```\n" + _FIELDS_BLOCK + "\n```"
        self.assertEqual(parse_fields(raw).qty_kg, 1500)

    def test_units_stripped_from_numbers(self):
        raw = _FIELDS_BLOCK.replace("qty_kg: 1500", "qty_kg: 1 500 кг").replace(
            "budget_rub: 400000", "budget_rub: 400000 ₽"
        )
        fields = parse_fields(raw)
        self.assertEqual(fields.qty_kg, 1500)
        self.assertEqual(fields.budget_rub, 400000)

    def test_unknown_synonyms_become_unknown(self):
        raw = _FIELDS_BLOCK.replace("qty_kg: 1500", "qty_kg: не указано").replace(
            "payment: prepay", "payment: -"
        )
        fields = parse_fields(raw)
        self.assertIsNone(fields.qty_kg)
        self.assertEqual(fields.payment, "unknown")
        self.assertEqual(fields.missing(), ["qty_kg"])

    def test_missing_line_is_format_error(self):
        raw = "\n".join(
            line for line in _FIELDS_BLOCK.splitlines() if not line.startswith("region")
        )
        with self.assertRaises(ValueError) as ctx:
            parse_fields(raw)
        self.assertIn("region", str(ctx.exception))

    def test_unknown_enum_value_is_format_error(self):
        raw = _FIELDS_BLOCK.replace("product: pipe_steel", "product: трубы")
        with self.assertRaises(ValueError) as ctx:
            parse_fields(raw)
        self.assertIn("product", str(ctx.exception))

    def test_non_iso_date_is_format_error(self):
        raw = _FIELDS_BLOCK.replace("deadline: 2026-08-20", "deadline: 20 августа")
        with self.assertRaises(ValueError):
            parse_fields(raw)

    def test_denormalized_phone_is_format_error(self):
        raw = _FIELDS_BLOCK.replace("contact: +79991234567", "contact: 8 (999) 123-45-67")
        with self.assertRaises(ValueError):
            parse_fields(raw)

    def test_email_lowercased(self):
        raw = _FIELDS_BLOCK.replace("contact: +79991234567", "contact: Sales@Romashka.RU")
        self.assertEqual(parse_fields(raw).contact, "sales@romashka.ru")

    def test_compact_roundtrip(self):
        self.assertEqual(parse_fields(_FIELDS_BLOCK).to_compact(), _FIELDS_BLOCK)


class ParseDecisionTest(unittest.TestCase):
    def test_canonical(self):
        decision = parse_decision(_DECISION_BLOCK)
        self.assertEqual(decision.decision, "accept")
        self.assertEqual(decision.reason, "ok")
        self.assertEqual(decision.missing, [])
        self.assertEqual(decision.source, "llm")

    def test_missing_list_parsed(self):
        raw = "DECISION: clarify\nREASON: missing_fields\nMISSING: budget_rub, contact"
        self.assertEqual(parse_decision(raw).missing, ["budget_rub", "contact"])

    def test_unknown_field_in_missing_is_error(self):
        raw = "DECISION: clarify\nREASON: missing_fields\nMISSING: budget_rub, inn"
        with self.assertRaises(ValueError) as ctx:
            parse_decision(raw)
        self.assertIn("inn", str(ctx.exception))

    def test_free_text_decision_is_error(self):
        with self.assertRaises(ValueError):
            parse_decision("Заявку стоит принять, всё в порядке.")

    def test_out_of_enum_reason_is_error(self):
        with self.assertRaises(ValueError):
            parse_decision("DECISION: accept\nREASON: looks_good\nMISSING: -")


class ParseReplyTest(unittest.TestCase):
    def test_subject_and_body(self):
        subject, body = parse_reply(_REPLY_BLOCK)
        self.assertEqual(subject, "Заявка принята")
        self.assertTrue(body.startswith("Подтверждаем"))

    def test_body_prefix_stripped(self):
        subject, body = parse_reply("SUBJECT: Тема\nBODY: Текст письма для клиента.")
        self.assertEqual(subject, "Тема")
        self.assertEqual(body, "Текст письма для клиента.")

    def test_without_subject_is_error(self):
        with self.assertRaises(ValueError):
            parse_reply("Здравствуйте! Ваша заявка принята.")

    def test_too_long_body_is_error(self):
        long_body = "SUBJECT: Тема\n" + ("слово " * 200)
        with self.assertRaises(ValueError) as ctx:
            parse_reply(long_body)
        self.assertIn("слов", str(ctx.exception))


class ParseMonolithicTest(unittest.TestCase):
    def test_three_blocks_in_one_answer(self):
        raw = f"{_FIELDS_BLOCK}\n{_DECISION_BLOCK}\n{_REPLY_BLOCK}"
        fields, decision, subject, body = parse_monolithic(raw)
        self.assertEqual(fields.region, "ural")
        self.assertEqual(decision.decision, "accept")
        self.assertEqual(subject, "Заявка принята")
        self.assertIn("менеджер", body)

    def test_reply_block_lost_is_error(self):
        with self.assertRaises(ValueError):
            parse_monolithic(f"{_FIELDS_BLOCK}\n{_DECISION_BLOCK}")


class PolicyRulesTest(unittest.TestCase):
    def test_full_request_accepted(self):
        decision = decide_by_rules(_fields(), _TODAY)
        self.assertEqual((decision.decision, decision.reason), ("accept", "ok"))
        self.assertEqual(decision.source, "rules")

    def test_product_outside_catalog_rejected(self):
        decision = decide_by_rules(_fields(product="other"), _TODAY)
        self.assertEqual((decision.decision, decision.reason), ("reject", "product_not_in_catalog"))

    def test_abroad_rejected(self):
        decision = decide_by_rules(_fields(region="abroad"), _TODAY)
        self.assertEqual((decision.decision, decision.reason), ("reject", "region_not_served"))

    def test_budget_below_min_order_rejected(self):
        decision = decide_by_rules(_fields(budget_rub=80000), _TODAY)
        self.assertEqual((decision.decision, decision.reason), ("reject", "below_min_order"))

    def test_missing_required_field_goes_to_clarify(self):
        decision = decide_by_rules(_fields(contact="unknown"), _TODAY)
        self.assertEqual((decision.decision, decision.reason), ("clarify", "missing_fields"))
        self.assertEqual(decision.missing, ["contact"])

    def test_deadline_shorter_than_region_sla_rejected(self):
        # Урал — 7 дней, а до срока 3 дня.
        decision = decide_by_rules(_fields(deadline="2026-08-02"), _TODAY)
        self.assertEqual((decision.decision, decision.reason), ("reject", "deadline_unrealistic"))

    def test_same_deadline_fits_moscow_sla(self):
        decision = decide_by_rules(_fields(region="moscow", deadline="2026-08-02"), _TODAY)
        self.assertEqual(decision.decision, "accept")

    def test_long_postpay_goes_to_clarify(self):
        decision = decide_by_rules(_fields(payment="postpay_60"), _TODAY)
        self.assertEqual((decision.decision, decision.reason), ("clarify", "payment_terms_review"))

    def test_rule_order_low_budget_beats_missing_fields(self):
        decision = decide_by_rules(_fields(budget_rub=50000, contact="unknown"), _TODAY)
        self.assertEqual(decision.reason, "below_min_order")

    def test_rule_order_missing_fields_beats_deadline(self):
        # Пока не хватает контакта, срок проверять не по чему — сначала уточнение.
        decision = decide_by_rules(_fields(contact="unknown", deadline="2026-07-31"), _TODAY)
        self.assertEqual(decision.reason, "missing_fields")

    def test_unknown_deadline_goes_to_clarify(self):
        decision = decide_by_rules(_fields(deadline="unknown"), _TODAY)
        self.assertEqual((decision.decision, decision.reason), ("clarify", "missing_fields"))
        self.assertEqual(decision.missing, ["deadline"])


if __name__ == "__main__":
    unittest.main()
