"""Декомпозиция инференса: монолит против цепочки этапов, ремонт формата и запасной путь решения."""

from __future__ import annotations

import asyncio
import sys
import unittest
from collections.abc import AsyncIterator
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent.intake_command import detect_intake_command, render_intake_card  # noqa: E402
from app.providers import AIProvider, Message, StreamResult  # noqa: E402
from app.staged.pipeline import run_intake  # noqa: E402

_NORMALIZE_MODEL = "openai/gpt-4.1-mini"
_CHEAP_MODEL = "openai/gpt-4.1-nano"
_MONO_MODEL = "openai/gpt-4.1"
_TODAY = "2026-07-30"
_LETTER = "Добрый день! Нужны трубы 1,5 т в Екатеринбург до 20 августа, бюджет 400к, +7 999 123-45-67"

_FIELDS = """company: ООО Ромашка
product: pipe_steel
qty_kg: 1500
budget_rub: 400000
deadline: 2026-08-20
region: ural
contact: +79991234567
payment: prepay"""
_DECISION = "DECISION: accept\nREASON: ok\nMISSING: -"
_REPLY = "SUBJECT: Заявка принята\nПодтверждаем приём заявки, менеджер свяжется с вами сегодня."
_MONO_ANSWER = f"{_FIELDS}\n{_DECISION}\n{_REPLY}"
_GARBAGE = "Кажется, клиенту нужны трубы. Наверное, стоит принять заявку."


class ScriptedProvider(AIProvider):
    """Провайдер-заглушка: на каждую модель — очередь ответов, вызовы записываются по порядку."""

    name = "scripted"
    models = [
        {"id": _NORMALIZE_MODEL, "label": "normalize"},
        {"id": _CHEAP_MODEL, "label": "cheap"},
        {"id": _MONO_MODEL, "label": "mono"},
    ]

    def __init__(self, by_model: dict[str, str | list[str]]):
        self.by_model = dict(by_model)
        self.calls: list[tuple[str, list[Message]]] = []

    def _next_answer(self, model: str) -> str:
        scripted = self.by_model.get(model, "нет ответа")
        if isinstance(scripted, list):
            return scripted.pop(0) if scripted else "нет ответа"
        return scripted

    def calls_for(self, model: str) -> int:
        return len([call for call in self.calls if call[0] == model])

    async def stream_chat(
        self, messages: list[Message], model: str, temperature: float = 0.7
    ) -> AsyncIterator[StreamResult]:
        self.calls.append((model, list(messages)))
        yield StreamResult(text=self._next_answer(model))
        yield StreamResult(meta={"time_ms": 10, "prompt_tokens": 200, "completion_tokens": 60})


def _intake(provider: ScriptedProvider, mode: str = "staged", letter: str = _LETTER):
    return asyncio.run(
        run_intake(
            provider,
            letter,
            mode=mode,
            today=_TODAY,
            mono_model=_MONO_MODEL,
            models={
                "normalize": _NORMALIZE_MODEL,
                "decide": _CHEAP_MODEL,
                "compose": _CHEAP_MODEL,
            },
        )
    )


class StagedPipelineTest(unittest.TestCase):
    def test_three_stages_happy_path(self):
        provider = ScriptedProvider({_NORMALIZE_MODEL: _FIELDS, _CHEAP_MODEL: [_DECISION, _REPLY]})
        result = _intake(provider)
        self.assertTrue(result.ok)
        self.assertEqual(result.mode, "staged")
        self.assertEqual(result.fields.qty_kg, 1500)
        self.assertEqual(result.decision.decision, "accept")
        self.assertEqual(result.decision.source, "llm")
        self.assertEqual(result.reply_subject, "Заявка принята")
        self.assertEqual([s.stage for s in result.stages], ["normalize", "decide", "compose"])
        self.assertEqual(result.metrics["llm_calls"], 3)
        self.assertEqual(result.metrics["repair_calls"], 0)
        self.assertGreater(result.metrics["cost_rub"], 0)

    def test_stages_use_their_own_models(self):
        provider = ScriptedProvider({_NORMALIZE_MODEL: _FIELDS, _CHEAP_MODEL: [_DECISION, _REPLY]})
        result = _intake(provider)
        self.assertEqual(result.metrics["models"], [_NORMALIZE_MODEL, _CHEAP_MODEL, _CHEAP_MODEL])
        self.assertEqual(provider.calls_for(_NORMALIZE_MODEL), 1)
        self.assertEqual(provider.calls_for(_CHEAP_MODEL), 2)

    def test_decide_stage_sees_fields_not_letter(self):
        """Смысл декомпозиции: на вход этапа 2 идут 8 строк, а не исходное письмо."""
        provider = ScriptedProvider({_NORMALIZE_MODEL: _FIELDS, _CHEAP_MODEL: [_DECISION, _REPLY]})
        _intake(provider)
        decide_prompt = provider.calls[1][1][-1].content
        self.assertIn("product: pipe_steel", decide_prompt)
        self.assertNotIn("Добрый день", decide_prompt)

    def test_format_repair_on_second_attempt(self):
        provider = ScriptedProvider(
            {_NORMALIZE_MODEL: [_GARBAGE, _FIELDS], _CHEAP_MODEL: [_DECISION, _REPLY]}
        )
        result = _intake(provider)
        self.assertTrue(result.ok)
        normalize = result.stages[0]
        self.assertTrue(normalize.repaired)
        self.assertEqual(normalize.calls, 2)
        self.assertIsNotNone(normalize.first_error)
        self.assertIsNone(normalize.error)
        self.assertEqual(result.metrics["repair_calls"], 1)

    def test_normalize_failure_stops_chain(self):
        provider = ScriptedProvider({_NORMALIZE_MODEL: _GARBAGE, _CHEAP_MODEL: [_DECISION, _REPLY]})
        result = _intake(provider)
        self.assertFalse(result.ok)
        self.assertEqual([s.stage for s in result.stages], ["normalize"])
        self.assertEqual(result.decision.decision, "clarify")
        self.assertEqual(result.decision.source, "failed")
        self.assertEqual(result.reply_body, "")
        self.assertEqual(provider.calls_for(_CHEAP_MODEL), 0)

    def test_decide_failure_falls_back_to_rules(self):
        """У этапа 2 есть запасной путь: поля уже нормализованы, политика применяется кодом."""
        provider = ScriptedProvider(
            {_NORMALIZE_MODEL: _FIELDS, _CHEAP_MODEL: [_GARBAGE, _GARBAGE, _REPLY]}
        )
        result = _intake(provider)
        self.assertEqual(result.decision.decision, "accept")
        self.assertEqual(result.decision.source, "rules_fallback")
        self.assertEqual(result.reply_subject, "Заявка принята")
        self.assertEqual([s.stage for s in result.stages], ["normalize", "decide", "compose"])

    def test_rules_mode_skips_decide_call(self):
        provider = ScriptedProvider({_NORMALIZE_MODEL: _FIELDS, _CHEAP_MODEL: [_REPLY]})
        result = _intake(provider, mode="staged_rules")
        self.assertEqual(result.mode, "staged_rules")
        self.assertEqual(result.decision.source, "rules")
        self.assertEqual(result.decision.decision, "accept")
        self.assertEqual([s.stage for s in result.stages], ["normalize", "compose"])
        self.assertEqual(result.metrics["llm_calls"], 2)

    def test_rules_mode_applies_policy_to_extracted_fields(self):
        fields = _FIELDS.replace("budget_rub: 400000", "budget_rub: 50000")
        provider = ScriptedProvider({_NORMALIZE_MODEL: fields, _CHEAP_MODEL: [_REPLY]})
        result = _intake(provider, mode="staged_rules")
        self.assertEqual(result.decision.decision, "reject")
        self.assertEqual(result.decision.reason, "below_min_order")


class MonolithicTest(unittest.TestCase):
    def test_single_call_returns_everything(self):
        provider = ScriptedProvider({_MONO_MODEL: _MONO_ANSWER})
        result = _intake(provider, mode="mono")
        self.assertTrue(result.ok)
        self.assertEqual(result.metrics["llm_calls"], 1)
        self.assertEqual(result.fields.region, "ural")
        self.assertEqual(result.decision.decision, "accept")
        self.assertEqual(result.reply_subject, "Заявка принята")
        self.assertEqual([s.stage for s in result.stages], ["monolithic"])

    def test_broken_answer_loses_everything(self):
        """У монолита нет частичного результата: сорванный формат уносит и поля, и решение."""
        provider = ScriptedProvider({_MONO_MODEL: _GARBAGE})
        result = _intake(provider, mode="mono")
        self.assertFalse(result.ok)
        self.assertEqual(result.fields.missing(), ["product", "qty_kg", "budget_rub", "deadline", "region", "contact"])
        self.assertEqual(result.decision.source, "failed")
        self.assertEqual(result.reply_body, "")
        self.assertEqual(result.stages[0].calls, 2)

    def test_partial_answer_without_reply_is_failure(self):
        provider = ScriptedProvider({_MONO_MODEL: f"{_FIELDS}\n{_DECISION}"})
        result = _intake(provider, mode="mono")
        self.assertFalse(result.ok)
        self.assertEqual(result.fields.product, "unknown")


class IntakeArgumentsTest(unittest.TestCase):
    def test_empty_letter_rejected(self):
        provider = ScriptedProvider({})
        with self.assertRaises(ValueError):
            _intake(provider, letter="   ")

    def test_unknown_mode_rejected(self):
        provider = ScriptedProvider({})
        with self.assertRaises(ValueError) as ctx:
            _intake(provider, mode="fast")
        self.assertIn("fast", str(ctx.exception))

    def test_broken_today_rejected(self):
        provider = ScriptedProvider({_NORMALIZE_MODEL: _FIELDS})
        with self.assertRaises(ValueError):
            asyncio.run(run_intake(provider, _LETTER, mode="staged", today="30.07.2026"))

    def test_today_reaches_normalize_prompt(self):
        provider = ScriptedProvider({_NORMALIZE_MODEL: _FIELDS, _CHEAP_MODEL: [_DECISION, _REPLY]})
        _intake(provider)
        self.assertIn(_TODAY, provider.calls[0][1][-1].content)


class IntakeCommandTest(unittest.TestCase):
    def test_plain_command(self):
        is_intake, mode, today, letter = detect_intake_command(f"/intake {_LETTER}")
        self.assertTrue(is_intake)
        self.assertEqual((mode, today), ("staged", ""))
        self.assertEqual(letter, _LETTER)

    def test_mode_and_date_prefix(self):
        is_intake, mode, today, letter = detect_intake_command(
            f"/intake mono today=2026-07-30 {_LETTER}"
        )
        self.assertTrue(is_intake)
        self.assertEqual((mode, today), ("mono", "2026-07-30"))
        self.assertEqual(letter, _LETTER)

    def test_rules_alias(self):
        _, mode, _, _ = detect_intake_command("/intake rules письмо")
        self.assertEqual(mode, "staged_rules")

    def test_other_text_is_not_command(self):
        is_intake, _, _, text = detect_intake_command("расскажи про /intake")
        self.assertFalse(is_intake)
        self.assertEqual(text, "расскажи про /intake")

    def test_command_without_letter(self):
        is_intake, _, _, letter = detect_intake_command("/intake")
        self.assertTrue(is_intake)
        self.assertEqual(letter, "")

    def test_card_shows_stages_and_decision(self):
        provider = ScriptedProvider({_NORMALIZE_MODEL: _FIELDS, _CHEAP_MODEL: [_DECISION, _REPLY]})
        card = render_intake_card(_intake(provider))
        self.assertIn("Заявка принята", card)
        self.assertIn("`accept` / `ok`", card)
        self.assertIn("1. Нормализация входа", card)
        self.assertIn("3. Формирование ответа", card)
        self.assertIn("Цена разбора", card)

    def test_card_marks_rules_decision(self):
        provider = ScriptedProvider({_NORMALIZE_MODEL: _FIELDS, _CHEAP_MODEL: [_REPLY]})
        card = render_intake_card(_intake(provider, mode="staged_rules"))
        self.assertIn("политика применена кодом", card)

    def test_card_survives_failed_parse(self):
        provider = ScriptedProvider({_MONO_MODEL: _GARBAGE})
        card = render_intake_card(_intake(provider, mode="mono"))
        self.assertIn("не сформировано", card)


if __name__ == "__main__":
    unittest.main()
