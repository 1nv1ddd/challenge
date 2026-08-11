"""Двухуровневый инференс: кого позвали, сколько это стоило и что попадает в карточку чата."""

from __future__ import annotations

import asyncio
import sys
import unittest
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent.intent_command import (  # noqa: E402
    detect_intent_command,
    render_intent_card,
    usage_markdown,
)
from app.agent_constants import MICRO_LLM_MODEL, MICRO_STRATEGIES  # noqa: E402
from app.micro.pipeline import classify_intent  # noqa: E402
from app.providers import AIProvider, Message, StreamResult  # noqa: E402

_BANK_PATH = ROOT / "data" / "micro_bank.jsonl"
# Обращения подобраны под офлайн-бэкенд: первое уверенно ложится на банк, второе — нет.
_CLEAR = "Дважды списали оплату за подписку, верните переплату на карту"
_UNCLEAR = "Здравствуйте, хотим обсудить сотрудничество и заодно уточнить пару моментов"
_ANSWER = '{"label": "other", "confidence": 0.82, "reason": "вопрос про сотрудничество"}'
_GARBAGE = "Похоже, это обращение про сотрудничество."
# Только tfidf: эмбеддинги ходят в сеть, а тесты должны проходить офлайн.
_OFFLINE_STRATEGIES = {
    **MICRO_STRATEGIES,
    "micro_only": ("tfidf", False),
    "llm_only": (None, True),
}


class ScriptedProvider(AIProvider):
    """Провайдер-заглушка: очередь ответов на модель, вызовы записываются по порядку."""

    name = "scripted"
    models = [{"id": MICRO_LLM_MODEL, "label": "large"}]

    def __init__(self, answers: list[str]):
        self.answers = list(answers)
        self.calls: list[list[Message]] = []

    async def stream_chat(
        self, messages: list[Message], model: str, temperature: float = 0.7
    ) -> AsyncIterator[StreamResult]:
        self.calls.append(list(messages))
        text = self.answers.pop(0) if self.answers else "нет ответа"
        yield StreamResult(text=text)
        yield StreamResult(meta={"time_ms": 900, "prompt_tokens": 320, "completion_tokens": 40})


def _classify(provider: AIProvider | None, text: str, strategy: str):
    with patch.dict("app.micro.pipeline.MICRO_STRATEGIES", _OFFLINE_STRATEGIES, clear=True):
        return asyncio.run(
            classify_intent(provider, text, strategy=strategy, bank_path=_BANK_PATH)
        )


class MicroFirstTests(unittest.TestCase):
    """Главное свойство дня: уверенный уровень 1 не даёт большой модели ни одного вызова."""

    def test_confident_micro_does_not_call_llm(self):
        provider = ScriptedProvider([_ANSWER])
        result = _classify(provider, _CLEAR, "micro_tfidf_first")
        self.assertEqual(result.source, "micro")
        self.assertEqual(result.label, "billing")
        self.assertEqual(provider.calls, [])
        self.assertEqual(result.metrics["llm_calls"], 0)
        self.assertEqual(result.metrics["cost_rub"], 0.0)
        self.assertFalse(result.metrics["escalated"])

    def test_unsure_micro_escalates_to_llm(self):
        provider = ScriptedProvider([_ANSWER])
        result = _classify(provider, _UNCLEAR, "micro_tfidf_first")
        self.assertEqual(result.source, "llm")
        self.assertEqual(result.label, "other")
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(result.metrics["llm_calls"], 1)
        self.assertGreater(result.metrics["cost_rub"], 0.0)
        self.assertIsNotNone(result.micro.escalate_reason)

    def test_micro_only_keeps_its_label_without_fallback(self):
        provider = ScriptedProvider([_ANSWER])
        result = _classify(provider, _UNCLEAR, "micro_only")
        self.assertEqual(result.source, "micro")
        self.assertEqual(provider.calls, [])
        self.assertEqual(result.micro.status, "UNSURE")
        self.assertEqual(result.label, result.micro.label)

    def test_llm_only_skips_micro_entirely(self):
        provider = ScriptedProvider([_ANSWER])
        result = _classify(provider, _CLEAR, "llm_only")
        self.assertIsNone(result.micro)
        self.assertEqual(result.source, "llm")
        self.assertEqual(len(provider.calls), 1)
        self.assertIsNone(result.metrics["backend"])

    def test_metrics_separate_micro_and_llm_time(self):
        result = _classify(ScriptedProvider([_ANSWER]), _UNCLEAR, "micro_tfidf_first")
        self.assertGreater(result.metrics["llm_ms"], 0)
        self.assertEqual(result.metrics["micro_calls"], 1)
        self.assertEqual(result.metrics["llm_model"], MICRO_LLM_MODEL)


class FallbackFormatTests(unittest.TestCase):
    """Уровень 2 тоже может нарушить формат — тогда его чинят, а в крайнем случае берут метку micro."""

    def test_broken_format_is_repaired(self):
        provider = ScriptedProvider([_GARBAGE, _ANSWER])
        result = _classify(provider, _UNCLEAR, "micro_tfidf_first")
        self.assertEqual(len(provider.calls), 2)
        self.assertTrue(result.llm.repaired)
        self.assertIsNone(result.llm.error)
        self.assertEqual(result.label, "other")
        self.assertEqual(result.metrics["repair_calls"], 1)

    def test_repair_prompt_names_the_error(self):
        provider = ScriptedProvider([_GARBAGE, _ANSWER])
        _classify(provider, _UNCLEAR, "micro_tfidf_first")
        repair = provider.calls[1][-1].content
        self.assertIn("не прошёл проверку", repair)

    def test_hopeless_format_falls_back_to_micro_label(self):
        provider = ScriptedProvider([_GARBAGE, _GARBAGE])
        result = _classify(provider, _UNCLEAR, "micro_tfidf_first")
        self.assertEqual(result.source, "micro_after_llm_error")
        self.assertEqual(result.label, result.micro.label)
        self.assertIsNotNone(result.llm.error)
        self.assertFalse(result.ok)


class ValidationTests(unittest.TestCase):
    """Вход проверяется до того, как потрачен хоть один вызов."""

    def test_empty_text_is_rejected(self):
        with self.assertRaises(ValueError):
            _classify(ScriptedProvider([]), "   ", "micro_tfidf_first")

    def test_unknown_strategy_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            _classify(ScriptedProvider([]), _CLEAR, "magic")
        self.assertIn("Неизвестная стратегия", str(ctx.exception))

    def test_fallback_without_provider_is_rejected(self):
        with self.assertRaises(ValueError):
            _classify(None, _UNCLEAR, "micro_tfidf_first")

    def test_micro_only_works_without_provider(self):
        result = _classify(None, _CLEAR, "micro_only")
        self.assertEqual(result.source, "micro")


class CommandTests(unittest.TestCase):
    """Команда `/intent` в веб-чате: разбор префикса и карточка результата."""

    def test_plain_command_uses_default_strategy(self):
        is_intent, strategy, text = detect_intent_command("/intent Не приходит счёт за июль")
        self.assertTrue(is_intent)
        self.assertEqual(strategy, "micro_embed_first")
        self.assertEqual(text, "Не приходит счёт за июль")

    def test_strategy_alias_is_stripped_from_text(self):
        _, strategy, text = detect_intent_command("/intent tfidf Не могу войти в кабинет")
        self.assertEqual(strategy, "micro_tfidf_first")
        self.assertEqual(text, "Не могу войти в кабинет")

    def test_llm_alias(self):
        _, strategy, _ = detect_intent_command("/intent llm Пропали файлы")
        self.assertEqual(strategy, "llm_only")

    def test_other_commands_are_not_intercepted(self):
        self.assertFalse(detect_intent_command("/intake Нужны трубы 1,5 т")[0])
        self.assertFalse(detect_intent_command("Обычное сообщение")[0])

    def test_empty_command_gets_usage(self):
        is_intent, _, text = detect_intent_command("/intent")
        self.assertTrue(is_intent)
        self.assertEqual(text, "")
        self.assertIn("/intent", usage_markdown())

    def test_card_for_micro_hit_says_llm_was_not_called(self):
        card = render_intent_card(_classify(ScriptedProvider([]), _CLEAR, "micro_tfidf_first"))
        self.assertIn("`billing`", card)
        self.assertIn("не понадобился", card)
        self.assertIn("0 вызов(ов) большой модели", card)

    def test_card_for_escalation_shows_both_levels(self):
        result = _classify(ScriptedProvider([_ANSWER]), _UNCLEAR, "micro_tfidf_first")
        card = render_intent_card(result)
        self.assertIn("UNSURE", card)
        self.assertIn("Уровень 2", card)
        self.assertIn("confidence 0.82", card)


if __name__ == "__main__":
    unittest.main()
