"""Гейт триажа: голосование, повторный инференс, self-check и статусы OK/UNSURE/FAIL."""

from __future__ import annotations

import asyncio
import json
import sys
import unittest
from collections.abc import AsyncIterator
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.confidence.pipeline import triage  # noqa: E402
from app.confidence.redundancy import consensus  # noqa: E402
from app.confidence.schema import Decision, SampleResult  # noqa: E402
from app.providers import AIProvider, Message, StreamResult  # noqa: E402


def _answer(category: str, priority: str, action: str, missing: list[str] | None = None) -> str:
    return json.dumps(
        {
            "category": category,
            "priority": priority,
            "action": action,
            "reason": "Тестовое обоснование решения.",
            "missing": missing or [],
        },
        ensure_ascii=False,
    )


_SELFCHECK_OK = '{"verdict": "confirm", "reason": "Решение соответствует правилам."}'
_SELFCHECK_BAD = '{"verdict": "reject", "reason": "Риск недооценён."}'


class ScriptedProvider(AIProvider):
    """Провайдер-заглушка: отдаёт заранее записанные ответы по порядку вызовов."""

    name = "scripted"
    models = [{"id": "test-model", "label": "Test"}]

    def __init__(self, answers: list[str]):
        self.answers = list(answers)
        self.calls: list[list[Message]] = []

    async def stream_chat(
        self, messages: list[Message], model: str, temperature: float = 0.7
    ) -> AsyncIterator[StreamResult]:
        self.calls.append(list(messages))
        text = self.answers.pop(0) if self.answers else "нет ответа"
        yield StreamResult(text=text)
        yield StreamResult(meta={"time_ms": 10, "prompt_tokens": 100, "completion_tokens": 20})


class TestTriageGate(unittest.TestCase):
    def _run(self, provider: ScriptedProvider, text: str, **kwargs):
        return asyncio.run(triage(provider, "test-model", text, **kwargs))

    def test_unanimous_valid_decision_accepted(self) -> None:
        answers = [_answer("billing", "high", "escalate")] * 3
        provider = ScriptedProvider(answers)
        v = self._run(provider, "С карты дважды списали 2400 рублей")
        self.assertEqual(v.status, "OK")
        self.assertEqual(v.confidence, 1.0)
        self.assertEqual(v.agreement, 1.0)
        self.assertEqual(v.applied_action, "escalate")
        self.assertIsNone(v.selfcheck)
        self.assertEqual(v.metrics["llm_calls"], 3)
        self.assertEqual(v.metrics["repair_calls"], 0)
        self.assertGreater(v.metrics["cost_rub"], 0)

    def test_split_votes_trigger_selfcheck_and_still_pass(self) -> None:
        answers = [
            _answer("billing", "high", "escalate"),
            _answer("billing", "high", "escalate"),
            _answer("technical", "normal", "request_info", ["номер заказа"]),
            _SELFCHECK_OK,
        ]
        provider = ScriptedProvider(answers)
        v = self._run(provider, "Оплата прошла, а заказ не создан")
        self.assertEqual(v.agreement, 0.67)
        self.assertIsNotNone(v.selfcheck)
        self.assertEqual(v.selfcheck.verdict, "confirm")
        self.assertEqual(v.status, "OK")
        self.assertEqual(v.metrics["selfcheck_calls"], 1)
        self.assertEqual(v.metrics["llm_calls"], 4)

    def test_selfcheck_reject_drops_to_fail(self) -> None:
        answers = [
            _answer("billing", "high", "escalate"),
            _answer("billing", "high", "escalate"),
            _answer("other", "low", "request_info", ["детали"]),
            _SELFCHECK_BAD,
        ]
        provider = ScriptedProvider(answers)
        v = self._run(provider, "Оплата прошла, а заказ не создан")
        self.assertEqual(v.selfcheck.verdict, "reject")
        self.assertLess(v.confidence, 0.75)
        self.assertIn(v.status, ("UNSURE", "FAIL"))
        self.assertEqual(v.applied_action, "escalate")

    def test_risky_action_always_selfchecked(self) -> None:
        answers = [*[_answer("feedback", "low", "close")] * 3, _SELFCHECK_OK]
        provider = ScriptedProvider(answers)
        v = self._run(provider, "Спасибо, всё работает отлично!")
        self.assertEqual(v.agreement, 1.0)
        self.assertIsNotNone(v.selfcheck)
        self.assertEqual(v.status, "OK")
        self.assertEqual(v.applied_action, "close")

    def test_broken_format_is_repaired_by_second_inference(self) -> None:
        provider = ScriptedProvider(
            ["Я думаю, это надо эскалировать.", _answer("billing", "high", "escalate")]
        )
        v = self._run(provider, "Списали деньги дважды", samples=1)
        self.assertEqual(v.status, "OK")
        self.assertEqual(v.metrics["llm_calls"], 2)
        self.assertEqual(v.metrics["repair_calls"], 1)
        self.assertTrue(v.samples[0].repaired)
        # Что именно не прошло на первой попытке — остаётся в отчётности.
        self.assertTrue(v.samples[0].first_violations)
        self.assertIn("формат", v.samples[0].first_violations[0])
        # В ремонтный запрос попало описание нарушения.
        self.assertIn("отклонён проверкой", provider.calls[1][-1].content)

    def test_persistent_violation_is_rejected(self) -> None:
        # Закрыть обращение про списание нельзя — и после ремонта тоже.
        answers = [_answer("billing", "normal", "close")] * 2
        provider = ScriptedProvider(answers)
        v = self._run(provider, "С карты списали лишние деньги", samples=1)
        self.assertEqual(v.status, "FAIL")
        self.assertIsNone(v.decision)
        self.assertEqual(v.applied_action, "escalate")
        self.assertTrue(v.violations)
        self.assertEqual(v.metrics["llm_calls"], 2)

    def test_blank_text_fails_without_inference(self) -> None:
        provider = ScriptedProvider([])
        v = self._run(provider, "   ")
        self.assertEqual(v.status, "FAIL")
        self.assertEqual(v.metrics["llm_calls"], 0)
        self.assertEqual(v.metrics["cost_rub"], 0)
        self.assertEqual(provider.calls, [])


class TestConsensus(unittest.TestCase):
    def _sample(self, index: int, decision: Decision | None) -> SampleResult:
        return SampleResult(index=index, decision=decision, violations=[], raw="")

    def test_majority_wins(self) -> None:
        d1 = Decision("billing", "high", "escalate", "r")
        d2 = Decision("other", "low", "close", "r")
        samples = [self._sample(0, d1), self._sample(1, d1), self._sample(2, d2)]
        decision, agreement, votes = consensus(samples)
        self.assertEqual(decision.action, "escalate")
        self.assertEqual(agreement, 0.67)
        self.assertEqual(votes["billing/escalate"], 2)

    def test_tie_breaks_to_safest_action(self) -> None:
        d_close = Decision("other", "low", "close", "r")
        d_escalate = Decision("billing", "high", "escalate", "r")
        samples = [self._sample(0, d_close), self._sample(1, d_escalate)]
        decision, agreement, _ = consensus(samples)
        self.assertEqual(decision.action, "escalate")
        self.assertEqual(agreement, 0.5)

    def test_all_invalid_gives_no_decision(self) -> None:
        samples = [self._sample(0, None), self._sample(1, None)]
        decision, agreement, votes = consensus(samples)
        self.assertIsNone(decision)
        self.assertEqual(agreement, 0.0)
        self.assertEqual(votes, {})


if __name__ == "__main__":
    unittest.main()
