"""Каскад роутинга: когда запрос остаётся на дешёвой модели, когда уходит на сильную."""

from __future__ import annotations

import asyncio
import sys
import unittest
from collections.abc import AsyncIterator
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from app.providers import AIProvider, Message, StreamResult  # noqa: E402
from app.routing.cascade import route_answer  # noqa: E402

_SMALL = "openai/gpt-4.1-nano"
_LARGE = "openai/gpt-4.1"
_GOOD = "Столица Франции — Париж, крупнейший город страны.\nCONFIDENCE: 0.96"
_WEAK = "Возможно, это Париж, но точно сказать нельзя.\nCONFIDENCE: 0.35"
_STRONG_ANSWER = "Столица Франции — Париж.\nCONFIDENCE: 0.99"


class ScriptedProvider(AIProvider):
    """Провайдер-заглушка: ответ зависит от модели, вызовы записываются по порядку.

    Для модели можно задать список ответов — так проверяется расхождение сэмплов дешёвой модели.
    """

    name = "scripted"
    models = [{"id": _SMALL, "label": "small"}, {"id": _LARGE, "label": "large"}]

    def __init__(
        self, by_model: dict[str, str | list[str]], fail_models: tuple[str, ...] = ()
    ):
        self.by_model = dict(by_model)
        self.fail_models = fail_models
        self.calls: list[tuple[str, list[Message]]] = []

    def _next_answer(self, model: str) -> str:
        scripted = self.by_model.get(model, "нет ответа")
        if isinstance(scripted, list):
            return scripted.pop(0) if scripted else "нет ответа"
        return scripted

    async def stream_chat(
        self, messages: list[Message], model: str, temperature: float = 0.7
    ) -> AsyncIterator[StreamResult]:
        self.calls.append((model, list(messages)))
        if model in self.fail_models:
            raise httpx.ConnectError("соединение разорвано")
        yield StreamResult(text=self._next_answer(model))
        yield StreamResult(meta={"time_ms": 10, "prompt_tokens": 100, "completion_tokens": 50})


def _route(provider: ScriptedProvider, question: str, samples: int = 1):
    return asyncio.run(
        route_answer(
            provider,
            question,
            small_model=_SMALL,
            large_model=_LARGE,
            consistency_samples=samples,
        )
    )


class TestCascade(unittest.TestCase):
    def test_confident_small_answer_stays_on_small(self) -> None:
        provider = ScriptedProvider({_SMALL: _GOOD})
        r = _route(provider, "Столица Франции?")
        self.assertEqual(r.tier, "small")
        self.assertFalse(r.escalated)
        self.assertEqual(r.model, _SMALL)
        self.assertEqual(r.metrics["llm_calls"], 1)
        self.assertEqual(r.metrics["path"], "small")
        self.assertEqual([m for m, _ in provider.calls], [_SMALL])
        # Служебная строка контракта пользователю не показывается.
        self.assertNotIn("CONFIDENCE", r.answer)

    def test_unsure_small_answer_escalates_to_large(self) -> None:
        provider = ScriptedProvider({_SMALL: _WEAK, _LARGE: _STRONG_ANSWER})
        r = _route(provider, "Столица Франции?")
        self.assertEqual(r.tier, "large")
        self.assertTrue(r.escalated)
        self.assertEqual(r.answer, "Столица Франции — Париж.")
        self.assertEqual(r.metrics["llm_calls"], 2)
        self.assertEqual(r.metrics["path"], "small→large")
        self.assertEqual([m for m, _ in provider.calls], [_SMALL, _LARGE])
        # Черновик дешёвой модели не пригодился — его цена учтена отдельно.
        self.assertGreater(r.metrics["wasted_rub"], 0)
        self.assertEqual(r.attempts[0].cost_rub, r.metrics["wasted_rub"])

    def test_hard_question_skips_small_entirely(self) -> None:
        provider = ScriptedProvider({_LARGE: _STRONG_ANSWER})
        r = _route(
            provider,
            "Спроектируй архитектуру очереди задач и подробно обоснуй выбор при условии 10k RPS",
        )
        self.assertEqual(r.preroute.tier, "large")
        self.assertEqual(r.metrics["path"], "large")
        self.assertEqual([m for m, _ in provider.calls], [_LARGE])
        self.assertIn("pre-routing", r.escalation_reason)
        # Дешёвую модель не звали — тратить на неё нечего.
        self.assertEqual(r.metrics["wasted_rub"], 0)

    def test_large_failure_falls_back_to_small_answer(self) -> None:
        provider = ScriptedProvider({_SMALL: _WEAK}, fail_models=(_LARGE,))
        r = _route(provider, "Столица Франции?")
        self.assertEqual(r.tier, "small")
        self.assertFalse(r.escalated)
        self.assertIn("Париж", r.answer)
        self.assertIn("эскалация не удалась", r.escalation_reason)
        self.assertEqual(r.attempts[-1].error, "соединение разорвано")
        self.assertEqual(r.metrics["path"], "small (fallback)")

    def test_cost_uses_per_model_price(self) -> None:
        cheap = ScriptedProvider({_SMALL: _GOOD})
        pricey = ScriptedProvider({_LARGE: _STRONG_ANSWER})
        r_small = _route(cheap, "Столица Франции?")
        r_large = _route(
            pricey,
            "Спроектируй архитектуру очереди задач и подробно обоснуй выбор при условии 10k RPS",
        )
        # Одинаковые токены, разные модели — цена обязана отличаться в разы.
        self.assertGreater(r_large.metrics["cost_rub"], r_small.metrics["cost_rub"] * 10)

    def test_agreeing_samples_stay_on_small(self) -> None:
        provider = ScriptedProvider({_SMALL: [_GOOD, _GOOD]})
        r = _route(provider, "Столица Франции?", samples=2)
        self.assertEqual(r.tier, "small")
        self.assertEqual(r.metrics["llm_calls"], 2)
        self.assertEqual([m for m, _ in provider.calls], [_SMALL, _SMALL])
        self.assertEqual(r.attempts[0].assessment.signals, [])

    def test_diverging_samples_escalate_despite_perfect_selfscore(self) -> None:
        # Обе выборки уверены в себе на 0.99, но называют разные числа — самооценка это не ловит.
        provider = ScriptedProvider(
            {
                _SMALL: [
                    "В слове «молокоотсос» 4 буквы «о».\nCONFIDENCE: 0.99",
                    "В слове «молокоотсос» 5 букв «о».\nCONFIDENCE: 0.99",
                ],
                _LARGE: "В слове «молокоотсос» 5 букв «о».\nCONFIDENCE: 0.98",
            }
        )
        r = _route(provider, "Сколько букв «о» в слове «молокоотсос»?", samples=2)
        self.assertTrue(r.escalated)
        self.assertEqual(r.tier, "large")
        self.assertEqual(r.metrics["llm_calls"], 3)
        self.assertIn("разошлись", " ".join(r.attempts[0].assessment.signals))
        # Оба черновика дешёвой модели оплачены и не пригодились.
        self.assertAlmostEqual(
            r.metrics["wasted_rub"], r.attempts[0].cost_rub + r.attempts[1].cost_rub, places=4
        )

    def test_blank_question_rejected_without_calls(self) -> None:
        provider = ScriptedProvider({})
        with self.assertRaises(ValueError):
            _route(provider, "   ")
        self.assertEqual(provider.calls, [])

    def test_prompt_carries_confidence_contract(self) -> None:
        provider = ScriptedProvider({_SMALL: _GOOD})
        _route(provider, "Столица Франции?")
        system = provider.calls[0][1][0]
        self.assertEqual(system.role, "system")
        self.assertIn("CONFIDENCE", system.content)


if __name__ == "__main__":
    unittest.main()
