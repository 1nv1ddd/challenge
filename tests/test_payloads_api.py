"""Разбор тел запросов (payloads) и валидация RAG-эндпоинтов до вызова провайдера."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import app  # noqa: E402
from app.payloads import (  # noqa: E402
    ChatRequestPayload,
    RagComparePayload,
    RagModesComparePayload,
    sse_error_line,
)


class TestChatRequestPayload(unittest.TestCase):
    def test_empty_body_gets_defaults(self) -> None:
        p = ChatRequestPayload.from_body({})
        self.assertEqual(p.provider_name, "")
        self.assertEqual(p.model, "")
        self.assertEqual(p.conversation_id, "default")
        self.assertEqual(p.branch_id, "main")
        self.assertEqual(p.raw_messages, [])
        self.assertAlmostEqual(p.temperature, 0.7)
        self.assertEqual(p.context_strategy, "sliding")
        self.assertIsNone(p.profile_id)
        self.assertFalse(p.resume)
        self.assertIsNone(p.rag)
        self.assertIsNone(p.task_workflow)

    def test_non_list_messages_become_empty_list(self) -> None:
        p = ChatRequestPayload.from_body({"messages": "привет"})
        self.assertEqual(p.raw_messages, [])

    def test_non_dict_rag_becomes_none(self) -> None:
        self.assertIsNone(ChatRequestPayload.from_body({"rag": True}).rag)
        self.assertEqual(ChatRequestPayload.from_body({"rag": {"enabled": True}}).rag, {"enabled": True})

    def test_task_workflow_false_is_not_confused_with_unset(self) -> None:
        self.assertIsNone(ChatRequestPayload.from_body({}).task_workflow)
        self.assertIs(ChatRequestPayload.from_body({"task_workflow": False}).task_workflow, False)
        self.assertIs(ChatRequestPayload.from_body({"task_workflow": True}).task_workflow, True)

    def test_numeric_fields_are_coerced(self) -> None:
        p = ChatRequestPayload.from_body({"temperature": "0.2", "conversation_id": 42})
        self.assertAlmostEqual(p.temperature, 0.2)
        self.assertEqual(p.conversation_id, "42")


class TestRagComparePayloads(unittest.TestCase):
    def test_strings_are_stripped_and_strategy_lowercased(self) -> None:
        p = RagComparePayload.from_body(
            {
                "provider": " routerai ",
                "model": " gpt ",
                "message": "  вопрос  ",
                "rag_strategy": " Structural ",
            }
        )
        self.assertEqual(p.provider_name, "routerai")
        self.assertEqual(p.model, "gpt")
        self.assertEqual(p.message, "вопрос")
        self.assertEqual(p.rag_strategy, "structural")

    def test_blank_index_path_becomes_none(self) -> None:
        self.assertIsNone(RagComparePayload.from_body({"index_path": "   "}).index_path)
        self.assertEqual(RagComparePayload.from_body({"index_path": " a.sqlite "}).index_path, "a.sqlite")

    def test_falsy_top_k_falls_back_to_default(self) -> None:
        self.assertEqual(RagComparePayload.from_body({"top_k": 0}).top_k, 8)
        self.assertEqual(RagComparePayload.from_body({"top_k": "3"}).top_k, 3)

    def test_defaults_of_modes_payload(self) -> None:
        p = RagModesComparePayload.from_body({})
        self.assertEqual(p.rag_strategy, "fixed")
        self.assertEqual(p.top_k, 8)
        self.assertAlmostEqual(p.temperature, 0.35)
        self.assertAlmostEqual(p.min_similarity, 0.28)

    def test_broken_min_similarity_falls_back_to_default(self) -> None:
        def min_sim(raw: object) -> float:
            return RagModesComparePayload.from_body({"min_similarity": raw}).min_similarity

        self.assertAlmostEqual(min_sim("abc"), 0.28)
        self.assertAlmostEqual(min_sim(None), 0.28)
        self.assertAlmostEqual(min_sim("0.5"), 0.5)


class TestSseErrorLine(unittest.TestCase):
    def test_error_line_has_sse_shape(self) -> None:
        self.assertEqual(sse_error_line(ValueError("плохой индекс")), "data: [ERROR] плохой индекс\n\n")

    def test_newlines_are_flattened(self) -> None:
        line = sse_error_line(RuntimeError("строка1\nстрока2"))
        self.assertEqual(line, "data: [ERROR] строка1 строка2\n\n")
        self.assertEqual(line.count("\n"), 2)

    def test_empty_message_falls_back_to_exception_type(self) -> None:
        self.assertEqual(sse_error_line(LookupError("")), "data: [ERROR] LookupError\n\n")


class TestRagCompareEndpointValidation(unittest.TestCase):
    """Валидация /api/rag/compare* должна отсекать запрос до похода в провайдера."""

    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_empty_message_returns_400_and_skips_agent(self) -> None:
        with patch("app.routers.hub.agent.compare_rag_answers", new_callable=AsyncMock) as m:
            m.return_value = {"ok": True}
            r = self.client.post(
                "/api/rag/compare",
                json={"provider": "routerai", "model": "openai/gpt-4o-mini", "message": "   "},
            )
        self.assertEqual(r.status_code, 400, r.text)
        self.assertIn("message", r.json()["detail"])
        m.assert_not_awaited()

    def test_missing_model_returns_400(self) -> None:
        with patch("app.routers.hub.agent.compare_rag_answers", new_callable=AsyncMock) as m:
            m.return_value = {"ok": True}
            r = self.client.post("/api/rag/compare", json={"provider": "routerai", "message": "вопрос"})
        self.assertEqual(r.status_code, 400, r.text)
        m.assert_not_awaited()

    def test_compare_modes_empty_message_returns_400_and_skips_agent(self) -> None:
        with patch("app.routers.hub.agent.compare_rag_modes", new_callable=AsyncMock) as m:
            m.return_value = {"ok": True}
            r = self.client.post(
                "/api/rag/compare_modes",
                json={"provider": "routerai", "model": "openai/gpt-4o-mini", "message": ""},
            )
        self.assertEqual(r.status_code, 400, r.text)
        m.assert_not_awaited()

    def test_missing_index_is_reported_as_400(self) -> None:
        with patch("app.routers.hub.agent.compare_rag_answers", new_callable=AsyncMock) as m:
            m.side_effect = LookupError("RAG-индекс не найден")
            r = self.client.post(
                "/api/rag/compare",
                json={"provider": "routerai", "model": "openai/gpt-4o-mini", "message": "вопрос"},
            )
        self.assertEqual(r.status_code, 400, r.text)
        self.assertEqual(r.json()["detail"], "RAG-индекс не найден")

    def test_bad_argument_is_reported_as_400(self) -> None:
        with patch("app.routers.hub.agent.compare_rag_modes", new_callable=AsyncMock) as m:
            m.side_effect = ValueError("неизвестная стратегия")
            r = self.client.post(
                "/api/rag/compare_modes",
                json={"provider": "routerai", "model": "openai/gpt-4o-mini", "message": "вопрос"},
            )
        self.assertEqual(r.status_code, 400, r.text)
        self.assertEqual(r.json()["detail"], "неизвестная стратегия")

    def test_payload_fields_are_forwarded_to_agent(self) -> None:
        with patch("app.routers.hub.agent.compare_rag_answers", new_callable=AsyncMock) as m:
            m.return_value = {"ok": True}
            r = self.client.post(
                "/api/rag/compare",
                json={
                    "provider": "routerai",
                    "model": "openai/gpt-4o-mini",
                    "message": " вопрос ",
                    "rag_strategy": "STRUCTURAL",
                    "top_k": 3,
                },
            )
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json(), {"ok": True})
        args, kwargs = m.await_args
        self.assertEqual(args, ("routerai", "openai/gpt-4o-mini", "вопрос"))
        self.assertEqual(kwargs["rag_strategy"], "structural")
        self.assertEqual(kwargs["top_k"], 3)
        self.assertIsNone(kwargs["index_path"])


if __name__ == "__main__":
    unittest.main()
