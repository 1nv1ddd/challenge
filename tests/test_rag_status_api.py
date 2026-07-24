"""HTTP-слой: GET /api/rag/status без реального RAG-индекса и сети."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import app  # noqa: E402


class TestRagStatusApi(unittest.TestCase):
    """GET /api/rag/status: обе ветки build_rag_status_response через TestClient."""

    # Клиент без контекст-менеджера: не запускаем lifespan (планировщик + возможную
    # автосборку RAG, которая ходит в сеть). Для чистого GET маршруты работают и так.
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_status_returns_200_and_stable_shape(self) -> None:
        # Реальный вызов: форма ответа не зависит от наличия индекса.
        r = self.client.get("/api/rag/status")

        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertIsInstance(body, dict)
        self.assertIs(body.get("ok"), True)
        self.assertIsInstance(body.get("indexed"), bool)
        self.assertIsInstance(body.get("path"), str)
        self.assertTrue(body["path"].endswith("chunks.sqlite"), body["path"])

    def test_status_without_index_reports_not_indexed_with_hint(self) -> None:
        # Индекса нет/пуст → indexed=False, есть подсказка, нет stats.
        with patch("app.rag.status_api.index_needs_build", return_value=True):
            r = self.client.get("/api/rag/status")

        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertIs(body["ok"], True)
        self.assertIs(body["indexed"], False)
        self.assertIsInstance(body["path"], str)
        self.assertIn("hint", body)
        self.assertNotIn("stats", body)

    def test_status_with_index_reports_stats_without_hint(self) -> None:
        # Индекс есть → indexed=True, отдаём агрегаты stats, без подсказки.
        stats = {"by_strategy": [{"strategy": "fixed", "count": 3, "avg_text_len": 42.0}]}
        with patch("app.rag.status_api.index_needs_build", return_value=False), patch(
            "app.rag.status_api.rag_index_stats", return_value=stats
        ):
            r = self.client.get("/api/rag/status")

        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertIs(body["ok"], True)
        self.assertIs(body["indexed"], True)
        self.assertEqual(body["stats"], stats)
        self.assertNotIn("hint", body)


if __name__ == "__main__":
    unittest.main()
