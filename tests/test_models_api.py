"""HTTP-слой роутера hub: GET /api/models без реальной сети/Ollama."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.bootstrap import providers  # noqa: E402
from app.main import app  # noqa: E402
from app.providers import OllamaProvider  # noqa: E402


class TestModelsApi(unittest.TestCase):
    """GET /api/models: словарь провайдеров, обновление Ollama замокано."""

    def test_list_models_returns_200_and_provider_dict(self) -> None:
        # Ollama всегда есть в providers, но её refresh_models ходит в сеть — мокаем.
        with patch.object(OllamaProvider, "refresh_models", new_callable=AsyncMock):
            with TestClient(app) as client:
                r = client.get("/api/models")

        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertIsInstance(body, dict)
        # Ollama-провайдер регистрируется всегда (_build_providers), ключ обязан быть.
        self.assertIn("ollama", body)
        for name, models in body.items():
            self.assertIsInstance(models, list, name)

    def test_list_models_refreshes_ollama_and_reflects_its_models(self) -> None:
        ollama = providers["ollama"]
        self.assertIsInstance(ollama, OllamaProvider)
        fixture = [{"id": "llama3", "label": "Llama 3 — Ollama"}]
        original = ollama.models
        # refresh_models в проде перезаписывает self.models из /api/tags; здесь
        # подменяем сеть и фиксируем список, чтобы поймать его в теле ответа.
        with patch.object(
            OllamaProvider, "refresh_models", new_callable=AsyncMock
        ) as refresh:
            ollama.models = fixture
            try:
                with TestClient(app) as client:
                    r = client.get("/api/models")
            finally:
                ollama.models = original

        self.assertEqual(r.status_code, 200, r.text)
        # Эндпоинт обязан дёрнуть refresh_models на Ollama-провайдере.
        refresh.assert_awaited()
        self.assertEqual(r.json().get("ollama"), fixture)


if __name__ == "__main__":
    unittest.main()
