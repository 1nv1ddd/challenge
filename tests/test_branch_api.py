"""HTTP-слой роутера hub: POST /api/branches требует checkpoint_id → 400."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import app  # noqa: E402


class TestCreateBranchApi(unittest.TestCase):
    """POST /api/branches: без checkpoint_id ветка не создаётся, ответ 400."""

    def test_missing_checkpoint_id_returns_400(self) -> None:
        # Тело без checkpoint_id: ветвиться не от чего — эндпоинт обязан отклонить.
        with TestClient(app) as client:
            r = client.post("/api/branches", json={"conversation_id": "default"})

        self.assertEqual(r.status_code, 400, r.text)
        detail = r.json().get("detail")
        self.assertIsInstance(detail, str)
        self.assertTrue(detail.strip(), "detail должен быть непустой строкой")
        # Сообщение об ошибке в проекте — на русском.
        self.assertTrue(
            any("а" <= ch.lower() <= "я" for ch in detail),
            f"detail не выглядит русским: {detail!r}",
        )

    def test_empty_checkpoint_id_returns_400(self) -> None:
        # Явно пустой checkpoint_id — та же ветка валидации, что и отсутствующий.
        with TestClient(app) as client:
            r = client.post(
                "/api/branches",
                json={"conversation_id": "default", "checkpoint_id": ""},
            )

        self.assertEqual(r.status_code, 400, r.text)
        self.assertTrue(str(r.json().get("detail", "")).strip())


if __name__ == "__main__":
    unittest.main()
