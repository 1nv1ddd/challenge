"""HTTP-слой роутера hub: POST /api/profiles не падает на не-объектном теле."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import app  # noqa: E402


class TestUpsertProfileApi(unittest.TestCase):
    """POST /api/profiles: тело-не-объект не должно давать 500."""

    def test_array_body_no_500(self) -> None:
        # JSON-массив вместо объекта — эндпоинт обязан не падать.
        with TestClient(app) as client:
            r = client.post("/api/profiles", json=[])

        self.assertNotEqual(r.status_code, 500, r.text)

    def test_string_body_no_500(self) -> None:
        # JSON-строка вместо объекта — та же ветка защиты.
        with TestClient(app) as client:
            r = client.post("/api/profiles", json="hello")

        self.assertNotEqual(r.status_code, 500, r.text)


if __name__ == "__main__":
    unittest.main()
