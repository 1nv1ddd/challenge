"""HTTP-слой MCP-панели: GET /api/mcp/status в чистом состоянии (без серверов)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import app  # noqa: E402
from app.mcp_panel import reset_mcp_servers_for_tests  # noqa: E402


class TestMcpStatusApi(unittest.TestCase):
    """GET /api/mcp/status без подключённых серверов: 200 и пустая форма."""

    def setUp(self) -> None:
        reset_mcp_servers_for_tests()

    def tearDown(self) -> None:
        reset_mcp_servers_for_tests()

    def test_status_empty_state_returns_200_and_shape(self) -> None:
        with TestClient(app) as client:
            r = client.get("/api/mcp/status")

        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        # В чистом состоянии панель отдаёт ровно _empty_public_state().
        self.assertEqual(
            body,
            {
                "connected": False,
                "servers": [],
                "tools": [],
                "error": None,
            },
        )

    def test_status_disconnected_has_no_connected_extras(self) -> None:
        # Пустое состояние не должно протекать полями connected-варианта
        # (multi_server / server_count / server_name появляются только с серверами).
        with TestClient(app) as client:
            body = client.get("/api/mcp/status").json()

        self.assertFalse(body["connected"])
        self.assertEqual(body["servers"], [])
        self.assertEqual(body["tools"], [])
        self.assertNotIn("server_count", body)
        self.assertNotIn("multi_server", body)


if __name__ == "__main__":
    unittest.main()
