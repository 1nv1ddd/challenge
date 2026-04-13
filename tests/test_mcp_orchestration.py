"""Мульти-сервер MCP: API статуса, disconnect, resolve_invocation (две машины)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent import _parse_mcp_tool_call  # noqa: E402
from app.main import app  # noqa: E402
from app.mcp_panel import (  # noqa: E402
    reset_mcp_servers_for_tests,
    resolve_invocation,
)

MINIMAL = ROOT / "scripts" / "minimal_mcp_server.py"
REL_MINIMAL = str(MINIMAL.relative_to(ROOT))


async def _fake_list_tools(_script: Path) -> tuple[dict, list[dict]]:
    return (
        {"name": "stub", "version": "0"},
        [
            {
                "name": "ping",
                "title": None,
                "description": None,
                "input_schema": {},
                "mcp_server_id": "",
            },
            {
                "name": "echo",
                "title": None,
                "description": None,
                "input_schema": {},
                "mcp_server_id": "",
            },
        ],
    )


class TestMcpOrchestration(unittest.TestCase):
    def setUp(self) -> None:
        reset_mcp_servers_for_tests()

    def tearDown(self) -> None:
        reset_mcp_servers_for_tests()

    @unittest.skipUnless(MINIMAL.is_file(), "minimal_mcp_server.py missing")
    def test_connect_two_servers_status_tools_and_disconnect(self) -> None:
        with patch("app.mcp_panel._list_tools_stdio", new_callable=AsyncMock) as m:
            m.side_effect = _fake_list_tools
            with TestClient(app) as client:
                r1 = client.post(
                    "/api/mcp/connect",
                    json={
                        "transport": "stdio",
                        "server_id": "alpha",
                        "script_path": REL_MINIMAL,
                        "server_name": "Alpha",
                    },
                )
                self.assertEqual(r1.status_code, 200, r1.text)
                self.assertTrue(r1.json().get("ok"))
                r2 = client.post(
                    "/api/mcp/connect",
                    json={
                        "transport": "stdio",
                        "server_id": "beta",
                        "script_path": REL_MINIMAL,
                        "server_name": "Beta",
                    },
                )
                self.assertEqual(r2.status_code, 200, r2.text)
                d = r2.json()
                self.assertTrue(d.get("ok"))
                self.assertTrue(d.get("multi_server"))
                self.assertEqual(d.get("server_count"), 2)
                servers = d.get("servers") or []
                self.assertEqual({s["id"] for s in servers}, {"alpha", "beta"})
                tools = d.get("tools") or []
                by_srv: dict[str, set[str]] = {}
                for t in tools:
                    sid = t.get("mcp_server_id")
                    by_srv.setdefault(str(sid), set()).add(t["name"])
                self.assertEqual(by_srv["alpha"], {"ping", "echo"})
                self.assertEqual(by_srv["beta"], {"ping", "echo"})

                st = client.get("/api/mcp/status").json()
                self.assertTrue(st["connected"])
                self.assertEqual(st["server_count"], 2)

                one = client.post(
                    "/api/mcp/disconnect", json={"server_id": "alpha"}
                ).json()
                self.assertTrue(one.get("ok"))
                self.assertEqual(one.get("server_count"), 1)
                self.assertEqual(
                    {s["id"] for s in (one.get("servers") or [])},
                    {"beta"},
                )

                cleared = client.post("/api/mcp/disconnect", json={}).json()
                self.assertFalse(cleared.get("connected"))

    @unittest.skipUnless(MINIMAL.is_file(), "minimal_mcp_server.py missing")
    def test_resolve_invocation_ambiguous_without_server(self) -> None:
        with patch("app.mcp_panel._list_tools_stdio", new_callable=AsyncMock) as m:
            m.side_effect = _fake_list_tools
            with TestClient(app) as client:
                client.post(
                    "/api/mcp/connect",
                    json={
                        "transport": "stdio",
                        "server_id": "alpha",
                        "script_path": REL_MINIMAL,
                    },
                )
                client.post(
                    "/api/mcp/connect",
                    json={
                        "transport": "stdio",
                        "server_id": "beta",
                        "script_path": REL_MINIMAL,
                    },
                )
        with self.assertRaises(ValueError) as ctx:
            resolve_invocation(None, "ping")
        self.assertIn("нескольких", str(ctx.exception).lower())

        p, name, sid = resolve_invocation("beta", "ping")
        self.assertEqual(name, "ping")
        self.assertEqual(sid, "beta")
        self.assertTrue(p.is_file())

    def test_parse_mcp_fence_includes_server(self) -> None:
        raw = (
            '```mcp\n{"name": "ping", "server": "alpha", "arguments": {}}\n```'
        )
        p = _parse_mcp_tool_call(raw)
        self.assertIsNotNone(p)
        assert p is not None
        self.assertEqual(p.get("server"), "alpha")
        self.assertEqual(p["name"], "ping")


if __name__ == "__main__":
    unittest.main()
