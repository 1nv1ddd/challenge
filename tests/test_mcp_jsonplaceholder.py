"""Вызов инструмента MCP-сервера jsonplaceholder (нужна сеть)."""

from __future__ import annotations

import asyncio
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent import _parse_mcp_tool_call  # noqa: E402
from app.mcp_stdio_client import call_tool_stdio  # noqa: E402

_SERVER = ROOT / "scripts" / "jsonplaceholder_mcp_server.py"


class TestParseMcpFence(unittest.TestCase):
    def test_parse_tool_json(self) -> None:
        raw = 'ok\n```mcp\n{"name": "get_post", "arguments": {"post_id": 7}}\n```'
        p = _parse_mcp_tool_call(raw)
        self.assertIsNotNone(p)
        assert p is not None
        self.assertEqual(p["name"], "get_post")
        self.assertEqual(p["arguments"], {"post_id": 7})


@unittest.skipUnless(_SERVER.is_file(), "jsonplaceholder_mcp_server.py missing")
class TestJsonPlaceholderMcpTool(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("RUN_MCP_NET"), "set RUN_MCP_NET=1 to run network test")
    def test_get_post_returns_title(self) -> None:
        async def _run() -> str:
            return await call_tool_stdio(_SERVER, "get_post", {"post_id": 1})

        out = asyncio.run(_run())
        self.assertIn("sunt aut facere", out.lower())
