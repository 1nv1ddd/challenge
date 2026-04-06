"""Проверка: MCP-соединение к локальному серверу и корректный list_tools."""

from __future__ import annotations

import asyncio
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp import StdioServerParameters  # noqa: E402

from scripts.mcp_list_tools import list_tools_once  # noqa: E402


class TestMcpLocalConnection(unittest.TestCase):
    @unittest.skipUnless(
        (ROOT / "scripts" / "minimal_mcp_server.py").is_file(),
        "minimal_mcp_server.py missing",
    )
    def test_list_tools_default_stdio_server(self) -> None:
        """Соединение устанавливается; список инструментов содержит ping и echo."""

        async def _probe() -> list[str]:
            params = StdioServerParameters(
                command=sys.executable,
                args=[str(ROOT / "scripts" / "minimal_mcp_server.py")],
                env=os.environ.copy(),
            )
            _info, tools = await list_tools_once(params)
            return sorted(t["name"] for t in tools)

        names = asyncio.run(_probe())
        self.assertEqual(names, ["echo", "ping"])


if __name__ == "__main__":
    unittest.main()
