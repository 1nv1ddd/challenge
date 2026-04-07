"""Локальный MCP git-сервер (без сети; нужен git и .git в проекте)."""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.mcp_stdio_client import call_tool_stdio  # noqa: E402

_SCRIPT = ROOT / "scripts" / "git_mcp_server.py"
_GIT_DIR = ROOT / ".git"


@unittest.skipUnless(_SCRIPT.is_file(), "git_mcp_server.py missing")
@unittest.skipUnless(_GIT_DIR.is_dir(), "not a git checkout")
class TestGitMcpTool(unittest.TestCase):
    def test_recent_commits_returns_json(self) -> None:
        async def _run() -> str:
            return await call_tool_stdio(_SCRIPT, "get_recent_commits", {"count": 5})

        out = asyncio.run(_run())
        self.assertNotIn("Git ошибка", out)
        self.assertIn("commits", out.lower())
