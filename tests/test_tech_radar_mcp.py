"""Tech radar MCP: цепочка search → summarize → saveToFile и совпадение с пошаговыми вызовами."""

from __future__ import annotations

import asyncio
import json
import socket
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.mcp_stdio_client import call_tool_stdio  # noqa: E402

_SCRIPT = ROOT / "scripts" / "tech_radar_mcp_server.py"
_OUT = ROOT / "data" / "tech_radar_outputs"


def _github_reachable() -> bool:
    try:
        socket.create_connection(("api.github.com", 443), timeout=5).close()
        return True
    except OSError:
        return False


@unittest.skipUnless(_SCRIPT.is_file(), "tech_radar_mcp_server.py missing")
@unittest.skipUnless(_github_reachable(), "needs api.github.com")
class TestTechRadarMcp(unittest.TestCase):
    def test_run_pipeline_github_chain(self) -> None:
        out_md = _OUT / "release_watch_fastapi_fastapi.md"
        if out_md.exists():
            out_md.unlink()

        async def _run() -> str:
            return await call_tool_stdio(
                _SCRIPT,
                "run_pipeline",
                {"repository": "fastapi/fastapi"},
            )

        raw = asyncio.run(_run())
        self.assertNotIn("Ошибка инструмента", raw)
        data = json.loads(raw)
        self.assertTrue(data.get("ok"), data)
        self.assertEqual(data.get("steps"), ["search", "summarize", "saveToFile"])
        self.assertIn("search_preview", data)
        self.assertIn("summary_preview", data)

        p = ROOT / str(data["saved_relative_path"])
        self.assertTrue(p.is_file())
        text = p.read_text(encoding="utf-8")
        self.assertIn("Tech radar", text)
        self.assertIn("fastapi", text.lower())

    def test_summary_matches_manual_chain(self) -> None:
        repo = "fastapi/fastapi"

        async def _seq() -> tuple[str, str]:
            found = await call_tool_stdio(_SCRIPT, "search", {"repository": repo})
            summ = await call_tool_stdio(
                _SCRIPT,
                "summarize",
                {"text": found, "max_sentences": 5},
            )
            return found, summ

        async def _pipe() -> str:
            return await call_tool_stdio(_SCRIPT, "run_pipeline", {"repository": repo})

        found, manual_summary = asyncio.run(_seq())
        pipe_raw = asyncio.run(_pipe())
        data = json.loads(pipe_raw)
        self.assertTrue(data.get("ok"), data)
        self.assertEqual(data.get("summary_preview"), manual_summary[:400])
        self.assertEqual(data.get("search_preview"), found[:500])
