"""Одиночный вызов MCP tool через stdio (новый процесс сервера на каждый вызов)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult


def format_call_tool_result(result: CallToolResult) -> str:
    if result.isError:
        parts: list[str] = []
        for block in result.content or []:
            if getattr(block, "type", None) == "text" and hasattr(block, "text"):
                parts.append(block.text)
            else:
                parts.append(str(block))
        return "Ошибка инструмента: " + ("\n".join(parts) if parts else "(нет текста)")
    parts = []
    for block in result.content or []:
        if getattr(block, "type", None) == "text" and hasattr(block, "text"):
            parts.append(block.text)
        else:
            parts.append(str(block))
    if parts:
        return "\n".join(parts)
    if result.structuredContent is not None:
        try:
            return json.dumps(result.structuredContent, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            return str(result.structuredContent)
    return ""


async def call_tool_stdio(script: Path, tool_name: str, arguments: dict[str, Any]) -> str:
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(script.resolve())],
        env=None,
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            res = await session.call_tool(tool_name, arguments)
            return format_call_tool_result(res)
