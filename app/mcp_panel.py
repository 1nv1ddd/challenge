"""MCP: состояние подключения для UI и вызов list_tools по stdio."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.exceptions import McpError
from mcp.types import Tool

from .mcp_stdio_client import call_tool_stdio

PROJECT_ROOT = Path(__file__).resolve().parent.parent

router = APIRouter(prefix="/api/mcp", tags=["mcp"])

_state: dict[str, Any] = {
    "connected": False,
    "transport": None,
    "server_name": None,
    "script_path": None,
    "server_info": None,
    "tools": [],
    "error": None,
}


def _empty_state() -> dict[str, Any]:
    return {
        "connected": False,
        "transport": None,
        "server_name": None,
        "script_path": None,
        "server_info": None,
        "tools": [],
        "error": None,
    }


def _resolve_script(user_path: str) -> Path:
    raw = (user_path or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="script_path is required")
    p = Path(raw)
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    else:
        p = p.resolve()
    root = PROJECT_ROOT.resolve()
    try:
        p.relative_to(root)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Path must be inside project directory",
        ) from None
    if not p.is_file():
        raise HTTPException(status_code=400, detail="File not found")
    if p.suffix.lower() != ".py":
        raise HTTPException(status_code=400, detail="Must be a .py file")
    return p


def _serialize_tool(t: Tool) -> dict[str, Any]:
    raw_schema = getattr(t, "inputSchema", None) or {}
    if hasattr(raw_schema, "model_dump"):
        input_schema: Any = raw_schema.model_dump(mode="json")
    elif isinstance(raw_schema, dict):
        input_schema = raw_schema
    else:
        input_schema = {}
    t_title = getattr(t, "title", None)
    title = t_title if isinstance(t_title, str) else None
    return {
        "name": t.name,
        "title": title,
        "description": (t.description or "").strip() or None,
        "input_schema": input_schema,
    }


async def _list_tools_stdio(script: Path) -> tuple[dict, list[dict]]:
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(script)],
        env=None,
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            server_info = {
                "name": init.serverInfo.name if init.serverInfo else None,
                "version": init.serverInfo.version if init.serverInfo else None,
            }
            listed = await session.list_tools()
            tools = [_serialize_tool(t) for t in listed.tools]
            return server_info, tools


@router.get("/status")
async def mcp_status() -> dict[str, Any]:
    return {**_state}

@router.post("/connect")
async def mcp_connect(request: Request) -> dict[str, Any]:
    global _state
    body = await request.json()
    transport = str(body.get("transport") or "stdio").lower().strip()
    server_name = str(body.get("server_name") or "MCP Server").strip() or "MCP Server"

    if transport != "stdio":
        raise HTTPException(
            status_code=400,
            detail="Сейчас поддерживается только transport=stdio (локальный Python-скрипт).",
        )

    script_path = _resolve_script(str(body.get("script_path") or ""))
    rel = str(script_path.relative_to(PROJECT_ROOT.resolve()))

    try:
        server_info, tools = await _list_tools_stdio(script_path)
    except (McpError, OSError, ValueError) as e:
        _state = _empty_state()
        _state["error"] = str(e)
        return {"ok": False, "error": str(e), **_state}
    except Exception as e:  # noqa: BLE001
        _state = _empty_state()
        _state["error"] = f"{type(e).__name__}: {e}"
        return {"ok": False, "error": _state["error"], **_state}

    _state = {
        "connected": True,
        "transport": "stdio",
        "server_name": server_name,
        "script_path": rel,
        "server_info": server_info,
        "tools": tools,
        "error": None,
    }
    return {"ok": True, **_state}


@router.post("/disconnect")
async def mcp_disconnect() -> dict[str, Any]:
    global _state
    _state = _empty_state()
    return {"ok": True, **_state}


def get_mcp_bridge() -> dict[str, Any] | None:
    """Снимок подключения для агента (инструкция + список инструментов)."""
    if not _state.get("connected") or not _state.get("script_path"):
        return None
    return {
        "server_name": _state.get("server_name") or "MCP",
        "tools": list(_state.get("tools") or []),
    }


def mcp_allowed_tool_names() -> frozenset[str]:
    if not _state.get("connected"):
        return frozenset()
    names: list[str] = []
    for t in _state.get("tools") or []:
        if isinstance(t, dict) and t.get("name"):
            names.append(str(t["name"]))
    return frozenset(names)


def tool_name_allowed(name: str) -> bool:
    return name in mcp_allowed_tool_names()


def _resolved_script_path() -> Path | None:
    rel = _state.get("script_path")
    if not rel:
        return None
    p = (PROJECT_ROOT / str(rel)).resolve()
    root = PROJECT_ROOT.resolve()
    try:
        p.relative_to(root)
    except ValueError:
        return None
    return p if p.is_file() else None


async def invoke_mcp_tool(tool_name: str, arguments: dict[str, Any]) -> str:
    if not tool_name_allowed(tool_name):
        raise ValueError(
            f"MCP: неизвестный инструмент «{tool_name}». "
            f"Доступны: {sorted(mcp_allowed_tool_names())}"
        )
    script = _resolved_script_path()
    if not script:
        raise RuntimeError("MCP: путь к серверу недоступен")
    return await call_tool_stdio(script, tool_name, dict(arguments or {}))
