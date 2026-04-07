"""MCP: состояние подключения для UI и вызов list_tools по stdio."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.exceptions import McpError

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
            tools = [
                {
                    "name": t.name,
                    "description": (t.description or "").strip() or None,
                }
                for t in listed.tools
            ]
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
