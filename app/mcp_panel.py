"""MCP: несколько stdio-серверов, маршрутизация вызовов по server_id."""

from __future__ import annotations

import re
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

# server_id -> запись подключения
_servers: dict[str, dict[str, Any]] = {}

_SERVER_ID_RE = re.compile(r"^[a-z][a-z0-9_-]{0,48}$")


def _empty_public_state() -> dict[str, Any]:
    return {
        "connected": False,
        "servers": [],
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


def _serialize_tool(t: Tool, server_id: str) -> dict[str, Any]:
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
        "mcp_server_id": server_id,
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
            return server_info, [_serialize_tool(t, "") for t in listed.tools]


def _tools_with_server_id(server_id: str, tools: list[dict]) -> list[dict]:
    out = []
    for t in tools:
        d = dict(t)
        d["mcp_server_id"] = server_id
        out.append(d)
    return out


def _public_status() -> dict[str, Any]:
    if not _servers:
        return _empty_public_state()
    servers_list = []
    flat_tools: list[dict] = []
    for sid in sorted(_servers.keys()):
        s = _servers[sid]
        entry = {
            "id": sid,
            "server_name": s["server_name"],
            "script_path": s["script_path"],
            "server_info": s["server_info"],
            "tools": s["tools"],
        }
        servers_list.append(entry)
        flat_tools.extend(s["tools"])
    return {
        "connected": True,
        "multi_server": len(_servers) > 1,
        "server_count": len(_servers),
        "servers": servers_list,
        "tools": flat_tools,
        "server_name": servers_list[0]["server_name"] if len(servers_list) == 1 else "multi",
        "error": None,
    }


@router.get("/status")
async def mcp_status() -> dict[str, Any]:
    return _public_status()


@router.post("/connect")
async def mcp_connect(request: Request) -> dict[str, Any]:
    global _servers
    body = await request.json()
    transport = str(body.get("transport") or "stdio").lower().strip()
    server_name = str(body.get("server_name") or "MCP Server").strip() or "MCP Server"
    raw_id = str(body.get("server_id") or "").strip().lower()

    if transport != "stdio":
        raise HTTPException(
            status_code=400,
            detail="Сейчас поддерживается только transport=stdio (локальный Python-скрипт).",
        )

    if not raw_id:
        raw_id = "default"
    if not _SERVER_ID_RE.match(raw_id):
        raise HTTPException(
            status_code=400,
            detail='server_id: латиница, цифры, _ и -; начало с буквы; пример: "git", "radar"',
        )

    script_path = _resolve_script(str(body.get("script_path") or ""))
    rel = str(script_path.relative_to(PROJECT_ROOT.resolve()))

    try:
        server_info, tools_raw = await _list_tools_stdio(script_path)
    except (McpError, OSError, ValueError) as e:
        return {**_empty_public_state(), "ok": False, "error": str(e)}
    except Exception as e:  # noqa: BLE001
        return {
            **_empty_public_state(),
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
        }

    tools = _tools_with_server_id(raw_id, tools_raw)
    _servers[raw_id] = {
        "id": raw_id,
        "server_name": server_name,
        "script_path": rel,
        "server_info": server_info,
        "tools": tools,
    }
    out = _public_status()
    return {"ok": True, **out}


@router.post("/disconnect")
async def mcp_disconnect(request: Request) -> dict[str, Any]:
    global _servers
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        body = {}
    sid = str(body.get("server_id") or "").strip().lower()
    if sid:
        _servers.pop(sid, None)
    else:
        _servers = {}
    return {"ok": True, **_public_status()}


def get_mcp_bridge() -> dict[str, Any] | None:
    if not _servers:
        return None
    st = _public_status()
    return {
        "multi_server": st.get("multi_server", False),
        "server_count": st.get("server_count", 0),
        "servers": st.get("servers", []),
        "tools": st.get("tools", []),
        "server_name": st.get("server_name", "MCP"),
    }


def resolve_invocation(
    server_id: str | None,
    tool_name: str,
) -> tuple[Path, str, str]:
    """
    Возвращает (путь_к_скрипту, имя_инструмента, server_id).
    """
    if not _servers:
        raise ValueError("MCP: нет подключённых серверов")
    name = (tool_name or "").strip()
    if not name:
        raise ValueError("MCP: пустое имя инструмента")

    sid_in = (server_id or "").strip().lower() or None
    ids = list(_servers.keys())

    if sid_in:
        if sid_in not in _servers:
            raise ValueError(
                f"MCP: неизвестный server_id «{sid_in}». Доступны: {sorted(ids)}",
            )
        srv = _servers[sid_in]
        if not any(t.get("name") == name for t in srv["tools"]):
            raise ValueError(
                f"MCP: на сервере «{sid_in}» нет инструмента «{name}»",
            )
        p = (PROJECT_ROOT / srv["script_path"]).resolve()
        if not p.is_file():
            raise ValueError("MCP: скрипт сервера не найден")
        return p, name, sid_in

    if len(ids) == 1:
        only = ids[0]
        srv = _servers[only]
        if not any(t.get("name") == name for t in srv["tools"]):
            raise ValueError(f"MCP: нет инструмента «{name}»")
        p = (PROJECT_ROOT / srv["script_path"]).resolve()
        return p, name, only

    matching = [
        sid for sid in ids if any(t.get("name") == name for t in _servers[sid]["tools"])
    ]
    if not matching:
        raise ValueError(
            f"MCP: инструмент «{name}» не найден ни на одном сервере. "
            f"Подключены: {sorted(ids)}",
        )
    if len(matching) > 1:
        raise ValueError(
            f"MCP: инструмент «{name}» есть на нескольких серверах {matching}. "
            f"Укажите в JSON поле \"server\" (например \"{matching[0]}\").",
        )
    sid = matching[0]
    srv = _servers[sid]
    p = (PROJECT_ROOT / srv["script_path"]).resolve()
    return p, name, sid


def mcp_call_allowed(server_id: str | None, tool_name: str) -> bool:
    try:
        resolve_invocation(server_id, tool_name)
        return True
    except ValueError:
        return False


def tool_name_allowed(name: str) -> bool:
    """Совместимость: разрешён ли tool при одном сервере или если имя уникально."""
    return mcp_call_allowed(None, name)


async def invoke_mcp_tool(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    server_id: str | None = None,
) -> str:
    path, tname, _ = resolve_invocation(server_id, tool_name)
    return await call_tool_stdio(path, tname, dict(arguments or {}))


def reset_mcp_servers_for_tests() -> None:
    """Только для тестов."""
    global _servers
    _servers = {}
