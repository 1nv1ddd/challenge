"""
MCP-клиент (stdio): соединение + list_tools.

По умолчанию подключается к локальному scripts/minimal_mcp_server.py.

Переопределение сервера (env):
  MCP_CMD — команда (по умолчанию: текущий Python)
  MCP_ARGS — JSON-массив аргументов, например:
    MCP_ARGS='["-y","@modelcontextprotocol/server-everything"]' MCP_CMD=npx python scripts/mcp_list_tools.py

Опции CLI:
  --json  — вывести список tools в JSON (stdout), служебные сообщения в stderr

Запуск проверки:
  python scripts/mcp_list_tools.py
  python -m unittest tests.test_mcp_connection -v
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.exceptions import McpError

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SERVER = ROOT / "scripts" / "minimal_mcp_server.py"


def _server_params() -> StdioServerParameters:
    cmd = os.environ.get("MCP_CMD", sys.executable).strip()
    raw_args = os.environ.get("MCP_ARGS")
    if raw_args:
        args: list[str] = json.loads(raw_args)
    else:
        args = [str(DEFAULT_SERVER)]
    return StdioServerParameters(command=cmd, args=args, env=None)


async def list_tools_once(
    params: StdioServerParameters,
) -> tuple[dict, list[dict]]:
    """
    Устанавливает сессию, initialize, list_tools.
    Возвращает (server_info dict, tools as list of {name, description}).
    """
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


async def run(json_out: bool) -> int:
    if not json_out and not os.environ.get("MCP_ARGS"):
        if not DEFAULT_SERVER.is_file():
            print(
                f"error: default server not found: {DEFAULT_SERVER}",
                file=sys.stderr,
            )
            return 1

    params = _server_params()
    print(f"MCP: spawning {params.command} {params.args!r}", file=sys.stderr)

    try:
        server_info, tools = await list_tools_once(params)
    except (McpError, json.JSONDecodeError, FileNotFoundError, OSError) as e:
        print(f"error: MCP connection failed: {e}", file=sys.stderr)
        return 1
    except Exception as e:  # noqa: BLE001 — показать любую сбойную цепочку при отладке
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    print(
        f"Connected. Server: {server_info.get('name') or '?'} "
        f"{server_info.get('version') or ''}".strip(),
        file=sys.stderr,
    )

    if json_out:
        print(json.dumps({"server": server_info, "tools": tools}, ensure_ascii=False, indent=2))
    else:
        print(f"Tools ({len(tools)}):")
        for t in tools:
            line = f"  - {t['name']}"
            if t.get("description"):
                desc = t["description"].replace("\n", " ")
                line += f": {desc}"
            print(line)

    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="MCP client: connect and list tools")
    p.add_argument(
        "--json",
        action="store_true",
        help="Print tools as JSON on stdout",
    )
    args = p.parse_args()
    raise SystemExit(asyncio.run(run(json_out=args.json)))


if __name__ == "__main__":
    main()
