"""Storage helpers for support tickets (shared with the MCP server).

Single JSON file `data/support_tickets.json` — both the bot writes and the
MCP server reads from it, so the AI agent gets up-to-date context.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Any


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_PATH = _PROJECT_ROOT / "data" / "support_tickets.json"
_TKT_RE = re.compile(r"TKT-(\d+)", re.IGNORECASE)


def data_path() -> Path:
    raw = (os.environ.get("SUPPORT_DATA_PATH") or "").strip()
    return Path(raw).expanduser().resolve() if raw else _DEFAULT_PATH


_io_lock = asyncio.Lock()


def _read_sync() -> dict:
    p = data_path()
    if not p.is_file():
        return {"users": [], "tickets": []}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"users": [], "tickets": []}


def _write_sync(db: dict) -> None:
    p = data_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(db, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)


async def read_db() -> dict:
    async with _io_lock:
        return await asyncio.to_thread(_read_sync)


async def write_db(db: dict) -> None:
    async with _io_lock:
        await asyncio.to_thread(_write_sync, db)


def _next_ticket_id(db: dict) -> str:
    max_n = 0
    for t in db.get("tickets", []):
        m = _TKT_RE.search(str(t.get("id", "")))
        if m:
            max_n = max(max_n, int(m.group(1)))
    return f"TKT-{max_n + 1:03d}"


async def create_ticket(
    *,
    title: str,
    user_text: str,
    ai_answer: str,
    tg_chat_id: int,
    tg_username: str | None,
    tg_full_name: str,
    priority: str = "medium",
) -> dict:
    db = await read_db()
    tid = _next_ticket_id(db)
    ticket = {
        "id": tid,
        "user_id": None,
        "title": title[:120],
        "status": "open",
        "priority": priority,
        "created_at": time.strftime("%Y-%m-%d"),
        "tg_chat_id": tg_chat_id,
        "tg_username": tg_username,
        "tg_full_name": tg_full_name,
        "messages": [
            {"role": "user", "text": user_text},
            {"role": "ai", "text": ai_answer},
        ],
    }
    db.setdefault("tickets", []).append(ticket)
    await write_db(db)
    return ticket


async def get_ticket(ticket_id: str) -> dict | None:
    tid = (ticket_id or "").strip().upper()
    db = await read_db()
    for t in db.get("tickets", []):
        if str(t.get("id", "")).upper() == tid:
            return t
    return None


async def list_open_tickets(limit: int = 20) -> list[dict]:
    db = await read_db()
    open_states = {"open", "in_progress"}
    rank = {"high": 0, "medium": 1, "low": 2}
    items = [
        t for t in db.get("tickets", [])
        if str(t.get("status", "")).lower() in open_states
    ]
    items.sort(key=lambda t: rank.get(str(t.get("priority", "")).lower(), 9))
    return items[:limit]


async def append_message(ticket_id: str, role: str, text: str) -> dict | None:
    tid = (ticket_id or "").strip().upper()
    db = await read_db()
    for t in db.get("tickets", []):
        if str(t.get("id", "")).upper() == tid:
            t.setdefault("messages", []).append({"role": role, "text": text})
            await write_db(db)
            return t
    return None


async def set_status(ticket_id: str, status: str) -> dict | None:
    tid = (ticket_id or "").strip().upper()
    db = await read_db()
    for t in db.get("tickets", []):
        if str(t.get("id", "")).upper() == tid:
            t["status"] = status
            await write_db(db)
            return t
    return None


def short_preview(t: dict[str, Any]) -> str:
    title = (t.get("title") or "—")[:60]
    return (
        f"{t.get('id','?')} · {t.get('priority','?')} · {t.get('status','?')}\n"
        f"  {title}"
    )
