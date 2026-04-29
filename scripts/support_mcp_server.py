"""MCP-сервер тикетов и пользователей для AI-ассистента поддержки.

Tools:
  - get_ticket(ticket_id)
  - get_user(user_id)
  - list_user_tickets(user_id)
  - list_open_tickets(limit)

Хранилище: data/support_tickets.json (JSON, никаких БД).
Путь к файлу можно переопределить env-переменной SUPPORT_DATA_PATH.

Запуск standalone:
  python scripts/support_mcp_server.py
"""

import json
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

app = FastMCP("polarline-support")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DATA = _PROJECT_ROOT / "data" / "support_tickets.json"


def _data_path() -> Path:
    raw = (os.environ.get("SUPPORT_DATA_PATH") or "").strip()
    return Path(raw).expanduser().resolve() if raw else _DEFAULT_DATA


def _load() -> dict:
    p = _data_path()
    if not p.is_file():
        return {"users": [], "tickets": []}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return {"error": f"cannot read {p}: {e}"}


def _err(msg: str) -> str:
    return json.dumps({"error": msg}, ensure_ascii=False)


@app.tool()
def get_ticket(ticket_id: str) -> str:
    """Вернуть тикет по id (формат `TKT-NNN`). JSON: {id, user_id, title, status,
    priority, created_at, messages:[{role,text},...]}."""
    tid = (ticket_id or "").strip().upper()
    if not tid:
        return _err("ticket_id обязателен")
    db = _load()
    if "error" in db:
        return _err(db["error"])
    for t in db.get("tickets", []):
        if str(t.get("id", "")).upper() == tid:
            return json.dumps(t, ensure_ascii=False)
    return _err(f"тикет {tid} не найден")


@app.tool()
def get_user(user_id: str) -> str:
    """Профиль пользователя по id (формат `user-NNN`)."""
    uid = (user_id or "").strip().lower()
    if not uid:
        return _err("user_id обязателен")
    db = _load()
    if "error" in db:
        return _err(db["error"])
    for u in db.get("users", []):
        if str(u.get("id", "")).lower() == uid:
            return json.dumps(u, ensure_ascii=False)
    return _err(f"пользователь {uid} не найден")


@app.tool()
def list_user_tickets(user_id: str) -> str:
    """Все тикеты юзера (краткая сводка: id, title, status, priority)."""
    uid = (user_id or "").strip().lower()
    db = _load()
    if "error" in db:
        return _err(db["error"])
    out = []
    for t in db.get("tickets", []):
        if str(t.get("user_id", "")).lower() == uid:
            out.append(
                {
                    "id": t.get("id"),
                    "title": t.get("title"),
                    "status": t.get("status"),
                    "priority": t.get("priority"),
                }
            )
    return json.dumps({"user_id": uid, "tickets": out, "count": len(out)}, ensure_ascii=False)


@app.tool()
def list_open_tickets(limit: int = 20) -> str:
    """Открытые/в работе тикеты, отсортированы по приоритету (high → low)."""
    n = max(1, min(100, int(limit)))
    db = _load()
    if "error" in db:
        return _err(db["error"])
    open_states = {"open", "in_progress"}
    rank = {"high": 0, "medium": 1, "low": 2}
    items = [
        t for t in db.get("tickets", [])
        if str(t.get("status", "")).lower() in open_states
    ]
    items.sort(key=lambda t: rank.get(str(t.get("priority", "")).lower(), 9))
    out = [
        {
            "id": t.get("id"),
            "user_id": t.get("user_id"),
            "title": t.get("title"),
            "status": t.get("status"),
            "priority": t.get("priority"),
        }
        for t in items[:n]
    ]
    return json.dumps({"tickets": out, "count": len(out)}, ensure_ascii=False)


if __name__ == "__main__":
    app.run()
