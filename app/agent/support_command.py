"""День 33: команда /support — AI-агент поддержки PolarLine.

Срабатывает на префикс `/support` в сообщении пользователя (как `/help`):
форсит RAG=structural top_k=8 по корпусу проекта (FAQ + handbook),
подмешивает system-промпт «ты support-агент», и при наличии в тексте
маркеров `TKT-###` или `user-###` — лезет через MCP в JSON-сервер
тикетов и подкладывает данные тикета/юзера в контекст.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from ..mcp_stdio_client import call_tool_stdio
from ..providers import Message

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SUPPORT_MCP_SCRIPT = _PROJECT_ROOT / "scripts" / "support_mcp_server.py"
_PREFIX = "/support"
_DEFAULT_QUESTION = "Подскажи, какие у нас типичные сценарии поддержки и как ими пользоваться?"

_TICKET_RE = re.compile(r"\bTKT-\d{1,6}\b", re.IGNORECASE)
_USER_RE = re.compile(r"\buser-\d{1,8}\b", re.IGNORECASE)

_log = logging.getLogger(__name__)


def detect_support_command(text: str) -> tuple[bool, str]:
    s = (text or "").lstrip()
    if not s.lower().startswith(_PREFIX):
        return False, text
    rest = s[len(_PREFIX):].lstrip(" :-—")
    return True, rest or _DEFAULT_QUESTION


def force_support_rag_cfg(rag_cfg: dict | None) -> dict:
    base = dict(rag_cfg or {})
    base.update(
        {"enabled": True, "strategy": "structural", "top_k": 8, "help_mode": True}
    )
    return base


async def _call_support_tool(tool: str, args: dict) -> dict | None:
    if not _SUPPORT_MCP_SCRIPT.is_file():
        return None
    try:
        raw = await call_tool_stdio(_SUPPORT_MCP_SCRIPT, tool, args)
    except Exception as e:  # noqa: BLE001
        _log.warning("support MCP %s failed: %s", tool, e)
        return None
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, AttributeError):
        return {"raw": raw}
    return data if isinstance(data, dict) else {"raw": data}


async def gather_ticket_context(user_text: str) -> dict:
    """Возвращает {tickets: [...], users: [...]} — что удалось вытащить через MCP."""
    tickets_found: list[dict] = []
    users_found: list[dict] = []

    seen_tickets: set[str] = set()
    for m in _TICKET_RE.finditer(user_text):
        tid = m.group(0).upper()
        if tid in seen_tickets:
            continue
        seen_tickets.add(tid)
        data = await _call_support_tool("get_ticket", {"ticket_id": tid})
        if data and not data.get("error"):
            tickets_found.append(data)

    seen_users: set[str] = set()
    for m in _USER_RE.finditer(user_text):
        uid = m.group(0).lower()
        if uid in seen_users:
            continue
        seen_users.add(uid)
        data = await _call_support_tool("get_user", {"user_id": uid})
        if data and not data.get("error"):
            users_found.append(data)

    return {"tickets": tickets_found, "users": users_found}


def _format_ticket(t: dict) -> str:
    msgs = t.get("messages") or []
    history = "\n".join(
        f"  - {m.get('role','?')}: {str(m.get('text',''))[:200]}" for m in msgs[-6:]
    )
    return (
        f"### Тикет `{t.get('id','?')}`\n"
        f"- статус: **{t.get('status','?')}**, приоритет: **{t.get('priority','?')}**\n"
        f"- тема: {t.get('title','—')}\n"
        f"- автор: `{t.get('user_id','?')}`\n"
        f"- создан: {t.get('created_at','?')}\n"
        f"- история (последние реплики):\n{history or '  (нет сообщений)'}"
    )


def _format_user(u: dict) -> str:
    return (
        f"### Пользователь `{u.get('id','?')}`\n"
        f"- имя: {u.get('name','—')}\n"
        f"- тариф: **{u.get('plan','—')}**\n"
        f"- зарегистрирован: {u.get('registered','?')}\n"
        f"- открытых тикетов: {u.get('open_tickets', '?')}"
    )


def support_system_message(ticket_ctx: dict) -> Message:
    parts = [
        "Ты — AI-агент поддержки **PolarLine**. Твоя задача: помочь пользователю по продукту, "
        "опираясь на FAQ из RAG-индекса.",
        "",
        "ПРАВИЛА:",
        "- Отвечай на русском, дружелюбно, по делу. 3-6 предложений или короткий список.",
        "- Опирайся на отрывки из «Набор отрывков» — там FAQ и handbook.",
        "- Если в контексте есть данные тикета/юзера (см. ниже) — учитывай их в ответе "
        "(тариф, статус тикета, история).",
        "- Если в FAQ нет ответа — честно скажи «не нашёл в базе» и подскажи нажать "
        "«❌ Не помогло» чтобы перевести на оператора.",
        "- Не выдумывай шаги, цены, тарифы — только то, что есть в отрывках.",
        "- В конце добавь короткую секцию **«Что попробовать»** — 1-3 пункта конкретных действий.",
    ]
    tickets = ticket_ctx.get("tickets") or []
    users = ticket_ctx.get("users") or []
    if tickets or users:
        parts.append("")
        parts.append("## Контекст из CRM (через MCP)")
        for t in tickets:
            parts.append(_format_ticket(t))
        for u in users:
            parts.append(_format_user(u))
    return Message(role="system", content="\n".join(parts))
