"""День 31: команда /help — ассистент-разработчик отвечает по проекту через RAG.

Текущая git-ветка тянется через **MCP**: бэкенд поднимает stdio-сессию с
`scripts/git_mcp_server.py` и зовёт tool `get_current_branch`. Subprocess
оставлен как fallback — если MCP-вызов упал (например, в Docker нет git
или скрипт переименовали), `/help` всё равно ответит, просто с пометкой
ветки `(unavailable)`.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

from ..mcp_stdio_client import call_tool_stdio
from ..providers import Message

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_GIT_MCP_SCRIPT = _PROJECT_ROOT / "scripts" / "git_mcp_server.py"
_HELP_PREFIX = "/help"
_DEFAULT_QUESTION = (
    "Расскажи про этот проект: что он делает, из каких слоёв состоит и как запустить."
)

_log = logging.getLogger(__name__)


def detect_help_command(text: str) -> tuple[bool, str]:
    """Возвращает (is_help, очищенный_от_префикса_вопрос)."""
    s = (text or "").lstrip()
    if not s.lower().startswith(_HELP_PREFIX):
        return False, text
    rest = s[len(_HELP_PREFIX):].lstrip(" :-—")
    return True, rest or _DEFAULT_QUESTION


def _git_branch_via_subprocess() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(_PROJECT_ROOT),
            text=True,
            timeout=2,
            stderr=subprocess.DEVNULL,
        ).strip()
        return out or "(detached)"
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return "(unavailable)"


async def get_current_git_branch_via_mcp() -> tuple[str, str]:
    """(branch, source) — `mcp` если получилось через MCP-сервер, иначе `subprocess`."""
    if _GIT_MCP_SCRIPT.is_file():
        try:
            raw = await call_tool_stdio(_GIT_MCP_SCRIPT, "get_current_branch", {})
            try:
                payload = json.loads(raw)
                br = str(payload.get("branch") or "").strip()
            except (json.JSONDecodeError, AttributeError):
                br = raw.strip()
            if br and not br.startswith("Git ошибка"):
                return br, "mcp"
            _log.warning("git MCP returned unexpected: %r", raw)
        except Exception as e:  # noqa: BLE001 — нам важно не упасть
            _log.warning("git MCP call failed (%s); fallback to subprocess", e)
    return _git_branch_via_subprocess(), "subprocess"


async def help_system_message(branch: str | None = None) -> Message:
    if branch:
        br, source = branch, "explicit"
    else:
        br, source = await get_current_git_branch_via_mcp()
    content = (
        "Ты ассистент-разработчик проекта **AI Chat Hub** (FastAPI + RAG + Ollama + MCP).\n"
        f"Текущая git-ветка проекта: `{br}` (источник: {source}).\n\n"
        "ПРАВИЛА:\n"
        "- Отвечай на вопросы про проект **только** на основе блоков «Набор отрывков».\n"
        "- Если факта нет в отрывках — честно скажи «в индексе не нашёл» и предложи где искать.\n"
        "- Не выдумывай файлы, эндпоинты, env-переменные, которых нет в отрывках.\n"
        "- Структура ответа:\n"
        "  1. Краткий ответ по делу (1–3 абзаца).\n"
        "  2. Секция **«Где смотреть»** — bullet-список путей файлов из отрывков, по 1 строке."
    )
    return Message(role="system", content=content)


def force_help_rag_cfg(rag_cfg: dict | None) -> dict:
    """`/help` всегда форсит RAG=structural top_k=8 даже если в UI выключен.

    `help_mode=True` нужен, чтобы Day-25 детектор meta-сообщений (`Цель:`,
    `term: …`) не вырубил retrieval, когда дефолтный /help-промпт начинается
    с «Расскажи про этот проект: …» — он формально проходит как `key: value`.
    """
    base = dict(rag_cfg or {})
    base.update(
        {"enabled": True, "strategy": "structural", "top_k": 8, "help_mode": True}
    )
    return base
