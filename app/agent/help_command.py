"""День 31: команда /help — ассистент-разработчик отвечает по проекту через RAG."""

from __future__ import annotations

import subprocess
from pathlib import Path

from ..providers import Message

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_HELP_PREFIX = "/help"
_DEFAULT_QUESTION = (
    "Расскажи про этот проект: что он делает, из каких слоёв состоит и как запустить."
)


def detect_help_command(text: str) -> tuple[bool, str]:
    """Возвращает (is_help, очищенный_от_префикса_вопрос)."""
    s = (text or "").lstrip()
    if not s.lower().startswith(_HELP_PREFIX):
        return False, text
    rest = s[len(_HELP_PREFIX):].lstrip(" :-—")
    return True, rest or _DEFAULT_QUESTION


def get_current_git_branch() -> str:
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


def help_system_message(branch: str | None = None) -> Message:
    br = branch or get_current_git_branch()
    content = (
        "Ты ассистент-разработчик проекта **AI Chat Hub** (FastAPI + RAG + Ollama + MCP).\n"
        f"Текущая git-ветка проекта: `{br}`.\n\n"
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
    """`/help` всегда форсит RAG=structural top_k=8 даже если в UI выключен."""
    base = dict(rag_cfg or {})
    base.update({"enabled": True, "strategy": "structural", "top_k": 8})
    return base
