"""
MCP вокруг локального Git: лог коммитов, ветка, статус.

Корень репозитория:
  - по умолчанию — каталог проекта (родитель scripts/);
  - переопределение: переменная окружения GIT_REPO_ROOT (путь ДОЛЖЕН быть внутри того же проекта).

Нужен установленный `git` в PATH.

Запуск: python scripts/git_mcp_server.py
"""

import json
import os
import subprocess
from pathlib import Path

from mcp.server.fastmcp import FastMCP

app = FastMCP("local-git")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _repo_root() -> Path:
    raw = (os.environ.get("GIT_REPO_ROOT") or "").strip()
    root = Path(raw).expanduser().resolve() if raw else _PROJECT_ROOT.resolve()
    proj = _PROJECT_ROOT.resolve()
    try:
        root.relative_to(proj)
    except ValueError as e:
        raise ValueError(
            f"GIT_REPO_ROOT ({root}) должен быть внутри каталога проекта ({proj})"
        ) from e
    return root


def _git(args: list[str], *, timeout: float = 60.0) -> str:
    root = _repo_root()
    cmd = ["git", "-C", str(root), *args]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip() or f"exit {proc.returncode}"
        return f"Git ошибка: {err}"
    return proc.stdout.strip()


@app.tool()
def get_recent_commits(count: int = 5) -> str:
    """
    Показать последние коммиты в репозитории (oneline: хеш, сообщение, автор, дата).
    Аргумент count — сколько коммитов (от 1 до 50).
    """
    n = max(1, min(50, int(count)))
    out = _git(
        [
            "log",
            f"-n{n}",
            "--pretty=format:%h | %s | %an | %ad",
            "--date=short",
        ]
    )
    if out.startswith("Git ошибка"):
        return out
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    return json.dumps(
        {"commits": lines, "count": len(lines)},
        ensure_ascii=False,
        indent=2,
    )


@app.tool()
def get_current_branch() -> str:
    """Текущая ветка (HEAD) в репозитории."""
    out = _git(["rev-parse", "--abbrev-ref", "HEAD"])
    if out.startswith("Git ошибка"):
        return out
    return json.dumps({"branch": out}, ensure_ascii=False)


@app.tool()
def get_short_status() -> str:
    """Краткий статус рабочей копии (porcelain v1, первые 30 строк)."""
    out = _git(["status", "--porcelain=v1"])
    if out.startswith("Git ошибка"):
        return out
    lines = out.splitlines()[:30]
    return json.dumps(
        {"lines": lines, "truncated": len(out.splitlines()) > 30},
        ensure_ascii=False,
        indent=2,
    )


if __name__ == "__main__":
    app.run(transport="stdio")
