"""MCP-сервер для работы с файлами проекта (День 34).

Tools:
  - list_files(pattern, max_results) — поиск файлов по glob-паттерну.
  - search_in_files(query, glob_filter, max_results) — поиск подстроки во всех md/py/yml/...
  - read_file(path, max_lines) — прочитать файл (с лимитом строк).
  - write_file(path, content) — создать/перезаписать файл (с whitelist'ом расширений
    и blacklist'ом путей).
  - preview_patch(path, old_str, new_str) — показать unified diff без записи.

Безопасность:
  - Все пути резолвятся внутри корня проекта (родитель scripts/), выход за границы
    запрещён (включая symlink'и).
  - Запись разрешена только для расширений из ALLOWED_WRITE_EXTS.
  - Запись запрещена для путей, начинающихся с любого префикса из BLOCKED_PREFIXES,
    и для конкретных файлов из BLOCKED_FILES.
"""

import difflib
import fnmatch
import json
import os
import re
import tempfile
from pathlib import Path

from mcp.server.fastmcp import FastMCP

app = FastMCP("project-files")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

ALLOWED_WRITE_EXTS = {
    ".py", ".md", ".yaml", ".yml", ".json", ".txt", ".toml",
    ".conf", ".sql", ".html", ".css", ".js", ".ts", ".cfg", ".ini",
}
ALLOWED_WRITE_FILENAMES = {
    "Dockerfile", "docker-compose.yml", ".gitignore", ".env.example",
    "Makefile", "requirements.txt",
}

BLOCKED_PREFIXES = (
    ".git/",
    ".venv/",
    "__pycache__/",
    "data/rag_index/",
    "node_modules/",
)
BLOCKED_FILES = {
    ".env",
    "data/agent_memory.json",
    "data/mcp_scheduler.sqlite",
    "data/support_tickets.json",  # пишет бот, MCP не должен
}

DEFAULT_SEARCH_GLOB = "**/*.{py,md,yml,yaml,json,html,css,js,ts,toml,conf}"


def _err(msg: str) -> str:
    return json.dumps({"error": msg}, ensure_ascii=False)


def _resolve_inside_root(rel_path: str) -> Path | None:
    """Резолвит rel_path относительно корня. Возвращает None если за пределами."""
    if not rel_path:
        return None
    p = (_PROJECT_ROOT / rel_path).resolve()
    try:
        p.relative_to(_PROJECT_ROOT.resolve())
    except ValueError:
        return None
    return p


def _is_blocked_for_write(rel_path: str, p: Path) -> str | None:
    """Возвращает причину блокировки или None если можно писать."""
    rel_norm = rel_path.replace("\\", "/").lstrip("./")
    if rel_norm in BLOCKED_FILES:
        return f"путь {rel_norm} в чёрном списке"
    for pref in BLOCKED_PREFIXES:
        if rel_norm.startswith(pref):
            return f"путь {rel_norm} начинается с заблокированного префикса {pref}"
    name = p.name
    if name in ALLOWED_WRITE_FILENAMES:
        return None
    if p.suffix.lower() not in ALLOWED_WRITE_EXTS:
        return f"расширение {p.suffix or '(нет)'} не в whitelist'е (см. ALLOWED_WRITE_EXTS)"
    return None


def _split_glob_braces(pattern: str) -> list[str]:
    """Развернуть `**/*.{py,md}` в `**/*.py`, `**/*.md` (rglob брейсы не понимает)."""
    m = re.match(r"^(.*)\{([^{}]+)\}(.*)$", pattern)
    if not m:
        return [pattern]
    prefix, alts, suffix = m.group(1), m.group(2), m.group(3)
    return [f"{prefix}{a.strip()}{suffix}" for a in alts.split(",")]


def _iter_files(pattern: str):
    for pat in _split_glob_braces(pattern):
        for p in _PROJECT_ROOT.glob(pat):
            if p.is_file() and not _is_excluded(p):
                yield p


def _is_excluded(p: Path) -> bool:
    rel = p.resolve().relative_to(_PROJECT_ROOT.resolve()).as_posix()
    for pref in BLOCKED_PREFIXES:
        if rel.startswith(pref):
            return True
    if rel in BLOCKED_FILES:
        return True
    return False


@app.tool()
def list_files(pattern: str = "**/*.py", max_results: int = 50) -> str:
    """Найти файлы по glob-паттерну от корня проекта. Поддерживает `**/*.{py,md}`.
    Возвращает JSON: {files: [...], count, truncated}."""
    n = max(1, min(500, int(max_results)))
    found: list[str] = []
    for p in _iter_files(pattern):
        rel = p.resolve().relative_to(_PROJECT_ROOT.resolve()).as_posix()
        found.append(rel)
        if len(found) >= n:
            break
    truncated = len(found) >= n
    return json.dumps(
        {"files": found, "count": len(found), "truncated": truncated},
        ensure_ascii=False,
    )


@app.tool()
def search_in_files(
    query: str,
    glob_filter: str = DEFAULT_SEARCH_GLOB,
    max_results: int = 40,
    case_sensitive: bool = False,
) -> str:
    """Поиск подстроки `query` по файлам, отобранным `glob_filter`. Возвращает
    JSON: {hits: [{path, line, text}, ...], count, truncated}.
    `glob_filter` — например `**/*.py`, `app/**/*.md`, `**/*.{py,yml}`."""
    if not query or not query.strip():
        return _err("query пустой")
    n = max(1, min(200, int(max_results)))
    needle = query if case_sensitive else query.lower()
    hits: list[dict] = []
    for p in _iter_files(glob_filter):
        try:
            text = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        haystack = text if case_sensitive else text.lower()
        if needle not in haystack:
            continue
        rel = p.resolve().relative_to(_PROJECT_ROOT.resolve()).as_posix()
        for i, line in enumerate(text.splitlines(), start=1):
            cmp = line if case_sensitive else line.lower()
            if needle in cmp:
                hits.append({"path": rel, "line": i, "text": line[:240]})
                if len(hits) >= n:
                    break
        if len(hits) >= n:
            break
    return json.dumps(
        {"hits": hits, "count": len(hits), "truncated": len(hits) >= n},
        ensure_ascii=False,
    )


@app.tool()
def read_file(path: str, max_lines: int = 400, offset: int = 0) -> str:
    """Прочитать файл проекта (UTF-8). `offset` — с какой строки начать (0-based),
    `max_lines` — сколько строк вернуть. Возвращает JSON: {path, total_lines,
    returned_range, content}."""
    p = _resolve_inside_root(path)
    if p is None:
        return _err(f"путь {path} вне корня проекта")
    if not p.is_file():
        return _err(f"файл {path} не найден")
    try:
        text = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        return _err(f"не удалось прочитать {path}: {e}")
    lines = text.splitlines()
    total = len(lines)
    start = max(0, int(offset))
    end = min(total, start + max(1, min(2000, int(max_lines))))
    chunk = "\n".join(lines[start:end])
    return json.dumps(
        {
            "path": path,
            "total_lines": total,
            "returned_range": [start, end],
            "content": chunk,
        },
        ensure_ascii=False,
    )


@app.tool()
def preview_patch(path: str, old_str: str, new_str: str) -> str:
    """Показать unified diff применения замены `old_str` → `new_str` в файле,
    БЕЗ записи. Полезно перед `write_file`, чтобы юзер увидел план.
    Возвращает {path, occurrences, diff}."""
    p = _resolve_inside_root(path)
    if p is None:
        return _err(f"путь {path} вне корня")
    if not p.is_file():
        return _err(f"файл {path} не найден")
    try:
        text = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        return _err(f"не удалось прочитать {path}: {e}")
    occurrences = text.count(old_str) if old_str else 0
    if occurrences == 0:
        return _err(f"old_str не найден в {path}")
    new_text = text.replace(old_str, new_str)
    diff = "".join(
        difflib.unified_diff(
            text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=path,
            tofile=path + " (proposed)",
            n=3,
        )
    )
    return json.dumps(
        {"path": path, "occurrences": occurrences, "diff": diff},
        ensure_ascii=False,
    )


@app.tool()
def write_file(path: str, content: str) -> str:
    """ЕДИНСТВЕННЫЙ способ создать или изменить файл проекта. Если пользователь
    просит «сохрани», «создай», «обнови», «запиши» — обязательно вызывай этот
    tool с готовым содержимым в `content`. Не пиши содержимое файла обычным
    текстом — без этого вызова файл реально не появится.

    Атомарная запись через temp+rename. Расширение должно быть в whitelist'е
    (py/md/yml/json/...), путь — не в blacklist'е (.env, .git/, .venv/,
    data/*.sqlite). Возвращает {path, bytes_written, created}
    ('created'=true если файла не было)."""
    p = _resolve_inside_root(path)
    if p is None:
        return _err(f"путь {path} вне корня проекта")
    block = _is_blocked_for_write(path, p)
    if block is not None:
        return _err(f"запись запрещена: {block}")
    p.parent.mkdir(parents=True, exist_ok=True)
    created = not p.is_file()
    try:
        # Атомарная запись: tmp в той же директории + rename.
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=str(p.parent),
            prefix=p.name + ".", suffix=".tmp", delete=False,
        ) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        tmp_path.replace(p)
    except OSError as e:
        return _err(f"не удалось записать {path}: {e}")
    return json.dumps(
        {"path": path, "bytes_written": len(content.encode("utf-8")), "created": created},
        ensure_ascii=False,
    )


if __name__ == "__main__":
    app.run()
