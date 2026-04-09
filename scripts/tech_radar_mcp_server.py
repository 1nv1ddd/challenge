"""
MCP «Tech radar»: GitHub релиз → конспект → файл.

Инструменты: search (последний релиз/тег), summarize, saveToFile, run_pipeline.
Вывод: data/tech_radar_outputs/release_watch_<owner>_<repo>.md

Запуск: python scripts/tech_radar_mcp_server.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

app = FastMCP("challenge-tech-radar")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "tech_radar_outputs"
_FNAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,120}\.(md|txt)$")

GITHUB_HEADERS = {
    "Accept": "application/vnd.github+json",
    "User-Agent": "challenge-tech-radar-mcp",
    "X-GitHub-Api-Version": "2022-11-28",
}


def _safe_filename(name: str) -> str:
    base = Path(str(name).strip()).name
    if not _FNAME_RE.match(base):
        raise ValueError(
            "filename: только имя файла .md или .txt, безопасные символы",
        )
    return base


def _ensure_out_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def _parse_repository(repository: str) -> tuple[str, str]:
    s = (repository or "").strip().strip("/")
    if "/" not in s:
        raise ValueError('repository: укажите "owner/repo", например fastapi/fastapi')
    owner, name = s.split("/", 1)
    owner, name = owner.strip(), name.strip()
    if not owner or not name or ".." in owner or ".." in name:
        raise ValueError("некорректный owner/repo")
    return owner, name


def _release_payload(data: dict[str, Any]) -> tuple[str, str, str, str]:
    tag = str(data.get("tag_name") or data.get("name") or "—")
    body = str(data.get("body") or "").strip() or "(описание релиза пустое)"
    published = str(data.get("published_at") or data.get("created_at") or "—")
    url = str(data.get("html_url") or "—")
    return tag, body, published, url


def search_impl(repository: str) -> str:
    """Текстовый отчёт: последний релиз, тег или (если их нет) последний коммит и карточка репо."""
    try:
        owner, repo = _parse_repository(repository)
    except ValueError as e:
        return str(e)

    base = f"https://api.github.com/repos/{owner}/{repo}"
    lines: list[str] = [f"Репозиторий: {owner}/{repo}", ""]

    try:
        with httpx.Client(timeout=20.0, headers=GITHUB_HEADERS) as client:
            r = client.get(f"{base}/releases/latest")
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, dict):
                    tag, body, published, url = _release_payload(data)
                    lines.append(f"Тег/версия: {tag}")
                    lines.append(f"Дата публикации: {published}")
                    lines.append(f"Страница релиза: {url}")
                    lines.append("")
                    lines.append("Заметки релиза:")
                    lines.append(body[:12000])
                    return "\n".join(lines)

            if r.status_code not in (404, 403):
                r.raise_for_status()

            r2 = client.get(f"{base}/releases", params={"per_page": "5"})
            r2.raise_for_status()
            arr = r2.json()
            if isinstance(arr, list) and arr and isinstance(arr[0], dict):
                tag, body, published, url = _release_payload(arr[0])
                lines.append("(latest недоступен; первый из списка релизов)")
                lines.append(f"Тег/версия: {tag}")
                lines.append(f"Дата публикации: {published}")
                lines.append(f"Страница релиза: {url}")
                lines.append("")
                lines.append("Заметки релиза:")
                lines.append(body[:12000])
                return "\n".join(lines)

            r3 = client.get(f"{base}/tags", params={"per_page": "1"})
            r3.raise_for_status()
            tags = r3.json()
            if isinstance(tags, list) and tags and isinstance(tags[0], dict):
                tname = tags[0].get("name", "—")
                lines.append("Релизов с заметками нет; последний тег:")
                lines.append(str(tname))
                lines.append(f"Ссылка: https://github.com/{owner}/{repo}/releases")
                return "\n".join(lines)

            # Нет releases и тегов — типично для новых репо; репо при этом существует.
            r_repo = client.get(base)
            if r_repo.status_code == 404:
                lines.append(
                    "Репозиторий не найден (404). Проверьте owner/repo и что репо public "
                    "или у вас есть доступ."
                )
                return "\n".join(lines)
            r_repo.raise_for_status()
            info = r_repo.json()
            if not isinstance(info, dict):
                lines.append("Не удалось разобрать ответ GitHub по репозиторию.")
                return "\n".join(lines)

            desc = str(info.get("description") or "—")
            branch = str(info.get("default_branch") or "main")
            pushed = str(info.get("pushed_at") or info.get("updated_at") or "—")
            stars = info.get("stargazers_count", "—")
            lines.append("На GitHub **нет** опубликованных Releases и нет **тегов** — это нормально.")
            lines.append("Ниже снимок репозитория и **последний коммит** на ветке по умолчанию.")
            lines.append("")
            lines.append(f"Описание: {desc}")
            lines.append(f"Ветка по умолчанию: {branch}")
            lines.append(f"Последний push (метаданные): {pushed}")
            lines.append(f"Звёзды: {stars}")
            lines.append(f"URL: https://github.com/{owner}/{repo}")
            lines.append("")

            r_c = client.get(f"{base}/commits", params={"per_page": "1"})
            r_c.raise_for_status()
            commits = r_c.json()
            if isinstance(commits, list) and commits and isinstance(commits[0], dict):
                c0 = commits[0]
                cmt = c0.get("commit") if isinstance(c0.get("commit"), dict) else {}
                msg = str(cmt.get("message", "")).strip().split("\n")[0] or "—"
                ad = cmt.get("author") if isinstance(cmt.get("author"), dict) else {}
                date = str(ad.get("date") or "—")
                sha = str(c0.get("sha", ""))[:12]
                curl = str(c0.get("html_url") or f"https://github.com/{owner}/{repo}/commits")
                lines.append("Последний коммит:")
                lines.append(f"  SHA: {sha}")
                lines.append(f"  Дата: {date}")
                lines.append(f"  Сообщение: {msg}")
                lines.append(f"  Ссылка: {curl}")
            else:
                lines.append("(Не удалось получить список коммитов.)")

            return "\n".join(lines)
    except httpx.HTTPStatusError as e:
        return (
            f"GitHub API: HTTP {e.response.status_code} для {owner}/{repo}. "
            f"{( e.response.text or '')[:400]}"
        )
    except httpx.HTTPError as e:
        return f"Сеть/HTTP: {e}"


def summarize_impl(text: str, max_sentences: int = 4) -> str:
    """Сжатие + короткая рекомендация по обновлению зависимости."""
    raw = (text or "").strip()
    if not raw:
        return "(нечего резюмировать)"

    n = max(1, min(12, int(max_sentences)))
    parts = re.split(r"(?<=[.!?])\s+|\n+", raw)
    sentences = [p.strip() for p in parts if p.strip() and len(p) > 2]
    picked = sentences[:n]
    out = " ".join(picked)
    if len(out) > 2000:
        out = out[:1997] + "..."

    advice = (
        "\n\n**Стоит ли трогать проект:** если это **прямая** зависимость — просмотрите полный changelog "
        "и прогоните тесты перед bump версии. Если пакет только транзитивный — обычно достаточно "
        "обновлять вместе с основным фреймворком или по security-алертам."
    )
    return (out if out else raw[:1500]) + advice


def save_impl(filename: str, content: str) -> Path:
    safe = _safe_filename(filename)
    out_dir = _ensure_out_dir()
    path = (out_dir / safe).resolve()
    if not str(path).startswith(str(out_dir.resolve())):
        raise ValueError("path traversal")
    path.write_text(content, encoding="utf-8")
    return path


@app.tool()
def search(repository: str) -> str:
    """
    Получить сведения о последнем релизе (или теге) GitHub-репозитория.
    Вход: repository — строка вида owner/repo (например fastapi/fastapi).
    """
    return search_impl(repository)


@app.tool()
def summarize(text: str, max_sentences: int = 4) -> str:
    """
    Кратко пересказать текст релиза и дать ориентир по обновлению зависимости.
    Вход: text — сырой текст (например вывод search); max_sentences — сколько фрагментов взять.
    """
    return summarize_impl(text, max_sentences)


@app.tool(name="saveToFile")
def save_to_file(filename: str, content: str) -> str:
    """
    Сохранить markdown/txt в каталог data/tech_radar_outputs/.
    Вход: filename — только имя файла (.md или .txt); content — текст.
    """
    path = save_impl(filename, content)
    rel = path.relative_to(PROJECT_ROOT.resolve())
    return json.dumps(
        {"ok": True, "path": str(rel), "bytes": path.stat().st_size},
        ensure_ascii=False,
    )


@app.tool()
def run_pipeline(repository: str) -> str:
    """
    Пайплайн: search(repository) → summarize → saveToFile(release_watch_owner_repo.md).
    Возвращает JSON: шаги, превью, путь к файлу.
    """
    try:
        owner, repo = _parse_repository(repository)
    except ValueError as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)
    slug = f"{owner}_{repo}"
    fname = f"release_watch_{slug}.md"

    found = search_impl(repository)
    summary = summarize_impl(found, max_sentences=5)
    body = (
        f"# Tech radar: `{owner}/{repo}`\n\n"
        f"## Краткий разбор\n\n{summary}\n\n"
        f"---\n\n## Данные с GitHub\n\n```\n{found[:8000]}\n```\n"
    )
    path = save_impl(fname, body)
    rel = str(path.relative_to(PROJECT_ROOT.resolve()))
    return json.dumps(
        {
            "ok": True,
            "steps": ["search", "summarize", "saveToFile"],
            "repository": f"{owner}/{repo}",
            "search_preview": found[:500],
            "summary_preview": summary[:400],
            "saved_relative_path": rel,
            "bytes": path.stat().st_size,
        },
        ensure_ascii=False,
        indent=2,
    )


if __name__ == "__main__":
    app.run(transport="stdio")
