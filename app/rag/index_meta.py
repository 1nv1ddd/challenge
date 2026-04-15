"""Метаданные SQLite-индекса RAG без тяжёлых зависимостей (только sqlite3)."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from .pipeline import default_index_path


def index_needs_build(path: Path | None = None) -> bool:
    """True если файла нет или таблица chunks пуста."""
    p = path or default_index_path()
    if not p.is_file():
        return True
    con = sqlite3.connect(p)
    try:
        cur = con.execute("SELECT COUNT(*) FROM chunks")
        n = int(cur.fetchone()[0])
        return n == 0
    except sqlite3.OperationalError:
        return True
    finally:
        con.close()


def rag_index_stats(path: Path) -> dict[str, Any]:
    """Агрегаты по чанкам (та же форма, что и store.stats)."""
    con = sqlite3.connect(path)
    try:
        cur = con.execute(
            "SELECT strategy, COUNT(*), AVG(LENGTH(text)) FROM chunks GROUP BY strategy"
        )
        rows = cur.fetchall()
    finally:
        con.close()
    return {
        "by_strategy": [
            {"strategy": r[0], "count": r[1], "avg_text_len": round(r[2] or 0, 1)}
            for r in rows
        ]
}
