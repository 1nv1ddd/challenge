"""SQLite-хранилище для периодических MCP-задач и их результатов."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from pathlib import Path

import httpx


def _db_file() -> Path:
    raw = (os.environ.get("SCHEDULER_SQLITE_PATH") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return Path(__file__).resolve().parent.parent / "data" / "mcp_scheduler.sqlite"

_TASK_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_ALLOWED_TYPES = frozenset({"reminder", "http_sample", "heartbeat_rollup"})


def _connect() -> sqlite3.Connection:
    path = _db_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_schema() -> None:
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS jobs (
              task_id TEXT PRIMARY KEY,
              interval_seconds INTEGER NOT NULL,
              task_type TEXT NOT NULL,
              payload TEXT,
              next_run REAL NOT NULL,
              last_run REAL,
              created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS results (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              task_id TEXT NOT NULL,
              ts REAL NOT NULL,
              content TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_results_task ON results(task_id);
            CREATE INDEX IF NOT EXISTS idx_jobs_next ON jobs(next_run);
            """
        )
        conn.commit()


def register_job(
    task_id: str,
    interval_seconds: int,
    task_type: str,
    payload: str = "",
    first_run_in_seconds: int = 15,
) -> dict:
    if not _TASK_ID_RE.match(task_id or ""):
        raise ValueError(
            "task_id: только латиница, цифры, _ и -, длина 1–64",
        )
    tt = (task_type or "").strip()
    if tt not in _ALLOWED_TYPES:
        raise ValueError(
            f"task_type должен быть одним из: {', '.join(sorted(_ALLOWED_TYPES))}",
        )
    iv = max(15, min(86400, int(interval_seconds)))
    delay = max(5, min(3600, int(first_run_in_seconds)))
    now = time.time()
    next_run = now + delay
    created = now
    with _connect() as conn:
        init_schema()
        exists = conn.execute(
            "SELECT 1 FROM jobs WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        if exists:
            conn.execute(
                """
                UPDATE jobs SET interval_seconds = ?, task_type = ?, payload = ?,
                  next_run = ?, last_run = NULL
                WHERE task_id = ?
                """,
                (iv, tt, payload or "", next_run, task_id),
            )
        else:
            conn.execute(
                """
                INSERT INTO jobs (task_id, interval_seconds, task_type, payload, next_run, last_run, created_at)
                VALUES (?, ?, ?, ?, ?, NULL, ?)
                """,
                (task_id, iv, tt, payload or "", next_run, created),
            )
        conn.commit()
    return {
        "ok": True,
        "task_id": task_id,
        "interval_seconds": iv,
        "task_type": tt,
        "next_run_epoch": next_run,
        "first_run_in_seconds": delay,
    }


def list_jobs() -> list[dict]:
    init_schema()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT task_id, interval_seconds, task_type, payload, next_run, last_run, created_at FROM jobs",
        ).fetchall()
    return [dict(r) for r in rows]


def _run_one_job_payload(job: sqlite3.Row) -> str:
    task_type = job["task_type"]
    payload = job["payload"] or ""
    if task_type == "reminder":
        body = {
            "kind": "reminder",
            "note": payload,
            "fired_at": time.time(),
        }
    elif task_type == "http_sample":
        url = payload.strip() or "https://jsonplaceholder.typicode.com/posts/1"
        try:
            r = httpx.get(url, timeout=15.0)
            snippet = ""
            if r.headers.get("content-type", "").startswith("application/json") and r.text:
                try:
                    data = r.json()
                    snippet = json.dumps(data, ensure_ascii=False)[:800]
                except (json.JSONDecodeError, ValueError):
                    snippet = r.text[:800]
            else:
                snippet = (r.text or "")[:800]
            body = {
                "kind": "http_sample",
                "url": url,
                "status_code": r.status_code,
                "snippet": snippet,
            }
        except Exception as e:  # noqa: BLE001
            body = {"kind": "http_sample", "url": url, "error": str(e)}
    elif task_type == "heartbeat_rollup":
        body = {
            "kind": "heartbeat_rollup",
            "tick": int(time.time()),
            "message": payload or "periodic tick",
        }
    else:
        body = {"kind": "unknown", "task_type": task_type}
    return json.dumps(body, ensure_ascii=False)


def _emit_job_event(job: sqlite3.Row, content: str) -> None:
    from . import scheduler_notify

    try:
        obj = json.loads(content)
        kind = str(obj.get("kind", "?"))
        if kind == "reminder":
            preview = f"напоминание: {(obj.get('note') or '')[:160]}"
        elif kind == "http_sample":
            preview = f"HTTP {obj.get('status_code', obj.get('error', '?'))} {str(obj.get('url', ''))[:80]}"
        elif kind == "heartbeat_rollup":
            preview = f"heartbeat tick={obj.get('tick')}"
        else:
            preview = content[:160]
    except json.JSONDecodeError:
        kind = "raw"
        preview = content[:160]
    scheduler_notify.emit_scheduler_tick_sync(
        task_id=str(job["task_id"]),
        task_type=str(job["task_type"]),
        preview=preview,
        content_json=content,
    )


def process_due_jobs() -> int:
    """Выполнить все задачи с next_run <= now. Возвращает число срабатываний."""
    init_schema()
    now = time.time()
    processed = 0
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT * FROM jobs WHERE next_run <= ? ORDER BY next_run",
            (now,),
        ).fetchall()
        for job in rows:
            content = _run_one_job_payload(job)
            conn.execute(
                "INSERT INTO results (task_id, ts, content) VALUES (?, ?, ?)",
                (job["task_id"], now, content),
            )
            interval = int(job["interval_seconds"])
            next_run = now + interval
            conn.execute(
                "UPDATE jobs SET last_run = ?, next_run = ? WHERE task_id = ?",
                (now, next_run, job["task_id"]),
            )
            conn.commit()
            processed += 1
            _emit_job_event(job, content)
    finally:
        conn.close()
    return processed


def get_aggregated_results(task_id: str, max_samples: int = 50) -> dict:
    init_schema()
    if not _TASK_ID_RE.match(task_id or ""):
        raise ValueError("некорректный task_id")
    lim = max(1, min(200, int(max_samples)))
    with _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM results WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        total = int(row["c"]) if row else 0
        rows = conn.execute(
            """
            SELECT content, ts, id FROM results
            WHERE task_id = ? ORDER BY id DESC LIMIT ?
            """,
            (task_id, lim),
        ).fetchall()
    parsed: list[dict] = []
    summary_bits: list[str] = []
    for r in rows:
        raw = r["content"]
        try:
            obj = json.loads(raw)
            parsed.append(obj)
            k = obj.get("kind", "?")
            if k == "reminder":
                summary_bits.append(f"reminder: {obj.get('note', '')[:120]}")
            elif k == "http_sample":
                summary_bits.append(
                    f"http {obj.get('url', '')}: status={obj.get('status_code', obj.get('error'))}",
                )
            elif k == "heartbeat_rollup":
                summary_bits.append(f"heartbeat tick={obj.get('tick')}")
            else:
                summary_bits.append(str(obj)[:120])
        except json.JSONDecodeError:
            parsed.append({"raw": raw})
            summary_bits.append(raw[:120])
    summary_text = " | ".join(reversed(summary_bits[:15])) if summary_bits else "(нет записей)"
    return {
        "task_id": task_id,
        "total_stored_runs": total,
        "returned_samples": len(parsed),
        "recent_newest_first": parsed,
        "summary_text": summary_text,
    }


def delete_job(task_id: str) -> bool:
    init_schema()
    if not _TASK_ID_RE.match(task_id or ""):
        raise ValueError("некорректный task_id")
    with _connect() as conn:
        conn.execute("DELETE FROM results WHERE task_id = ?", (task_id,))
        cur = conn.execute("DELETE FROM jobs WHERE task_id = ?", (task_id,))
        conn.commit()
        return cur.rowcount > 0
