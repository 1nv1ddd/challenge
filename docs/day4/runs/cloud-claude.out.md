# Облачный ассистент (Claude) — тот же промпт, те же правила

Задача просит вернуть `app/scheduler_routes.py` целиком, но «домашний» ответ
требует и тонкой функции в сторе: инвариант reuse-first + запрет сырого SQL в роутере
(п.1, п.6) означают, что доступ к одной джобе должен жить в `scheduler_store.py`
рядом с `list_jobs()` / `delete_job()`, а роутер её только вызывает. Поэтому два файла.

## 1. `app/scheduler_store.py` — добавить рядом с `list_jobs()`

```python
def get_job(task_id: str) -> dict | None:
    """Вернуть одну джобу по task_id или None, если её нет."""
    init_schema()
    with _connect() as conn:
        row = conn.execute(
            "SELECT task_id, interval_seconds, task_type, payload, next_run, last_run, created_at "
            "FROM jobs WHERE task_id = ?",
            (task_id,),
        ).fetchone()
    return dict(row) if row else None
```

## 2. `app/scheduler_routes.py` — целиком

```python
"""HTTP: SSE-стрим тиков планировщика + ping для проверки деплоя."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from .scheduler_notify import sse_scheduler_subscribe
from .scheduler_store import get_job

router = APIRouter(prefix="/api/scheduler", tags=["scheduler"])


@router.get("/ping")
async def scheduler_ping() -> dict:
    return {"ok": True, "sse_path": "/api/scheduler/stream"}


@router.get("/jobs/{task_id}")
async def get_scheduler_job(task_id: str) -> dict:
    """Вернуть одну джобу планировщика по task_id или 404, если её нет."""
    job = get_job(task_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Задача {task_id!r} не найдена")
    return job


@router.get("/stream")
async def scheduler_event_stream():
    async def gen():
        async for chunk in sse_scheduler_subscribe():
            yield chunk

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
```

## Почему так

- **reuse-first + слой стора (п.1, п.6):** доступ к БД — новая функция `get_job()` в
  `scheduler_store.py` через `_connect()`, не сырой SQL в роутере. Имя симметрично
  соседям (`register_job`/`delete_job`/`list_jobs`).
- **параметризованный SQL (п.7):** `WHERE task_id = ?`, никаких f-string.
- **`from __future__` + docstring модуля** сохранены (файл возвращён целиком, без `# ...`).
- **русский `HTTPException(404)` (п.5)** с человекочитаемым detail.
- **аннотации возврата** `-> dict` / `-> dict | None`, имя эндпоинта по действию
  (`get_scheduler_job`), не затирает приватный `get_job` из стора.
