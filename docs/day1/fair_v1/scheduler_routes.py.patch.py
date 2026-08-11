# ВСТАВИТЬ В: app/scheduler_routes.py
#
# Что делает: добавляет HTTP-эндпоинт `GET /api/scheduler/jobs`, который
# отдаёт список зарегистрированных задач планировщика. По каждой задаче —
# поля task_id, task_type, next_run, interval_seconds. Данные берутся из
# слоя хранилища (app/scheduler_store.list_jobs), маршрут только проецирует
# нужные поля и ничего не знает про SQLite напрямую.
#
# Как вставлять:
#   1) В блок импортов файла app/scheduler_routes.py добавить строку
#      `from .scheduler_store import list_jobs` (см. секцию "ИМПОРТ" ниже).
#   2) В тело файла, рядом с остальными @router-хендлерами, добавить функцию
#      `scheduler_jobs` (см. секцию "ХЕНДЛЕР" ниже).
#
# Ниже — как файл выглядит целиком после правки (можно скопировать целиком).

"""HTTP: SSE-стрим тиков планировщика + ping для проверки деплоя + список задач."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from .scheduler_notify import sse_scheduler_subscribe

# --- ИМПОРТ (добавить) ---------------------------------------------------
from .scheduler_store import list_jobs

router = APIRouter(prefix="/api/scheduler", tags=["scheduler"])


@router.get("/ping")
async def scheduler_ping() -> dict:
    return {"ok": True, "sse_path": "/api/scheduler/stream"}


# --- ХЕНДЛЕР (добавить) --------------------------------------------------
@router.get("/jobs")
async def scheduler_jobs() -> dict:
    """Список зарегистрированных задач планировщика.

    Источник данных — слой хранилища (scheduler_store.list_jobs). Наружу
    отдаём только стабильный публичный контракт из четырёх полей и не тянем
    payload/last_run/created_at, чтобы не раскрывать внутренние детали.
    """
    jobs = [
        {
            "task_id": row["task_id"],
            "task_type": row["task_type"],
            "next_run": row["next_run"],
            "interval_seconds": row["interval_seconds"],
        }
        for row in list_jobs()
    ]
    return {"jobs": jobs, "count": len(jobs)}


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
