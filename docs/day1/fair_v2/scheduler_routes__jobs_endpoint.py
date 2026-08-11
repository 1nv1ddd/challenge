# ВСТАВИТЬ В: app/scheduler_routes.py
#
# Что делает: добавляет HTTP-эндпоинт `GET /api/scheduler/jobs` — отдаёт список
# зарегистрированных задач планировщика. По каждой задаче четыре поля:
# task_id, task_type, next_run, interval_seconds.
#
# Ключевое решение по реализации:
#   - Данные берём ТОЛЬКО из слоя хранилища через scheduler_store.list_jobs().
#     Роутер не знает про SQLite и не трогает БД напрямую — он лишь проецирует
#     нужные поля. list_jobs() уже возвращает все четыре поля (+ payload/
#     last_run/created_at), поэтому МЕНЯТЬ scheduler_store.py НЕ НУЖНО.
#   - Наружу отдаём ровно публичный контракт из четырёх полей; внутренние
#     payload/last_run/created_at не утекают.
#   - Сортируем по next_run (ближайшие к запуску — первыми): list_jobs() порядок
#     не гарантирует, а так ответ детерминирован и удобнее для UI/отладки.
#
# Как вставлять:
#   1) В блок импортов добавить строку из секции «ИМПОРТ».
#   2) Рядом с остальными @router-хендлерами добавить функцию scheduler_jobs
#      из секции «ХЕНДЛЕР».
#
# Ниже — как app/scheduler_routes.py выглядит целиком после правки
# (можно скопировать файл целиком).

"""HTTP: SSE-стрим тиков планировщика + ping + список зарегистрированных задач."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from .scheduler_notify import sse_scheduler_subscribe

# --- ИМПОРТ (добавить) ---------------------------------------------------
from .scheduler_store import list_jobs

router = APIRouter(prefix="/api/scheduler", tags=["scheduler"])

# Публичный контракт эндпоинта — какие поля задачи отдаём наружу.
_JOB_PUBLIC_FIELDS = ("task_id", "task_type", "next_run", "interval_seconds")


@router.get("/ping")
async def scheduler_ping() -> dict:
    return {"ok": True, "sse_path": "/api/scheduler/stream"}


# --- ХЕНДЛЕР (добавить) --------------------------------------------------
@router.get("/jobs")
async def scheduler_jobs() -> dict:
    """Список зарегистрированных задач планировщика.

    Источник данных — слой хранилища (scheduler_store.list_jobs). Роутер только
    проецирует стабильный публичный контракт из четырёх полей и не обращается к
    SQLite напрямую. Порядок — по next_run (ближайшие к запуску первыми).
    """
    jobs = sorted(
        ({field: row[field] for field in _JOB_PUBLIC_FIELDS} for row in list_jobs()),
        key=lambda job: job["next_run"],
    )
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
