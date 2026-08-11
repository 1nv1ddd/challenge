"""Роут-обработчик `GET /api/scheduler/jobs`.

Предназначен для вставки в `app/scheduler_routes.py`. Отдаёт список
зарегистрированных задач планировщика, читая их из слоя хранилища
(`app/scheduler_store.py`, функция `list_scheduler_jobs`).

Для вставки в существующий модуль нужно:
  1. добавить импорт хранилища, например:
         from . import scheduler_store
     (в текущем `scheduler_routes.py` импортируется только
      `sse_scheduler_subscribe` из `.scheduler_notify`);
  2. добавить обработчик ниже к уже существующему `router`
     (APIRouter с prefix="/api/scheduler").

Полный путь: /api/scheduler/jobs
"""

from __future__ import annotations

from . import scheduler_store

# router уже объявлен в scheduler_routes.py:
#   router = APIRouter(prefix="/api/scheduler", tags=["scheduler"])


@router.get("/jobs")
async def scheduler_jobs() -> dict:
    """Список зарегистрированных задач планировщика.

    По каждой задаче: task_id, task_type, next_run, interval_seconds.
    """
    jobs = scheduler_store.list_scheduler_jobs()
    return {"ok": True, "count": len(jobs), "jobs": jobs}
