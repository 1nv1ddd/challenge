"""Роут-обработчик GET /api/scheduler/jobs.

Это то, что было бы добавлено в app/scheduler_routes.py. Роутер там уже создан
с префиксом /api/scheduler (`router = APIRouter(prefix="/api/scheduler", ...)`),
поэтому декоратор ниже даёт итоговый путь GET /api/scheduler/jobs.

Данные читаются из слоя хранилища планировщика (SQLite) через
`scheduler_store.list_scheduler_jobs()`.
"""

from __future__ import annotations

from . import scheduler_store


@router.get("/jobs")  # noqa: F821 — router определён в app/scheduler_routes.py
async def scheduler_jobs() -> dict:
    """Список зарегистрированных задач планировщика.

    По каждой задаче: task_id, task_type, next_run (epoch, сек),
    interval_seconds.
    """
    jobs = scheduler_store.list_scheduler_jobs()
    return {"ok": True, "count": len(jobs), "jobs": jobs}
