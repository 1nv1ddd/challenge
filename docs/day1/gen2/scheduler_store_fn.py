"""Функция слоя хранилища планировщика: список зарегистрированных задач.

Предназначена для вставки в `app/scheduler_store.py`. Читает задачи напрямую
из таблицы `jobs` в SQLite и отдаёт только те поля, которые нужны публичному
эндпоинту `GET /api/scheduler/jobs`: task_id, task_type, next_run,
interval_seconds. Задачи отсортированы по ближайшему запуску (next_run).

В модуле уже есть `_connect()` и `init_schema()` — здесь используются они же.
"""

from __future__ import annotations


def list_scheduler_jobs() -> list[dict]:
    """Вернуть зарегистрированные задачи планировщика.

    По каждой задаче — компактный набор полей для API:
      - task_id (str)
      - task_type (str)
      - next_run (float, epoch-секунды следующего запуска)
      - interval_seconds (int)

    Список отсортирован по next_run (ближайшие запуски первыми).
    """
    init_schema()
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT task_id, task_type, next_run, interval_seconds
            FROM jobs
            ORDER BY next_run ASC
            """,
        ).fetchall()
    return [
        {
            "task_id": r["task_id"],
            "task_type": r["task_type"],
            "next_run": r["next_run"],
            "interval_seconds": int(r["interval_seconds"]),
        }
        for r in rows
    ]
