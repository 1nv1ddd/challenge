"""Функция слоя хранилища планировщика для чтения списка задач.

Это то, что было бы добавлено в app/scheduler_store.py. Возвращает по каждой
зарегистрированной задаче только поля, нужные API: task_id, task_type,
next_run, interval_seconds — отсортированные по ближайшему next_run.

Реализация опирается на уже существующие в модуле хелперы `init_schema()` и
`_connect()` (см. app/scheduler_store.py), поэтому при вставке в проект
дублировать их не нужно.
"""

from __future__ import annotations


def list_scheduler_jobs() -> list[dict]:
    """Вернуть зарегистрированные задачи планировщика.

    По каждой задаче: task_id, task_type, next_run (epoch, сек),
    interval_seconds. Отсортировано по next_run (ближайшие — первыми).
    """
    init_schema()  # noqa: F821 — определён в app/scheduler_store.py
    with _connect() as conn:  # noqa: F821 — определён в app/scheduler_store.py
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
