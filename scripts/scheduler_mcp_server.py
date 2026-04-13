"""
MCP: отложенные и периодические задачи (напоминания, опрос URL, heartbeat-сводка).

Данные в SQLite: data/mcp_scheduler.sqlite (создаётся автоматически).
Фоновые срабатывания выполняет процесс **веб-приложения** (uvicorn), не этот скрипт.

Типы task_type:
  - reminder — в payload текст напоминания; в лог пишется JSON с отметкой времени
  - http_sample — в payload URL (или дефолт JSONPlaceholder); сохраняется статус и фрагмент ответа
  - heartbeat_rollup — периодический «тик» для накопления сводки

Инструменты:
  - register_interval_job
  - list_scheduled_jobs
  - get_aggregated_results
  - remove_scheduled_job

Запуск: python scripts/scheduler_mcp_server.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp.server.fastmcp import FastMCP

from app.scheduler_store import (
    delete_job,
    get_aggregated_results as store_aggregate,
    list_jobs,
    register_job,
)

app = FastMCP("challenge-scheduler")


@app.tool()
def register_interval_job(
    task_id: str,
    interval_seconds: int,
    task_type: str,
    payload: str = "",
    first_run_in_seconds: int = 15,
) -> str:
    """
    Зарегистрировать периодическую задачу. Первый запуск — через first_run_in_seconds (5–3600 с),
    далее каждые interval_seconds (15–86400 с). Сохраняется в SQLite; исполняет основное приложение.
    """
    data = register_job(
        task_id=task_id,
        interval_seconds=interval_seconds,
        task_type=task_type,
        payload=payload,
        first_run_in_seconds=first_run_in_seconds,
    )
    return json.dumps(data, ensure_ascii=False, indent=2)


@app.tool()
def list_scheduled_jobs() -> str:
    """Список всех зарегистрированных задач и их next_run (unix epoch)."""
    jobs = list_jobs()
    return json.dumps({"jobs": jobs}, ensure_ascii=False, indent=2)


@app.tool()
def get_aggregated_results(task_id: str, max_samples: int = 30) -> str:
    """
    Агрегированная сводка по накопленным срабатываниям задачи: счётчики, последние записи, summary_text.
    """
    agg = store_aggregate(task_id, max_samples=max_samples)
    return json.dumps(agg, ensure_ascii=False, indent=2)


@app.tool()
def remove_scheduled_job(task_id: str) -> str:
    """Удалить задачу и все её сохранённые результаты."""
    ok = delete_job(task_id)
    return json.dumps({"ok": ok, "task_id": task_id}, ensure_ascii=False)


if __name__ == "__main__":
    app.run(transport="stdio")
