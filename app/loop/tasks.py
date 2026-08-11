"""Корпус задач цикла: загрузка из jsonl и выборка по id."""

from __future__ import annotations

import json
from pathlib import Path

from ..agent_constants import LOOP_TASKS_PATH
from .schema import LoopTask

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def tasks_path(path: str | Path | None = None) -> Path:
    p = Path(path) if path else Path(LOOP_TASKS_PATH)
    return p if p.is_absolute() else _PROJECT_ROOT / p


def load_tasks(path: str | Path | None = None) -> list[LoopTask]:
    """Читает корпус; дубли id и битые строки — ошибка загрузки, а не тихий пропуск."""
    file = tasks_path(path)
    if not file.is_file():
        raise ValueError(f"Корпус задач цикла не найден: {file}")
    tasks: list[LoopTask] = []
    seen: set[str] = set()
    for lineno, raw in enumerate(file.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{file.name}:{lineno} — не JSON: {exc}") from exc
        task = LoopTask.from_dict(data)
        if task.id in seen:
            raise ValueError(f"{file.name}:{lineno} — дубль id {task.id!r}.")
        seen.add(task.id)
        tasks.append(task)
    if not tasks:
        raise ValueError(f"Корпус задач цикла пуст: {file}")
    return tasks


def select_tasks(tasks: list[LoopTask], ids: tuple[str, ...] = ()) -> list[LoopTask]:
    """Фильтр по id. Пустой фильтр ничего не отсекает; неизвестный id — ошибка."""
    known = {t.id for t in tasks}
    unknown = [i for i in ids if i not in known]
    if unknown:
        raise ValueError(f"Нет таких задач в корпусе: {', '.join(unknown)}.")
    return [t for t in tasks if t.id in ids] if ids else tasks
