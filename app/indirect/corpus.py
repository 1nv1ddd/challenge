"""Загрузка корпуса ловушек из jsonl и выборки по сценарию, носителю и технике сокрытия."""

from __future__ import annotations

import json
from pathlib import Path

from ..agent_constants import INDIRECT_CASES_PATH
from .schema import IndirectCase

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def cases_path(path: str | Path | None = None) -> Path:
    p = Path(path) if path else Path(INDIRECT_CASES_PATH)
    return p if p.is_absolute() else _PROJECT_ROOT / p


def load_cases(path: str | Path | None = None) -> list[IndirectCase]:
    """Читает корпус; дубли id и битые строки — ошибка загрузки, а не тихий пропуск."""
    file = cases_path(path)
    if not file.is_file():
        raise ValueError(f"Корпус ловушек не найден: {file}")
    cases: list[IndirectCase] = []
    seen: set[str] = set()
    for lineno, raw in enumerate(file.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{file.name}:{lineno} — не JSON: {exc}") from exc
        case = IndirectCase.from_dict(data)
        if case.id in seen:
            raise ValueError(f"{file.name}:{lineno} — дубль id {case.id!r}.")
        seen.add(case.id)
        cases.append(case)
    if not cases:
        raise ValueError(f"Корпус ловушек пуст: {file}")
    return cases


def select_cases(
    cases: list[IndirectCase],
    *,
    ids: tuple[str, ...] = (),
    scenario: str = "",
    source: str = "",
    hiding: str = "",
) -> list[IndirectCase]:
    """Фильтр корпуса. Пустой фильтр ничего не отсекает; неизвестный id — ошибка."""
    known = {c.id for c in cases}
    unknown = [i for i in ids if i not in known]
    if unknown:
        raise ValueError(f"Нет таких ловушек в корпусе: {', '.join(unknown)}.")
    out = cases
    if ids:
        out = [c for c in out if c.id in ids]
    if scenario:
        out = [c for c in out if c.scenario == scenario]
    if source:
        out = [c for c in out if c.source == source]
    if hiding:
        out = [c for c in out if c.hiding == hiding]
    return out


def corpus_stats(cases: list[IndirectCase]) -> dict[str, dict[str, int]]:
    """Состав корпуса: сколько ловушек по сценарию, носителю и технике сокрытия."""
    stats: dict[str, dict[str, int]] = {"scenario": {}, "source": {}, "hiding": {}}
    for case in cases:
        for key, value in (
            ("scenario", case.scenario),
            ("source", case.source),
            ("hiding", case.hiding),
        ):
            stats[key][value] = stats[key].get(value, 0) + 1
    return stats
