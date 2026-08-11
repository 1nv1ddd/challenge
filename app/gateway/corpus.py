"""Корпус тест-кейсов шлюза и его офлайн-прогон по детекторам: что поймали, что пропустили."""

from __future__ import annotations

import json
from pathlib import Path

from ..agent_constants import GATEWAY_CASES_PATH
from .detectors import guard_input, scan_text
from .schema import CaseOutcome, CorpusRun, GatewayCase

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def cases_path(path: str | Path | None = None) -> Path:
    p = Path(path) if path else Path(GATEWAY_CASES_PATH)
    return p if p.is_absolute() else _PROJECT_ROOT / p


def load_cases(path: str | Path | None = None) -> list[GatewayCase]:
    """Читает корпус; дубли id и битые строки — ошибка загрузки, а не тихий пропуск."""
    file = cases_path(path)
    if not file.is_file():
        raise ValueError(f"Корпус кейсов шлюза не найден: {file}")
    cases: list[GatewayCase] = []
    seen: set[str] = set()
    for lineno, raw in enumerate(file.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{file.name}:{lineno} — не JSON: {exc}") from exc
        case = GatewayCase.from_dict(data)
        if case.id in seen:
            raise ValueError(f"{file.name}:{lineno} — дубль id {case.id!r}.")
        seen.add(case.id)
        cases.append(case)
    if not cases:
        raise ValueError(f"Корпус кейсов шлюза пуст: {file}")
    return cases


def select_cases(cases: list[GatewayCase], ids: tuple[str, ...] = ()) -> list[GatewayCase]:
    """Фильтр по id. Пустой фильтр ничего не отсекает; неизвестный id — ошибка."""
    known = {c.id for c in cases}
    unknown = [i for i in ids if i not in known]
    if unknown:
        raise ValueError(f"Нет таких кейсов в корпусе: {', '.join(unknown)}.")
    return [c for c in cases if c.id in ids] if ids else cases


def run_case(case: GatewayCase) -> CaseOutcome:
    """Прогон одного кейса через input guard — без сети и без вызова модели."""
    findings = scan_text(case.text)
    verdict = guard_input(case.text, case.mode)
    return CaseOutcome(
        case_id=case.id,
        title=case.title,
        mode=case.mode,
        expect_kinds=case.expect_kinds,
        found_kinds=tuple(sorted({f.kind for f in findings})),
        expect_action=case.expect_action,
        action=verdict.action,
        variants=tuple(sorted({f.variant for f in findings})),
        note=case.note,
    )


def run_corpus(cases: list[GatewayCase]) -> CorpusRun:
    """Весь корпус разом: результат — таблица «ожидали / нашли / пропустили» по каждому кейсу."""
    return CorpusRun(outcomes=[run_case(case) for case in cases])
