"""Загрузка корпуса инъекций из jsonl и выборки по вектору, технике и цели."""

from __future__ import annotations

import json
from pathlib import Path

from ..agent_constants import SECURITY_ATTACKS_PATH
from .schema import Attack

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def attacks_path(path: str | Path | None = None) -> Path:
    p = Path(path) if path else Path(SECURITY_ATTACKS_PATH)
    return p if p.is_absolute() else _PROJECT_ROOT / p


def load_attacks(path: str | Path | None = None) -> list[Attack]:
    """Читает корпус; дубли id и битые строки — ошибка загрузки, а не тихий пропуск."""
    file = attacks_path(path)
    if not file.is_file():
        raise ValueError(f"Корпус атак не найден: {file}")
    attacks: list[Attack] = []
    seen: set[str] = set()
    for lineno, raw in enumerate(file.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{file.name}:{lineno} — не JSON: {exc}") from exc
        attack = Attack.from_dict(data)
        if attack.id in seen:
            raise ValueError(f"{file.name}:{lineno} — дубль id {attack.id!r}.")
        seen.add(attack.id)
        attacks.append(attack)
    if not attacks:
        raise ValueError(f"Корпус атак пуст: {file}")
    return attacks


def select_attacks(
    attacks: list[Attack],
    *,
    ids: tuple[str, ...] = (),
    target: str = "",
    vector: str = "",
    technique: str = "",
) -> list[Attack]:
    """Фильтр корпуса. Пустой фильтр ничего не отсекает; неизвестный id — ошибка."""
    known = {a.id for a in attacks}
    unknown = [i for i in ids if i not in known]
    if unknown:
        raise ValueError(f"Нет таких атак в корпусе: {', '.join(unknown)}.")
    out = attacks
    if ids:
        out = [a for a in out if a.id in ids]
    if target:
        out = [a for a in out if a.target == target]
    if vector:
        out = [a for a in out if a.vector == vector]
    if technique:
        out = [a for a in out if a.technique == technique]
    return out


def corpus_stats(attacks: list[Attack]) -> dict[str, dict[str, int]]:
    """Состав корпуса: сколько атак по каждому вектору, технике и цели."""
    stats: dict[str, dict[str, int]] = {"vector": {}, "technique": {}, "target": {}}
    for attack in attacks:
        for key, value in (
            ("vector", attack.vector),
            ("technique", attack.technique),
            ("target", attack.target),
        ):
            stats[key][value] = stats[key].get(value, 0) + 1
    return stats
