"""Свести метрики прогона execution loop из runN.jsonl (streak, avg time, first-try %)."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _load(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def summarize(rows: list[dict]) -> dict:
    """Посчитать сводку; streak — ведущая серия done до первого не-done."""
    total = len(rows)
    done = [r for r in rows if r.get("outcome") == "done"]
    streak = 0
    for r in rows:
        if r.get("outcome") == "done":
            streak += 1
        else:
            break
    break_row = rows[streak] if streak < total else None
    seconds = [float(r.get("seconds", 0.0)) for r in rows]
    avg = sum(seconds) / total if total else 0.0
    first_try = [r for r in rows if r.get("first_try")]
    outcomes: dict[str, int] = {}
    for r in rows:
        outcomes[r.get("outcome", "?")] = outcomes.get(r.get("outcome", "?"), 0) + 1
    return {
        "total": total,
        "done": len(done),
        "streak": streak,
        "break_at": None if break_row is None else break_row.get("id"),
        "break_reason": None if break_row is None else break_row.get("reason"),
        "avg_seconds": round(avg, 1),
        "first_try_pct": round(100.0 * len(first_try) / total, 1) if total else 0.0,
        "outcomes": outcomes,
    }


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: metrics.py <run.jsonl>", file=sys.stderr)
        return 2
    rows = _load(Path(argv[1]))
    s = summarize(rows)
    out = [
        f"Задач всего:        {s['total']}",
        f"Выполнено (done):   {s['done']}",
        f"Streak без паузы:   {s['streak']}"
        + (f"  (слом на {s['break_at']}: {s['break_reason']})" if s["break_at"] else "  (весь пул подряд)"),
        f"Среднее на задачу:  {s['avg_seconds']} c",
        f"С первого раза:     {s['first_try_pct']}%",
        f"Исходы:             {s['outcomes']}",
    ]
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
