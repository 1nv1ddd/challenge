"""Отчёт по прогону: runs/runN.json → runs/runN.md (таблицы baseline vs gated)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_RUNS_DIR = Path(__file__).resolve().parent / "runs"
_GROUP_TITLE = {"clean": "Корректные", "edge": "Пограничные", "noisy": "Шумные/сложные"}


def _row_line(row: dict) -> str:
    gated = row["gated"]
    base_action = row["baseline"]["action"] or "ошибка формата"
    base_mark = "✅" if row["baseline"]["action"] in row["allowed_actions"] else "❌"
    applied = gated["applied_action"]
    if gated["status"] == "OK":
        gate_mark = "✅" if applied in row["allowed_actions"] else "❌"
    else:
        gate_mark = "🚧"  # автоматика не применена, обращение ушло человеку
    sc = (gated["selfcheck"] or {}).get("verdict", "—")
    return (
        f"| {row['id']} | {base_mark} `{base_action}` | {gated['status']} "
        f"{gate_mark} `{applied}` | {gated['confidence']} | {gated['agreement']} | {sc} | "
        f"{gated['metrics']['llm_calls']} | {gated['metrics']['time_ms'] / 1000:.1f} с |"
    )


def render(payload: dict) -> str:
    s = payload["summary"]
    base, gate, delta = s["baseline"], s["gated"], s["delta"]
    lines = [
        f"# Прогон: {payload['model']}, {payload['samples']} сэмпла, "
        f"t={payload['temperature']} (baseline t={payload.get('baseline_temperature', 0.0)})",
        "",
        f"Кейсов: **{s['total']}**.",
        "",
        "## Сводка",
        "",
        "| Метрика | Baseline (1 вызов, без гейта) | Gated (redundancy + constraints + self-check) |",
        "|---|---|---|",
        f"| Решение совпало с допустимым | {base['correct']}/{s['total']} "
        f"({base['correct_pct']}%) | {gate['accepted_correct']}/{gate['accepted']} принятых "
        f"({gate['accepted_correct_pct']}%) |",
        f"| Недопустимых автодействий | **{base['unsafe_auto']}** | **{gate['unsafe_auto']}** |",
        f"| Сломанный формат ответа | {base['format_errors']} | 0 (не проходит гейт) |",
        f"| Принято автоматически | {s['total']} (всё) | {gate['accepted']} "
        f"({gate['accepted_pct']}%) |",
        f"| Отклонено / отдано человеку | 0 | {gate['rejected']} ({gate['rejected_pct']}%) |",
        f"| Вызовов LLM | {base['llm_calls']} | {gate['llm_calls']} |",
        f"| Повторных инференсов (ремонт) | — | {gate['repair_calls']} "
        f"в {gate['cases_with_repair']} кейсах |",
        f"| Self-check вызовов | — | {gate['cases_with_selfcheck']} |",
        f"| Latency avg / p50 / p95 | {base['latency_ms_avg']} / {base['latency_ms']['p50']} / "
        f"{base['latency_ms']['p95']} мс | {gate['latency_ms_avg']} / {gate['latency_ms']['p50']} / "
        f"{gate['latency_ms']['p95']} мс |",
        f"| Стоимость прогона | {base['cost_rub']} ₽ | {gate['cost_rub']} ₽ |",
        "",
        f"**Цена гейта:** latency ×{delta['latency_x']}, деньги ×{delta['cost_x']}.",
        f"**Опасных решений baseline остановлено гейтом:** {delta['blocked_unsafe']}"
        + (f" ({', '.join(delta['blocked_unsafe_ids'])})" if delta["blocked_unsafe_ids"] else ""),
        f"**Ложных блокировок (baseline был прав, гейт не принял):** {delta['over_blocked']}"
        + (f" ({', '.join(delta['over_blocked_ids'])})" if delta["over_blocked_ids"] else ""),
        "",
        "## Что поймал constraint-слой на первой попытке",
        "",
    ]
    catches = gate.get("constraint_catches") or {}
    if catches:
        lines += ["| Тип проверки | Сработала раз |", "|---|---|"]
        lines += [f"| {key} | {count} |" for key, count in sorted(catches.items())]
        lines.append("")
        lines.append("Кейсы: " + ", ".join(gate.get("caught_ids", [])))
    else:
        lines.append("Ни одна проверка не сработала — все сэмплы были валидны с первой попытки.")
    lines.append("")

    for group in ("clean", "edge", "noisy"):
        rows = [r for r in payload["rows"] if r["group"] == group]
        if not rows:
            continue
        lines += [
            f"## {_GROUP_TITLE[group]} ({len(rows)})",
            "",
            "| ID | Baseline | Гейт | conf | agree | self-check | вызовов | время |",
            "|---|---|---|---|---|---|---|---|",
        ]
        lines += [_row_line(r) for r in rows]
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "run1"
    src = _RUNS_DIR / f"{name}.json"
    if not src.is_file():
        print(f"Нет файла {src}", file=sys.stderr)
        return 2
    payload = json.loads(src.read_text(encoding="utf-8"))
    out = _RUNS_DIR / f"{name}.md"
    out.write_text(render(payload), encoding="utf-8")
    print(f"Записано: {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
