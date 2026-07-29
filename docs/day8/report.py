"""Отчёт по прогону: runs/runN.json → runs/runN.md (три стратегии + поимённые маршруты)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_RUNS_DIR = Path(__file__).resolve().parent / "runs"
_GROUP_TITLE = {
    "simple": "Простые",
    "medium": "Средние",
    "hard": "Сложные",
    "trap": "Ловушки",
}
_PATH_LABEL = {
    "small": "осталось на дешёвой",
    "small→large": "эскалация",
    "large": "pre-routing на сильную",
    "small (fallback)": "дешёвая (эскалация не удалась)",
}


def _mark(ok: bool) -> str:
    return "✅" if ok else "❌"


def _row_line(row: dict) -> str:
    r = row["router"]
    assessment = r.get("assessment") or {}
    conf = assessment.get("confidence", "—")
    self_rep = assessment.get("self_reported")
    self_rep = "—" if self_rep is None else self_rep
    return (
        f"| {row['id']} | {_mark(row['small_only']['correct'])} | "
        f"{_mark(row['large_only']['correct'])} | {_mark(r['correct'])} | "
        f"{_PATH_LABEL.get(r['path'], r['path'])} | {self_rep} | {conf} | "
        f"{r['llm_calls']} | {r['cost_rub']:.4f} ₽ |"
    )


def _reason_line(row: dict) -> str:
    r = row["router"]
    reason = r["escalation_reason"]
    return f"| {row['id']} | {_PATH_LABEL.get(r['path'], r['path'])} | {reason} |"


def render(payload: dict) -> str:
    s = payload["summary"]
    small, large, router, delta = s["small_only"], s["large_only"], s["router"], s["delta"]
    total = s["total"]
    lines = [
        f"# Прогон: `{payload['small_model']}` → `{payload['large_model']}`, t={payload['temperature']}",
        "",
        f"Запросов: **{total}**. Три стратегии на одних и тех же вопросах. "
        f"Дубль для проверки согласованности берётся при "
        f"t={payload.get('consistency_temperature', payload['temperature'])}.",
        "",
        "## Сводка",
        "",
        "| Метрика | Только дешёвая | Только сильная | Роутер |",
        "|---|---|---|---|",
        f"| Верных ответов | {small['correct']}/{total} ({small['correct_pct']}%) | "
        f"{large['correct']}/{total} ({large['correct_pct']}%) | "
        f"**{router['correct']}/{total} ({router['correct_pct']}%)** |",
        f"| Вызовов LLM | {total} | {total} | {router['llm_calls']} |",
        f"| Стоимость прогона | {small['cost_rub']} ₽ | {large['cost_rub']} ₽ | "
        f"**{router['cost_rub']} ₽** |",
        f"| Latency avg / p50 / p95 | {small['latency_ms_avg']} / {small['latency_ms']['p50']} / "
        f"{small['latency_ms']['p95']} мс | {large['latency_ms_avg']} / {large['latency_ms']['p50']} / "
        f"{large['latency_ms']['p95']} мс | {router['latency_ms_avg']} / "
        f"{router['latency_ms']['p50']} / {router['latency_ms']['p95']} мс |",
        "",
        f"**Роутер стоит {delta['cost_vs_large_pct']}% от цены «всегда сильная» "
        f"и ×{delta['cost_vs_small_x']} от цены «всегда дешёвая».**",
        "",
        "## Куда ушли запросы",
        "",
        "| Маршрут | Запросов | Доля |",
        "|---|---|---|",
    ]
    for path, count in sorted(router["paths"].items(), key=lambda kv: -kv[1]):
        share = round(100.0 * count / total, 1) if total else 0.0
        lines.append(f"| {_PATH_LABEL.get(path, path)} | {count} | {share}% |")
    lines += [
        "",
        f"- осталось на дешёвой модели: **{router['kept_small']}** ({router['kept_small_pct']}%)",
        f"- эскалировано после дешёвой: **{router['escalated_after_small']}**",
        f"- отправлено на сильную сразу (pre-routing): **{router['prerouted_large']}**",
        f"- потрачено на черновики, которые не пригодились: **{router['wasted_rub']} ₽**",
        "",
        "## Качество решений роутера",
        "",
        "| Событие | Кейсов | ID |",
        "|---|---|---|",
        f"| Эскалация спасла ответ (дешёвая ошиблась, сильная исправила) | {delta['rescued']} | "
        f"{', '.join(delta['rescued_ids']) or '—'} |",
        f"| Лишняя эскалация (дешёвая была права) | {delta['over_escalated']} | "
        f"{', '.join(delta['over_escalated_ids']) or '—'} |",
        f"| Пропущенная эскалация (оставили на дешёвой, ответ неверный) | "
        f"{delta['missed_escalation']} | {', '.join(delta['missed_escalation_ids']) or '—'} |",
        f"| Лишний pre-routing (дешёвая знала ответ, но её не спросили) | "
        f"{delta.get('preroute_overspend', 0)} | "
        f"{', '.join(delta.get('preroute_overspend_ids', [])) or '—'} |",
        "",
    ]

    for group in ("simple", "medium", "hard", "trap"):
        rows = [r for r in payload["rows"] if r["group"] == group]
        if not rows:
            continue
        lines += [
            f"## {_GROUP_TITLE[group]} ({len(rows)})",
            "",
            "| ID | дешёвая | сильная | роутер | маршрут | самооценка | итог conf | вызовов | цена |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        lines += [_row_line(r) for r in rows]
        lines.append("")

    lines += [
        "## Причины маршрутизации по каждому запросу",
        "",
        "| ID | Маршрут | Причина |",
        "|---|---|---|",
    ]
    lines += [_reason_line(r) for r in payload["rows"]]
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
