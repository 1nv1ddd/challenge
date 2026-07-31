"""Отчёт по прогону: runs/runN.json → runs/runN.md (четыре стратегии на одних письмах)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent_constants import INTAKE_FIELD_KEYS  # noqa: E402

_RUNS_DIR = Path(__file__).resolve().parent / "runs"
_GROUP_TITLE = {
    "simple": "Чистые письма",
    "messy": "Шумные формулировки",
    "conditional": "Решение по условиям",
    "trap": "Ловушки",
}
_STRATEGY_TITLE = {
    "mono_large": "A. Монолит, сильная",
    "mono_small": "A. Монолит, дешёвая",
    "staged": "B. Этапы, решение моделью",
    "staged_rules": "B. Этапы, решение кодом",
}
_FIELD_TITLE = {
    "company": "company",
    "product": "product",
    "qty_kg": "qty_kg",
    "budget_rub": "budget_rub",
    "deadline": "deadline",
    "region": "region",
    "contact": "contact",
    "payment": "payment",
}


def _mark(ok: bool) -> str:
    return "✅" if ok else "❌"


def _ids(values: list[str]) -> str:
    return ", ".join(values) if values else "—"


def _summary_table(payload: dict) -> list[str]:
    strategies = payload["strategies"]
    stats = payload["summary"]["strategies"]
    total = payload["summary"]["total"]
    header = "| Метрика | " + " | ".join(_STRATEGY_TITLE.get(s, s) for s in strategies) + " |"
    lines = [header, "|---" * (len(strategies) + 1) + "|"]

    def row(title: str, fn) -> str:
        return f"| {title} | " + " | ".join(fn(stats[s]) for s in strategies) + " |"

    lines += [
        row(
            "Вердикт верен",
            lambda s: f"**{s['verdict_correct']}/{total}** ({s['verdict_pct']}%)",
        ),
        row("Все 8 полей верны", lambda s: f"{s['all_fields_correct']}/{total}"),
        row("Полей верно всего", lambda s: f"{s['fields_correct']}/{s['fields_total']} ({s['fields_pct']}%)"),
        row("Сбоев формата", lambda s: str(s["format_failures"])),
        row("Ремонтных вызовов", lambda s: str(s["repairs"])),
        row("Вызовов LLM", lambda s: str(s["llm_calls"])),
        row("Стоимость прогона", lambda s: f"**{s['cost_rub']} ₽**"),
        row(
            "Latency avg / p50 / p95, мс",
            lambda s: f"{s['latency_ms_avg']} / {s['latency_ms']['p50']} / {s['latency_ms']['p95']}",
        ),
    ]
    return lines


def _per_field_table(payload: dict) -> list[str]:
    strategies = payload["strategies"]
    stats = payload["summary"]["strategies"]
    total = payload["summary"]["total"]
    lines = [
        "| Поле | " + " | ".join(_STRATEGY_TITLE.get(s, s) for s in strategies) + " |",
        "|---" * (len(strategies) + 1) + "|",
    ]
    for field in INTAKE_FIELD_KEYS:
        cells = " | ".join(f"{stats[s]['per_field'][field]}/{total}" for s in strategies)
        lines.append(f"| `{_FIELD_TITLE[field]}` | {cells} |")
    return lines


def _rows_table(payload: dict, group: str) -> list[str]:
    strategies = payload["strategies"]
    rows = [r for r in payload["rows"] if r["group"] == group]
    if not rows:
        return []
    lines = [
        f"## {_GROUP_TITLE.get(group, group)} ({len(rows)})",
        "",
        "| ID | Эталон | " + " | ".join(_STRATEGY_TITLE.get(s, s) for s in strategies) + " |",
        "|---" * (len(strategies) + 2) + "|",
    ]
    for row in rows:
        expected = f"`{row['expected']['decision']}` / `{row['expected']['reason']}`"
        cells = []
        for mode in strategies:
            cell = row[mode]
            got = f"{_mark(cell['verdict_correct'])} {cell['decision']}/{cell['reason']}"
            if not cell["all_fields_correct"]:
                wrong = [k for k, ok in cell["grades"].items() if not ok]
                got += f" · поля: {', '.join(wrong)}"
            cells.append(got)
        lines.append(f"| {row['id']} | {expected} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def render(payload: dict) -> str:
    summary = payload["summary"]
    stats = summary["strategies"]
    total = summary["total"]
    delta = summary.get("delta", {})
    stage_models = payload["stage_models"]
    lines = [
        f"# Прогон: монолит `{payload['mono_model']}` против цепочки этапов, t={payload['temperature']}",
        "",
        f"Писем: **{total}**. Каждое прогоняется всеми стратегиями на одних и тех же правилах "
        "нормализации и одной и той же политике приёма.",
        "",
        f"- монолит, сильная модель — `{payload['mono_model']}`",
        f"- монолит, дешёвая модель — `{payload['small_model']}`",
        f"- этапы — нормализация `{stage_models['normalize']}`, решение "
        f"`{stage_models['decide']}`, письмо `{stage_models['compose']}`",
        "- этапы с решением кодом — те же модели на этапах 1 и 3, политика применяется без LLM",
        "",
        "## Сводка",
        "",
    ]
    lines += _summary_table(payload)
    if delta:
        lines += [
            "",
            f"**Цепочка этапов стоит {delta['cost_vs_mono_large_pct']}% от монолита на сильной модели.** "
            f"Разница по вердиктам: {delta['verdict_delta']:+d}, по полям: {delta['fields_delta']:+d}.",
            "",
            f"- этапы починили: {_ids(delta['fixed_by_staged'])}",
            f"- этапы сломали: {_ids(delta['broken_by_staged'])}",
        ]
    lines += [
        "",
        "## Где ошибается извлечение (верных значений на 24 письма)",
        "",
    ]
    lines += _per_field_table(payload)
    lines += [
        "",
        "## Согласие решения с политикой на своих же полях",
        "",
        "Решение считается согласованным, если совпадает с политикой, применённой кодом к тем "
        "полям, которые стратегия сама извлекла. Это отделяет ошибку извлечения от ошибки "
        "применения правил.",
        "",
        "| Стратегия | Согласовано с политикой | Вердикт верен |",
        "|---|---|---|",
    ]
    for mode in payload["strategies"]:
        s = stats[mode]
        lines.append(
            f"| {_STRATEGY_TITLE.get(mode, mode)} | {s['decision_matches_rules']}/{total} | "
            f"{s['verdict_correct']}/{total} |"
        )
    lines.append("")

    for group in ("simple", "messy", "conditional", "trap"):
        lines += _rows_table(payload, group)

    lines += [
        "## Кейсы с расхождениями",
        "",
        "| Стратегия | Неверный вердикт | Неполные поля |",
        "|---|---|---|",
    ]
    for mode in payload["strategies"]:
        s = stats[mode]
        lines.append(
            f"| {_STRATEGY_TITLE.get(mode, mode)} | {_ids(s['wrong_verdict_ids'])} | "
            f"{_ids(s['wrong_fields_ids'])} |"
        )
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
