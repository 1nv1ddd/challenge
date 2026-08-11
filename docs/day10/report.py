"""Отчёт по прогону: runs/runN.json → runs/runN.md (четыре стратегии на одних обращениях)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_RUNS_DIR = Path(__file__).resolve().parent / "runs"
_GROUP_TITLE = {
    "simple": "Простые",
    "borderline": "Пограничные",
    "hard": "Сложные",
}
_STRATEGY_TITLE = {
    "llm_only": "Базовая линия: сразу большая модель",
    "micro_embed_first": "Micro-first: эмбеддинги + fallback",
    "micro_tfidf_first": "Micro-first: tfidf + fallback",
    "micro_only": "Только micro-model, без fallback",
}
_REASON_TITLE = {
    "short_input": "слишком короткий вход",
    "no_close_neighbor": "нет похожего примера",
    "fallback_label": "победил класс other",
    "low_margin": "нет отрыва от второго класса",
    "low_score": "score ниже порога",
}


def _strategies_table(summary: dict, strategies: list[str]) -> list[str]:
    lines = [
        "| Стратегия | Точность | Закрыла micro | Вызовов большой LLM | Стоимость | Latency avg / p50 / p95 |",
        "|---|---|---|---|---|---|",
    ]
    for name in strategies:
        st = summary["strategies"][name]
        lines.append(
            f"| {_STRATEGY_TITLE.get(name, name)} | {st['accuracy_pct']}% "
            f"({st['correct']}/{st['total']}) | {st['handled_by_micro_pct']}% | "
            f"{st['llm_calls']} | {st['cost_rub']} ₽ | "
            f"{st['latency_ms_avg']} / {st['latency_ms']['p50']} / {st['latency_ms']['p95']} мс |"
        )
    return lines


def _groups_table(summary: dict, strategies: list[str]) -> list[str]:
    groups = list(_GROUP_TITLE)
    header = " | ".join(_GROUP_TITLE[g] for g in groups)
    lines = [f"| Стратегия | {header} |", "|---" * (len(groups) + 1) + "|"]
    for name in strategies:
        st = summary["strategies"][name]
        cells = []
        for group in groups:
            cell = st["by_group"].get(group)
            cells.append(
                f"{cell['accuracy_pct']}% ({cell['correct']}/{cell['total']})" if cell else "—"
            )
        lines.append(f"| {_STRATEGY_TITLE.get(name, name)} | {' | '.join(cells)} |")
    return lines


def _reasons_table(summary: dict, strategies: list[str]) -> list[str]:
    lines = ["| Стратегия | Причина эскалации | Кейсов |", "|---|---|---|"]
    for name in strategies:
        reasons = summary["strategies"][name]["escalate_reasons"]
        if not reasons:
            continue
        for reason, count in reasons.items():
            lines.append(
                f"| {_STRATEGY_TITLE.get(name, name)} | {_REASON_TITLE.get(reason, reason)} "
                f"(`{reason}`) | {count} |"
            )
    return lines


def _calibration_table(rows: list[dict]) -> list[str]:
    lines = [
        "| Порог accept | Закрыла micro | Точность принятых | Вызовов LLM | Итоговая точность связки |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['accept']} | {row['micro_share_pct']}% | {row['micro_accuracy_pct']}% | "
            f"{row['llm_calls']} | {row['overall_accuracy_pct']}% |"
        )
    return lines


def _errors_table(payload: dict, name: str) -> list[str]:
    errors = payload["summary"]["strategies"][name]["errors"]
    if not errors:
        return ["_Ошибок нет._"]
    by_id = {row["id"]: row for row in payload["rows"]}
    lines = ["| Кейс | Группа | Ожидали | Получили | Кто решил | Обращение |", "|---|---|---|---|---|---|"]
    for error in errors:
        text = by_id[error["id"]]["text"].replace("\n", " ")
        preview = text[:70] + ("…" if len(text) > 70 else "")
        lines.append(
            f"| `{error['id']}` | {_GROUP_TITLE.get(error['group'], error['group'])} | "
            f"`{error['expected']}` | `{error['got']}` | {error['source']} | {preview} |"
        )
    return lines


def _micro_detail_table(payload: dict, name: str) -> list[str]:
    """Кейсы, которые micro-model закрыла сама: видно, на чём она выигрывает и где ошибается."""
    lines = ["| Кейс | Метка micro | Верно | score | Близость | Отрыв |", "|---|---|---|---|---|---|"]
    for row in payload["rows"]:
        cell = row[name]
        if cell["source"] != "micro" or cell["micro_status"] != "OK":
            continue
        lines.append(
            f"| `{row['id']}` | `{cell['micro_label']}` | {'✅' if cell['correct'] else '❌'} | "
            f"{cell['micro_score']} | {cell['micro_top_similarity']} | {cell['micro_margin']} |"
        )
    return lines


def build_report(payload: dict, run_name: str) -> str:
    summary = payload["summary"]
    strategies = payload["strategies"]
    baseline = summary["strategies"].get("llm_only")
    lines = [
        f"# Прогон {run_name} — micro-model first",
        "",
        f"**Кейсов:** {summary['total']} · **большая модель:** `{payload['llm_model']}` · "
        f"**эмбеддинги:** `{payload['embed_model']}` · **k:** {payload['k']} · "
        f"**temperature:** {payload['temperature']}",
        "",
        "**Пороги гейта:** "
        + " · ".join(
            f"`{backend}`: accept {limits['accept']}, sim_floor {limits['sim_floor']}, "
            f"margin_min {limits['margin_min']}"
            for backend, limits in payload["thresholds"].items()
        ),
        "",
        "## Стратегии",
        "",
    ]
    lines += _strategies_table(summary, strategies)
    if "delta" in summary:
        delta = summary["delta"]
        lines += [
            "",
            f"**Micro-first (эмбеддинги) против базовой линии:** вызовов большой модели меньше на "
            f"{delta['llm_calls_saved']} из {baseline['llm_calls'] if baseline else '—'} "
            f"({delta['llm_calls_saved_pct']}%), стоимость — {delta['cost_vs_llm_only_pct']}% от "
            f"базовой, средняя задержка — {delta['latency_ratio']}× от базовой, точность "
            f"{'+' if delta['accuracy_delta'] >= 0 else ''}{delta['accuracy_delta']} кейса.",
        ]
        if delta["broken_by_micro"]:
            lines.append(
                f"Сломано micro-моделью: {', '.join(f'`{i}`' for i in delta['broken_by_micro'])}. "
                f"Починено: {', '.join(f'`{i}`' for i in delta['fixed_by_micro']) or '—'}."
            )
    lines += ["", "## Точность по группам", ""]
    lines += _groups_table(summary, strategies)
    lines += ["", "## Почему micro-model отдавала кейсы наверх", ""]
    lines += _reasons_table(summary, strategies)
    for backend_key, rows in (summary.get("calibration") or {}).items():
        lines += ["", f"## Калибровка порога accept — {_STRATEGY_TITLE.get(backend_key, backend_key)}", ""]
        lines += [
            "Порог двигали задним числом по данным прогона: «жёсткие» причины эскалации им не",
            "отменяются, эскалированные кейсы считаются по фактическому ответу большой модели.",
            "",
        ]
        lines += _calibration_table(rows)
    lines += ["", "## Что micro-model закрыла сама (эмбеддинги)", ""]
    lines += _micro_detail_table(payload, "micro_embed_first")
    for name in strategies:
        lines += ["", f"## Ошибки — {_STRATEGY_TITLE.get(name, name)}", ""]
        lines += _errors_table(payload, name)
    return "\n".join(lines) + "\n"


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "run1"
    source = _RUNS_DIR / f"{name}.json"
    if not source.exists():
        print(f"Нет файла прогона: {source}", file=sys.stderr)
        return 2
    payload = json.loads(source.read_text(encoding="utf-8"))
    target = _RUNS_DIR / f"{name}.md"
    target.write_text(build_report(payload, name), encoding="utf-8")
    print(f"Записано: {target}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
