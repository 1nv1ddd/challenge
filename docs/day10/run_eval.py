"""Прогон датасета День 10: одно и то же обращение через micro-model и через большую LLM."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from app.agent_constants import (  # noqa: E402
    MICRO_EMBED_MODEL,
    MICRO_KNN_K,
    MICRO_LABELS,
    MICRO_LLM_MODEL,
    MICRO_STRATEGIES,
    MICRO_TEMPERATURE,
    MICRO_THRESHOLDS,
)
from app.micro.pipeline import classify_intent  # noqa: E402
from app.micro.schema import IntentResult  # noqa: E402
from app.providers import AIProvider, RouterAIProvider  # noqa: E402

_DAY_DIR = Path(__file__).resolve().parent
_MAX_ATTEMPTS = 3
# Сетка порога accept для таблицы калибровки: что мы получаем, сдвигая гейт в обе стороны.
_CALIBRATION_GRID = [round(0.30 + 0.05 * step, 2) for step in range(13)]
# Причины, которые не зависят от порога accept: их сдвигом гейта не отменить.
_HARD_REASONS = ("short_input", "no_close_neighbor", "fallback_label", "low_margin")


def _load_cases(path: Path, limit: int | None) -> list[dict]:
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    cases = [json.loads(ln) for ln in lines]
    return cases[:limit] if limit else cases


def check_dataset(cases: list[dict]) -> list[str]:
    """Датасет должен быть валиден сам по себе: уникальные id и метки из того же enum."""
    problems: list[str] = []
    seen: set[str] = set()
    for case in cases:
        case_id = str(case.get("id") or "")
        if case_id in seen:
            problems.append(f"{case_id}: дубль id")
        seen.add(case_id)
        if case.get("label") not in MICRO_LABELS:
            problems.append(f"{case_id}: метка {case.get('label')!r} вне списка {list(MICRO_LABELS)}")
        if not str(case.get("text") or "").strip():
            problems.append(f"{case_id}: пустой текст обращения")
        if case.get("group") not in ("simple", "borderline", "hard"):
            problems.append(f"{case_id}: неизвестная группа {case.get('group')!r}")
    return problems


def _row_for(case: dict, result: IntentResult) -> dict:
    micro = result.micro
    return {
        "label": result.label,
        "correct": result.label == case["label"],
        "source": result.source,
        "ok": result.ok,
        "micro_label": micro.label if micro else None,
        "micro_correct": (micro.label == case["label"]) if micro else None,
        "micro_status": micro.status if micro else None,
        "micro_score": micro.score if micro else None,
        "micro_top_similarity": round(micro.top_similarity, 4) if micro else None,
        "micro_margin": round(micro.margin, 4) if micro else None,
        "escalate_reason": micro.escalate_reason if micro else None,
        # Порогом accept можно двигать только «мягкие» отказы — это нужно таблице калибровки.
        "hard_escalate": bool(micro and micro.escalate_reason in _HARD_REASONS),
        "llm_label": result.llm_answer.label if result.llm_answer else None,
        "llm_confidence": result.llm_answer.confidence if result.llm_answer else None,
        "escalated": result.metrics["escalated"],
        "llm_calls": result.metrics["llm_calls"],
        "repairs": result.metrics["repair_calls"],
        "time_ms": result.metrics["time_ms"],
        "micro_ms": result.metrics["micro_ms"],
        "llm_ms": result.metrics["llm_ms"],
        "cost_rub": result.metrics["cost_rub"],
    }


def _pct(part: int, total: int) -> float:
    return round(100.0 * part / total, 1) if total else 0.0


def _percentiles(values: list[int]) -> dict:
    if not values:
        return {"p50": 0, "p95": 0}
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {"p50": int(statistics.median(ordered)), "p95": ordered[idx]}


def _by_group(rows: list[dict], key: str) -> dict:
    groups: dict[str, dict] = {}
    for row in rows:
        cell = groups.setdefault(row["group"], {"total": 0, "correct": 0, "escalated": 0})
        cell["total"] += 1
        cell["correct"] += int(row[key]["correct"])
        cell["escalated"] += int(row[key]["escalated"])
    for cell in groups.values():
        cell["accuracy_pct"] = _pct(cell["correct"], cell["total"])
        cell["escalated_pct"] = _pct(cell["escalated"], cell["total"])
    return groups


def _confusion(rows: list[dict], key: str) -> list[dict]:
    return [
        {"id": row["id"], "group": row["group"], "expected": row["expected"], "got": row[key]["label"],
         "source": row[key]["source"]}
        for row in rows
        if not row[key]["correct"]
    ]


def _strategy_stats(rows: list[dict], key: str) -> dict:
    cells = [row[key] for row in rows]
    total = len(cells)
    handled_by_micro = [c for c in cells if c["source"] in ("micro", "micro_after_llm_error")]
    escalated = [c for c in cells if c["escalated"]]
    times = [c["time_ms"] for c in cells]
    micro_decided = [c for c in cells if c["source"] == "micro" and c["micro_status"] == "OK"]
    return {
        "total": total,
        "correct": sum(1 for c in cells if c["correct"]),
        "accuracy_pct": _pct(sum(1 for c in cells if c["correct"]), total),
        "handled_by_micro": len(handled_by_micro),
        "handled_by_micro_pct": _pct(len(handled_by_micro), total),
        "escalated": len(escalated),
        "escalated_pct": _pct(len(escalated), total),
        # Точность именно тех кейсов, которые micro-model закрыла сама — цена отсечения.
        "micro_accepted_correct": sum(1 for c in micro_decided if c["correct"]),
        "micro_accepted_total": len(micro_decided),
        "micro_accepted_accuracy_pct": _pct(
            sum(1 for c in micro_decided if c["correct"]), len(micro_decided)
        ),
        "llm_calls": sum(c["llm_calls"] for c in cells),
        "repairs": sum(c["repairs"] for c in cells),
        "format_failures": sum(1 for c in cells if not c["ok"]),
        "cost_rub": round(sum(c["cost_rub"] for c in cells), 4),
        "latency_ms_avg": int(statistics.fmean(times)) if times else 0,
        "latency_ms": _percentiles(times),
        "micro_ms_avg": int(statistics.fmean([c["micro_ms"] for c in cells])) if cells else 0,
        "escalate_reasons": _reason_counts(cells),
        "by_group": _by_group(rows, key),
        "wrong_ids": [row["id"] for row in rows if not row[key]["correct"]],
        "errors": _confusion(rows, key),
    }


def _reason_counts(cells: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for cell in cells:
        reason = cell["escalate_reason"]
        if reason:
            counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: -item[1]))


def _calibration(rows: list[dict], micro_key: str, llm_key: str) -> list[dict]:
    """Что даёт сдвиг порога accept: доля кейсов у micro, её точность и итоговая точность связки."""
    table: list[dict] = []
    for threshold in _CALIBRATION_GRID:
        accepted = [
            row
            for row in rows
            if not row[micro_key]["hard_escalate"] and (row[micro_key]["micro_score"] or 0) >= threshold
        ]
        escalated = [row for row in rows if row not in accepted]
        correct = sum(1 for row in accepted if row[micro_key]["micro_correct"])
        # Эскалированные кейсы решает большая модель — берём её фактический ответ из llm_only.
        correct += sum(1 for row in escalated if row[llm_key]["correct"])
        table.append(
            {
                "accept": threshold,
                "micro_share_pct": _pct(len(accepted), len(rows)),
                "micro_accuracy_pct": _pct(
                    sum(1 for row in accepted if row[micro_key]["micro_correct"]), len(accepted)
                ),
                "llm_calls": len(escalated),
                "overall_accuracy_pct": _pct(correct, len(rows)),
            }
        )
    return table


def _summarize(rows: list[dict], strategies: list[str]) -> dict:
    summary = {"total": len(rows), "strategies": {s: _strategy_stats(rows, s) for s in strategies}}
    if "micro_embed_first" in strategies and "llm_only" in strategies:
        micro, llm = summary["strategies"]["micro_embed_first"], summary["strategies"]["llm_only"]
        summary["delta"] = {
            "llm_calls_saved": llm["llm_calls"] - micro["llm_calls"],
            "llm_calls_saved_pct": _pct(llm["llm_calls"] - micro["llm_calls"], llm["llm_calls"]),
            "cost_vs_llm_only_pct": (
                round(100.0 * micro["cost_rub"] / llm["cost_rub"], 1) if llm["cost_rub"] else 0.0
            ),
            "latency_ratio": (
                round(micro["latency_ms_avg"] / llm["latency_ms_avg"], 2)
                if llm["latency_ms_avg"]
                else 0.0
            ),
            "accuracy_delta": micro["correct"] - llm["correct"],
            "fixed_by_micro": [
                r["id"] for r in rows
                if r["micro_embed_first"]["correct"] and not r["llm_only"]["correct"]
            ],
            "broken_by_micro": [
                r["id"] for r in rows
                if not r["micro_embed_first"]["correct"] and r["llm_only"]["correct"]
            ],
        }
    for backend_key in ("micro_embed_first", "micro_tfidf_first"):
        if backend_key in strategies and "llm_only" in strategies:
            summary.setdefault("calibration", {})[backend_key] = _calibration(
                rows, backend_key, "llm_only"
            )
    return summary


async def _run_case(
    provider: AIProvider, case: dict, strategies: list[str], args: argparse.Namespace
) -> dict:
    row = {
        "id": case["id"],
        "group": case["group"],
        "text": case["text"],
        "expected": case["label"],
        "note": case.get("note", ""),
    }
    for strategy in strategies:
        result = await classify_intent(
            provider,
            case["text"],
            strategy=strategy,
            llm_model=args.llm_model,
            temperature=args.temperature,
            k=args.k,
            embed_model=args.embed_model,
        )
        row[strategy] = _row_for(case, result)
    return row


async def _run(args: argparse.Namespace) -> int:
    load_dotenv(ROOT / ".env")
    cases = _load_cases(_DAY_DIR / args.dataset, args.limit)
    problems = check_dataset(cases)
    if problems:
        print("Датасет невалиден:", file=sys.stderr)
        for line in problems:
            print(f"  {line}", file=sys.stderr)
        return 3
    if args.check_only:
        print(f"Датасет валиден: {len(cases)} кейсов", file=sys.stderr)
        return 0

    key = os.getenv("ROUTERAI_API_KEY")
    if not key:
        print("Нет ROUTERAI_API_KEY в окружении/.env", file=sys.stderr)
        return 2
    provider = RouterAIProvider(key)
    strategies = args.strategies.split(",")
    rows: list[dict] = []
    for case in cases:
        row = None
        # RouterAI изредка рвёт соединение — ретраим кейс целиком, чтобы прогон не терял данные.
        for attempt in range(_MAX_ATTEMPTS):
            try:
                row = await _run_case(provider, case, strategies, args)
                break
            except httpx.HTTPError as exc:
                print(f"{case['id']}: сетевая ошибка ({exc}), попытка {attempt + 1}", file=sys.stderr)
                await asyncio.sleep(3)
        if row is None:
            print(f"{case['id']}: пропущен — сеть не ответила", file=sys.stderr)
            continue
        rows.append(row)
        marks = " ".join(
            f"{s.replace('micro_', '').replace('_first', '')}="
            f"{'ok ' if row[s]['correct'] else 'ERR'}({row[s]['source'][:5]})"
            for s in strategies
        )
        print(f"{row['id']:>4} {row['group']:<11} {marks}", file=sys.stderr)

    summary = _summarize(rows, strategies)
    out_dir = _DAY_DIR / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "llm_model": args.llm_model,
        "embed_model": args.embed_model,
        "k": args.k,
        "temperature": args.temperature,
        "thresholds": MICRO_THRESHOLDS,
        "strategies": strategies,
        "summary": summary,
        "rows": rows,
    }
    (out_dir / args.out).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary["strategies"], ensure_ascii=False, indent=2)[:2000], file=sys.stderr)
    print(f"\nЗаписано: {out_dir / args.out}", file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="День 10: micro-model first vs большая LLM")
    parser.add_argument("--dataset", default="dataset.jsonl")
    parser.add_argument("--llm-model", default=MICRO_LLM_MODEL)
    parser.add_argument("--embed-model", default=MICRO_EMBED_MODEL)
    parser.add_argument("--k", type=int, default=MICRO_KNN_K)
    parser.add_argument("--temperature", type=float, default=MICRO_TEMPERATURE)
    parser.add_argument("--strategies", default=",".join(MICRO_STRATEGIES))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--check-only", action="store_true", help="только проверить датасет")
    parser.add_argument("--out", default="run1.json")
    return asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
