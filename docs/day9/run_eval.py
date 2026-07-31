"""Прогон датасета День 9: одно и то же письмо через монолит и через цепочку этапов."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from app.agent_constants import (  # noqa: E402
    INTAKE_FIELD_KEYS,
    INTAKE_MONO_MODEL,
    INTAKE_NUMERIC_FIELDS,
    INTAKE_STAGE_MODELS,
    INTAKE_TEMPERATURE,
    ROUTING_SMALL_MODEL,
)
from app.providers import AIProvider, RouterAIProvider  # noqa: E402
from app.staged.pipeline import run_intake  # noqa: E402
from app.staged.policy import decide_by_rules  # noqa: E402
from app.staged.schema import IntakeFields, IntakeResult  # noqa: E402

_DAY_DIR = Path(__file__).resolve().parent
_MAX_ATTEMPTS = 3
_QUOTES_RE = re.compile(r"[«»\"'`]")


def _load_cases(path: Path, limit: int | None) -> list[dict]:
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    cases = [json.loads(ln) for ln in lines]
    return cases[:limit] if limit else cases


def _expected_fields(case: dict) -> IntakeFields:
    values = dict(case["fields"])
    for key in INTAKE_FIELD_KEYS:
        if key not in INTAKE_NUMERIC_FIELDS:
            values[key] = values.get(key) or "unknown"
    return IntakeFields(**values)


def check_dataset(cases: list[dict]) -> list[str]:
    """Датасет должен быть согласован сам с собой: политика на эталонных полях даёт эталонное решение."""
    problems: list[str] = []
    for case in cases:
        rules = decide_by_rules(_expected_fields(case), date.fromisoformat(case["today"]))
        expected = (case["decision"], case["reason"], case["missing"])
        actual = (rules.decision, rules.reason, rules.missing)
        if expected != actual:
            problems.append(f"{case['id']}: ожидание {expected}, политика даёт {actual}")
    return problems


def _same_text(expected: str, got: str) -> bool:
    """Сравнение текстовых полей: кавычки, регистр и «ё» не считаются ошибкой извлечения."""
    def norm(value: str) -> str:
        return " ".join(_QUOTES_RE.sub("", str(value)).lower().replace("ё", "е").split())

    return norm(expected) == norm(got)


def _grade_fields(case: dict, result: IntakeResult) -> dict[str, bool]:
    expected = case["fields"]
    got = result.fields.to_dict()
    grades: dict[str, bool] = {}
    for key in INTAKE_FIELD_KEYS:
        exp, act = expected.get(key), got.get(key)
        if key in INTAKE_NUMERIC_FIELDS:
            grades[key] = exp == act
        else:
            grades[key] = _same_text(exp or "unknown", act or "unknown")
    return grades


def _row_for(case: dict, result: IntakeResult) -> dict:
    grades = _grade_fields(case, result)
    decision_ok = result.decision.decision == case["decision"]
    reason_ok = result.decision.reason == case["reason"]
    # Решение по уже извлечённым полям: отделяет ошибку извлечения от ошибки применения политики.
    rules_on_extracted = decide_by_rules(result.fields, date.fromisoformat(case["today"]))
    return {
        "mode": result.mode,
        "fields": result.fields.to_dict(),
        "grades": grades,
        "fields_correct": sum(1 for ok in grades.values() if ok),
        "all_fields_correct": all(grades.values()),
        "decision": result.decision.decision,
        "reason": result.decision.reason,
        "missing": result.decision.missing,
        "decision_source": result.decision.source,
        "decision_correct": decision_ok,
        "reason_correct": reason_ok,
        "verdict_correct": decision_ok and reason_ok,
        "decision_matches_rules": (
            result.decision.decision == rules_on_extracted.decision
            and result.decision.reason == rules_on_extracted.reason
        ),
        "reply_subject": result.reply_subject,
        "reply_body": result.reply_body,
        "has_reply": bool(result.reply_body),
        "ok": result.ok,
        "repairs": result.metrics["repair_calls"],
        "llm_calls": result.metrics["llm_calls"],
        "time_ms": result.metrics["time_ms"],
        "cost_rub": result.metrics["cost_rub"],
        "stages": [s.to_dict() for s in result.stages],
    }


def _pct(part: int, total: int) -> float:
    return round(100.0 * part / total, 1) if total else 0.0


def _percentiles(values: list[int]) -> dict:
    if not values:
        return {"p50": 0, "p95": 0}
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {"p50": int(statistics.median(ordered)), "p95": ordered[idx]}


def _strategy_stats(rows: list[dict], key: str) -> dict:
    cells = [r[key] for r in rows]
    total_fields = len(cells) * len(INTAKE_FIELD_KEYS)
    per_field = {
        field: sum(1 for c in cells if c["grades"][field]) for field in INTAKE_FIELD_KEYS
    }
    times = [c["time_ms"] for c in cells]
    return {
        "fields_correct": sum(c["fields_correct"] for c in cells),
        "fields_total": total_fields,
        "fields_pct": _pct(sum(c["fields_correct"] for c in cells), total_fields),
        "all_fields_correct": sum(1 for c in cells if c["all_fields_correct"]),
        "per_field": per_field,
        "decision_correct": sum(1 for c in cells if c["decision_correct"]),
        "verdict_correct": sum(1 for c in cells if c["verdict_correct"]),
        "verdict_pct": _pct(sum(1 for c in cells if c["verdict_correct"]), len(cells)),
        "decision_matches_rules": sum(1 for c in cells if c["decision_matches_rules"]),
        "format_failures": sum(1 for c in cells if not c["ok"]),
        "no_reply": sum(1 for c in cells if not c["has_reply"]),
        "repairs": sum(c["repairs"] for c in cells),
        "llm_calls": sum(c["llm_calls"] for c in cells),
        "cost_rub": round(sum(c["cost_rub"] for c in cells), 4),
        "latency_ms": _percentiles(times),
        "latency_ms_avg": int(statistics.fmean(times)) if times else 0,
        "wrong_verdict_ids": [r["id"] for r in rows if not r[key]["verdict_correct"]],
        "wrong_fields_ids": [r["id"] for r in rows if not r[key]["all_fields_correct"]],
    }


def _summarize(rows: list[dict], strategies: list[str]) -> dict:
    summary = {"total": len(rows), "strategies": {s: _strategy_stats(rows, s) for s in strategies}}
    if "mono_large" in summary["strategies"] and "staged" in summary["strategies"]:
        mono, staged = summary["strategies"]["mono_large"], summary["strategies"]["staged"]
        summary["delta"] = {
            "cost_vs_mono_large_pct": (
                round(100.0 * staged["cost_rub"] / mono["cost_rub"], 1) if mono["cost_rub"] else 0.0
            ),
            "verdict_delta": staged["verdict_correct"] - mono["verdict_correct"],
            "fields_delta": staged["fields_correct"] - mono["fields_correct"],
            "fixed_by_staged": [
                r["id"]
                for r in rows
                if r["staged"]["verdict_correct"] and not r["mono_large"]["verdict_correct"]
            ],
            "broken_by_staged": [
                r["id"]
                for r in rows
                if not r["staged"]["verdict_correct"] and r["mono_large"]["verdict_correct"]
            ],
        }
    return summary


async def _strategy_result(
    provider: AIProvider, case: dict, mode: str, args: argparse.Namespace
) -> IntakeResult:
    mono_model = args.small_model if mode == "mono_small" else args.mono_model
    return await run_intake(
        provider,
        case["letter"],
        mode="mono" if mode.startswith("mono") else mode,
        today=case["today"],
        mono_model=mono_model,
        models={
            "normalize": args.normalize_model,
            "decide": args.decide_model,
            "compose": args.compose_model,
        },
        temperature=args.temperature,
    )


async def _run_case(
    provider: AIProvider, case: dict, strategies: list[str], args: argparse.Namespace
) -> dict:
    row = {
        "id": case["id"],
        "group": case["group"],
        "today": case["today"],
        "letter": case["letter"],
        "expected": {
            "fields": case["fields"],
            "decision": case["decision"],
            "reason": case["reason"],
            "missing": case["missing"],
        },
        "note": case.get("note", ""),
    }
    for mode in strategies:
        result = await _strategy_result(provider, case, mode, args)
        row[mode] = _row_for(case, result)
    return row


async def _run(args: argparse.Namespace) -> int:
    load_dotenv(ROOT / ".env")
    cases = _load_cases(_DAY_DIR / args.dataset, args.limit)
    problems = check_dataset(cases)
    if problems:
        print("Датасет противоречит политике:", file=sys.stderr)
        for line in problems:
            print(f"  {line}", file=sys.stderr)
        return 3
    if args.check_only:
        print(f"Датасет согласован: {len(cases)} кейсов", file=sys.stderr)
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
                print(
                    f"{case['id']}: сетевая ошибка ({exc}), попытка {attempt + 1}", file=sys.stderr
                )
                await asyncio.sleep(3)
        if row is None:
            print(f"{case['id']}: пропущен — сеть не ответила", file=sys.stderr)
            continue
        rows.append(row)
        marks = " ".join(
            f"{mode}={'ok ' if row[mode]['verdict_correct'] else 'ERR'}"
            f"({row[mode]['fields_correct']}/{len(INTAKE_FIELD_KEYS)})"
            for mode in strategies
        )
        print(f"{row['id']:>4} {row['group']:<11} {marks}", file=sys.stderr)

    summary = _summarize(rows, strategies)
    out_dir = _DAY_DIR / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "mono_model": args.mono_model,
        "small_model": args.small_model,
        "stage_models": {
            "normalize": args.normalize_model,
            "decide": args.decide_model,
            "compose": args.compose_model,
        },
        "temperature": args.temperature,
        "strategies": strategies,
        "summary": summary,
        "rows": rows,
    }
    (out_dir / args.out).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary["strategies"], ensure_ascii=False, indent=2), file=sys.stderr)
    print(f"\nЗаписано: {out_dir / args.out}", file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="День 9: monolithic vs multi-stage")
    parser.add_argument("--dataset", default="dataset.jsonl")
    parser.add_argument("--mono-model", default=INTAKE_MONO_MODEL)
    parser.add_argument("--small-model", default=ROUTING_SMALL_MODEL)
    parser.add_argument("--normalize-model", default=INTAKE_STAGE_MODELS["normalize"])
    parser.add_argument("--decide-model", default=INTAKE_STAGE_MODELS["decide"])
    parser.add_argument("--compose-model", default=INTAKE_STAGE_MODELS["compose"])
    parser.add_argument("--temperature", type=float, default=INTAKE_TEMPERATURE)
    parser.add_argument("--strategies", default="mono_large,mono_small,staged,staged_rules")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--check-only", action="store_true", help="только проверить датасет")
    parser.add_argument("--out", default="run1.json")
    return asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
