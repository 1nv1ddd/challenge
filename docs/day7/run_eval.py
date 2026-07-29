"""Прогон датасета День 7: baseline (один вызов без гейта) против gated-пайплайна."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from app.agent_constants import TRIAGE_SAMPLES, TRIAGE_TEMPERATURE  # noqa: E402
from app.confidence.constraints import parse_decision  # noqa: E402
from app.confidence.inference import complete, cost_rub  # noqa: E402
from app.confidence.pipeline import triage  # noqa: E402
from app.confidence.prompts import triage_messages  # noqa: E402
from app.providers import AIProvider, RouterAIProvider  # noqa: E402

_DAY_DIR = Path(__file__).resolve().parent
_DEFAULT_MODEL = "openai/gpt-4o-mini"
_MAX_ATTEMPTS = 3


def _load_cases(path: Path, limit: int | None) -> list[dict]:
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return cases[:limit] if limit else cases


async def _baseline(provider: AIProvider, model: str, text: str, temperature: float) -> dict:
    """Как это выглядит без гейта: один вызов, ответ применяется как есть."""
    t_start = time.monotonic()
    if not text.strip():
        return {
            "action": None,
            "parse_error": "пустое обращение",
            "time_ms": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cost_rub": 0.0,
        }
    call = await complete(provider, model, triage_messages(text), temperature)
    try:
        decision = parse_decision(call.text)
        action, err = decision.action, None
    except ValueError as exc:
        action, err = None, str(exc)
    return {
        "action": action,
        "decision": decision.to_dict() if err is None else None,
        "parse_error": err,
        "time_ms": call.time_ms or round((time.monotonic() - t_start) * 1000),
        "prompt_tokens": call.prompt_tokens,
        "completion_tokens": call.completion_tokens,
        "cost_rub": cost_rub(call.prompt_tokens, call.completion_tokens),
    }


def _pct(part: int, total: int) -> float:
    return round(100.0 * part / total, 1) if total else 0.0


def _percentiles(values: list[int]) -> dict:
    if not values:
        return {"p50": 0, "p95": 0}
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {"p50": int(statistics.median(ordered)), "p95": ordered[idx]}


def _summarize(rows: list[dict]) -> dict:
    total = len(rows)
    base_correct = [r for r in rows if r["baseline"]["action"] in r["allowed_actions"]]
    base_unsafe = [r for r in rows if r["baseline"]["action"] in r["unsafe_actions"]]
    base_broken = [r for r in rows if r["baseline"]["parse_error"]]

    statuses = {"OK": 0, "UNSURE": 0, "FAIL": 0}
    for r in rows:
        statuses[r["gated"]["status"]] = statuses.get(r["gated"]["status"], 0) + 1
    accepted = [r for r in rows if r["gated"]["status"] == "OK"]
    accepted_correct = [r for r in accepted if r["gated"]["applied_action"] in r["allowed_actions"]]
    accepted_unsafe = [r for r in accepted if r["gated"]["applied_action"] in r["unsafe_actions"]]
    repaired = [r for r in rows if r["gated"]["metrics"]["repair_calls"] > 0]
    catches: dict[str, int] = {}
    caught_ids: list[str] = []
    for r in rows:
        first = [v for s in r["gated"]["samples"] for v in s.get("first_violations", [])]
        if first:
            caught_ids.append(r["id"])
        for v in first:
            key = v.split(":", 1)[0]
            catches[key] = catches.get(key, 0) + 1
    selfchecked = [r for r in rows if r["gated"]["metrics"]["selfcheck_calls"] > 0]
    blocked_unsafe = [
        r
        for r in rows
        if r["baseline"]["action"] in r["unsafe_actions"]
        and r["gated"]["applied_action"] not in r["unsafe_actions"]
    ]
    over_blocked = [
        r
        for r in rows
        if r["baseline"]["action"] in r["allowed_actions"] and r["gated"]["status"] != "OK"
    ]

    base_cost = round(sum(r["baseline"]["cost_rub"] for r in rows), 4)
    gate_cost = round(sum(r["gated"]["metrics"]["cost_rub"] for r in rows), 4)
    base_ms = [r["baseline"]["time_ms"] for r in rows]
    gate_ms = [r["gated"]["metrics"]["time_ms"] for r in rows]
    return {
        "total": total,
        "baseline": {
            "correct": len(base_correct),
            "correct_pct": _pct(len(base_correct), total),
            "unsafe_auto": len(base_unsafe),
            "format_errors": len(base_broken),
            "latency_ms": _percentiles(base_ms),
            "latency_ms_avg": int(statistics.fmean(base_ms)) if base_ms else 0,
            "cost_rub": base_cost,
            "llm_calls": sum(1 for r in rows if r["baseline"]["time_ms"] > 0),
        },
        "gated": {
            "statuses": statuses,
            "accepted": len(accepted),
            "accepted_pct": _pct(len(accepted), total),
            "accepted_correct": len(accepted_correct),
            "accepted_correct_pct": _pct(len(accepted_correct), len(accepted)),
            "unsafe_auto": len(accepted_unsafe),
            "rejected": total - len(accepted),
            "rejected_pct": _pct(total - len(accepted), total),
            "cases_with_repair": len(repaired),
            "constraint_catches": catches,
            "caught_ids": sorted(set(caught_ids)),
            "repair_calls": sum(r["gated"]["metrics"]["repair_calls"] for r in rows),
            "cases_with_selfcheck": len(selfchecked),
            "llm_calls": sum(r["gated"]["metrics"]["llm_calls"] for r in rows),
            "latency_ms": _percentiles(gate_ms),
            "latency_ms_avg": int(statistics.fmean(gate_ms)) if gate_ms else 0,
            "cost_rub": gate_cost,
        },
        "delta": {
            "blocked_unsafe": len(blocked_unsafe),
            "blocked_unsafe_ids": [r["id"] for r in blocked_unsafe],
            "over_blocked": len(over_blocked),
            "over_blocked_ids": [r["id"] for r in over_blocked],
            "latency_x": round(
                (statistics.fmean(gate_ms) / statistics.fmean(base_ms)) if base_ms else 0.0, 2
            ),
            "cost_x": round(gate_cost / base_cost, 2) if base_cost else 0.0,
        },
    }


async def _run(args: argparse.Namespace) -> int:
    load_dotenv(ROOT / ".env")
    key = os.getenv("ROUTERAI_API_KEY")
    if not key:
        print("Нет ROUTERAI_API_KEY в окружении/.env", file=sys.stderr)
        return 2
    provider = RouterAIProvider(key)
    cases = _load_cases(_DAY_DIR / args.dataset, args.limit)
    rows: list[dict] = []
    for case in cases:
        text = case["text"]
        # RouterAI изредка рвёт соединение — ретраим кейс целиком, чтобы прогон не терял данные.
        base, verdict = None, None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                base = await _baseline(provider, args.model, text, args.baseline_temperature)
                verdict = await triage(
                    provider,
                    args.model,
                    text,
                    samples=args.samples,
                    temperature=args.temperature,
                )
                break
            except httpx.HTTPError as exc:
                base, verdict = None, None
                print(f"{case['id']}: сетевая ошибка ({exc}), попытка {attempt + 1}", file=sys.stderr)
                await asyncio.sleep(3)
        if verdict is None or base is None:
            print(f"{case['id']}: пропущен — сеть не ответила", file=sys.stderr)
            continue
        row = {
            "id": case["id"],
            "group": case["group"],
            "text": text,
            "allowed_actions": case["allowed_actions"],
            "unsafe_actions": case["unsafe_actions"],
            "baseline": base,
            "gated": verdict.to_dict(),
        }
        rows.append(row)
        print(
            f"{case['id']:>4} {case['group']:<5} baseline={base['action'] or 'ERR':<12} "
            f"gate={verdict.status:<6} conf={verdict.confidence:<4} "
            f"applied={verdict.applied_action}",
            file=sys.stderr,
        )

    summary = _summarize(rows)
    out_dir = _DAY_DIR / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "samples": args.samples,
        "temperature": args.temperature,
        "baseline_temperature": args.baseline_temperature,
        "summary": summary,
        "rows": rows,
    }
    (out_dir / args.out).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), file=sys.stderr)
    print(f"\nЗаписано: {out_dir / args.out}", file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="День 7: baseline vs gated triage")
    parser.add_argument("--dataset", default="dataset.jsonl")
    parser.add_argument("--model", default=_DEFAULT_MODEL)
    parser.add_argument("--samples", type=int, default=TRIAGE_SAMPLES)
    parser.add_argument("--temperature", type=float, default=TRIAGE_TEMPERATURE)
    parser.add_argument("--baseline-temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="run1.json")
    return asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
