"""Прогон датасета День 8: три стратегии на одних запросах — только дешёвая, только сильная, роутер."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from app.agent_constants import (  # noqa: E402
    ROUTING_CONSISTENCY_TEMPERATURE,
    ROUTING_LARGE_MODEL,
    ROUTING_SMALL_MODEL,
    ROUTING_TEMPERATURE,
)
from app.confidence.inference import complete  # noqa: E402
from app.providers import AIProvider, RouterAIProvider  # noqa: E402
from app.routing.cascade import route_answer  # noqa: E402
from app.routing.pricing import cost_rub_model  # noqa: E402
from app.routing.prompts import answer_messages  # noqa: E402
from app.routing.signals import split_confidence  # noqa: E402

_DAY_DIR = Path(__file__).resolve().parent
_MAX_ATTEMPTS = 3


def _load_cases(path: Path, limit: int | None) -> list[dict]:
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    cases = [json.loads(ln) for ln in lines]
    return cases[:limit] if limit else cases


def _is_correct(answer: str, expect: list[str]) -> bool:
    """Ответ засчитан, если совпали все ожидаемые шаблоны (регистр и переводы строк не важны)."""
    return all(re.search(pattern, answer, re.I | re.S) for pattern in expect)


async def _single_model(provider: AIProvider, model: str, question: str, temperature: float) -> dict:
    """Стратегия без роутинга: один вызов фиксированной модели."""
    t_start = time.monotonic()
    call = await complete(provider, model, answer_messages(question), temperature)
    body, self_reported = split_confidence(call.text)
    return {
        "model": model,
        "answer": body,
        "self_reported": self_reported,
        "time_ms": call.time_ms or round((time.monotonic() - t_start) * 1000),
        "prompt_tokens": call.prompt_tokens,
        "completion_tokens": call.completion_tokens,
        "cost_rub": cost_rub_model(model, call.prompt_tokens, call.completion_tokens),
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
    correct = [r for r in rows if r[key]["correct"]]
    costs = [r[key]["cost_rub"] for r in rows]
    times = [r[key]["time_ms"] for r in rows]
    return {
        "correct": len(correct),
        "correct_pct": _pct(len(correct), len(rows)),
        "wrong_ids": [r["id"] for r in rows if not r[key]["correct"]],
        "cost_rub": round(sum(costs), 4),
        "latency_ms": _percentiles(times),
        "latency_ms_avg": int(statistics.fmean(times)) if times else 0,
    }


def _summarize(rows: list[dict]) -> dict:
    total = len(rows)
    paths: dict[str, int] = {}
    for r in rows:
        path = r["router"]["path"]
        paths[path] = paths.get(path, 0) + 1

    kept = [r for r in rows if r["router"]["path"] == "small"]
    escalated = [r for r in rows if r["router"]["path"] == "small→large"]
    prerouted = [r for r in rows if r["router"]["path"] == "large"]
    # Точная проверка решения роутера: сравниваем с тем самым черновиком, который он оценивал.
    over_escalated = [r for r in escalated if r["router"]["small_draft_correct"]]
    rescued = [
        r for r in escalated if not r["router"]["small_draft_correct"] and r["router"]["correct"]
    ]
    missed = [r for r in kept if not r["router"]["correct"]]
    # Pre-routing тоже может промахнуться: отправил на дорогую модель то, что дешёвая знала.
    preroute_overspend = [r for r in prerouted if r["small_only"]["correct"]]

    small_stats = _strategy_stats(rows, "small_only")
    large_stats = _strategy_stats(rows, "large_only")
    router_stats = _strategy_stats(rows, "router")
    return {
        "total": total,
        "small_only": small_stats,
        "large_only": large_stats,
        "router": {
            **router_stats,
            "paths": paths,
            "kept_small": len(kept),
            "kept_small_pct": _pct(len(kept), total),
            "escalated_after_small": len(escalated),
            "prerouted_large": len(prerouted),
            "llm_calls": sum(r["router"]["llm_calls"] for r in rows),
            "wasted_rub": round(sum(r["router"]["wasted_rub"] for r in rows), 4),
        },
        "delta": {
            "over_escalated": len(over_escalated),
            "over_escalated_ids": [r["id"] for r in over_escalated],
            "rescued": len(rescued),
            "rescued_ids": [r["id"] for r in rescued],
            "missed_escalation": len(missed),
            "missed_escalation_ids": [r["id"] for r in missed],
            "preroute_overspend": len(preroute_overspend),
            "preroute_overspend_ids": [r["id"] for r in preroute_overspend],
            "cost_vs_large_pct": (
                round(100.0 * router_stats["cost_rub"] / large_stats["cost_rub"], 1)
                if large_stats["cost_rub"]
                else 0.0
            ),
            "cost_vs_small_x": (
                round(router_stats["cost_rub"] / small_stats["cost_rub"], 2)
                if small_stats["cost_rub"]
                else 0.0
            ),
        },
    }


async def _run_case(provider: AIProvider, case: dict, args: argparse.Namespace) -> dict:
    expect = case["expect"]
    small = await _single_model(provider, args.small_model, case["question"], args.temperature)
    large = await _single_model(provider, args.large_model, case["question"], args.temperature)
    result = await route_answer(
        provider,
        case["question"],
        small_model=args.small_model,
        large_model=args.large_model,
        temperature=args.temperature,
        consistency_temperature=args.consistency_temperature,
    )
    draft = next((a for a in result.attempts if a.tier == "small" and a.answer), None)
    draft_body = split_confidence(draft.answer)[0] if draft else ""
    return {
        "id": case["id"],
        "group": case["group"],
        "question": case["question"],
        "expect": expect,
        "note": case.get("note", ""),
        "small_only": {**small, "correct": _is_correct(small["answer"], expect)},
        "large_only": {**large, "correct": _is_correct(large["answer"], expect)},
        "router": {
            "answer": result.answer,
            "correct": _is_correct(result.answer, expect),
            "model": result.model,
            "tier": result.tier,
            "path": result.metrics["path"],
            "escalated": result.escalated,
            "escalation_reason": result.escalation_reason,
            "preroute": result.preroute.to_dict(),
            "assessment": (
                draft.assessment.to_dict() if draft and draft.assessment else None
            ),
            # None, а не False: при pre-routing черновика дешёвой модели не существует —
            # это не «дешёвая ошиблась», это «её не спрашивали».
            "small_draft_correct": _is_correct(draft_body, expect) if draft_body else None,
            "llm_calls": result.metrics["llm_calls"],
            "time_ms": result.metrics["time_ms"],
            "cost_rub": result.metrics["cost_rub"],
            "wasted_rub": result.metrics["wasted_rub"],
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
        row = None
        # RouterAI изредка рвёт соединение — ретраим кейс целиком, чтобы прогон не терял данные.
        for attempt in range(_MAX_ATTEMPTS):
            try:
                row = await _run_case(provider, case, args)
                break
            except httpx.HTTPError as exc:
                print(f"{case['id']}: сетевая ошибка ({exc}), попытка {attempt + 1}", file=sys.stderr)
                await asyncio.sleep(3)
        if row is None:
            print(f"{case['id']}: пропущен — сеть не ответила", file=sys.stderr)
            continue
        rows.append(row)
        r = row["router"]
        print(
            f"{row['id']:>4} {row['group']:<6} small={'ok ' if row['small_only']['correct'] else 'ERR'} "
            f"large={'ok ' if row['large_only']['correct'] else 'ERR'} "
            f"router={'ok ' if r['correct'] else 'ERR'} path={r['path']:<12} {r['cost_rub']:.4f} ₽",
            file=sys.stderr,
        )

    summary = _summarize(rows)
    out_dir = _DAY_DIR / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "small_model": args.small_model,
        "large_model": args.large_model,
        "temperature": args.temperature,
        "consistency_temperature": args.consistency_temperature,
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
    parser = argparse.ArgumentParser(description="День 8: small vs large vs router")
    parser.add_argument("--dataset", default="dataset.jsonl")
    parser.add_argument("--small-model", default=ROUTING_SMALL_MODEL)
    parser.add_argument("--large-model", default=ROUTING_LARGE_MODEL)
    parser.add_argument("--temperature", type=float, default=ROUTING_TEMPERATURE)
    parser.add_argument(
        "--consistency-temperature", type=float, default=ROUTING_CONSISTENCY_TEMPERATURE
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="run1.json")
    return asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
