"""Подбор порогов гейта: сетка sim_floor × margin_min × accept на датасете, без вызовов LLM."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from app.agent_constants import MICRO_BACKENDS, MICRO_KNN_K, MICRO_THRESHOLDS  # noqa: E402
from app.micro.backends import get_backend  # noqa: E402
from app.micro.gate import judge  # noqa: E402
from app.micro.schema import Neighbor  # noqa: E402

_DAY_DIR = Path(__file__).resolve().parent
# Шкалы близости у бэкендов разные (косинус эмбеддингов заметно выше косинуса tfidf),
# поэтому сетка пола идёт с запасом в обе стороны.
_SIM_FLOOR_GRID = [round(0.10 + 0.05 * step, 2) for step in range(13)]
_MARGIN_GRID = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08]
_ACCEPT_GRID = [round(0.30 + 0.05 * step, 2) for step in range(11)]


def _load_cases(path: Path) -> list[dict]:
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [json.loads(ln) for ln in lines]


async def _neighbors_per_case(backend: str, cases: list[dict], k: int) -> list[list[Neighbor]]:
    """Соседи считаются один раз: сетка порогов на них уже не тратит ни сети, ни денег."""
    impl = await get_backend(backend)
    return [await impl.neighbors(case["text"], k) for case in cases]


def _grid_row(
    backend: str, cases: list[dict], neighbors: list[list[Neighbor]], limits: dict
) -> dict:
    accepted = 0
    accepted_correct = 0
    for case, found in zip(cases, neighbors):
        verdict = judge(backend, case["text"], found, limits=limits)
        if verdict.ok:
            accepted += 1
            accepted_correct += int(verdict.label == case["label"])
    total = len(cases)
    return {
        **limits,
        "micro_share_pct": round(100.0 * accepted / total, 1) if total else 0.0,
        "micro_accuracy_pct": round(100.0 * accepted_correct / accepted, 1) if accepted else 0.0,
        "accepted": accepted,
        "accepted_correct": accepted_correct,
        "llm_calls": total - accepted,
    }


async def _run(args: argparse.Namespace) -> int:
    load_dotenv(ROOT / ".env")
    cases = _load_cases(_DAY_DIR / args.dataset)
    out: dict[str, list[dict]] = {}
    for backend in args.backends.split(","):
        neighbors = await _neighbors_per_case(backend, cases, args.k)
        rows: list[dict] = []
        for sim_floor in _SIM_FLOOR_GRID:
            for margin_min in _MARGIN_GRID:
                for accept in _ACCEPT_GRID:
                    limits = {
                        "accept": accept,
                        "sim_floor": sim_floor,
                        "margin_min": margin_min,
                        "margin_full": MICRO_THRESHOLDS[backend]["margin_full"],
                    }
                    rows.append(_grid_row(backend, cases, neighbors, limits))
        out[backend] = rows
        best = [r for r in rows if r["micro_accuracy_pct"] >= args.min_accuracy]
        best.sort(key=lambda r: (-r["micro_share_pct"], -r["micro_accuracy_pct"]))
        print(f"\n=== {backend}: точность принятых ≥ {args.min_accuracy}%", file=sys.stderr)
        for row in best[:8]:
            print(
                f"  accept={row['accept']:<5} sim_floor={row['sim_floor']:<5} "
                f"margin_min={row['margin_min']:<5} → micro {row['micro_share_pct']}% "
                f"({row['accepted_correct']}/{row['accepted']}), LLM {row['llm_calls']}",
                file=sys.stderr,
            )
    out_dir = _DAY_DIR / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nЗаписано: {out_dir / args.out}", file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="День 10: подбор порогов гейта micro-model")
    parser.add_argument("--dataset", default="dataset.jsonl")
    parser.add_argument("--backends", default=",".join(MICRO_BACKENDS))
    parser.add_argument("--k", type=int, default=MICRO_KNN_K)
    parser.add_argument("--min-accuracy", type=float, default=90.0)
    parser.add_argument("--out", default="calibration.json")
    return asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
