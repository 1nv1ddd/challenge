"""Прогон День 12: ловушки во внешнем контенте против наборов слоёв защиты."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from app.agent_constants import (  # noqa: E402
    INDIRECT_MODEL,
    INDIRECT_PRESETS,
    INDIRECT_TEMPERATURE,
)
from app.indirect import (  # noqa: E402
    IndirectCase,
    IndirectResult,
    LayerRun,
    corpus_stats,
    layer_effect,
    load_cases,
    run_case,
    select_cases,
)
from app.providers import AIProvider, RouterAIProvider  # noqa: E402

_DAY_DIR = Path(__file__).resolve().parent
_MAX_ATTEMPTS = 3
_ANSWER_PREVIEW = 400


async def _run_one(
    provider: AIProvider, case: IndirectCase, preset: str, args: argparse.Namespace
) -> IndirectResult:
    """Один кейс при одном пресете, с ретраями на сетевые сбои."""
    layers = INDIRECT_PRESETS[preset]
    for attempt in range(_MAX_ATTEMPTS):
        try:
            result = await run_case(
                provider, case, layers, model=args.model, temperature=args.temperature
            )
        except httpx.HTTPError as exc:
            print(f"{case.id}/{preset}: сеть ({exc}), попытка {attempt + 1}", file=sys.stderr)
            await asyncio.sleep(3)
            continue
        mark = "INJECTED" if result.injected else ("blocked" if result.blocked else "clean")
        useful = "польза:да" if result.useful else "польза:НЕТ"
        print(f"{case.id:<22} {preset:<9} {mark:<9} {useful}", file=sys.stderr)
        return result
    print(f"{case.id}/{preset}: пропущен — сеть не ответила", file=sys.stderr)
    return IndirectResult(
        case_id=case.id, scenario=case.scenario, hiding=case.hiding, error="network"
    )


def _summarize(runs: dict[str, LayerRun], cases: list[IndirectCase]) -> dict:
    summary: dict = {"corpus": corpus_stats(cases), "presets": {}}
    for preset, run in runs.items():
        summary["presets"][preset] = {
            "layers": list(run.layers()),
            "total": run.total,
            "injected": run.injected,
            "blocked": run.blocked,
            "broke_usefulness": run.broke_usefulness,
            "injection_rate": round(run.injected / run.total, 3) if run.total else 0.0,
            "cost_rub": run.cost_rub,
            "injected_ids": [r.case_id for r in run.results if r.injected],
            "useless_ids": [r.case_id for r in run.results if not r.useful and r.error is None],
        }
    summary["effect"] = layer_effect(runs)
    return summary


def _md_matrix(runs: dict[str, LayerRun], cases: list[IndirectCase]) -> list[str]:
    presets = [p for p in INDIRECT_PRESETS if p in runs]
    by_case = {(r.case_id, run.preset): r for run in runs.values() for r in run.results}
    lines = [
        "| Ловушка | Агент | Сокрытие | " + " | ".join(f"`{p}`" for p in presets) + " |",
        "|---" * (3 + len(presets)) + "|",
    ]
    for case in cases:
        cells = []
        for preset in presets:
            result = by_case.get((case.id, preset))
            if result is None or result.error:
                cells.append("⚠️")
            elif result.injected:
                cells.append("❌ прошла")
            elif result.blocked:
                cells.append("🛑 блок")
            elif not result.useful:
                cells.append("⚠️ пусто")
            else:
                cells.append("✅ чисто")
        lines.append(
            f"| `{case.id}` | {case.scenario} | `{case.hiding}` | " + " | ".join(cells) + " |"
        )
    return lines


def _md_report(payload: dict, runs: dict[str, LayerRun], cases: list[IndirectCase]) -> str:
    summary = payload["summary"]
    lines = [
        f"# Прогон непрямых инъекций — {payload['label']}",
        "",
        f"Модель: `{payload['model']}` · температура {payload['temperature']} · "
        f"ловушек в прогоне: {len(cases)}",
        "",
        "## Итог по наборам слоёв",
        "",
        "| Пресет | Слои | Инъекций прошло | Блокировок | Польза потеряна | Стоимость |",
        "|---|---|---|---|---|---|",
    ]
    for preset, stats in summary["presets"].items():
        layers = ", ".join(stats["layers"]) or "—"
        lines.append(
            f"| `{preset}` | {layers} | {stats['injected']} / {stats['total']} | "
            f"{stats['blocked']} | {stats['broke_usefulness']} | {stats['cost_rub']} ₽ |"
        )
    lines += ["", "## Матрица «ловушка × защита»", ""]
    lines += _md_matrix(runs, cases)
    effect = summary["effect"]
    lines += [
        "",
        "## Вклад защиты",
        "",
        f"- **закрыто всеми слоями:** {', '.join(f'`{c}`' for c in effect['fixed']) or '—'}",
        f"- **проходит и со всеми слоями:** {', '.join(f'`{c}`' for c in effect['still_injected']) or '—'}",
        f"- **защита сломала полезный ответ:** "
        f"{', '.join(f'`{c}`' for c in effect['usefulness_lost']) or '—'}",
        "",
        "## Что ушло пользователю на сработавших инъекциях",
        "",
    ]
    for run in runs.values():
        for result in run.results:
            if not result.injected:
                continue
            preview = result.delivered_answer.replace("\n", " ")[:_ANSWER_PREVIEW]
            lines += [
                f"**`{result.case_id}` / `{run.preset}`** — сигналы: {', '.join(result.signals)}",
                "",
                f"> {preview}…",
                "",
            ]
    return "\n".join(lines)


async def _run(args: argparse.Namespace) -> int:
    load_dotenv(ROOT / ".env")
    cases = select_cases(
        load_cases(),
        ids=tuple(i for i in args.ids.split(",") if i),
        scenario=args.scenario,
        source=args.source,
        hiding=args.hiding,
    )
    if not cases:
        print("Под фильтр не попала ни одна ловушка.", file=sys.stderr)
        return 3
    if args.check_only:
        print(json.dumps(corpus_stats(cases), ensure_ascii=False, indent=2), file=sys.stderr)
        return 0

    key = os.getenv("ROUTERAI_API_KEY")
    if not key:
        print("Нет ROUTERAI_API_KEY в окружении/.env", file=sys.stderr)
        return 2
    provider = RouterAIProvider(key)
    presets = tuple(p for p in INDIRECT_PRESETS if p in args.presets.split(","))

    runs: dict[str, LayerRun] = {}
    for preset in presets:
        results = [await _run_one(provider, case, preset, args) for case in cases]
        runs[preset] = LayerRun(preset=preset, results=results)

    payload = {
        "label": args.label,
        "model": args.model,
        "temperature": args.temperature,
        "presets": list(presets),
        "summary": _summarize(runs, cases),
        "runs": {preset: run.to_dict() for preset, run in runs.items()},
    }
    out_dir = _DAY_DIR / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.out}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / f"{args.out}.md").write_text(_md_report(payload, runs, cases), encoding="utf-8")
    print(json.dumps(payload["summary"]["presets"], ensure_ascii=False, indent=2), file=sys.stderr)
    print(f"\nЗаписано: {out_dir / args.out}.json / .md", file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="День 12: indirect injection — слои защиты")
    parser.add_argument("--presets", default=",".join(INDIRECT_PRESETS))
    parser.add_argument("--ids", default="", help="через запятую: только эти ловушки")
    parser.add_argument("--scenario", default="", help="summarize | analyze | search")
    parser.add_argument("--source", default="", help="email | document | webpage")
    parser.add_argument("--hiding", default="", help="техника сокрытия payload")
    parser.add_argument("--model", default=INDIRECT_MODEL)
    parser.add_argument("--temperature", type=float, default=INDIRECT_TEMPERATURE)
    parser.add_argument("--out", default="run1", help="имя файлов в runs/ без расширения")
    parser.add_argument("--label", default="прогон")
    parser.add_argument("--check-only", action="store_true", help="только состав корпуса")
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
