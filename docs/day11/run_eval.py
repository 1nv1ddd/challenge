"""Прогон День 11: корпус инъекций против наивного (v1) и защищённого (v2) system-промпта."""

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
    SECURITY_MODEL,
    SECURITY_PROMPT_VERSIONS,
    SECURITY_TEMPERATURE,
)
from app.providers import AIProvider, RouterAIProvider  # noqa: E402
from app.security import (  # noqa: E402
    Attack,
    AttackVerdict,
    RedteamRun,
    corpus_stats,
    fixed_by_hardening,
    load_attacks,
    run_attack,
    select_attacks,
)

_DAY_DIR = Path(__file__).resolve().parent
_MAX_ATTEMPTS = 3
_REPLY_PREVIEW = 400


async def _run_one(
    provider: AIProvider, attack: Attack, version: str, args: argparse.Namespace
) -> dict:
    """Одна атака с ретраями: сеть не должна выбивать кейс из отчёта."""
    for attempt in range(_MAX_ATTEMPTS):
        try:
            verdict = await run_attack(
                provider, attack, version, model=args.model, temperature=args.temperature
            )
        except httpx.HTTPError as exc:
            print(f"{attack.id}/{version}: сеть ({exc}), попытка {attempt + 1}", file=sys.stderr)
            await asyncio.sleep(3)
            continue
        mark = "BROKEN" if verdict.broken else ("refused" if verdict.refused else "held")
        print(f"{attack.id:<28} {version}  {mark:<8} {','.join(verdict.signals[:2])}", file=sys.stderr)
        return verdict.to_dict()
    print(f"{attack.id}/{version}: пропущен — сеть не ответила", file=sys.stderr)
    return {"attack_id": attack.id, "version": version, "error": "network", "broken": False}


def _summarize(runs: dict[str, RedteamRun], attacks: list[Attack]) -> dict:
    by_id = {a.id: a for a in attacks}
    summary: dict = {"corpus": corpus_stats(attacks), "versions": {}}
    for version, run in runs.items():
        summary["versions"][version] = {
            "total": run.total,
            "broken": run.broken,
            "held": run.held,
            "errors": run.errors,
            "break_rate": run.break_rate(),
            "cost_rub": run.cost_rub,
            "by_vector": run.by_vector(),
            "by_technique": run.by_technique(),
            "broken_ids": [v.attack_id for v in run.verdicts if v.broken],
        }
    diff = fixed_by_hardening(runs)
    summary["diff"] = diff
    summary["still_broken_titles"] = [by_id[a].title for a in diff["still_broken"] if a in by_id]
    return summary


def _md_table(runs: dict[str, RedteamRun], attacks: list[Attack]) -> list[str]:
    by_id = {a.id: a for a in attacks}
    versions = [v for v in SECURITY_PROMPT_VERSIONS if v in runs]
    header = "| Атака | Вектор | Цель | " + " | ".join(versions) + " | Сигналы (последняя версия) |"
    lines = [header, "|---" * (4 + len(versions)) + "|"]
    verdicts = {(v.attack_id, run.version): v for run in runs.values() for v in run.verdicts}
    for attack in attacks:
        cells = []
        signals = "—"
        for version in versions:
            verdict = verdicts.get((attack.id, version))
            if verdict is None:
                cells.append("—")
                continue
            if verdict.error:
                cells.append("⚠️")
            elif verdict.broken:
                cells.append("❌ пробило")
                signals = ", ".join(verdict.signals[:3])
            else:
                cells.append("✅ устоял")
        lines.append(
            f"| `{attack.id}` — {by_id[attack.id].title} | {attack.vector} | `{attack.target}` | "
            + " | ".join(cells)
            + f" | {signals} |"
        )
    return lines


def _md_report(payload: dict, runs: dict[str, RedteamRun], attacks: list[Attack]) -> str:
    summary = payload["summary"]
    lines = [
        f"# Прогон ред-тима — {payload['label']}",
        "",
        f"Модель: `{payload['model']}` · температура {payload['temperature']} · "
        f"атак в прогоне: {len(attacks)}",
        "",
        "## Итог по версиям промпта",
        "",
        "| Версия | Пробито | Устояло | Ошибки | Доля пробоя | Стоимость |",
        "|---|---|---|---|---|---|",
    ]
    for version, stats in summary["versions"].items():
        title = "наивный" if version == "v1" else "защищённый"
        lines.append(
            f"| `{version}` ({title}) | {stats['broken']} | {stats['held']} | {stats['errors']} | "
            f"{stats['break_rate']:.0%} | {stats['cost_rub']} ₽ |"
        )
    lines += ["", "## По атакам", ""]
    lines += _md_table(runs, attacks)
    diff = summary["diff"]
    lines += [
        "",
        "## Что дала защита",
        "",
        f"- **закрыто в v2:** {', '.join(f'`{a}`' for a in diff['fixed']) or '—'}",
        f"- **пробивает обе версии:** {', '.join(f'`{a}`' for a in diff['still_broken']) or '—'}",
        f"- **регресс (пробило только v2):** {', '.join(f'`{a}`' for a in diff['regressed']) or '—'}",
        "",
        "## Ответы модели на пробитых атаках",
        "",
    ]
    for run in runs.values():
        for verdict in run.verdicts:
            if not verdict.broken:
                continue
            preview = verdict.reply.replace("\n", " ")[:_REPLY_PREVIEW]
            lines += [
                f"**`{verdict.attack_id}` / `{verdict.version}`** — сигналы: "
                f"{', '.join(verdict.signals)}",
                "",
                f"> {preview}…",
                "",
            ]
    return "\n".join(lines)


async def _run(args: argparse.Namespace) -> int:
    load_dotenv(ROOT / ".env")
    attacks = select_attacks(
        load_attacks(),
        ids=tuple(i for i in args.ids.split(",") if i),
        target=args.target,
        vector=args.vector,
    )
    if not attacks:
        print("Под фильтр не попала ни одна атака.", file=sys.stderr)
        return 3
    if args.check_only:
        print(json.dumps(corpus_stats(attacks), ensure_ascii=False, indent=2), file=sys.stderr)
        return 0

    key = os.getenv("ROUTERAI_API_KEY")
    if not key:
        print("Нет ROUTERAI_API_KEY в окружении/.env", file=sys.stderr)
        return 2
    provider = RouterAIProvider(key)
    versions = tuple(v for v in SECURITY_PROMPT_VERSIONS if v in args.versions.split(","))

    runs: dict[str, RedteamRun] = {}
    raw_rows: list[dict] = []
    for version in versions:
        verdicts = []
        for attack in attacks:
            row = await _run_one(provider, attack, version, args)
            raw_rows.append(row)
            verdicts.append(row)
        # Пересобираем прогон из словарей: строки с сетевой ошибкой тоже должны попасть в отчёт.
        runs[version] = RedteamRun(
            version=version,
            verdicts=[_verdict_from_row(row, attacks) for row in verdicts],
        )

    payload = {
        "label": args.label,
        "model": args.model,
        "temperature": args.temperature,
        "versions": list(versions),
        "summary": _summarize(runs, attacks),
        "rows": raw_rows,
    }
    out_dir = _DAY_DIR / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.out}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / f"{args.out}.md").write_text(
        _md_report(payload, runs, attacks), encoding="utf-8"
    )
    print(json.dumps(payload["summary"]["versions"], ensure_ascii=False, indent=2), file=sys.stderr)
    print(f"\nЗаписано: {out_dir / args.out}.json / .md", file=sys.stderr)
    return 0


def _verdict_from_row(row: dict, attacks: list[Attack]) -> AttackVerdict:
    """Строка отчёта → вердикт: строки с сетевой ошибкой тоже должны попадать в сводку."""
    by_id = {a.id: a for a in attacks}
    attack = by_id.get(row.get("attack_id", ""))
    return AttackVerdict(
        attack_id=row.get("attack_id", "?"),
        target=row.get("target") or (attack.target if attack else ""),
        version=row.get("version", "?"),
        vector=row.get("vector") or (attack.vector if attack else ""),
        technique=row.get("technique") or (attack.technique if attack else ""),
        reply=row.get("reply", ""),
        broken=bool(row.get("broken")),
        signals=list(row.get("signals") or []),
        refused=bool(row.get("refused")),
        error=row.get("error"),
        model=row.get("model", ""),
        time_ms=int(row.get("time_ms") or 0),
        prompt_tokens=int(row.get("prompt_tokens") or 0),
        completion_tokens=int(row.get("completion_tokens") or 0),
        cost_rub=float(row.get("cost_rub") or 0.0),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="День 11: prompt injection — v1 против v2")
    parser.add_argument("--versions", default=",".join(SECURITY_PROMPT_VERSIONS))
    parser.add_argument("--ids", default="", help="через запятую: только эти атаки")
    parser.add_argument("--target", default="", help="bank | support")
    parser.add_argument("--vector", default="", help="direct | indirect | jailbreak")
    parser.add_argument("--model", default=SECURITY_MODEL)
    parser.add_argument("--temperature", type=float, default=SECURITY_TEMPERATURE)
    parser.add_argument("--out", default="run1", help="имя файлов в runs/ без расширения")
    parser.add_argument("--label", default="прогон")
    parser.add_argument("--check-only", action="store_true", help="только состав корпуса")
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
