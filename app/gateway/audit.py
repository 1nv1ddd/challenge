"""Аудит шлюза: append-only jsonl со всеми запросами, находками guard'ов и стоимостью."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ..agent_constants import GATEWAY_AUDIT_PATH, GATEWAY_AUDIT_TEXT_CHARS
from .schema import GatewayResult

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def audit_path(path: str | Path | None = None) -> Path:
    p = Path(path) if path else Path(GATEWAY_AUDIT_PATH)
    return p if p.is_absolute() else _PROJECT_ROOT / p


def _trim(text: str) -> str:
    body = (text or "").strip()
    if len(body) <= GATEWAY_AUDIT_TEXT_CHARS:
        return body
    return f"{body[:GATEWAY_AUDIT_TEXT_CHARS]}… (обрезано)"


def build_record(result: GatewayResult) -> dict:
    """Запись аудита. В лог кладём только маскированный промпт: сырой секрет не хранится нигде —
    ни при блокировке, ни при маскировании, от него остаются вид, префикс и sha256."""
    return {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "request_id": result.request_id,
        "client_ip": result.client_ip,
        "model": result.model,
        "mode": result.mode,
        "status": result.status,
        "llm_called": result.llm_called,
        "input_action": result.input.action,
        "input_findings": [f.to_dict() for f in result.input.findings],
        "output_action": result.output.action,
        "output_enforced": result.output_enforced,
        "output_findings": [f.to_dict() for f in result.output.findings],
        "prompt_masked": _trim(result.input.prompt),
        "answer": _trim(result.answer),
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "tokens_estimated": result.tokens_estimated,
        "cost_rub": result.cost_rub,
        "time_ms": result.time_ms,
        "rate": result.rate.to_dict() if result.rate else None,
        "error": result.error,
    }


def log_request(result: GatewayResult, path: str | Path | None = None) -> dict:
    """Дописывает запись в аудит-лог и возвращает её (удобно тестам и API)."""
    record = build_record(result)
    file = audit_path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def read_records(limit: int = 50, path: str | Path | None = None) -> list[dict]:
    """Последние записи аудита, новые в конце. Битые строки пропускаем — лог append-only."""
    file = audit_path(path)
    if not file.is_file():
        return []
    records: list[dict] = []
    for raw in file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records[-limit:] if limit > 0 else records


def audit_stats(records: list[dict]) -> dict:
    """Сводка по логу: сколько чего заблокировано, какие секреты ловились и во что это обошлось."""
    secrets: dict[str, int] = {}
    statuses: dict[str, int] = {}
    for record in records:
        statuses[record.get("status", "?")] = statuses.get(record.get("status", "?"), 0) + 1
        for finding in record.get("input_findings") or []:
            kind = str(finding.get("kind", "?"))
            secrets[kind] = secrets.get(kind, 0) + 1
    llm_calls = sum(1 for r in records if r.get("llm_called"))
    return {
        "requests": len(records),
        "statuses": statuses,
        "llm_calls": llm_calls,
        "saved_calls": len(records) - llm_calls,
        "secrets_by_kind": dict(sorted(secrets.items(), key=lambda kv: -kv[1])),
        "prompt_tokens": sum(int(r.get("prompt_tokens") or 0) for r in records),
        "completion_tokens": sum(int(r.get("completion_tokens") or 0) for r in records),
        "cost_rub": round(sum(float(r.get("cost_rub") or 0.0) for r in records), 4),
    }
