"""Клиент запуска файнтюна в OpenAI: upload file → create job → poll status.

ВНИМАНИЕ: по умолчанию режим --dry-run — ничего в сеть не отправляется, только
печатается план. Реальный запуск (расходует деньги на API) — с флагом --go.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import httpx

_API_BASE = "https://api.openai.com/v1"
_BASE_MODEL = "gpt-4o-mini-2024-07-18"  # актуальная база с поддержкой fine-tuning
_POLL_INTERVAL_SECONDS = 30
_TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled"})


def _load_env_file(path: Path) -> None:
    """Подтянуть KEY=VALUE из .env в окружение, не перетирая уже заданные."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")


def _headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def upload_file(client: httpx.Client, api_key: str, path: Path) -> str:
    """Загрузить JSONL с purpose=fine-tune; вернуть file_id."""
    with path.open("rb") as f:
        resp = client.post(
            f"{_API_BASE}/files",
            headers=_headers(api_key),
            data={"purpose": "fine-tune"},
            files={"file": (path.name, f, "application/jsonl")},
        )
    resp.raise_for_status()
    file_id = resp.json()["id"]
    print(f"Файл загружен: {file_id} ({path.name})", file=sys.stderr)
    return file_id


def create_job(
    client: httpx.Client, api_key: str, training_file: str,
    validation_file: str | None,
) -> str:
    """Создать fine-tuning job; вернуть job_id."""
    body: dict[str, object] = {"training_file": training_file, "model": _BASE_MODEL}
    if validation_file:
        body["validation_file"] = validation_file
    resp = client.post(
        f"{_API_BASE}/fine_tuning/jobs",
        headers={**_headers(api_key), "Content-Type": "application/json"},
        json=body,
    )
    resp.raise_for_status()
    job_id = resp.json()["id"]
    print(f"Job создан: {job_id} (base {_BASE_MODEL})", file=sys.stderr)
    return job_id


def poll_job(client: httpx.Client, api_key: str, job_id: str) -> dict:
    """Опрашивать статус job до терминального; вернуть финальный объект job."""
    while True:
        resp = client.get(
            f"{_API_BASE}/fine_tuning/jobs/{job_id}", headers=_headers(api_key)
        )
        resp.raise_for_status()
        job = resp.json()
        status = job.get("status", "unknown")
        print(f"[{time.strftime('%H:%M:%S')}] status={status}", file=sys.stderr)
        if status in _TERMINAL_STATUSES:
            return job
        time.sleep(_POLL_INTERVAL_SECONDS)


def _dry_run(train: Path, ev: Path) -> int:
    print("DRY-RUN — реальные запросы не отправлены. Что произойдёт при --go:", file=sys.stderr)
    print(f"  1) upload {train}  (purpose=fine-tune)", file=sys.stderr)
    print(f"  2) upload {ev}     (purpose=fine-tune, как validation_file)", file=sys.stderr)
    print(f"  3) create fine_tuning job на базе {_BASE_MODEL}", file=sys.stderr)
    print(f"  4) poll статуса каждые {_POLL_INTERVAL_SECONDS} c до succeeded/failed/cancelled", file=sys.stderr)
    print("\nЗапуск: OPENAI_API_KEY=... python finetune/run_finetune.py --go", file=sys.stderr)
    return 0


def main() -> int:
    here = Path(__file__).resolve().parent
    _load_env_file(here.parent / ".env")
    train_path = here / "data" / "train.jsonl"
    eval_path = here / "data" / "eval.jsonl"

    go = "--go" in sys.argv[1:]
    if not go:
        return _dry_run(train_path, eval_path)

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("Нет OPENAI_API_KEY — реальный запуск невозможен.", file=sys.stderr)
        return 1

    with httpx.Client(timeout=120) as client:
        training_file = upload_file(client, api_key, train_path)
        validation_file = upload_file(client, api_key, eval_path)
        job_id = create_job(client, api_key, training_file, validation_file)
        job = poll_job(client, api_key, job_id)

    status = job.get("status")
    print(json.dumps(job, ensure_ascii=False, indent=2))
    if status == "succeeded":
        print(f"Готовая модель: {job.get('fine_tuned_model')}", file=sys.stderr)
        return 0
    print(f"Job завершился со статусом {status}.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
