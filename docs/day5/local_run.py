"""Прогон 4 задач пула Day 5 на локальной модели (Vikhr-12B через Ollama) для сравнения с облаком."""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

_MODEL = "huggingface.co/bartowski/Vikhr-Nemo-12B-Instruct-R-21-09-24-GGUF:Q4_K_M"
_URL = "http://localhost:11434/api/generate"
_OUT = Path(__file__).parent / "runs" / "local"
_RULES = (Path(__file__).parent / "rules-local.snapshot.md").read_text(encoding="utf-8")

# Контекст даём в промпте: у локальной модели нет инструментов, чтобы исследовать репозиторий.
_TASKS = [
    {
        "id": "T06",
        "type": "feature",
        "task": "Добавь эндпоинт GET /api/scheduler/jobs, отдающий список задач планировщика.",
        "context": '''# app/scheduler_routes.py (текущий)
"""HTTP: SSE-стрим тиков планировщика + ping для проверки деплоя."""
from __future__ import annotations
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from .scheduler_notify import sse_scheduler_subscribe
router = APIRouter(prefix="/api/scheduler", tags=["scheduler"])

@router.get("/ping")
async def scheduler_ping() -> dict:
    return {"ok": True, "sse_path": "/api/scheduler/stream"}

# В app/scheduler_store.py УЖЕ ЕСТЬ готовая функция:
#     def list_jobs() -> list[dict]: ...   # возвращает все задачи из SQLite
''',
    },
    {
        "id": "T04",
        "type": "refactor",
        "task": ("Отрефактори обработчики create_checkpoint и create_branch: убери ручной разбор body.get, "
                 "заведи payload-классы CheckpointPayload и BranchPayload с классметодом from_body(body) и "
                 "используй их (по образцу ChatRequestPayload). Поведение 400 без checkpoint_id сохрани."),
        "context": '''# app/payloads.py — образец стиля payload-класса:
@dataclass(frozen=True)
class ChatRequestPayload:
    provider_name: str
    model: str
    @classmethod
    def from_body(cls, body: dict[str, Any]) -> "ChatRequestPayload":
        return cls(provider_name=str(body.get("provider", "")), model=str(body.get("model", "")))

# app/routers/hub.py — текущие обработчики (ручной разбор):
@router.post("/api/checkpoints")
async def create_checkpoint(request: Request):
    body = await request.json()
    conversation_id: str = body.get("conversation_id", "default")
    branch_id: str = body.get("branch_id", "main")
    return agent.create_checkpoint(conversation_id=conversation_id, branch_id=branch_id)

@router.post("/api/branches")
async def create_branch(request: Request):
    body = await request.json()
    conversation_id: str = body.get("conversation_id", "default")
    checkpoint_id: str = body.get("checkpoint_id", "")
    branch_name: str | None = body.get("branch_name")
    if not checkpoint_id:
        raise HTTPException(status_code=400, detail="Нужно поле checkpoint_id.")
    return agent.create_branch(conversation_id=conversation_id, checkpoint_id=checkpoint_id, branch_name=branch_name)
''',
    },
    {
        "id": "T14",
        "type": "refactor",
        "task": ("Паттерн `body if isinstance(body, dict) else {}` повторяется в нескольких обработчиках. "
                 "Вынеси его в один переиспользуемый хелпер as_dict(body) в app/payloads.py и покажи, как обработчик его использует."),
        "context": '''# app/payloads.py (шапка):
"""Разбор JSON-тел HTTP-запросов для API (тонкий слой без FastAPI)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

# app/routers/hub.py — повторяющийся guard в разных обработчиках:
p = ChatRequestPayload.from_body(body if isinstance(body, dict) else {})
rc = RagComparePayload.from_body(body if isinstance(body, dict) else {})
pm = RagModesComparePayload.from_body(body if isinstance(body, dict) else {})
''',
    },
    {
        "id": "T02",
        "type": "test",
        "task": ("Напиши тест на HTTP-слой: POST /api/branches с телом без checkpoint_id должен вернуть 400 "
                 "с непустым русским detail. pytest в проекте НЕ установлен — используй unittest + fastapi.testclient."),
        "context": '''# app/routers/hub.py:
@router.post("/api/branches")
async def create_branch(request: Request):
    body = await request.json()
    checkpoint_id: str = body.get("checkpoint_id", "")
    if not checkpoint_id:
        raise HTTPException(status_code=400, detail="Нужно поле checkpoint_id.")
    ...

# app-объект: from app.main import app
# стиль тестов проекта: unittest.TestCase; from fastapi.testclient import TestClient
''',
    },
]


def _generate(prompt: str) -> tuple[str, float, int]:
    body = json.dumps({
        "model": _MODEL,
        "system": _RULES,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "num_ctx": 8192, "num_predict": 1024},
    }).encode("utf-8")
    req = urllib.request.Request(_URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    dt = time.time() - t0
    return data.get("response", ""), dt, data.get("eval_count", 0)


def main() -> int:
    _OUT.mkdir(parents=True, exist_ok=True)
    log = []
    for t in _TASKS:
        prompt = (
            f"ЗАДАЧА ({t['type']}): {t['task']}\n\n"
            f"КОНТЕКСТ (существующий код проекта):\n{t['context']}\n"
            "Верни ОДИН код-блок с готовым файлом или патчем, без рассуждений вне блока."
        )
        out, dt, toks = _generate(prompt)
        tps = round(toks / dt, 1) if dt else 0.0
        (_OUT / f"{t['id']}.out.md").write_text(
            f"# {t['id']} ({t['type']}) — локальный вывод Vikhr-12B\n\n"
            f"_latency: {dt:.1f}s · {toks} tok · {tps} tok/s · temp=0.2 · num_ctx=8192_\n\n"
            f"**Задача:** {t['task']}\n\n---\n\n{out}\n",
            encoding="utf-8",
        )
        log.append({"id": t["id"], "type": t["type"], "seconds": round(dt, 1), "tokens": toks, "tok_s": tps})
        print(f"{t['id']}: {dt:.1f}s, {toks} tok, {tps} tok/s", flush=True)
    (_OUT / "local.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in log) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
