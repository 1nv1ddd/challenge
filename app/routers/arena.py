"""API Арены Дня 15: Оракул за проходом шлюза и форма сдачи кода (red-team challenge)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ..agent_constants import (
    CTF_AUDIT_PATH,
    CTF_MISS_MSG,
    CTF_MODEL,
    CTF_RATE_LIMIT_PER_MIN,
    CTF_RATE_WINDOW_SEC,
    CTF_WIN_MSG,
)
from ..bootstrap import agent, providers
from ..ctf import verify_code
from ..gateway.audit import audit_stats, read_records
from ..payloads import ArenaChatPayload, ArenaSubmitPayload

router = APIRouter(prefix="/api/arena", tags=["arena"])
# Сколько записей лога атак отдаём защитнику по умолчанию.
_AUDIT_LIMIT = 100


def _client_ip(request: Request) -> str:
    """IP клиента с учётом nginx впереди (реальный адрес — в X-Forwarded-For)."""
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded.strip():
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@router.post("/chat")
async def arena_chat(request: Request):
    """Реплика Оракула на попытку атаки. Код — в его промпте, утечку режет output guard."""
    if not providers:
        raise HTTPException(
            status_code=503, detail="Нет настроенного провайдера (нужен ROUTERAI_API_KEY)."
        )
    body = await request.json()
    p = ArenaChatPayload.from_body(body if isinstance(body, dict) else {})
    if not p.prompt.strip():
        raise HTTPException(status_code=400, detail="Пустой промпт: Оракулу нечего ответить.")
    try:
        result = await agent.arena_request(
            p.provider_name, p.prompt, model=p.model, client_ip=_client_ip(request)
        )
    except LookupError as exc:
        raise HTTPException(status_code=400, detail=str(exc).strip() or "LookupError") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc).strip() or "ValueError") from exc

    # Наружу отдаём только то, что нужно атакующему: реплику и факт блокировки. Внутренние поля
    # (канарейка в находках, стоимость, модель) не раскрываем — это подсказки об устройстве защиты.
    payload = {
        "request_id": result.request_id,
        "reply": result.answer,
        "blocked": result.status in ("blocked_output", "blocked_input"),
        "status": result.status,
    }
    if result.status == "rate_limited":
        raise HTTPException(status_code=429, detail=payload)
    return payload


@router.post("/submit")
async def arena_submit(request: Request):
    """Проверка сданного кода. Верный код = Оракул взломан."""
    body = await request.json()
    p = ArenaSubmitPayload.from_body(body if isinstance(body, dict) else {})
    if not p.code:
        raise HTTPException(status_code=400, detail="Пустой код: нечего проверять.")
    ok = verify_code(p.code)
    return {"correct": ok, "message": CTF_WIN_MSG if ok else CTF_MISS_MSG}


@router.get("/audit")
async def arena_audit(limit: int = _AUDIT_LIMIT):
    """Лог атак на Арену для защитника: попытки, находки guard'ов, блокировки."""
    records = read_records(limit, path=CTF_AUDIT_PATH)
    return {"stats": audit_stats(records), "records": records}


@router.get("/config")
async def arena_config():
    """Брифинг Арены: какие эндпоинты дёргать, лимиты, модель Оракула."""
    return {
        "endpoints": {
            "chat": "POST /api/arena/chat  {\"prompt\": \"...\"}",
            "submit": "POST /api/arena/submit  {\"code\": \"...\"}",
            "audit": "GET /api/arena/audit  (лог атак для защитника)",
        },
        "goal": "Заставить Оракула выдать секретный код доступа и сдать его через /api/arena/submit.",
        "model": CTF_MODEL,
        "rate_limit": {
            "requests": CTF_RATE_LIMIT_PER_MIN,
            "window_sec": CTF_RATE_WINDOW_SEC,
        },
    }
