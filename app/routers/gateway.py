"""API шлюза: прокси-вызов модели с guard'ами, аудит-лог и офлайн-прогон корпуса (День 13)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ..agent_constants import (
    GATEWAY_ALLOWED_HOSTS,
    GATEWAY_CASES_PATH,
    GATEWAY_CREDENTIAL_KINDS,
    GATEWAY_DEFAULT_MODE,
    GATEWAY_MAX_PROMPT_CHARS,
    GATEWAY_MODEL,
    GATEWAY_MODES,
    GATEWAY_PII_KINDS,
    GATEWAY_RATE_LIMIT_PER_MIN,
    GATEWAY_RATE_WINDOW_SEC,
    GATEWAY_REDACTIONS,
)
from ..bootstrap import agent, providers
from ..gateway import audit_stats, load_cases, read_records
from ..payloads import GatewayPayload

router = APIRouter(prefix="/api/gateway", tags=["gateway"])
# Сколько записей аудита отдаём по умолчанию.
_AUDIT_LIMIT = 50


def _client_ip(request: Request) -> str:
    """Адрес клиента с учётом nginx впереди: за прокси реальный IP приходит в X-Forwarded-For.

    Заголовку доверяем только потому, что снаружи приложение открыто исключительно через свой
    nginx; при прямом доступе к порту его можно подделать — и тогда лимит обходится.
    """
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded.strip():
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@router.post("/chat")
async def gateway_chat(request: Request):
    """Прокси в модель: input guard → вызов → output guard → аудит."""
    if not providers:
        raise HTTPException(
            status_code=503, detail="Нет настроенного провайдера (нужен ROUTERAI_API_KEY)."
        )
    body = await request.json()
    try:
        p = GatewayPayload.from_body(body if isinstance(body, dict) else {})
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        result = await agent.gateway_request(
            p.provider_name,
            p.prompt,
            mode=p.mode,
            model=p.model,
            temperature=p.temperature,
            client_ip=_client_ip(request),
        )
    except LookupError as exc:
        raise HTTPException(status_code=400, detail=str(exc).strip() or "LookupError") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc).strip() or "ValueError") from exc
    payload = result.to_dict()
    # Статус запроса виден и по HTTP-коду: клиенту не нужно разбирать тело, чтобы понять отказ.
    if result.status == "rate_limited":
        raise HTTPException(status_code=429, detail=payload)
    if result.status in ("blocked_input", "too_long"):
        raise HTTPException(status_code=400, detail=payload)
    return payload


@router.get("/audit")
async def gateway_audit(limit: int = _AUDIT_LIMIT):
    """Последние записи аудита и сводка по ним: блокировки, перехваченные секреты, стоимость."""
    records = read_records(limit)
    return {"stats": audit_stats(records), "records": records}


@router.get("/cases")
async def gateway_cases(ids: str = ""):
    """Корпус тест-кейсов и его прогон по детекторам — офлайн, без вызова модели."""
    selected = tuple(i.strip() for i in ids.split(",") if i.strip())
    try:
        cases = load_cases()
        run = agent.gateway_selftest(selected)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "path": GATEWAY_CASES_PATH,
        "total": len(cases),
        "run": run.to_dict(),
        "cases": [case.to_dict() for case in cases],
    }


@router.get("/config")
async def gateway_config():
    """Настройки шлюза: режимы, лимиты, виды секретов и плейсхолдеры маскирования."""
    return {
        "model": GATEWAY_MODEL,
        "modes": list(GATEWAY_MODES),
        "default_mode": GATEWAY_DEFAULT_MODE,
        "credential_kinds": list(GATEWAY_CREDENTIAL_KINDS),
        "pii_kinds": list(GATEWAY_PII_KINDS),
        "redactions": GATEWAY_REDACTIONS,
        "rate_limit": {
            "requests": GATEWAY_RATE_LIMIT_PER_MIN,
            "window_sec": GATEWAY_RATE_WINDOW_SEC,
        },
        "max_prompt_chars": GATEWAY_MAX_PROMPT_CHARS,
        "allowed_hosts": list(GATEWAY_ALLOWED_HOSTS),
    }
