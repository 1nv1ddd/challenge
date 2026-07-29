"""API триажа обращений поддержки с явной оценкой уверенности (День 7 advance)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ..bootstrap import agent, providers
from ..payloads import TriagePayload

router = APIRouter(prefix="/api/triage", tags=["triage"])


@router.post("")
async def run_triage(request: Request):
    """Обращение → решение + статус приёмки OK/UNSURE/FAIL с метриками вызовов."""
    if not providers:
        raise HTTPException(
            status_code=503, detail="Нет настроенного провайдера (нужен ROUTERAI_API_KEY)."
        )
    body = await request.json()
    p = TriagePayload.from_body(body if isinstance(body, dict) else {})
    if not p.provider_name or not p.model or not p.text:
        raise HTTPException(
            status_code=400, detail="Нужны поля provider, model и непустой text."
        )
    try:
        verdict = await agent.triage_message(
            p.provider_name,
            p.model,
            p.text,
            samples=p.samples,
            temperature=p.temperature,
        )
    except LookupError as exc:
        raise HTTPException(status_code=400, detail=str(exc).strip() or "LookupError") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc).strip() or "ValueError") from exc
    return verdict.to_dict()
