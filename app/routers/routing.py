"""API маршрутизации запроса между дешёвой и сильной моделью (День 8 advance)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ..bootstrap import agent, providers
from ..payloads import RoutePayload

router = APIRouter(prefix="/api/route", tags=["routing"])


@router.post("")
async def run_route(request: Request):
    """Вопрос → ответ плюс маршрут: кто отвечал, почему эскалировали и во что это обошлось."""
    if not providers:
        raise HTTPException(
            status_code=503, detail="Нет настроенного провайдера (нужен ROUTERAI_API_KEY)."
        )
    body = await request.json()
    p = RoutePayload.from_body(body if isinstance(body, dict) else {})
    if not p.question:
        raise HTTPException(status_code=400, detail="Нужно непустое поле question.")
    try:
        result = await agent.route_question(
            p.provider_name,
            p.question,
            small_model=p.small_model,
            large_model=p.large_model,
            temperature=p.temperature,
        )
    except LookupError as exc:
        raise HTTPException(status_code=400, detail=str(exc).strip() or "LookupError") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc).strip() or "ValueError") from exc
    return result.to_dict()
