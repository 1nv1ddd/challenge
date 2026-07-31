"""API разбора письма-заявки: один большой запрос против цепочки этапов (День 9 advance)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ..bootstrap import agent, providers
from ..payloads import IntakePayload

router = APIRouter(prefix="/api/intake", tags=["intake"])


@router.post("")
async def run_intake_request(request: Request):
    """Письмо → нормализованные поля, решение по политике, ответ клиенту и цена выбранного режима."""
    if not providers:
        raise HTTPException(
            status_code=503, detail="Нет настроенного провайдера (нужен ROUTERAI_API_KEY)."
        )
    body = await request.json()
    p = IntakePayload.from_body(body if isinstance(body, dict) else {})
    if not p.letter:
        raise HTTPException(status_code=400, detail="Нужно непустое поле letter.")
    try:
        result = await agent.parse_intake(
            p.provider_name,
            p.letter,
            mode=p.mode,
            today=p.today,
            mono_model=p.mono_model,
            stage_models=p.stage_models,
            temperature=p.temperature,
        )
    except LookupError as exc:
        raise HTTPException(status_code=400, detail=str(exc).strip() or "LookupError") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc).strip() or "ValueError") from exc
    return result.to_dict()
