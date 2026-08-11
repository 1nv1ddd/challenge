"""API двухуровневой классификации интента: micro-model с fallback на большую LLM (День 10 advance)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ..agent_constants import MICRO_BANK_PATH, MICRO_LABELS, MICRO_STRATEGIES
from ..bootstrap import agent, providers
from ..micro import bank_labels, load_bank
from ..payloads import IntentPayload

router = APIRouter(prefix="/api/intent", tags=["intent"])


@router.post("")
async def classify_intent_request(request: Request):
    """Обращение → метка интента плюс разбор: решила micro-model или понадобилась большая."""
    if not providers:
        raise HTTPException(
            status_code=503, detail="Нет настроенного провайдера (нужен ROUTERAI_API_KEY)."
        )
    body = await request.json()
    p = IntentPayload.from_body(body if isinstance(body, dict) else {})
    if not p.text:
        raise HTTPException(status_code=400, detail="Нужно непустое поле text.")
    try:
        result = await agent.classify_intent(
            p.provider_name,
            p.text,
            strategy=p.strategy,
            llm_model=p.llm_model,
            temperature=p.temperature,
        )
    except LookupError as exc:
        raise HTTPException(status_code=400, detail=str(exc).strip() or "LookupError") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc).strip() or "ValueError") from exc
    return result.to_dict()


@router.get("/bank")
async def intent_bank_status():
    """Из чего состоит уровень 1: сколько примеров каждой метки лежит в банке."""
    try:
        items = load_bank()
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "path": MICRO_BANK_PATH,
        "total": len(items),
        "labels": bank_labels(items),
        "known_labels": list(MICRO_LABELS),
        "strategies": list(MICRO_STRATEGIES),
    }
