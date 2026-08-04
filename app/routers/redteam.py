"""API ред-тима промптов: прогон корпуса инъекций по версиям system-промпта (День 11 advance)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ..agent_constants import (
    SECURITY_ATTACKS_PATH,
    SECURITY_PROMPT_VERSIONS,
    SECURITY_TARGETS,
    SECURITY_TECHNIQUES,
    SECURITY_VECTORS,
)
from ..bootstrap import agent, providers
from ..payloads import RedteamPayload
from ..security import corpus_stats, fixed_by_hardening, load_attacks

router = APIRouter(prefix="/api/redteam", tags=["redteam"])


@router.post("")
async def run_redteam(request: Request):
    """Корпус инъекций против выбранных версий промпта: что пробило, что закрыла защита."""
    if not providers:
        raise HTTPException(
            status_code=503, detail="Нет настроенного провайдера (нужен ROUTERAI_API_KEY)."
        )
    body = await request.json()
    p = RedteamPayload.from_body(body if isinstance(body, dict) else {})
    try:
        runs = await agent.redteam_prompts(
            p.provider_name,
            versions=p.versions,
            ids=p.ids,
            target=p.target,
            vector=p.vector,
            technique=p.technique,
            model=p.model,
            temperature=p.temperature,
        )
    except LookupError as exc:
        raise HTTPException(status_code=400, detail=str(exc).strip() or "LookupError") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc).strip() or "ValueError") from exc
    return {
        "runs": {version: run.to_dict() for version, run in runs.items()},
        "diff": fixed_by_hardening(runs),
    }


@router.get("/corpus")
async def redteam_corpus():
    """Состав корпуса: атаки с классификацией и разбором «почему работает / как защититься»."""
    try:
        attacks = load_attacks()
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "path": SECURITY_ATTACKS_PATH,
        "total": len(attacks),
        "stats": corpus_stats(attacks),
        "known_vectors": list(SECURITY_VECTORS),
        "known_techniques": list(SECURITY_TECHNIQUES),
        "known_targets": list(SECURITY_TARGETS),
        "known_versions": list(SECURITY_PROMPT_VERSIONS),
        "attacks": [a.to_dict() for a in attacks],
    }
