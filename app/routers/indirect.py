"""API непрямых инъекций: прогон ловушек по слоям защиты и состав корпуса (День 12 advance)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ..agent_constants import (
    INDIRECT_CASES_PATH,
    INDIRECT_HIDING,
    INDIRECT_LAYERS,
    INDIRECT_PRESETS,
    INDIRECT_SCENARIOS,
    INDIRECT_SOURCES,
)
from ..bootstrap import agent, providers
from ..indirect import build_document, corpus_stats, layer_effect, load_cases, visible_part
from ..payloads import IndirectPayload

router = APIRouter(prefix="/api/indirect", tags=["indirect"])


@router.post("")
async def run_indirect(request: Request):
    """Ловушки во внешнем контенте против выбранных наборов слоёв защиты."""
    if not providers:
        raise HTTPException(
            status_code=503, detail="Нет настроенного провайдера (нужен ROUTERAI_API_KEY)."
        )
    body = await request.json()
    p = IndirectPayload.from_body(body if isinstance(body, dict) else {})
    try:
        runs = await agent.run_indirect_cases(
            p.provider_name,
            presets=p.presets,
            ids=p.ids,
            scenario=p.scenario,
            source=p.source,
            hiding=p.hiding,
            model=p.model,
            temperature=p.temperature,
        )
    except LookupError as exc:
        raise HTTPException(status_code=400, detail=str(exc).strip() or "LookupError") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc).strip() or "ValueError") from exc
    return {
        "runs": {preset: run.to_dict() for preset, run in runs.items()},
        "effect": layer_effect(runs),
    }


@router.get("/corpus")
async def indirect_corpus():
    """Состав корпуса ловушек: сценарий, носитель, техника сокрытия и разбор каждой."""
    try:
        cases = load_cases()
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "path": INDIRECT_CASES_PATH,
        "total": len(cases),
        "stats": corpus_stats(cases),
        "known_scenarios": list(INDIRECT_SCENARIOS),
        "known_sources": list(INDIRECT_SOURCES),
        "known_hiding": list(INDIRECT_HIDING),
        "known_layers": list(INDIRECT_LAYERS),
        "known_presets": list(INDIRECT_PRESETS),
        "cases": [
            {
                **case.to_dict(),
                # Видно, что человек payload не увидит: длина документа против видимой части.
                "document_chars": len(build_document(case)),
                "visible_chars": len(visible_part(case)),
            }
            for case in cases
        ],
    }
