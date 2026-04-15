"""Основной REST/SSE API чата, памяти, веток, RAG."""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from ..bootstrap import STATIC_DIR, agent, providers
from ..payloads import ChatRequestPayload, RagComparePayload, RagModesComparePayload, sse_error_line
from ..rag.status_api import build_rag_status_response

router = APIRouter()


@router.get("/api/models")
async def list_models():
    return agent.list_models()


@router.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    p = ChatRequestPayload.from_body(body if isinstance(body, dict) else {})

    async def event_stream():
        try:
            async for result in agent.stream_reply(
                provider_name=p.provider_name,
                model=p.model,
                conversation_id=p.conversation_id,
                raw_messages=p.raw_messages,
                temperature=p.temperature,
                context_strategy=p.context_strategy,
                branch_id=p.branch_id,
                profile_id=p.profile_id,
                resume=p.resume,
                rag=p.rag,
                task_workflow=p.task_workflow,
            ):
                if result.text is not None:
                    escaped = json.dumps(result.text, ensure_ascii=False)
                    yield f"data: {escaped}\n\n"
                if result.meta is not None:
                    yield f"data: [META]{json.dumps(result.meta)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:
            yield sse_error_line(exc)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/api/rag/compare")
async def rag_compare(request: Request):
    """День 22: один вопрос — два ответа (без контекста из индекса и с RAG)."""
    if not providers:
        raise HTTPException(status_code=503, detail="Нет настроенного провайдера (нужен ROUTERAI_API_KEY).")
    body = await request.json()
    rc = RagComparePayload.from_body(body if isinstance(body, dict) else {})
    if not rc.provider_name or not rc.model or not rc.message:
        raise HTTPException(
            status_code=400,
            detail="Нужны поля provider, model и непустой message.",
        )
    try:
        out = await agent.compare_rag_answers(
            rc.provider_name,
            rc.model,
            rc.message,
            temperature=rc.temperature,
            rag_strategy=rc.rag_strategy,
            top_k=rc.top_k,
            index_path=rc.index_path,
        )
    except LookupError as exc:
        raise HTTPException(status_code=400, detail=str(exc).strip() or "LookupError") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc).strip() or "ValueError") from exc
    return out


@router.post("/api/rag/compare_modes")
async def rag_compare_modes(request: Request):
    """День 23: сравнение базового RAG и режима с фильтром/реранком/query rewrite."""
    if not providers:
        raise HTTPException(status_code=503, detail="Нет настроенного провайдера (нужен ROUTERAI_API_KEY).")
    body = await request.json()
    pm = RagModesComparePayload.from_body(body if isinstance(body, dict) else {})
    if not pm.provider_name or not pm.model or not pm.message:
        raise HTTPException(
            status_code=400,
            detail="Нужны поля provider, model и непустой message.",
        )
    try:
        return await agent.compare_rag_modes(
            pm.provider_name,
            pm.model,
            pm.message,
            temperature=pm.temperature,
            rag_strategy=pm.rag_strategy,
            top_k=pm.top_k,
            index_path=pm.index_path,
            min_similarity=pm.min_similarity,
        )
    except LookupError as exc:
        raise HTTPException(status_code=400, detail=str(exc).strip() or "LookupError") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc).strip() or "ValueError") from exc


@router.get("/api/rag/status")
async def rag_status():
    return build_rag_status_response()


@router.get("/api/branches")
async def list_branches(conversation_id: str):
    return agent.list_branches(conversation_id)


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
        return {"error": "checkpoint_id is required"}
    return agent.create_branch(
        conversation_id=conversation_id,
        checkpoint_id=checkpoint_id,
        branch_name=branch_name,
    )


@router.get("/api/memory")
async def list_memory(conversation_id: str, branch_id: str = "main"):
    return agent.list_memory_layers(conversation_id, branch_id=branch_id)


@router.get("/api/profiles")
async def list_profiles():
    return agent.list_profiles()


@router.post("/api/profiles")
async def upsert_profile(request: Request):
    body = await request.json()
    profile_id: str = body.get("profile_id", "")
    name: str = body.get("name", "")
    style: str = body.get("style", "")
    format_pref: str = body.get("format", "")
    constraints: str = body.get("constraints", "")
    return agent.upsert_profile(
        profile_id=profile_id,
        name=name,
        style=style,
        format_pref=format_pref,
        constraints=constraints,
    )


@router.get("/api/task-state")
async def list_task_state(conversation_id: str):
    return agent.list_task_state(conversation_id)


@router.get("/api/invariants")
async def list_invariants(conversation_id: str):
    return agent.list_invariants(conversation_id)


@router.post("/api/invariants")
async def set_invariants(request: Request):
    body = await request.json()
    conversation_id: str = body.get("conversation_id", "default")
    replace: bool = bool(body.get("replace", True))
    raw = body.get("invariants")
    invariants = raw if isinstance(raw, dict) else {}
    return agent.set_invariants(
        conversation_id=conversation_id,
        invariants=invariants,
        replace=replace,
    )


@router.post("/api/task-state")
async def update_task_state(request: Request):
    body = await request.json()
    conversation_id: str = body.get("conversation_id", "default")
    phase: str | None = body.get("phase")
    current_step: str | None = body.get("current_step")
    expected_action: str | None = body.get("expected_action")
    action: str | None = body.get("action")
    return agent.update_task_state(
        conversation_id=conversation_id,
        phase=phase,
        current_step=current_step,
        expected_action=expected_action,
        action=action,
    )


@router.get("/", response_class=HTMLResponse)
async def index():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(
        content=html,
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )
