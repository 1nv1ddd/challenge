from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .agent import SimpleChatAgent
from .mcp_panel import router as mcp_router
from .providers import AIProvider, RouterAIProvider
from .scheduler_notify import attach_loop
from .scheduler_routes import router as scheduler_router
from .scheduler_runner import scheduler_loop

load_dotenv()

_rag_log = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    attach_loop(asyncio.get_running_loop())
    sched_task = asyncio.create_task(scheduler_loop())

    auto_rag = os.getenv("RAG_AUTO_BUILD", "1").strip().lower() in ("1", "true", "yes")
    if auto_rag and (os.getenv("ROUTERAI_API_KEY") or "").strip():
        from .rag.build_index import build_rag_index, index_needs_build
        from .rag.pipeline import default_index_path

        idx = default_index_path()
        if index_needs_build(idx):
            _rag_log.warning("RAG: индекс отсутствует или пуст — автосборка (может занять 1–3 мин)…")
            try:
                await asyncio.to_thread(
                    lambda: build_rag_index(include_extra=True, write_report=True),
                )
                _rag_log.warning("RAG: индекс готов — %s", idx)
            except Exception as exc:  # noqa: BLE001
                _rag_log.exception("RAG: автосборка не удалась: %s", exc)

    yield
    sched_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await sched_task


app = FastAPI(title="AI Chat Hub", lifespan=lifespan)
app.include_router(mcp_router)
app.include_router(scheduler_router)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
MEMORY_PATH = Path(__file__).resolve().parent.parent / "data" / "agent_memory.json"

providers: dict[str, AIProvider] = {}

if key := os.getenv("ROUTERAI_API_KEY"):
    providers["routerai"] = RouterAIProvider(key)

agent = SimpleChatAgent(providers, memory_path=MEMORY_PATH)


@app.get("/api/models")
async def list_models():
    return agent.list_models()


@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()

    provider_name: str = body.get("provider", "")
    model: str = body.get("model", "")
    conversation_id: str = body.get("conversation_id", "default")
    raw_messages: list[dict] = body.get("messages", [])
    temperature: float = body.get("temperature", 0.7)
    context_strategy: str = body.get("context_strategy", "sliding")
    branch_id: str = body.get("branch_id", "main")
    profile_id: str | None = body.get("profile_id")
    resume: bool = bool(body.get("resume", False))
    rag = body.get("rag")
    raw_tw = body.get("task_workflow")
    task_workflow: bool | None = None if raw_tw is None else bool(raw_tw)

    async def event_stream():
        try:
            async for result in agent.stream_reply(
                provider_name=provider_name,
                model=model,
                conversation_id=conversation_id,
                raw_messages=raw_messages,
                temperature=temperature,
                context_strategy=context_strategy,
                branch_id=branch_id,
                profile_id=profile_id,
                resume=resume,
                rag=rag if isinstance(rag, dict) else None,
                task_workflow=task_workflow,
            ):
                if result.text is not None:
                    escaped = json.dumps(result.text, ensure_ascii=False)
                    yield f"data: {escaped}\n\n"
                if result.meta is not None:
                    yield f"data: [META]{json.dumps(result.meta)}\n\n"
            yield "data: [DONE]\n\n"
        except LookupError as exc:
            msg = (str(exc).strip() or type(exc).__name__)
            yield f"data: [ERROR] {msg}\n\n"
        except ValueError as exc:
            msg = (str(exc).strip() or type(exc).__name__)
            yield f"data: [ERROR] {msg}\n\n"
        except Exception as exc:
            msg = str(exc).replace("\n", " ").strip() or type(exc).__name__
            yield f"data: [ERROR] {msg}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/rag/compare")
async def rag_compare(request: Request):
    """День 22: один вопрос — два ответа (без контекста из индекса и с RAG)."""
    if not providers:
        raise HTTPException(status_code=503, detail="Нет настроенного провайдера (нужен ROUTERAI_API_KEY).")
    body = await request.json()
    provider_name = str(body.get("provider") or "").strip()
    model = str(body.get("model") or "").strip()
    message = str(body.get("message") or "").strip()
    if not provider_name or not model or not message:
        raise HTTPException(
            status_code=400,
            detail="Нужны поля provider, model и непустой message.",
        )
    temperature = float(body.get("temperature", 0.35))
    rag_strategy = str(body.get("rag_strategy") or "fixed").lower().strip()
    top_k = int(body.get("top_k") or 8)
    raw_idx = body.get("index_path")
    index_path = str(raw_idx).strip() if isinstance(raw_idx, str) and raw_idx.strip() else None
    try:
        out = await agent.compare_rag_answers(
            provider_name,
            model,
            message,
            temperature=temperature,
            rag_strategy=rag_strategy,
            top_k=top_k,
            index_path=index_path,
        )
    except LookupError as exc:
        raise HTTPException(status_code=400, detail=str(exc).strip() or "LookupError") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc).strip() or "ValueError") from exc
    return out


@app.get("/api/rag/status")
async def rag_status():
    from app.rag.build_index import index_needs_build
    from app.rag.pipeline import default_index_path
    from app.rag.store import stats

    p = default_index_path()
    if index_needs_build(p):
        return {
            "ok": True,
            "indexed": False,
            "path": str(p),
            "hint": "Индекс пуст или отсутствует. При RAG_AUTO_BUILD=1 и ROUTERAI_API_KEY он соберётся при старте; иначе: python scripts/build_rag_index.py",
        }
    return {"ok": True, "indexed": True, "path": str(p), "stats": stats(p)}


@app.get("/api/branches")
async def list_branches(conversation_id: str):
    return agent.list_branches(conversation_id)


@app.post("/api/checkpoints")
async def create_checkpoint(request: Request):
    body = await request.json()
    conversation_id: str = body.get("conversation_id", "default")
    branch_id: str = body.get("branch_id", "main")
    return agent.create_checkpoint(conversation_id=conversation_id, branch_id=branch_id)


@app.post("/api/branches")
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


@app.get("/api/memory")
async def list_memory(conversation_id: str, branch_id: str = "main"):
    return agent.list_memory_layers(conversation_id, branch_id=branch_id)


@app.get("/api/profiles")
async def list_profiles():
    return agent.list_profiles()


@app.post("/api/profiles")
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


@app.get("/api/task-state")
async def list_task_state(conversation_id: str):
    return agent.list_task_state(conversation_id)


@app.get("/api/invariants")
async def list_invariants(conversation_id: str):
    return agent.list_invariants(conversation_id)


@app.post("/api/invariants")
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


@app.post("/api/task-state")
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


@app.get("/", response_class=HTMLResponse)
async def index():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(
        content=html,
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
