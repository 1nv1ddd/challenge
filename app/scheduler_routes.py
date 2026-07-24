"""HTTP: SSE-стрим тиков планировщика + ping для проверки деплоя."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from .scheduler_notify import sse_scheduler_subscribe
from .scheduler_store import list_jobs

router = APIRouter(prefix="/api/scheduler", tags=["scheduler"])


@router.get("/ping")
async def scheduler_ping() -> dict:
    return {"ok": True, "sse_path": "/api/scheduler/stream"}


@router.get("/jobs")
async def scheduler_jobs() -> dict:
    """Список задач планировщика из стора (без прямого доступа к SQLite)."""
    jobs = list_jobs()
    return {"ok": True, "jobs": jobs, "count": len(jobs)}


@router.get("/stream")
async def scheduler_event_stream():
    async def gen():
        async for chunk in sse_scheduler_subscribe():
            yield chunk

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
