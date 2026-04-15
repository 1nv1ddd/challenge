"""Жизненный цикл FastAPI: планировщик и опциональная автосборка RAG."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .scheduler_notify import attach_loop
from .scheduler_runner import scheduler_loop

_rag_log = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    attach_loop(asyncio.get_running_loop())
    sched_task = asyncio.create_task(scheduler_loop())

    auto_rag = os.getenv("RAG_AUTO_BUILD", "1").strip().lower() in ("1", "true", "yes")
    if auto_rag and (os.getenv("ROUTERAI_API_KEY") or "").strip():
        from .rag.build_index import build_rag_index
        from .rag.index_meta import index_needs_build
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
