"""SSE-рассылка событий планировщика в браузер + опциональный webhook (мессенджеры и т.п.)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from collections.abc import AsyncIterator
from typing import Any

import httpx

_loop: asyncio.AbstractEventLoop | None = None
_clients: list[asyncio.Queue] = []

_MAX_QUEUE = 50


def attach_loop(loop: asyncio.AbstractEventLoop) -> None:
    global _loop
    _loop = loop


def _broadcast_payload(payload: dict[str, Any]) -> None:
    dead: list[asyncio.Queue] = []
    for q in _clients:
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                dead.append(q)
        except Exception:
            dead.append(q)
    for q in dead:
        with contextlib.suppress(ValueError):
            _clients.remove(q)

    url = (os.environ.get("SCHEDULER_WEBHOOK_URL") or "").strip()
    if url:
        try:
            httpx.post(url, json=payload, timeout=8.0)
        except httpx.HTTPError:
            pass


def emit_scheduler_tick_sync(
    task_id: str,
    task_type: str,
    preview: str,
    content_json: str,
) -> None:
    if _loop is None or not _loop.is_running():
        return
    payload = {
        "type": "scheduler_tick",
        "task_id": task_id,
        "task_type": task_type,
        "preview": preview,
        "raw": content_json[:2000],
    }
    _loop.call_soon_threadsafe(_broadcast_payload, payload)


async def sse_scheduler_subscribe() -> AsyncIterator[str]:
    q: asyncio.Queue = asyncio.Queue(maxsize=_MAX_QUEUE)
    _clients.append(q)
    try:
        while True:
            item = await q.get()
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
    finally:
        with contextlib.suppress(ValueError):
            _clients.remove(q)
