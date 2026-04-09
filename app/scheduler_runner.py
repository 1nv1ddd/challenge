"""Фоновый цикл: срабатывание отложенных/периодических MCP-задач."""

from __future__ import annotations

import asyncio
import contextlib

from .scheduler_store import process_due_jobs

_TICK_SEC = 5


async def scheduler_loop() -> None:
    while True:
        await asyncio.sleep(_TICK_SEC)
        with contextlib.suppress(Exception):
            await asyncio.to_thread(process_due_jobs)
