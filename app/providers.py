from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx


@dataclass
class Message:
    role: str  # "user" | "assistant" | "system"
    content: str


@dataclass
class StreamResult:
    text: str | None = None
    meta: dict | None = None


class AIProvider(ABC):
    name: str
    models: list[dict]

    @abstractmethod
    async def stream_chat(
        self, messages: list[Message], model: str, temperature: float = 0.7
    ) -> AsyncIterator[StreamResult]:
        yield StreamResult()


class RouterAIProvider(AIProvider):
    name = "routerai"
    models = [
        {"id": "openai/gpt-4o-mini", "label": "GPT-4o Mini — RouterAI"},
    ]

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def stream_chat(
        self, messages: list[Message], model: str, temperature: float = 0.7
    ) -> AsyncIterator[StreamResult]:
        url = "https://routerai.ru/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
        }

        t_start = time.monotonic()

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(url, json=body, headers=headers)
            resp.raise_for_status()
            payload = resp.json()

        elapsed_ms = round((time.monotonic() - t_start) * 1000)
        usage = payload.get("usage", {})
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens))
        text = (
            payload.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )

        if text:
            yield StreamResult(text=text)

        yield StreamResult(meta={
            "time_ms": elapsed_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        })
