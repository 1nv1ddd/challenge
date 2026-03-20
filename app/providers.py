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


GROQ_PRICING = {
    "llama-3.1-8b-instant":    {"input": 0.05, "output": 0.08},
    "qwen/qwen3-32b":         {"input": 0.29, "output": 0.39},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
}


class GroqProvider(AIProvider):
    name = "groq"
    models = [
        {"id": "llama-3.1-8b-instant",    "label": "Llama 3.1 8B — Fast"},
        {"id": "qwen/qwen3-32b",          "label": "Qwen 3 32B — Balanced"},
        {"id": "llama-3.3-70b-versatile",  "label": "Llama 3.3 70B — Smart"},
    ]

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def stream_chat(
        self, messages: list[Message], model: str, temperature: float = 0.7
    ) -> AsyncIterator[StreamResult]:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
            "temperature": temperature,
            "stream_options": {"include_usage": True},
        }

        usage = None
        t_start = time.monotonic()

        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST", url, json=body, headers=headers
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        if u := chunk.get("usage"):
                            usage = u
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        if text := delta.get("content"):
                            yield StreamResult(text=text)
                    except (json.JSONDecodeError, IndexError, KeyError):
                        continue

        elapsed_ms = round((time.monotonic() - t_start) * 1000)
        pricing = GROQ_PRICING.get(model, {"input": 0, "output": 0})
        prompt_tokens = (usage or {}).get("prompt_tokens", 0)
        completion_tokens = (usage or {}).get("completion_tokens", 0)
        cost = (prompt_tokens * pricing["input"] + completion_tokens * pricing["output"]) / 1_000_000

        yield StreamResult(meta={
            "time_ms": elapsed_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost_usd": round(cost, 6),
        })
