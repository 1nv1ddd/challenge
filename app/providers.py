from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx


@dataclass
class Message:
    role: str  # "user" | "assistant" | "system"
    content: str


class AIProvider(ABC):
    name: str
    models: list[str]

    @abstractmethod
    async def stream_chat(
        self, messages: list[Message], model: str
    ) -> AsyncIterator[str]:
        yield ""


# ---------------------------------------------------------------------------
# Google Gemini  (free tier: 15 RPM Pro, 1500 RPD Flash)
# ---------------------------------------------------------------------------
class GeminiProvider(AIProvider):
    name = "gemini"
    models = [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    ]

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def stream_chat(
        self, messages: list[Message], model: str
    ) -> AsyncIterator[str]:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:streamGenerateContent?alt=sse&key={self.api_key}"
        )

        contents = []
        system_instruction = None
        for m in messages:
            if m.role == "system":
                system_instruction = {"parts": [{"text": m.content}]}
                continue
            contents.append({
                "role": "model" if m.role == "assistant" else "user",
                "parts": [{"text": m.content}],
            })

        body: dict = {"contents": contents}
        if system_instruction:
            body["systemInstruction"] = system_instruction

        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("POST", url, json=body) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        parts = (
                            chunk.get("candidates", [{}])[0]
                            .get("content", {})
                            .get("parts", [])
                        )
                        for part in parts:
                            if text := part.get("text"):
                                yield text
                    except (json.JSONDecodeError, IndexError, KeyError):
                        continue


# ---------------------------------------------------------------------------
# Groq  (free tier: 30 RPM, 14 400 RPD for most models)
# ---------------------------------------------------------------------------
class GroqProvider(AIProvider):
    name = "groq"
    models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ]

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def stream_chat(
        self, messages: list[Message], model: str
    ) -> AsyncIterator[str]:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
        }

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
                        delta = chunk["choices"][0].get("delta", {})
                        if text := delta.get("content"):
                            yield text
                    except (json.JSONDecodeError, IndexError, KeyError):
                        continue


# ---------------------------------------------------------------------------
# Anthropic Claude
# ---------------------------------------------------------------------------
class ClaudeProvider(AIProvider):
    name = "claude"
    models = [
        "claude-sonnet-4-20250514",
        "claude-3-5-haiku-20241022",
    ]

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def stream_chat(
        self, messages: list[Message], model: str
    ) -> AsyncIterator[str]:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        system_text = None
        api_messages = []
        for m in messages:
            if m.role == "system":
                system_text = m.content
                continue
            api_messages.append({"role": m.role, "content": m.content})

        body: dict = {
            "model": model,
            "max_tokens": 4096,
            "messages": api_messages,
            "stream": True,
        }
        if system_text:
            body["system"] = system_text

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
                        if chunk.get("type") == "content_block_delta":
                            text = chunk.get("delta", {}).get("text", "")
                            if text:
                                yield text
                    except (json.JSONDecodeError, KeyError):
                        continue
