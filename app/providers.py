from __future__ import annotations

import json
import os
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


def _normalize_stream_content(content: object) -> str:
    """Текст из message/delta content: строка или список частей (OpenAI-совместимо)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                t = item.get("text")
                if isinstance(t, str):
                    parts.append(t)
                else:
                    nested = item.get("content")
                    if isinstance(nested, str):
                        parts.append(nested)
        return "".join(parts)
    return str(content)


def _stream_text_from_chunk(chunk: dict) -> str:
    choices = chunk.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return ""
    ch0 = choices[0]
    delta = ch0.get("delta") if isinstance(ch0.get("delta"), dict) else {}
    msg = ch0.get("message") if isinstance(ch0.get("message"), dict) else {}
    t = _normalize_stream_content(delta.get("content"))
    if t:
        return t
    return _normalize_stream_content(msg.get("content"))


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
        body_base = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
        }

        t_start = time.monotonic()
        usage: dict = {}
        any_text = False
        # Стрим RouterAI нередко даёт HTTP 200 и тело вроде «HTTP/1.0 500...» без строк data:
        # По умолчанию — один запрос stream=false. SSE: ROUTERAI_USE_STREAM=1
        use_sse = os.getenv("ROUTERAI_USE_STREAM", "").lower() in ("1", "true", "yes")

        async with httpx.AsyncClient(timeout=120) as client:
            if use_sse:
                async with client.stream(
                    "POST",
                    url,
                    json={**body_base, "stream": True},
                    headers=headers,
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
                        except json.JSONDecodeError:
                            continue
                        if u := chunk.get("usage"):
                            usage = u
                        if text := _stream_text_from_chunk(chunk):
                            any_text = True
                            yield StreamResult(text=text)

            if not any_text:
                r = await client.post(
                    url,
                    json={**body_base, "stream": False},
                    headers=headers,
                )
                if r.status_code >= 400:
                    try:
                        detail = r.json().get("error", r.text)
                    except (json.JSONDecodeError, TypeError):
                        detail = (r.text or "").strip() or r.reason_phrase
                    yield StreamResult(
                        text=(
                            f"Ошибка RouterAI (HTTP {r.status_code}): {detail}\n\n"
                            "Если указано credits — проверьте баланс и ключ на routerai.ru."
                        ),
                    )
                else:
                    try:
                        payload = r.json()
                    except json.JSONDecodeError:
                        yield StreamResult(
                            text=f"Ответ API не JSON: {(r.text or '')[:800]}",
                        )
                    else:
                        if u := payload.get("usage"):
                            usage = u
                        ch0 = (payload.get("choices") or [{}])[0]
                        if isinstance(ch0, dict):
                            msg = ch0.get("message") or {}
                            reply = _normalize_stream_content(msg.get("content"))
                            if reply:
                                yield StreamResult(text=reply)
                            else:
                                yield StreamResult(
                                    text=(
                                        "Модель вернула пустое сообщение. "
                                        f"Фрагмент ответа: {json.dumps(payload, ensure_ascii=False)[:600]}"
                                    ),
                                )
                        else:
                            yield StreamResult(
                                text=f"Неожиданный ответ API: {str(payload)[:800]}",
                            )

        elapsed_ms = round((time.monotonic() - t_start) * 1000)
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens))

        yield StreamResult(meta={
            "time_ms": elapsed_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        })


_EMBEDDING_MODEL_PATTERNS = (
    "bge-",
    "nomic-embed",
    "snowflake-arctic-embed",
    "-embed",
    "embedding",
    "e5-",
    "gte-",
    "mxbai-embed",
)


def _is_embedding_model(model_id: str) -> bool:
    mid = model_id.rsplit("/", 1)[-1].lower()
    return any(p in mid for p in _EMBEDDING_MODEL_PATTERNS)


def _label_for_ollama_model(model_id: str) -> str:
    """huggingface.co/bartowski/Vikhr-Nemo-12B-Instruct-R-21-09-24-GGUF:Q4_K_M
    → 'Vikhr-Nemo-12B-Instruct (Q4_K_M) — Local'."""
    tag = ""
    base = model_id
    if ":" in model_id:
        base, tag = model_id.split(":", 1)
    short = base.rsplit("/", 1)[-1]
    for suf in ("-GGUF", "-gguf"):
        if short.endswith(suf):
            short = short[: -len(suf)]
    short = short.replace("-R-21-09-24", "")
    qpart = f" ({tag})" if tag else ""
    return f"{short}{qpart} — Local"


class OllamaProvider(AIProvider):
    """Локальные модели через Ollama (OpenAI-совместимый эндпоинт /v1)."""

    name = "ollama"

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")
        self.models: list[dict] = []

    async def refresh_models(self) -> list[dict]:
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                r = await client.get(f"{self.base_url}/api/tags")
                r.raise_for_status()
                data = r.json()
        except (httpx.HTTPError, json.JSONDecodeError):
            self.models = []
            return self.models
        items = data.get("models") or []
        self.models = [
            {"id": m["name"], "label": _label_for_ollama_model(m["name"])}
            for m in items
            if isinstance(m, dict) and m.get("name") and not _is_embedding_model(m["name"])
        ]
        return self.models

    async def stream_chat(
        self, messages: list[Message], model: str, temperature: float = 0.7
    ) -> AsyncIterator[StreamResult]:
        url = f"{self.base_url}/v1/chat/completions"
        body = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "stream": True,
        }
        t_start = time.monotonic()
        usage: dict = {}
        any_text = False

        async with httpx.AsyncClient(timeout=600) as client:
            try:
                async with client.stream("POST", url, json=body) as resp:
                    if resp.status_code >= 400:
                        detail = (await resp.aread()).decode("utf-8", "ignore")[:800]
                        yield StreamResult(
                            text=f"Ошибка Ollama (HTTP {resp.status_code}): {detail}"
                        )
                        return
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[6:]
                        if data.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        if u := chunk.get("usage"):
                            usage = u
                        if text := _stream_text_from_chunk(chunk):
                            any_text = True
                            yield StreamResult(text=text)
            except httpx.HTTPError as e:
                yield StreamResult(
                    text=(
                        f"Ollama недоступна по {url}: {e}. "
                        "Убедитесь, что `ollama serve` запущен и модель скачана."
                    ),
                )
                return

        elapsed_ms = round((time.monotonic() - t_start) * 1000)
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens))

        yield StreamResult(meta={
            "time_ms": elapsed_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "local": True,
        })
        if not any_text:
            yield StreamResult(text="Модель не вернула текст.")
