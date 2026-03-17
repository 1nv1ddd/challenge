from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .providers import (
    AIProvider,
    GeminiProvider,
    GroqProvider,
    Message,
    OpenRouterProvider,
)

load_dotenv()

app = FastAPI(title="AI Chat Hub")

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

providers: dict[str, AIProvider] = {}

if key := os.getenv("GEMINI_API_KEY"):
    providers["gemini"] = GeminiProvider(key)
if key := os.getenv("GROQ_API_KEY"):
    providers["groq"] = GroqProvider(key)
if key := os.getenv("OPENROUTER_API_KEY"):
    providers["openrouter"] = OpenRouterProvider(key)


@app.get("/api/models")
async def list_models():
    """Return available providers and their models."""
    result = {}
    for name, prov in providers.items():
        result[name] = prov.models
    return result


@app.post("/api/chat")
async def chat(request: Request):
    """Stream a chat completion."""
    body = await request.json()

    provider_name: str = body.get("provider", "")
    model: str = body.get("model", "")
    raw_messages: list[dict] = body.get("messages", [])

    if provider_name not in providers:
        raise HTTPException(404, f"Provider '{provider_name}' not configured")

    prov = providers[provider_name]
    if model not in prov.models:
        raise HTTPException(400, f"Model '{model}' not available for {provider_name}")

    messages = [Message(role=m["role"], content=m["content"]) for m in raw_messages]

    async def event_stream():
        try:
            async for token in prov.stream_chat(messages, model):
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:
            yield f"data: [ERROR] {exc}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")



if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
