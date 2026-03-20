from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .providers import AIProvider, GroqProvider, Message

load_dotenv()

app = FastAPI(title="AI Chat Hub")

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

providers: dict[str, AIProvider] = {}

if key := os.getenv("GROQ_API_KEY"):
    providers["groq"] = GroqProvider(key)


@app.get("/api/models")
async def list_models():
    result = {}
    for name, prov in providers.items():
        result[name] = prov.models
    return result


@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()

    provider_name: str = body.get("provider", "")
    model: str = body.get("model", "")
    raw_messages: list[dict] = body.get("messages", [])
    temperature: float = body.get("temperature", 0.7)

    if provider_name not in providers:
        raise HTTPException(404, f"Provider '{provider_name}' not configured")

    prov = providers[provider_name]
    model_ids = [m["id"] for m in prov.models]
    if model not in model_ids:
        raise HTTPException(400, f"Model '{model}' not available for {provider_name}")

    messages = [Message(role=m["role"], content=m["content"]) for m in raw_messages]

    async def event_stream():
        try:
            async for result in prov.stream_chat(messages, model, temperature):
                if result.text is not None:
                    escaped = json.dumps(result.text, ensure_ascii=False)
                    yield f"data: {escaped}\n\n"
                if result.meta is not None:
                    yield f"data: [META]{json.dumps(result.meta)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:
            msg = str(exc).replace("\n", " ")
            yield f"data: [ERROR] {msg}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/", response_class=HTMLResponse)
async def index():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(
        content=html,
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
