from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .agent import SimpleChatAgent
from .providers import AIProvider, RouterAIProvider

load_dotenv()

app = FastAPI(title="AI Chat Hub")

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
MEMORY_PATH = Path(__file__).resolve().parent.parent / "data" / "agent_memory.json"

providers: dict[str, AIProvider] = {}

if key := os.getenv("ROUTERAI_API_KEY"):
    providers["routerai"] = RouterAIProvider(key)

agent = SimpleChatAgent(providers, memory_path=MEMORY_PATH)


@app.get("/api/models")
async def list_models():
    return agent.list_models()


@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()

    provider_name: str = body.get("provider", "")
    model: str = body.get("model", "")
    conversation_id: str = body.get("conversation_id", "default")
    raw_messages: list[dict] = body.get("messages", [])
    temperature: float = body.get("temperature", 0.7)

    async def event_stream():
        try:
            async for result in agent.stream_reply(
                provider_name=provider_name,
                model=model,
                conversation_id=conversation_id,
                raw_messages=raw_messages,
                temperature=temperature,
            ):
                if result.text is not None:
                    escaped = json.dumps(result.text, ensure_ascii=False)
                    yield f"data: {escaped}\n\n"
                if result.meta is not None:
                    yield f"data: [META]{json.dumps(result.meta)}\n\n"
            yield "data: [DONE]\n\n"
        except LookupError as exc:
            yield f"data: [ERROR] {str(exc)}\n\n"
        except ValueError as exc:
            yield f"data: [ERROR] {str(exc)}\n\n"
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
