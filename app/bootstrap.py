"""Инициализация окружения, провайдеров и агента (один раз при импорте приложения)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from .agent import SimpleChatAgent
from .providers import AIProvider, OllamaProvider, RouterAIProvider

load_dotenv()

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
MEMORY_PATH = Path(__file__).resolve().parent.parent / "data" / "agent_memory.json"


def _build_providers() -> dict[str, AIProvider]:
    out: dict[str, AIProvider] = {}
    if key := os.getenv("ROUTERAI_API_KEY"):
        out["routerai"] = RouterAIProvider(key)
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    out["ollama"] = OllamaProvider(ollama_url)
    return out


providers: dict[str, AIProvider] = _build_providers()
agent = SimpleChatAgent(providers, memory_path=MEMORY_PATH)
