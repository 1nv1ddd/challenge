"""Эмбеддинги через OpenAI-совместимый POST /v1/embeddings (RouterAI)."""

from __future__ import annotations

import os
from typing import Any

import httpx

DEFAULT_EMBED_URL = "https://routerai.ru/api/v1/embeddings"
DEFAULT_EMBED_MODEL = "openai/text-embedding-3-small"


def _parse_embeddings_payload(data: dict[str, Any]) -> list[list[float]]:
    items = data.get("data")
    if not isinstance(items, list):
        raise ValueError("embeddings: нет поля data")
    dict_items = [x for x in items if isinstance(x, dict)]
    if dict_items and all("index" in x for x in dict_items):
        dict_items.sort(key=lambda x: int(x.get("index", 0)))
    out: list[list[float]] = []
    for it in dict_items:
        emb = it.get("embedding")
        if not isinstance(emb, list):
            continue
        out.append([float(v) for v in emb])
    if len(out) != len(items):
        raise ValueError("embeddings: не удалось разобрать все векторы")
    return out


def embed_texts_sync(
    texts: list[str],
    *,
    api_key: str | None = None,
    model: str | None = None,
    url: str | None = None,
    batch_size: int = 64,
) -> list[list[float]]:
    """Синхронный батчинг (для scripts/build_rag_index)."""
    key = api_key or os.getenv("ROUTERAI_API_KEY") or ""
    if not key.strip():
        raise ValueError("Нужен ROUTERAI_API_KEY для эмбеддингов")
    m = model or os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_EMBED_MODEL)
    u = url or os.getenv("RAG_EMBEDDINGS_URL", DEFAULT_EMBED_URL)
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    all_vec: list[list[float]] = []
    with httpx.Client(timeout=120.0) as client:
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            r = client.post(u, headers=headers, json={"model": m, "input": batch})
            r.raise_for_status()
            all_vec.extend(_parse_embeddings_payload(r.json()))
    return all_vec


async def embed_texts_async(
    texts: list[str],
    *,
    api_key: str | None = None,
    model: str | None = None,
    url: str | None = None,
    batch_size: int = 32,
) -> list[list[float]]:
    key = api_key or os.getenv("ROUTERAI_API_KEY") or ""
    if not key.strip():
        raise ValueError("Нужен ROUTERAI_API_KEY для эмбеддингов")
    m = model or os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_EMBED_MODEL)
    u = url or os.getenv("RAG_EMBEDDINGS_URL", DEFAULT_EMBED_URL)
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    all_vec: list[list[float]] = []
    async with httpx.AsyncClient(timeout=120.0) as client:
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            r = await client.post(u, headers=headers, json={"model": m, "input": batch})
            r.raise_for_status()
            all_vec.extend(_parse_embeddings_payload(r.json()))
    return all_vec
