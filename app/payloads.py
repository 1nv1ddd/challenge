"""Разбор JSON-тел HTTP-запросов для API (тонкий слой без FastAPI)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ChatRequestPayload:
    provider_name: str
    model: str
    conversation_id: str
    raw_messages: list[dict]
    temperature: float
    context_strategy: str
    branch_id: str
    profile_id: str | None
    resume: bool
    rag: dict[str, Any] | None
    task_workflow: bool | None

    @classmethod
    def from_body(cls, body: dict[str, Any]) -> ChatRequestPayload:
        raw_msgs = body.get("messages", [])
        if not isinstance(raw_msgs, list):
            raw_msgs = []
        raw_tw = body.get("task_workflow")
        rag = body.get("rag")
        return cls(
            provider_name=str(body.get("provider", "")),
            model=str(body.get("model", "")),
            conversation_id=str(body.get("conversation_id", "default")),
            raw_messages=raw_msgs,
            temperature=float(body.get("temperature", 0.7)),
            context_strategy=str(body.get("context_strategy", "sliding")),
            branch_id=str(body.get("branch_id", "main")),
            profile_id=body.get("profile_id"),
            resume=bool(body.get("resume", False)),
            rag=rag if isinstance(rag, dict) else None,
            task_workflow=None if raw_tw is None else bool(raw_tw),
        )


@dataclass(frozen=True)
class RagComparePayload:
    provider_name: str
    model: str
    message: str
    temperature: float
    rag_strategy: str
    top_k: int
    index_path: str | None

    @classmethod
    def from_body(cls, body: dict[str, Any]) -> RagComparePayload:
        raw_idx = body.get("index_path")
        ip = str(raw_idx).strip() if isinstance(raw_idx, str) and raw_idx.strip() else None
        return cls(
            provider_name=str(body.get("provider") or "").strip(),
            model=str(body.get("model") or "").strip(),
            message=str(body.get("message") or "").strip(),
            temperature=float(body.get("temperature", 0.35)),
            rag_strategy=str(body.get("rag_strategy") or "fixed").lower().strip(),
            top_k=int(body.get("top_k") or 8),
            index_path=ip,
        )


@dataclass(frozen=True)
class RagModesComparePayload:
    provider_name: str
    model: str
    message: str
    temperature: float
    rag_strategy: str
    top_k: int
    index_path: str | None
    min_similarity: float

    @classmethod
    def from_body(cls, body: dict[str, Any]) -> RagModesComparePayload:
        raw_idx = body.get("index_path")
        ip = str(raw_idx).strip() if isinstance(raw_idx, str) and raw_idx.strip() else None
        raw_min = body.get("min_similarity", 0.28)
        try:
            min_sim = float(raw_min)
        except (TypeError, ValueError):
            min_sim = 0.28
        return cls(
            provider_name=str(body.get("provider") or "").strip(),
            model=str(body.get("model") or "").strip(),
            message=str(body.get("message") or "").strip(),
            temperature=float(body.get("temperature", 0.35)),
            rag_strategy=str(body.get("rag_strategy") or "fixed").lower().strip(),
            top_k=int(body.get("top_k") or 8),
            index_path=ip,
            min_similarity=min_sim,
        )


def sse_error_line(exc: BaseException) -> str:
    """Одна строка SSE с префиксом [ERROR] для стрима чата."""
    if isinstance(exc, (LookupError, ValueError)):
        msg = str(exc).strip() or type(exc).__name__
    else:
        msg = str(exc).replace("\n", " ").strip() or type(exc).__name__
    return f"data: [ERROR] {msg}\n\n"
