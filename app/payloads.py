"""Разбор JSON-тел HTTP-запросов для API (тонкий слой без FastAPI)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .agent_constants import (
    INDIRECT_MODEL,
    INDIRECT_PRESETS,
    INDIRECT_TEMPERATURE,
    INTAKE_MONO_MODEL,
    INTAKE_STAGE_MODELS,
    INTAKE_TEMPERATURE,
    MICRO_DEFAULT_STRATEGY,
    MICRO_LLM_MODEL,
    MICRO_TEMPERATURE,
    ROUTING_LARGE_MODEL,
    ROUTING_SMALL_MODEL,
    ROUTING_TEMPERATURE,
    SECURITY_MODEL,
    SECURITY_PROMPT_VERSIONS,
    SECURITY_TEMPERATURE,
    TRIAGE_SAMPLES,
    TRIAGE_TEMPERATURE,
)


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


@dataclass(frozen=True)
class TriagePayload:
    provider_name: str
    model: str
    text: str
    samples: int
    temperature: float

    @classmethod
    def from_body(cls, body: dict[str, Any]) -> TriagePayload:
        try:
            samples = int(body.get("samples") or TRIAGE_SAMPLES)
        except (TypeError, ValueError):
            samples = TRIAGE_SAMPLES
        try:
            temperature = float(body.get("temperature", TRIAGE_TEMPERATURE))
        except (TypeError, ValueError):
            temperature = TRIAGE_TEMPERATURE
        return cls(
            provider_name=str(body.get("provider") or "").strip(),
            model=str(body.get("model") or "").strip(),
            text=str(body.get("text") or "").strip(),
            samples=max(1, min(5, samples)),
            temperature=temperature,
        )


@dataclass(frozen=True)
class RoutePayload:
    provider_name: str
    question: str
    small_model: str
    large_model: str
    temperature: float

    @classmethod
    def from_body(cls, body: dict[str, Any]) -> RoutePayload:
        try:
            temperature = float(body.get("temperature", ROUTING_TEMPERATURE))
        except (TypeError, ValueError):
            temperature = ROUTING_TEMPERATURE
        return cls(
            provider_name=str(body.get("provider") or "routerai").strip(),
            question=str(body.get("question") or "").strip(),
            small_model=str(body.get("small_model") or ROUTING_SMALL_MODEL).strip(),
            large_model=str(body.get("large_model") or ROUTING_LARGE_MODEL).strip(),
            temperature=temperature,
        )


@dataclass(frozen=True)
class IntakePayload:
    provider_name: str
    letter: str
    mode: str
    today: str
    mono_model: str
    stage_models: dict[str, str]
    temperature: float

    @classmethod
    def from_body(cls, body: dict[str, Any]) -> IntakePayload:
        try:
            temperature = float(body.get("temperature", INTAKE_TEMPERATURE))
        except (TypeError, ValueError):
            temperature = INTAKE_TEMPERATURE
        raw_models = body.get("stage_models")
        # Переопределять можно только известные этапы: чужие ключи молча игнорируем.
        stage_models = (
            {
                str(stage): str(model).strip()
                for stage, model in raw_models.items()
                if stage in INTAKE_STAGE_MODELS and str(model).strip()
            }
            if isinstance(raw_models, dict)
            else {}
        )
        return cls(
            provider_name=str(body.get("provider") or "routerai").strip(),
            letter=str(body.get("letter") or "").strip(),
            mode=str(body.get("mode") or "staged").strip(),
            today=str(body.get("today") or "").strip(),
            mono_model=str(body.get("mono_model") or INTAKE_MONO_MODEL).strip(),
            stage_models=stage_models,
            temperature=temperature,
        )


@dataclass(frozen=True)
class IntentPayload:
    provider_name: str
    text: str
    strategy: str
    llm_model: str
    temperature: float

    @classmethod
    def from_body(cls, body: dict[str, Any]) -> IntentPayload:
        try:
            temperature = float(body.get("temperature", MICRO_TEMPERATURE))
        except (TypeError, ValueError):
            temperature = MICRO_TEMPERATURE
        return cls(
            provider_name=str(body.get("provider") or "routerai").strip(),
            text=str(body.get("text") or "").strip(),
            strategy=str(body.get("strategy") or MICRO_DEFAULT_STRATEGY).strip(),
            llm_model=str(body.get("llm_model") or MICRO_LLM_MODEL).strip(),
            temperature=temperature,
        )


@dataclass(frozen=True)
class IndirectPayload:
    provider_name: str
    presets: tuple[str, ...]
    ids: tuple[str, ...]
    scenario: str
    source: str
    hiding: str
    model: str
    temperature: float

    @classmethod
    def from_body(cls, body: dict[str, Any]) -> IndirectPayload:
        try:
            temperature = float(body.get("temperature", INDIRECT_TEMPERATURE))
        except (TypeError, ValueError):
            temperature = INDIRECT_TEMPERATURE
        raw_presets = body.get("presets")
        # Порядок пресетов фиксируем константой: в отчёте «без защиты» всегда идёт первым.
        presets = (
            tuple(p for p in INDIRECT_PRESETS if p in set(map(str, raw_presets)))
            if isinstance(raw_presets, list) and raw_presets
            else ("none", "all")
        )
        raw_ids = body.get("ids")
        ids = (
            tuple(str(i).strip() for i in raw_ids if str(i).strip())
            if isinstance(raw_ids, list)
            else ()
        )
        return cls(
            provider_name=str(body.get("provider") or "routerai").strip(),
            presets=presets or ("none", "all"),
            ids=ids,
            scenario=str(body.get("scenario") or "").strip(),
            source=str(body.get("source") or "").strip(),
            hiding=str(body.get("hiding") or "").strip(),
            model=str(body.get("model") or INDIRECT_MODEL).strip(),
            temperature=temperature,
        )


@dataclass(frozen=True)
class RedteamPayload:
    provider_name: str
    versions: tuple[str, ...]
    ids: tuple[str, ...]
    target: str
    vector: str
    technique: str
    model: str
    temperature: float

    @classmethod
    def from_body(cls, body: dict[str, Any]) -> RedteamPayload:
        try:
            temperature = float(body.get("temperature", SECURITY_TEMPERATURE))
        except (TypeError, ValueError):
            temperature = SECURITY_TEMPERATURE
        raw_versions = body.get("versions")
        # Порядок версий фиксируем константой: в отчёте v1 всегда идёт перед v2.
        versions = (
            tuple(v for v in SECURITY_PROMPT_VERSIONS if v in set(map(str, raw_versions)))
            if isinstance(raw_versions, list) and raw_versions
            else SECURITY_PROMPT_VERSIONS
        )
        raw_ids = body.get("ids")
        ids = (
            tuple(str(i).strip() for i in raw_ids if str(i).strip())
            if isinstance(raw_ids, list)
            else ()
        )
        return cls(
            provider_name=str(body.get("provider") or "routerai").strip(),
            versions=versions or SECURITY_PROMPT_VERSIONS,
            ids=ids,
            target=str(body.get("target") or "").strip(),
            vector=str(body.get("vector") or "").strip(),
            technique=str(body.get("technique") or "").strip(),
            model=str(body.get("model") or SECURITY_MODEL).strip(),
            temperature=temperature,
        )


def sse_error_line(exc: BaseException) -> str:
    """Одна строка SSE с префиксом [ERROR] для стрима чата."""
    if isinstance(exc, (LookupError, ValueError)):
        msg = str(exc).strip() or type(exc).__name__
    else:
        msg = str(exc).replace("\n", " ").strip() or type(exc).__name__
    return f"data: [ERROR] {msg}\n\n"
