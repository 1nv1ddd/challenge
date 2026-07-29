"""DTO роутинга: пре-маршрут, попытка на конкретном тире, оценка ответа и итог каскада."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PreRoute:
    """Решение до единого вызова LLM: с какого тира начинать и почему."""

    tier: str  # "small" | "large"
    score: int
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"tier": self.tier, "score": self.score, "reasons": list(self.reasons)}


@dataclass
class Assessment:
    """Оценка ответа дешёвой модели: во что сложились сигналы и нужна ли эскалация."""

    confidence: float
    self_reported: float | None
    signals: list[str]
    escalate: bool
    reason: str

    def to_dict(self) -> dict:
        return {
            "confidence": self.confidence,
            "self_reported": self.self_reported,
            "signals": list(self.signals),
            "escalate": self.escalate,
            "reason": self.reason,
        }


@dataclass
class RouteAttempt:
    """Один вызов модели внутри каскада — с ценой именно этой модели."""

    tier: str
    model: str
    answer: str
    assessment: Assessment | None = None
    time_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_rub: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "tier": self.tier,
            "model": self.model,
            "answer": self.answer,
            "assessment": self.assessment.to_dict() if self.assessment else None,
            "time_ms": self.time_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_rub": self.cost_rub,
            "error": self.error,
        }


@dataclass
class RouteResult:
    """Итог маршрутизации: чей ответ ушёл пользователю и во что обошёлся путь."""

    question: str
    answer: str
    model: str
    tier: str
    escalated: bool
    escalation_reason: str
    preroute: PreRoute
    attempts: list[RouteAttempt]
    metrics: dict

    @property
    def stayed_small(self) -> bool:
        return self.tier == "small"

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "model": self.model,
            "tier": self.tier,
            "escalated": self.escalated,
            "escalation_reason": self.escalation_reason,
            "preroute": self.preroute.to_dict(),
            "attempts": [a.to_dict() for a in self.attempts],
            "metrics": dict(self.metrics),
        }
