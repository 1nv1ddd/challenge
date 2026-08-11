"""DTO двухуровневого инференса: сосед из банка, вердикт micro-model и итог классификации."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..staged.schema import StageCall

# Длина примера банка в карточке соседа: полный текст в ответе не нужен, нужен опознавательный кусок.
_NEIGHBOR_PREVIEW = 90


@dataclass
class Neighbor:
    """Пример из банка, оказавшийся близко к запросу."""

    label: str
    similarity: float
    text: str

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "similarity": round(self.similarity, 4),
            "text": self.text[:_NEIGHBOR_PREVIEW],
        }


@dataclass
class MicroVerdict:
    """Ответ уровня 1: метка, уверенность и статус — пускать дальше или звать большую модель."""

    backend: str
    label: str
    score: float
    # "OK" — решение micro-model принимается, "UNSURE" — идём на уровень 2.
    status: str
    escalate_reason: str | None = None
    top_similarity: float = 0.0
    margin: float = 0.0
    votes: float = 0.0
    neighbors: list[Neighbor] = field(default_factory=list)
    time_ms: int = 0

    @property
    def ok(self) -> bool:
        return self.status == "OK"

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "label": self.label,
            "score": round(self.score, 4),
            "status": self.status,
            "escalate_reason": self.escalate_reason,
            "top_similarity": round(self.top_similarity, 4),
            "margin": round(self.margin, 4),
            "votes": round(self.votes, 4),
            "neighbors": [n.to_dict() for n in self.neighbors],
            "time_ms": self.time_ms,
        }


@dataclass
class LabelAnswer:
    """Разобранный ответ большой модели: метка плюс её собственная оценка уверенности."""

    label: str
    confidence: float
    reason: str

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
        }


@dataclass
class IntentResult:
    """Итог классификации: кто ответил, чем это обошлось и что показала micro-model."""

    strategy: str
    text: str
    label: str
    # "micro" — вопрос закрыт уровнем 1, "llm" — понадобился уровень 2.
    source: str
    micro: MicroVerdict | None
    llm: StageCall | None
    llm_answer: LabelAnswer | None
    metrics: dict

    @property
    def ok(self) -> bool:
        """Классификация состоялась, если метка получена и уровень 2 не остался с ошибкой формата."""
        return bool(self.label) and (self.llm is None or self.llm.error is None)

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "text": self.text,
            "label": self.label,
            "source": self.source,
            "ok": self.ok,
            "micro": self.micro.to_dict() if self.micro else None,
            "llm": self.llm.to_dict() if self.llm else None,
            "llm_answer": self.llm_answer.to_dict() if self.llm_answer else None,
            "metrics": dict(self.metrics),
        }
