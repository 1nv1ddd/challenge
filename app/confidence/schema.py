"""DTO триажа: решение модели, отдельный сэмпл, self-check и итоговый вердикт гейта."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Decision:
    """Структурированное решение по обращению (то, что модель обязана вернуть JSON-ом)."""

    category: str
    priority: str
    action: str
    reason: str
    missing: list[str] = field(default_factory=list)

    @property
    def vote_key(self) -> str:
        """Ключ голосования redundancy: важны именно категория и действие."""
        return f"{self.category}/{self.action}"

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "priority": self.priority,
            "action": self.action,
            "reason": self.reason,
            "missing": list(self.missing),
        }


@dataclass
class SampleResult:
    """Один прогон модели: что вернулось, прошло ли проверки, чего стоило."""

    index: int
    decision: Decision | None
    violations: list[str]
    raw: str
    # Что не прошло на первой попытке — даже если ремонт потом удался (нужно для отчётности).
    first_violations: list[str] = field(default_factory=list)
    repaired: bool = False
    calls: int = 1
    time_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "decision": self.decision.to_dict() if self.decision else None,
            "violations": list(self.violations),
            "first_violations": list(self.first_violations),
            "repaired": self.repaired,
            "calls": self.calls,
            "time_ms": self.time_ms,
        }


@dataclass
class SelfCheck:
    """Результат перепроверки решения самой моделью."""

    verdict: str  # "confirm" | "reject" | "unknown"
    reason: str = ""
    time_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def to_dict(self) -> dict:
        return {"verdict": self.verdict, "reason": self.reason, "time_ms": self.time_ms}


@dataclass
class Verdict:
    """Итог гейта: решение + статус приёмки + чем этот статус обоснован."""

    status: str  # "OK" | "UNSURE" | "FAIL"
    confidence: float
    decision: Decision | None
    applied_action: str
    agreement: float
    votes: dict[str, int]
    violations: list[str]
    selfcheck: SelfCheck | None
    samples: list[SampleResult]
    metrics: dict

    @property
    def accepted(self) -> bool:
        return self.status == "OK"

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "confidence": self.confidence,
            "decision": self.decision.to_dict() if self.decision else None,
            "applied_action": self.applied_action,
            "agreement": self.agreement,
            "votes": dict(self.votes),
            "violations": list(self.violations),
            "selfcheck": self.selfcheck.to_dict() if self.selfcheck else None,
            "samples": [s.to_dict() for s in self.samples],
            "metrics": dict(self.metrics),
        }
