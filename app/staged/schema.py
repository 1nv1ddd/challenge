"""DTO декомпозиции инференса: нормализованные поля заявки, решение, один этап и итог разбора."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..agent_constants import INTAKE_FIELD_KEYS, INTAKE_NUMERIC_FIELDS, INTAKE_REQUIRED_FIELDS

_UNKNOWN = "unknown"


@dataclass
class IntakeFields:
    """Заявка после нормализации: единицы, даты и регион уже приведены к канону."""

    company: str = _UNKNOWN
    product: str = _UNKNOWN
    qty_kg: int | None = None
    budget_rub: int | None = None
    deadline: str = _UNKNOWN
    region: str = _UNKNOWN
    contact: str = _UNKNOWN
    payment: str = _UNKNOWN

    def value(self, key: str) -> str:
        """Значение поля строкой в том же виде, в каком его выдаёт модель."""
        raw = getattr(self, key)
        if key in INTAKE_NUMERIC_FIELDS:
            return _UNKNOWN if raw is None else str(raw)
        return str(raw or _UNKNOWN)

    def missing(self) -> list[str]:
        """Обязательные поля, которых в письме не нашлось."""
        return [key for key in INTAKE_REQUIRED_FIELDS if self.value(key) == _UNKNOWN]

    def to_compact(self) -> str:
        """Компактный формат `ключ: значение` — выход этапа 1 и вход этапа 2."""
        return "\n".join(f"{key}: {self.value(key)}" for key in INTAKE_FIELD_KEYS)

    def to_dict(self) -> dict:
        return {key: getattr(self, key) for key in INTAKE_FIELD_KEYS}


@dataclass
class StageDecision:
    """Решение по заявке: строгий enum плюс чего не хватает, если решение — уточнять."""

    decision: str
    reason: str
    missing: list[str] = field(default_factory=list)
    # "llm" — решение приняла модель, "rules" — детерминированная политика по нормализованным полям.
    source: str = "llm"

    def to_dict(self) -> dict:
        return {
            "decision": self.decision,
            "reason": self.reason,
            "missing": list(self.missing),
            "source": self.source,
        }


@dataclass
class StageCall:
    """Один вызов модели внутри разбора — со своей моделью, ценой и историей ремонта формата."""

    stage: str
    model: str
    raw: str = ""
    calls: int = 0
    repaired: bool = False
    # Что не прошло на первой попытке — даже если ремонт потом удался.
    first_error: str | None = None
    error: str | None = None
    time_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_rub: float = 0.0

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "model": self.model,
            "raw": self.raw,
            "calls": self.calls,
            "repaired": self.repaired,
            "first_error": self.first_error,
            "error": self.error,
            "time_ms": self.time_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_rub": self.cost_rub,
        }


@dataclass
class IntakeResult:
    """Итог разбора письма: поля, решение, ответ клиенту и цена выбранного способа инференса."""

    mode: str
    letter: str
    today: str
    fields: IntakeFields
    decision: StageDecision
    reply_subject: str
    reply_body: str
    stages: list[StageCall]
    metrics: dict

    @property
    def ok(self) -> bool:
        """Разбор состоялся, если ни один этап не остался с ошибкой формата."""
        return all(stage.error is None for stage in self.stages)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "letter": self.letter,
            "today": self.today,
            "ok": self.ok,
            "fields": self.fields.to_dict(),
            "decision": self.decision.to_dict(),
            "reply": {"subject": self.reply_subject, "body": self.reply_body},
            "stages": [stage.to_dict() for stage in self.stages],
            "metrics": dict(self.metrics),
        }
