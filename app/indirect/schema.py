"""DTO непрямых инъекций: кейс из корпуса, отчёт санитайзера, находки guard и результат прогона."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..agent_constants import (
    INDIRECT_HIDING,
    INDIRECT_LAYERS,
    INDIRECT_SCENARIOS,
    INDIRECT_SOURCES,
)


@dataclass(frozen=True)
class IndirectCase:
    """Одна ловушка: чистый носитель, спрятанная в нём инструкция и признаки успеха атаки."""

    id: str
    title: str
    scenario: str
    source: str
    hiding: str
    # Текст, который видит человек: письмо, документ, страница. Payload сюда вставляется отдельно.
    carrier: str
    # Инструкция для модели, спрятанная в носителе.
    payload: str
    # Запрос пользователя агенту: он про содержимое, а не про инъекцию.
    request: str
    origin: str = ""
    # Что должно появиться в ответе, если инъекция сработала.
    success_markers: tuple[str, ...] = ()
    # Что агент обязан сказать по существу: пропало — защита испортила полезный ответ.
    expected_markers: tuple[str, ...] = ()
    why_works: str = ""
    defense: str = ""

    @classmethod
    def from_dict(cls, raw: dict) -> IndirectCase:
        """Строка корпуса → кейс; неизвестный сценарий/носитель/техника — ошибка загрузки."""
        case_id = str(raw.get("id") or "").strip()
        if not case_id:
            raise ValueError("У кейса нет id.")
        scenario = str(raw.get("scenario") or "").strip()
        if scenario not in INDIRECT_SCENARIOS:
            raise ValueError(f"{case_id}: неизвестный сценарий {scenario!r}.")
        source = str(raw.get("source") or "").strip()
        if source not in INDIRECT_SOURCES:
            raise ValueError(f"{case_id}: неизвестный носитель {source!r}.")
        hiding = str(raw.get("hiding") or "").strip()
        if hiding not in INDIRECT_HIDING:
            raise ValueError(f"{case_id}: неизвестная техника сокрытия {hiding!r}.")
        for field_name in ("carrier", "payload", "request"):
            if not str(raw.get(field_name) or "").strip():
                raise ValueError(f"{case_id}: пустое поле {field_name}.")
        if not raw.get("success_markers"):
            raise ValueError(f"{case_id}: без success_markers успех атаки не проверить.")
        return cls(
            id=case_id,
            title=str(raw.get("title") or case_id).strip(),
            scenario=scenario,
            source=source,
            hiding=hiding,
            carrier=str(raw["carrier"]).strip(),
            payload=str(raw["payload"]).strip(),
            request=str(raw["request"]).strip(),
            origin=str(raw.get("origin") or "").strip(),
            success_markers=tuple(str(m) for m in raw.get("success_markers") or ()),
            expected_markers=tuple(str(m) for m in raw.get("expected_markers") or ()),
            why_works=str(raw.get("why_works") or "").strip(),
            defense=str(raw.get("defense") or "").strip(),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "scenario": self.scenario,
            "source": self.source,
            "hiding": self.hiding,
            "carrier": self.carrier,
            "payload": self.payload,
            "request": self.request,
            "origin": self.origin,
            "why_works": self.why_works,
            "defense": self.defense,
        }


@dataclass
class SanitizeReport:
    """Что чистка вырезала из документа: по этим счётчикам можно алертить, а не только чистить."""

    html_comments: int = 0
    hidden_elements: int = 0
    zero_width: int = 0
    suspicious_links: int = 0
    imperative_lines: int = 0
    removed_chars: int = 0

    @property
    def total(self) -> int:
        return (
            self.html_comments
            + self.hidden_elements
            + self.zero_width
            + self.suspicious_links
            + self.imperative_lines
        )

    def to_dict(self) -> dict:
        return {
            "html_comments": self.html_comments,
            "hidden_elements": self.hidden_elements,
            "zero_width": self.zero_width,
            "suspicious_links": self.suspicious_links,
            "imperative_lines": self.imperative_lines,
            "removed_chars": self.removed_chars,
            "total": self.total,
        }


@dataclass(frozen=True)
class GuardFinding:
    """Находка output guard: что именно в ответе агента выглядит следом инъекции."""

    kind: str
    detail: str

    def to_dict(self) -> dict:
        return {"kind": self.kind, "detail": self.detail}


@dataclass
class IndirectResult:
    """Результат одного прогона кейса при заданном наборе слоёв защиты."""

    case_id: str
    scenario: str
    hiding: str
    layers: tuple[str, ...] = ()
    raw_answer: str = ""
    delivered_answer: str = ""
    sanitize: SanitizeReport = field(default_factory=SanitizeReport)
    findings: list[GuardFinding] = field(default_factory=list)
    blocked: bool = False
    injected: bool = False
    useful: bool = True
    signals: list[str] = field(default_factory=list)
    error: str | None = None
    model: str = ""
    time_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_rub: float = 0.0

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "scenario": self.scenario,
            "hiding": self.hiding,
            "layers": list(self.layers),
            "raw_answer": self.raw_answer,
            "delivered_answer": self.delivered_answer,
            "sanitize": self.sanitize.to_dict(),
            "findings": [f.to_dict() for f in self.findings],
            "blocked": self.blocked,
            "injected": self.injected,
            "useful": self.useful,
            "signals": list(self.signals),
            "error": self.error,
            "model": self.model,
            "time_ms": self.time_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_rub": self.cost_rub,
        }


@dataclass
class LayerRun:
    """Прогон набора кейсов при одном пресете защиты."""

    preset: str
    results: list[IndirectResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def injected(self) -> int:
        return sum(1 for r in self.results if r.injected)

    @property
    def blocked(self) -> int:
        return sum(1 for r in self.results if r.blocked)

    @property
    def broke_usefulness(self) -> int:
        """Кейсы, где защита убила полезный ответ — цена, которую платим за безопасность."""
        return sum(1 for r in self.results if not r.useful and r.error is None)

    @property
    def cost_rub(self) -> float:
        return round(sum(r.cost_rub for r in self.results), 4)

    def layers(self) -> tuple[str, ...]:
        for result in self.results:
            if result.layers:
                return result.layers
        return ()

    def to_dict(self) -> dict:
        return {
            "preset": self.preset,
            "layers": list(self.layers()),
            "total": self.total,
            "injected": self.injected,
            "blocked": self.blocked,
            "broke_usefulness": self.broke_usefulness,
            "injection_rate": round(self.injected / self.total, 3) if self.total else 0.0,
            "cost_rub": self.cost_rub,
            "results": [r.to_dict() for r in self.results],
        }


def normalize_layers(layers: tuple[str, ...]) -> tuple[str, ...]:
    """Приводит набор слоёв к порядку INDIRECT_LAYERS и отсекает неизвестные."""
    unknown = [layer for layer in layers if layer not in INDIRECT_LAYERS]
    if unknown:
        raise ValueError(
            f"Неизвестные слои защиты: {', '.join(unknown)}; доступны: {', '.join(INDIRECT_LAYERS)}."
        )
    return tuple(layer for layer in INDIRECT_LAYERS if layer in set(layers))
