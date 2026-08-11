"""DTO шлюза: находки guard'ов, вердикты входа и выхода, запись аудита и кейс корпуса."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from ..agent_constants import (
    GATEWAY_CREDENTIAL_KINDS,
    GATEWAY_MODES,
    GATEWAY_SECRET_HASH_CHARS,
    GATEWAY_SECRET_KINDS,
    GATEWAY_VARIANTS,
)

# Действия шлюза над запросом/ответом — они же значения поля action в аудите.
GATEWAY_ACTIONS = ("pass", "mask", "block")


def secret_digest(value: str) -> str:
    """Отпечаток секрета для лога: сам секрет в аудит не попадает никогда."""
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()[:GATEWAY_SECRET_HASH_CHARS]


def secret_preview(value: str) -> str:
    """Хвост секрета отрезан, начало оставлено: по нему опознают, какой именно ключ утёк."""
    head = (value or "")[:6]
    return f"{head}…" if len(value or "") > 6 else head


@dataclass(frozen=True)
class SecretFinding:
    """Одна находка input guard: что нашли, в каком представлении текста и где именно."""

    kind: str
    variant: str = "direct"
    # Границы в исходном тексте; (-1, -1) — находка видна только после преобразования.
    start: int = -1
    end: int = -1
    preview: str = ""
    digest: str = ""

    @classmethod
    def make(cls, kind: str, value: str, *, variant: str = "direct", span: tuple[int, int] = (-1, -1)):
        """Собирает находку из найденного значения: превью и хэш считаются здесь, не по месту."""
        if kind not in GATEWAY_SECRET_KINDS:
            raise ValueError(f"Неизвестный вид секрета: {kind!r}.")
        if variant not in GATEWAY_VARIANTS:
            raise ValueError(f"Неизвестный вариант текста: {variant!r}.")
        return cls(
            kind=kind,
            variant=variant,
            start=span[0],
            end=span[1],
            preview=secret_preview(value),
            digest=secret_digest(value),
        )

    @property
    def severity(self) -> str:
        """credential — доступ в чужую систему, pii — персональные данные."""
        return "credential" if self.kind in GATEWAY_CREDENTIAL_KINDS else "pii"

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "variant": self.variant,
            "severity": self.severity,
            "start": self.start,
            "end": self.end,
            "preview": self.preview,
            "digest": self.digest,
        }


@dataclass
class InputVerdict:
    """Что input guard решил сделать с промптом до того, как он ушёл в модель."""

    action: str = "pass"
    prompt: str = ""
    findings: list[SecretFinding] = field(default_factory=list)
    masked: int = 0
    warning: str = ""

    @property
    def blocked(self) -> bool:
        return self.action == "block"

    def kinds(self) -> list[str]:
        return sorted({f.kind for f in self.findings})

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "blocked": self.blocked,
            "kinds": self.kinds(),
            "masked": self.masked,
            "warning": self.warning,
            "findings": [f.to_dict() for f in self.findings],
            # Промпт отдаём уже безопасным: в block-режиме — то, что было бы отправлено.
            "prompt_to_model": self.prompt,
        }


@dataclass(frozen=True)
class OutputFinding:
    """Находка output guard в ответе модели: вид проблемы и её фрагмент."""

    kind: str
    detail: str = ""

    def to_dict(self) -> dict:
        return {"kind": self.kind, "detail": self.detail}


@dataclass
class OutputVerdict:
    """Что уходит пользователю после проверки ответа модели."""

    action: str = "pass"
    answer: str = ""
    raw_answer: str = ""
    findings: list[OutputFinding] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return self.action == "block"

    def kinds(self) -> list[str]:
        return sorted({f.kind for f in self.findings})

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "blocked": self.blocked,
            "kinds": self.kinds(),
            "findings": [f.to_dict() for f in self.findings],
        }


@dataclass
class RateDecision:
    """Решение лимитера по одному клиенту."""

    allowed: bool
    used: int = 0
    limit: int = 0
    retry_after_sec: int = 0

    def to_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "used": self.used,
            "limit": self.limit,
            "retry_after_sec": self.retry_after_sec,
        }


@dataclass
class GatewayResult:
    """Результат одного прохода через шлюз — он же тело ответа API и строка аудита."""

    request_id: str
    client_ip: str = ""
    model: str = ""
    mode: str = ""
    # ok | blocked_input | blocked_output | rate_limited | too_long | error
    status: str = "ok"
    answer: str = ""
    input: InputVerdict = field(default_factory=InputVerdict)
    output: OutputVerdict = field(default_factory=OutputVerdict)
    rate: RateDecision | None = None
    # False — ответ отдан как есть, находки только зафиксированы (машинный потребитель).
    output_enforced: bool = True
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tokens_estimated: bool = False
    cost_rub: float = 0.0
    time_ms: int = 0
    error: str | None = None

    @property
    def llm_called(self) -> bool:
        """Дошёл ли запрос до модели: по этому полю в аудите видно сэкономленные вызовы."""
        return self.status in ("ok", "blocked_output")

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "status": self.status,
            "model": self.model,
            "mode": self.mode,
            "answer": self.answer,
            "input": self.input.to_dict(),
            "output": {**self.output.to_dict(), "enforced": self.output_enforced},
            "rate": self.rate.to_dict() if self.rate else None,
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens,
                "estimated": self.tokens_estimated,
                "cost_rub": self.cost_rub,
            },
            "time_ms": self.time_ms,
            "llm_called": self.llm_called,
            "error": self.error,
        }


@dataclass(frozen=True)
class GatewayCase:
    """Тест-кейс корпуса: промпт, ожидаемые виды секретов и ожидаемое действие шлюза."""

    id: str
    title: str
    text: str
    expect_kinds: tuple[str, ...] = ()
    expect_action: str = "pass"
    mode: str = "hybrid"
    note: str = ""

    @classmethod
    def from_dict(cls, raw: dict) -> GatewayCase:
        """Строка корпуса → кейс; неизвестный вид секрета или действие — ошибка загрузки."""
        case_id = str(raw.get("id") or "").strip()
        if not case_id:
            raise ValueError("У кейса нет id.")
        text = str(raw.get("text") or "")
        if not text.strip():
            raise ValueError(f"{case_id}: пустой промпт.")
        kinds = tuple(str(k).strip() for k in raw.get("expect_kinds") or ())
        unknown = [k for k in kinds if k not in GATEWAY_SECRET_KINDS]
        if unknown:
            raise ValueError(f"{case_id}: неизвестные виды секретов: {', '.join(unknown)}.")
        action = str(raw.get("expect_action") or "pass").strip()
        if action not in GATEWAY_ACTIONS:
            raise ValueError(f"{case_id}: неизвестное действие {action!r}.")
        mode = str(raw.get("mode") or "hybrid").strip()
        if mode not in GATEWAY_MODES:
            raise ValueError(f"{case_id}: неизвестный режим {mode!r}.")
        return cls(
            id=case_id,
            title=str(raw.get("title") or case_id).strip(),
            text=text,
            expect_kinds=kinds,
            expect_action=action,
            mode=mode,
            note=str(raw.get("note") or "").strip(),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "text": self.text,
            "expect_kinds": list(self.expect_kinds),
            "expect_action": self.expect_action,
            "mode": self.mode,
            "note": self.note,
        }


@dataclass
class CaseOutcome:
    """Итог прогона кейса по детекторам: что поймали, что пропустили, что нашли сверх ожидания."""

    case_id: str
    title: str
    mode: str
    expect_kinds: tuple[str, ...] = ()
    found_kinds: tuple[str, ...] = ()
    expect_action: str = "pass"
    action: str = "pass"
    variants: tuple[str, ...] = ()
    note: str = ""

    @property
    def caught(self) -> list[str]:
        return sorted(set(self.expect_kinds) & set(self.found_kinds))

    @property
    def missed(self) -> list[str]:
        return sorted(set(self.expect_kinds) - set(self.found_kinds))

    @property
    def extra(self) -> list[str]:
        """Находки сверх ожидаемых: не всегда ошибка, но всегда повод посмотреть глазами."""
        return sorted(set(self.found_kinds) - set(self.expect_kinds))

    @property
    def ok(self) -> bool:
        return not self.missed and self.action == self.expect_action

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "title": self.title,
            "mode": self.mode,
            "expect_kinds": list(self.expect_kinds),
            "found_kinds": list(self.found_kinds),
            "expect_action": self.expect_action,
            "action": self.action,
            "variants": list(self.variants),
            "caught": self.caught,
            "missed": self.missed,
            "extra": self.extra,
            "ok": self.ok,
            "note": self.note,
        }


@dataclass
class CorpusRun:
    """Прогон всего корпуса по детекторам — офлайн, без вызовов модели."""

    outcomes: list[CaseOutcome] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.outcomes)

    @property
    def passed(self) -> int:
        return sum(1 for o in self.outcomes if o.ok)

    @property
    def missed_cases(self) -> list[str]:
        return [o.case_id for o in self.outcomes if o.missed]

    @property
    def wrong_action(self) -> list[str]:
        return [o.case_id for o in self.outcomes if not o.missed and o.action != o.expect_action]

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "passed": self.passed,
            "missed_cases": self.missed_cases,
            "wrong_action": self.wrong_action,
            "outcomes": [o.to_dict() for o in self.outcomes],
        }
