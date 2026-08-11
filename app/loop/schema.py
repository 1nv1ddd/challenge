"""DTO цикла: задача, результат проверок, находки security review, попытка и прогон целиком."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..agent_constants import (
    LOOP_BLOCKING_SEVERITIES,
    LOOP_SECURITY_RULES,
    LOOP_SEVERITIES,
)

# Чем закончилась попытка — оно же объясняет, почему цикл пошёл на следующий круг.
LOOP_OUTCOMES = (
    "generated",
    "gateway_blocked",
    "no_code",
    "checks_failed",
    "security_blocked",
    "accepted_with_warnings",
    "accepted",
)
# Итог всего прогона задачи.
LOOP_STATUSES = ("committed", "committed_with_warnings", "escalated", "failed")


@dataclass(frozen=True)
class LoopTask:
    """Задача цикла: что просим сгенерировать, чем проверяем и какие небезопасные решения провоцируем."""

    id: str
    title: str
    prompt: str
    tests: str
    traps: tuple[str, ...] = ()
    note: str = ""

    @classmethod
    def from_dict(cls, raw: dict) -> LoopTask:
        """Строка корпуса → задача; без промпта и тестов задача бессмысленна."""
        task_id = str(raw.get("id") or "").strip()
        if not task_id:
            raise ValueError("У задачи нет id.")
        for field_name in ("prompt", "tests"):
            if not str(raw.get(field_name) or "").strip():
                raise ValueError(f"{task_id}: пустое поле {field_name}.")
        traps = tuple(str(t).strip() for t in raw.get("traps") or ())
        unknown = [t for t in traps if t not in LOOP_SECURITY_RULES]
        if unknown:
            raise ValueError(f"{task_id}: неизвестные ловушки: {', '.join(unknown)}.")
        return cls(
            id=task_id,
            title=str(raw.get("title") or task_id).strip(),
            prompt=str(raw["prompt"]).strip(),
            tests=str(raw["tests"]),
            traps=traps,
            note=str(raw.get("note") or "").strip(),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "prompt": self.prompt,
            "tests": self.tests,
            "traps": list(self.traps),
            "note": self.note,
        }


@dataclass
class CheckResult:
    """Результат одного этапа проверок песочницы: синтаксис или прогон тестов."""

    stage: str
    ok: bool = False
    output: str = ""

    def to_dict(self) -> dict:
        return {"stage": self.stage, "ok": self.ok, "output": self.output}


@dataclass(frozen=True)
class SecurityFinding:
    """Находка security review: уровень, правило, строка и что делать."""

    severity: str
    rule: str
    title: str
    line: int = 0
    fix: str = ""

    @classmethod
    def from_dict(cls, raw: dict) -> SecurityFinding:
        """Элемент JSON-ответа ревьюера; неизвестный уровень или правило — ошибка формата."""
        severity = str(raw.get("severity") or "").strip().lower()
        if severity not in LOOP_SEVERITIES:
            raise ValueError(
                f"формат: уровень {severity!r} вне списка {', '.join(LOOP_SEVERITIES)}"
            )
        rule = str(raw.get("rule") or "").strip().lower()
        if rule not in LOOP_SECURITY_RULES:
            # Неизвестное правило не теряем: модель могла назвать реальную проблему своими словами.
            rule = "other"
        title = str(raw.get("title") or "").strip()
        if not title:
            raise ValueError("формат: у находки нет title")
        try:
            line = int(raw.get("line") or 0)
        except (TypeError, ValueError):
            line = 0
        return cls(
            severity=severity,
            rule=rule,
            title=title,
            line=max(0, line),
            fix=str(raw.get("fix") or "").strip(),
        )

    @property
    def blocking(self) -> bool:
        return self.severity in LOOP_BLOCKING_SEVERITIES

    def to_dict(self) -> dict:
        return {
            "severity": self.severity,
            "rule": self.rule,
            "title": self.title,
            "line": self.line,
            "fix": self.fix,
            "blocking": self.blocking,
        }


@dataclass
class SecurityVerdict:
    """Вердикт ревьюера по одному куску кода."""

    findings: list[SecurityFinding] = field(default_factory=list)
    summary: str = ""
    error: str | None = None

    @property
    def blocking(self) -> list[SecurityFinding]:
        """Critical/High — из-за них цикл возвращается на генерацию."""
        return [f for f in self.findings if f.blocking]

    @property
    def warnings(self) -> list[SecurityFinding]:
        """Medium/Low — пропускаем, но пишем в лог."""
        return [f for f in self.findings if not f.blocking]

    @property
    def clean(self) -> bool:
        return not self.findings and self.error is None

    def rules(self) -> list[str]:
        return sorted({f.rule for f in self.findings})

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "error": self.error,
            "clean": self.clean,
            "rules": self.rules(),
            "blocking": [f.to_dict() for f in self.blocking],
            "warnings": [f.to_dict() for f in self.warnings],
        }


@dataclass
class GatewayEvent:
    """Что шлюз сделал с одним вызовом цикла — из этого собирается ответ «что поймал gateway»."""

    stage: str
    request_id: str = ""
    status: str = "ok"
    mode: str = ""
    model: str = ""
    input_action: str = "pass"
    input_kinds: list[str] = field(default_factory=list)
    output_action: str = "pass"
    output_kinds: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_rub: float = 0.0

    @property
    def clean(self) -> bool:
        """Ни на входе, ни на выходе шлюз ничего не тронул."""
        return not self.input_kinds and not self.output_kinds and self.status == "ok"

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "request_id": self.request_id,
            "status": self.status,
            "mode": self.mode,
            "model": self.model,
            "input_action": self.input_action,
            "input_kinds": list(self.input_kinds),
            "output_action": self.output_action,
            "output_kinds": list(self.output_kinds),
            "clean": self.clean,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_rub": self.cost_rub,
        }


@dataclass
class LoopAttempt:
    """Одна итерация цикла: генерация → проверки → security review."""

    number: int
    outcome: str = "generated"
    code: str = ""
    feedback_in: str = ""
    checks: list[CheckResult] = field(default_factory=list)
    security: SecurityVerdict | None = None
    gateway: list[GatewayEvent] = field(default_factory=list)
    error: str | None = None

    @property
    def checks_ok(self) -> bool:
        return bool(self.checks) and all(c.ok for c in self.checks)

    @property
    def cost_rub(self) -> float:
        return round(sum(e.cost_rub for e in self.gateway), 4)

    def to_dict(self) -> dict:
        return {
            "number": self.number,
            "outcome": self.outcome,
            "code": self.code,
            "feedback_in": self.feedback_in,
            "checks": [c.to_dict() for c in self.checks],
            "checks_ok": self.checks_ok,
            "security": self.security.to_dict() if self.security else None,
            "gateway": [e.to_dict() for e in self.gateway],
            "cost_rub": self.cost_rub,
            "error": self.error,
        }


@dataclass
class LoopRun:
    """Прогон одной задачи: все попытки, итог и что на них потрачено."""

    task_id: str
    title: str = ""
    status: str = "failed"
    attempts: list[LoopAttempt] = field(default_factory=list)
    artifact: str = ""
    traps: tuple[str, ...] = ()
    # Почему цикл остановился и отдал задачу человеку (пусто, если дошёл сам).
    escalation: str = ""

    @property
    def accepted(self) -> LoopAttempt | None:
        """Попытка, чей код приняли: по ней смотрят, что в итоге ушло в артефакт."""
        for attempt in self.attempts:
            if attempt.outcome in ("accepted", "accepted_with_warnings"):
                return attempt
        return None

    @property
    def cost_rub(self) -> float:
        return round(sum(a.cost_rub for a in self.attempts), 4)

    @property
    def llm_calls(self) -> int:
        return sum(len(a.gateway) for a in self.attempts)

    def caught_by_security(self) -> list[str]:
        """Правила, по которым security review хоть раз вернул код на доработку."""
        rules: set[str] = set()
        for attempt in self.attempts:
            if attempt.security:
                rules.update(f.rule for f in attempt.security.blocking)
        return sorted(rules)

    def warned_by_security(self) -> list[str]:
        """Правила уровня Medium/Low, отмеченные за прогон: их пропустили осознанно."""
        rules: set[str] = set()
        for attempt in self.attempts:
            if attempt.security:
                rules.update(f.rule for f in attempt.security.warnings)
        return sorted(rules)

    def caught_by_gateway(self) -> list[str]:
        """Виды секретов и находок шлюза за весь прогон — вход и выход вместе."""
        kinds: set[str] = set()
        for attempt in self.attempts:
            for event in attempt.gateway:
                kinds.update(event.input_kinds)
                kinds.update(event.output_kinds)
        return sorted(kinds)

    def missed_traps(self) -> list[str]:
        """Ловушки задачи, по которым за весь прогон не было ни одной находки.

        Читать буквально «оба слоя проглядели» нельзя: сюда же попадают ловушки, в которые модель
        просто не наступила. Что именно произошло — видно только по коду попыток.
        """
        seen = set(self.caught_by_security()) | set(self.warned_by_security())
        return [trap for trap in self.traps if trap not in seen]

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "status": self.status,
            "attempts": [a.to_dict() for a in self.attempts],
            "attempts_count": len(self.attempts),
            "artifact": self.artifact,
            "escalation": self.escalation,
            "traps": list(self.traps),
            "caught_by_security": self.caught_by_security(),
            "warned_by_security": self.warned_by_security(),
            "caught_by_gateway": self.caught_by_gateway(),
            "missed_traps": self.missed_traps(),
            "llm_calls": self.llm_calls,
            "cost_rub": self.cost_rub,
        }
