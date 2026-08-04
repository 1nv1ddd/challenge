"""DTO ред-тима: атака из корпуса, вердикт одного прогона и сводка по прогону корпуса."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..agent_constants import SECURITY_TARGETS, SECURITY_TECHNIQUES, SECURITY_VECTORS


@dataclass(frozen=True)
class Attack:
    """Одна инъекция: чем бьём, куда и по каким признакам считаем, что пробило."""

    id: str
    title: str
    vector: str
    technique: str
    target: str
    prompt: str
    source: str = ""
    # Внешний «документ» для indirect: письмо, тикет, отрывок из базы. Уходит в контекст как данные.
    document: str = ""
    # Строки, наличие которых в ответе означает, что атака удалась (регистр не важен).
    success_markers: tuple[str, ...] = ()
    # Строки отказа: нужны для отчёта, сам вердикт по ним не выносится.
    refusal_markers: tuple[str, ...] = ()
    why_works: str = ""
    defense: str = ""

    @classmethod
    def from_dict(cls, raw: dict) -> Attack:
        """Строка корпуса → Attack. Неизвестный вектор/техника/цель — ошибка, а не тихий пропуск."""
        attack_id = str(raw.get("id") or "").strip()
        if not attack_id:
            raise ValueError("У атаки нет id.")
        vector = str(raw.get("vector") or "").strip()
        if vector not in SECURITY_VECTORS:
            raise ValueError(f"{attack_id}: неизвестный вектор {vector!r}.")
        technique = str(raw.get("technique") or "").strip()
        if technique not in SECURITY_TECHNIQUES:
            raise ValueError(f"{attack_id}: неизвестная техника {technique!r}.")
        target = str(raw.get("target") or "").strip()
        if target not in SECURITY_TARGETS:
            raise ValueError(f"{attack_id}: неизвестная цель {target!r}.")
        prompt = str(raw.get("prompt") or "").strip()
        if not prompt:
            raise ValueError(f"{attack_id}: пустой prompt.")
        return cls(
            id=attack_id,
            title=str(raw.get("title") or attack_id).strip(),
            vector=vector,
            technique=technique,
            target=target,
            prompt=prompt,
            source=str(raw.get("source") or "").strip(),
            document=str(raw.get("document") or "").strip(),
            success_markers=tuple(str(m) for m in raw.get("success_markers") or ()),
            refusal_markers=tuple(str(m) for m in raw.get("refusal_markers") or ()),
            why_works=str(raw.get("why_works") or "").strip(),
            defense=str(raw.get("defense") or "").strip(),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "vector": self.vector,
            "technique": self.technique,
            "target": self.target,
            "prompt": self.prompt,
            "source": self.source,
            "document": self.document,
            "why_works": self.why_works,
            "defense": self.defense,
        }


@dataclass
class AttackVerdict:
    """Результат одного прогона: пробило или нет, по каким сигналам и во что обошлось."""

    attack_id: str
    target: str
    version: str
    vector: str = ""
    technique: str = ""
    reply: str = ""
    broken: bool = False
    signals: list[str] = field(default_factory=list)
    refused: bool = False
    error: str | None = None
    model: str = ""
    time_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_rub: float = 0.0

    def to_dict(self) -> dict:
        return {
            "attack_id": self.attack_id,
            "target": self.target,
            "version": self.version,
            "vector": self.vector,
            "technique": self.technique,
            "reply": self.reply,
            "broken": self.broken,
            "signals": list(self.signals),
            "refused": self.refused,
            "error": self.error,
            "model": self.model,
            "time_ms": self.time_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_rub": self.cost_rub,
        }


@dataclass
class RedteamRun:
    """Прогон набора атак по одной версии промпта."""

    version: str
    verdicts: list[AttackVerdict] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.verdicts)

    @property
    def broken(self) -> int:
        return sum(1 for v in self.verdicts if v.broken)

    @property
    def held(self) -> int:
        return sum(1 for v in self.verdicts if not v.broken and v.error is None)

    @property
    def errors(self) -> int:
        return sum(1 for v in self.verdicts if v.error is not None)

    @property
    def cost_rub(self) -> float:
        return round(sum(v.cost_rub for v in self.verdicts), 4)

    def break_rate(self) -> float:
        """Доля пробитых атак. Считаем от всех прогонов, ошибки вызова — не пробой."""
        return round(self.broken / self.total, 3) if self.total else 0.0

    def by_vector(self) -> dict[str, dict[str, int]]:
        """Сводка «вектор → сколько пробило из скольких» — чтобы видеть, что держится хуже."""
        return self._group("vector")

    def by_technique(self) -> dict[str, dict[str, int]]:
        """То же самое в разрезе техники: ролевая игра, override, extraction, отравление контекста."""
        return self._group("technique")

    def _group(self, key: str) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for verdict in self.verdicts:
            bucket = out.setdefault(getattr(verdict, key) or "?", {"total": 0, "broken": 0})
            bucket["total"] += 1
            if verdict.broken:
                bucket["broken"] += 1
        return out

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "total": self.total,
            "broken": self.broken,
            "held": self.held,
            "errors": self.errors,
            "break_rate": self.break_rate(),
            "cost_rub": self.cost_rub,
            "verdicts": [v.to_dict() for v in self.verdicts],
        }
