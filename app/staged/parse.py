"""Строгий разбор компактных ответов этапов: поля заявки, enum-решение, ответ клиенту."""

from __future__ import annotations

import re
from datetime import date

from ..agent_constants import (
    INTAKE_DECISIONS,
    INTAKE_FIELD_KEYS,
    INTAKE_NUMERIC_FIELDS,
    INTAKE_PAYMENTS,
    INTAKE_PRODUCTS,
    INTAKE_REASONS,
    INTAKE_REGIONS,
    INTAKE_REPLY_MAX_WORDS,
)
from .schema import IntakeFields, StageDecision

_FENCE_RE = re.compile(r"^\s*```[a-zA-Z]*\s*|\s*```\s*$")
_PAIR_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)$")
_SUBJECT_RE = re.compile(r"^\s*subject\s*:\s*(.+)$", re.I | re.M)
_NUMBER_RE = re.compile(r"^\d+([.,]0+)?$")
_PHONE_RE = re.compile(r"^\+7\d{10}$")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[a-z]{2,}$")
_UNKNOWN_WORDS = {"unknown", "-", "—", "", "none", "null", "n/a", "нет", "не указано"}
_ENUMS = {"product": INTAKE_PRODUCTS, "region": INTAKE_REGIONS, "payment": INTAKE_PAYMENTS}
# Мусор, который модели любят приписывать к числам вопреки формату.
_NUMBER_JUNK_RE = re.compile(r"(кг|kg|руб\.?|рублей|₽|\s| )", re.I)


def _strip_fences(raw: str) -> str:
    return _FENCE_RE.sub("", (raw or "").strip())


def parse_compact(raw: str) -> dict[str, str]:
    """Строки `ключ: значение` → dict; первое вхождение ключа побеждает, остальное игнорируется."""
    pairs: dict[str, str] = {}
    for line in _strip_fences(raw).splitlines():
        match = _PAIR_RE.match(line)
        if not match:
            continue
        key = match.group(1).lower()
        if key not in pairs:
            pairs[key] = match.group(2).strip()
    return pairs


def _is_unknown(value: str) -> bool:
    return value.strip().lower() in _UNKNOWN_WORDS


def _as_int(key: str, value: str) -> int | None:
    if _is_unknown(value):
        return None
    cleaned = _NUMBER_JUNK_RE.sub("", value)
    if not _NUMBER_RE.match(cleaned):
        raise ValueError(f"формат: поле {key} должно быть целым числом или unknown, получено «{value}»")
    return int(cleaned.split(".")[0].split(",")[0])


def _as_enum(key: str, value: str, allowed: tuple[str, ...]) -> str:
    cleaned = value.strip().lower()
    if _is_unknown(cleaned):
        return "unknown"
    if cleaned not in allowed:
        raise ValueError(
            f"формат: поле {key} вне списка {', '.join(allowed)} — получено «{value}»"
        )
    return cleaned


def _as_date(value: str) -> str:
    if _is_unknown(value):
        return "unknown"
    cleaned = value.strip()
    try:
        date.fromisoformat(cleaned)
    except ValueError as exc:
        raise ValueError(f"формат: deadline должен быть YYYY-MM-DD или unknown, получено «{value}»") from exc
    return cleaned


def _as_contact(value: str) -> str:
    if _is_unknown(value):
        return "unknown"
    cleaned = value.strip().lower()
    if _PHONE_RE.match(cleaned) or _EMAIL_RE.match(cleaned):
        return cleaned
    raise ValueError(
        f"формат: contact должен быть +7XXXXXXXXXX, email или unknown — получено «{value}»"
    )


def _as_text(value: str) -> str:
    return "unknown" if _is_unknown(value) else " ".join(value.split())


def parse_fields(raw: str) -> IntakeFields:
    """Ответ этапа 1 → нормализованные поля; любое отступление от формата — ValueError."""
    pairs = parse_compact(raw)
    absent = [key for key in INTAKE_FIELD_KEYS if key not in pairs]
    if absent:
        raise ValueError(f"формат: нет обязательных строк {', '.join(absent)}")

    values: dict[str, str | int | None] = {}
    for key in INTAKE_FIELD_KEYS:
        value = pairs[key]
        if key in INTAKE_NUMERIC_FIELDS:
            values[key] = _as_int(key, value)
        elif key in _ENUMS:
            values[key] = _as_enum(key, value, _ENUMS[key])
        elif key == "deadline":
            values[key] = _as_date(value)
        elif key == "contact":
            values[key] = _as_contact(value)
        else:
            values[key] = _as_text(value)
    return IntakeFields(**values)


def parse_decision(raw: str) -> StageDecision:
    """Ответ этапа 2 → решение; строки DECISION и REASON обязательны, MISSING — по смыслу."""
    pairs = parse_compact(raw)
    absent = [key for key in ("decision", "reason") if key not in pairs]
    if absent:
        raise ValueError(f"формат: нет обязательных строк {', '.join(k.upper() for k in absent)}")

    decision = _as_enum("DECISION", pairs["decision"], INTAKE_DECISIONS)
    reason = _as_enum("REASON", pairs["reason"], INTAKE_REASONS)
    if decision == "unknown" or reason == "unknown":
        raise ValueError("формат: DECISION и REASON не могут быть unknown")

    missing: list[str] = []
    if not _is_unknown(pairs.get("missing", "")):
        raw_items = [item.strip().lower() for item in re.split(r"[,\s]+", pairs["missing"]) if item.strip()]
        unexpected = [item for item in raw_items if item not in INTAKE_FIELD_KEYS]
        if unexpected:
            raise ValueError(f"формат: MISSING содержит неизвестные поля {', '.join(unexpected)}")
        missing = raw_items
    return StageDecision(decision=decision, reason=reason, missing=missing)


def parse_reply(raw: str) -> tuple[str, str]:
    """Ответ этапа 3 → (тема, тело письма) с проверкой лимита длины."""
    text = _strip_fences(raw)
    match = _SUBJECT_RE.search(text)
    if not match:
        raise ValueError("формат: нет строки SUBJECT с темой письма")
    subject = " ".join(match.group(1).split())
    body = text[match.end():].strip()
    body = re.sub(r"^\s*body\s*:\s*", "", body, flags=re.I).strip()
    if not body:
        raise ValueError("формат: после SUBJECT нет текста письма")
    words = len(body.split())
    if words > INTAKE_REPLY_MAX_WORDS:
        raise ValueError(f"формат: письмо длиннее {INTAKE_REPLY_MAX_WORDS} слов ({words})")
    return subject, body


def parse_monolithic(raw: str) -> tuple[IntakeFields, StageDecision, str, str]:
    """Ответ монолитного варианта: те же три блока в одном ответе — разбираем теми же парсерами."""
    fields = parse_fields(raw)
    decision = parse_decision(raw)
    subject, body = parse_reply(raw)
    return fields, decision, subject, body
