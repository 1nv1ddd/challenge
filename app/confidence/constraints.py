"""Constraint-based слой: разбор формата решения и логические инварианты триажа."""

from __future__ import annotations

import json
import re

from ..agent_constants import (
    TRIAGE_ACTIONS,
    TRIAGE_CATEGORIES,
    TRIAGE_PRIORITIES,
    TRIAGE_REASON_MAX_LEN,
)
from ..mcp_tool_parse import _extract_json_candidate
from .schema import Decision

# Маркеры «дорогих» тем: деньги, потеря данных, угон доступа. Их нельзя закрывать автоматом.
_RISK_RE = re.compile(
    r"списал|списан|списыв|двойн\w*\s+оплат|оплат|платеж|платёж|возврат|рефанд|refund|"
    r"деньг|рубл|\bруб\b|карт[аыуе]\b|счёт|счет|"
    r"пропал|исчез|удалил|удалён|удален|потер|данные\s+пропа|бэкап|backup|"
    r"взлом|утечк|доступ\s+к\s+аккаунт|не\s+мой\s+вход|чужой\s+вход",
    re.IGNORECASE,
)
# Инъекция в тексте обращения: не инвариант, но признак, который поднимаем в нарушения self-check.
_INJECTION_RE = re.compile(
    r"игнорируй\s+(все\s+)?(предыдущие\s+)?(правила|инструкции)|ignore\s+(all\s+)?previous|"
    r"ты\s+обязан\s+(закрыть|вернуть)|закрой\s+(этот\s+)?тикет|system\s*:|"
    r"верни\s+json\s+с\s+action",
    re.IGNORECASE,
)
_BALANCED_JSON_RE = re.compile(r"\{[\s\S]*\}")


def extract_json_object(raw: str) -> str | None:
    """Кандидат JSON: сначала общий парсер MCP-ответов, потом первый {...} в прозе."""
    if candidate := _extract_json_candidate(raw):
        return candidate
    if m := _BALANCED_JSON_RE.search(raw or ""):
        return m.group(0)
    return None


def parse_decision(raw: str) -> Decision:
    """Проверка формата: строгий JSON и только допустимые значения. Иначе ValueError."""
    candidate = extract_json_object(raw)
    if candidate is None:
        raise ValueError("формат: в ответе нет JSON-объекта")
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"формат: JSON не разбирается ({exc.msg})") from exc
    if not isinstance(data, dict):
        raise ValueError("формат: ожидался JSON-объект")

    category = str(data.get("category") or "").strip().lower()
    priority = str(data.get("priority") or "").strip().lower()
    action = str(data.get("action") or "").strip().lower()
    reason = str(data.get("reason") or "").strip()
    raw_missing = data.get("missing")

    if category not in TRIAGE_CATEGORIES:
        raise ValueError(f"значение: category={category!r} вне списка {list(TRIAGE_CATEGORIES)}")
    if priority not in TRIAGE_PRIORITIES:
        raise ValueError(f"значение: priority={priority!r} вне списка {list(TRIAGE_PRIORITIES)}")
    if action not in TRIAGE_ACTIONS:
        raise ValueError(f"значение: action={action!r} вне списка {list(TRIAGE_ACTIONS)}")
    if not reason:
        raise ValueError("формат: пустое поле reason")
    if len(reason) > TRIAGE_REASON_MAX_LEN:
        raise ValueError(f"формат: reason длиннее {TRIAGE_REASON_MAX_LEN} символов")
    if raw_missing is None:
        missing: list[str] = []
    elif isinstance(raw_missing, list):
        missing = [str(item).strip() for item in raw_missing if str(item).strip()]
    else:
        raise ValueError("формат: поле missing должно быть списком строк")

    return Decision(
        category=category, priority=priority, action=action, reason=reason, missing=missing
    )


def has_risk_markers(text: str) -> bool:
    """Есть ли в обращении темы, где автоматическое close/auto_reply недопустимо."""
    return bool(_RISK_RE.search(text or ""))


def has_injection_markers(text: str) -> bool:
    """Похоже ли обращение на попытку управлять триажем через собственный текст."""
    return bool(_INJECTION_RE.search(text or ""))


def check_invariants(decision: Decision, text: str) -> list[str]:
    """Логические инварианты решения. Пустой список = решение непротиворечиво."""
    violations: list[str] = []
    if decision.priority == "critical" and decision.action != "escalate":
        violations.append(
            f"инвариант: priority=critical требует action=escalate, получено {decision.action}"
        )
    if decision.category == "data_loss" and decision.priority not in ("high", "critical"):
        violations.append(
            f"инвариант: category=data_loss требует priority high|critical, "
            f"получено {decision.priority}"
        )
    if decision.action == "request_info" and not decision.missing:
        violations.append("инвариант: action=request_info требует непустой список missing")
    if decision.action != "request_info" and decision.missing:
        violations.append("инвариант: missing заполняется только для action=request_info")
    if has_risk_markers(text) and decision.action in ("close", "auto_reply"):
        violations.append(
            f"инвариант: обращение про деньги/данные/доступ нельзя обрабатывать "
            f"автоматически ({decision.action})"
        )
    if has_injection_markers(text) and decision.action in ("close", "auto_reply"):
        violations.append(
            "инвариант: в тексте обращения есть инструкции для триажа — "
            "автоматическое действие запрещено"
        )
    return violations
