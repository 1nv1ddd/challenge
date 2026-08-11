"""Разбор ответа security-ревьюера и фидбек генератору по блокирующим находкам."""

from __future__ import annotations

import json

from ..confidence.constraints import extract_json_object
from .schema import SecurityFinding, SecurityVerdict


def parse_verdict(raw: str) -> SecurityVerdict:
    """Строгий разбор JSON-вердикта. Битый формат — не «чисто», а вердикт с ошибкой.

    Это принципиально: если ревьюер ответил мусором, код нельзя считать проверенным. Ошибка
    разбора возвращает цикл на генерацию так же, как блокирующая находка.
    """
    candidate = extract_json_object(raw)
    if candidate is None:
        return SecurityVerdict(error="формат: в ответе ревьюера нет JSON-объекта")
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return SecurityVerdict(error=f"формат: JSON не разбирается ({exc.msg})")
    if not isinstance(data, dict):
        return SecurityVerdict(error="формат: ожидался JSON-объект")

    raw_findings = data.get("findings")
    if raw_findings is None:
        return SecurityVerdict(error="формат: нет поля findings")
    if not isinstance(raw_findings, list):
        return SecurityVerdict(error="формат: findings должен быть списком")

    findings: list[SecurityFinding] = []
    for item in raw_findings:
        if not isinstance(item, dict):
            return SecurityVerdict(error="формат: элемент findings должен быть объектом")
        try:
            findings.append(SecurityFinding.from_dict(item))
        except ValueError as exc:
            return SecurityVerdict(error=str(exc))
    return SecurityVerdict(findings=findings, summary=str(data.get("summary") or "").strip())


def security_feedback(verdict: SecurityVerdict) -> str:
    """Что вернуть генератору: перечень блокирующих находок со строками и способом починки."""
    if verdict.error:
        return (
            "Security review не дал разбираемого вердикта "
            f"({verdict.error}). Перепроверь модуль сам и верни исправленную версию: секреты — "
            "из окружения, только https, в логи ничего чувствительного."
        )
    lines = ["Security review вернул блокирующие находки. Исправь:", ""]
    for finding in verdict.blocking:
        where = f" в строке {finding.line}" if finding.line else ""
        fix = f" — {finding.fix}" if finding.fix else ""
        lines.append(f"- [{finding.severity.upper()}] {finding.title}{where}{fix}")
    if verdict.warnings:
        lines += ["", "Заодно, если не ломает тесты:"]
        lines += [f"- [{f.severity}] {f.title}" for f in verdict.warnings]
    return "\n".join(lines)
