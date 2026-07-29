"""Эвристики уверенности: самооценка, согласие сэмплов, длина, обрыв, hedging, отказ."""

from __future__ import annotations

import re

from ..agent_constants import (
    ROUTING_CONSISTENCY_JACCARD,
    ROUTING_DISAGREE_PENALTY,
    ROUTING_ESCALATE_BELOW,
    ROUTING_HEDGE_PENALTY,
    ROUTING_MIN_ANSWER_CHARS,
    ROUTING_NO_CONFIDENCE_BASE,
    ROUTING_REFUSAL_PENALTY,
    ROUTING_SHORT_PENALTY,
    ROUTING_TRUNCATED_PENALTY,
)
from .preroute import classify_question
from .schema import Assessment

# Строка контракта: `CONFIDENCE: 0.82`, в том числе в **жирном**, `в бэктиках` и с запятой.
_CONFIDENCE_LINE = re.compile(
    r"^[\s>*_`#-]*confidence[\s*_`]*[:=][\s*_`]*([01](?:[.,]\d+)?|[.,]\d+)[\s*_`.]*$",
    re.I | re.M,
)
_HEDGE_MARKERS = re.compile(
    r"\bне\s+уверен|\bзатрудня\w+\s+ответить|\bскорее\s+всего\b|\bпредположительно\b"
    r"|\bесли\s+я\s+не\s+ошибаюсь\b|\bточно\s+(не\s+)?сказать\s+нельзя\b|\bвозможно,?\s+что\b"
    r"|\bпо-?видимому\b|\bкажется,\s+что\b",
    re.I,
)
_REFUSAL_MARKERS = re.compile(
    r"\bне\s+могу\s+(ответить|помочь|определить|сказать)\b|\bне\s+знаю\b"
    r"|\bнедостаточно\s+(данных|информации)\b|\bнет\s+(данных|информации)\b"
    r"|\bя\s+не\s+располагаю\b",
    re.I,
)
# Чем ответ может законно заканчиваться. Проверяем только когда строки CONFIDENCE нет:
# если модель успела её вывести, ответ точно не оборван.
_TERMINAL_TAIL = re.compile(r"[.!?…»)\]}\"'`%\d]\s*$")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
_WORD = re.compile(r"[^\W\d_]+", re.U)
# Разряды в числах пишут по-разному: «12 300», «12300», «12,300» — считаем это одним числом.
_DIGIT_GROUPS = re.compile(r"(?<=\d)[\s ,](?=\d{3}\b)")


def split_confidence(answer: str) -> tuple[str, float | None]:
    """Отделяет самооценку от текста ответа: (ответ без служебной строки, оценка или None)."""
    text = answer or ""
    matches = list(_CONFIDENCE_LINE.finditer(text))
    if not matches:
        return text.strip(), None
    last = matches[-1]
    try:
        value = float(last.group(1).replace(",", "."))
    except ValueError:
        return text.strip(), None
    body = (text[: last.start()] + text[last.end() :]).strip()
    return body, max(0.0, min(1.0, value))


def _normalize_number(raw: str) -> str:
    value = raw.replace(",", ".")
    return value.rstrip("0").rstrip(".") if "." in value else value


def _final_number(text: str) -> str | None:
    """Последнее число в ответе — обычно это и есть итог, а не промежуточный шаг расчёта."""
    found = _NUMBER.findall(_DIGIT_GROUPS.sub("", text))
    return _normalize_number(found[-1]) if found else None


def _tokens(text: str) -> set[str]:
    normalized = _DIGIT_GROUPS.sub("", text)
    words = {w.lower() for w in _WORD.findall(normalized)}
    return words | {_normalize_number(n) for n in _NUMBER.findall(normalized)}


def _jaccard(first: str, second: str) -> float:
    a, b = _tokens(first), _tokens(second)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def answers_agree(first: str, second: str) -> tuple[bool, str]:
    """Сошлись ли два ответа дешёвой модели: сначала по итоговому числу, затем по составу текста."""
    num_a, num_b = _final_number(first), _final_number(second)
    if num_a is not None and num_b is not None:
        # Числовой ответ: решает итог, а не многословность — одна выборка может показывать расчёт.
        if num_a != num_b:
            return False, f"итог разошёлся: {num_a} против {num_b}"
        return True, ""
    overlap = _jaccard(first, second)
    if overlap < ROUTING_CONSISTENCY_JACCARD:
        return False, f"ответы совпадают лишь на {round(overlap, 2)} по составу"
    return True, ""


def assess_answer(question: str, answer: str, *, peers: list[str] | None = None) -> Assessment:
    """Уверенность в ответе дешёвой модели и вердикт «принять или эскалировать».

    `peers` — другие сэмплы той же дешёвой модели: расхождение между ними ловит ошибки,
    которых самооценка не видит (мелкие модели ставят себе 1.0 и на неверных ответах).
    """
    body, self_reported = split_confidence(answer)
    signals: list[str] = []
    penalty = 0.0

    for peer in peers or []:
        agree, why = answers_agree(body, split_confidence(peer)[0])
        if not agree:
            penalty += ROUTING_DISAGREE_PENALTY
            signals.append(f"сэмплы дешёвой модели разошлись — {why}")
            break

    base = self_reported
    if base is None:
        base = ROUTING_NO_CONFIDENCE_BASE
        signals.append("нет строки CONFIDENCE — контракт ответа нарушен")
        if body and not _TERMINAL_TAIL.search(body):
            penalty += ROUTING_TRUNCATED_PENALTY
            signals.append("ответ оборван на полуслове")

    if not body:
        penalty += ROUTING_REFUSAL_PENALTY
        signals.append("пустой ответ")
    elif len(body) < ROUTING_MIN_ANSWER_CHARS and classify_question(question).score > 0:
        # На простой вопрос короткий ответ — норма («Au»), подозрителен он только там,
        # где вопрос требовал развёрнутого разбора.
        penalty += ROUTING_SHORT_PENALTY
        signals.append(f"слишком короткий ответ ({len(body)} символов) на непростой вопрос")

    if _REFUSAL_MARKERS.search(body):
        penalty += ROUTING_REFUSAL_PENALTY
        signals.append("модель признаёт, что не знает ответа")
    elif _HEDGE_MARKERS.search(body):
        penalty += ROUTING_HEDGE_PENALTY
        signals.append("формулировки неуверенности в тексте")

    confidence = round(max(0.0, min(1.0, base - penalty)), 2)
    escalate = confidence < ROUTING_ESCALATE_BELOW
    if escalate:
        head = signals[0] if signals else f"самооценка {self_reported}"
        reason = f"confidence {confidence} < {ROUTING_ESCALATE_BELOW}: {head}"
    else:
        reason = f"confidence {confidence} — ответа дешёвой модели достаточно"
    return Assessment(
        confidence=confidence,
        self_reported=self_reported,
        signals=signals,
        escalate=escalate,
        reason=reason,
    )
