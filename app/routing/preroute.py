"""Пре-маршрутизация: оценка сложности запроса по тексту, без единого вызова LLM."""

from __future__ import annotations

import re

from ..agent_constants import ROUTING_LONG_QUESTION_CHARS, ROUTING_PREROUTE_HARD_SCORE
from .schema import PreRoute

# Маркеры, каждый из которых даёт +1 к сложности. Один маркер сам по себе на сильную
# модель не отправляет — порог `ROUTING_PREROUTE_HARD_SCORE` требует совпадения нескольких.
_HARD_MARKERS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("код", re.compile(r"```|\bнапиши\s+(функцию|класс|скрипт|код)|\bотрефактор|\bдебаж", re.I)),
    ("многошаговость", re.compile(r"\bпошагов|\bпо\s+шагам|\bсначала\b.*\bзатем\b", re.I | re.S)),
    ("доказательство", re.compile(r"\bдокажи|\bобоснуй|\bвыведи\s+формул|\bпочему\s+именно\b", re.I)),
    ("проектирование", re.compile(r"\bспроектируй|\bархитектур|\btrade-?off|\bкомпромисс", re.I)),
    ("сравнение", re.compile(r"\bсравни\b|\bпроанализируй|\bоптимизируй|\bплюсы\s+и\s+минусы", re.I)),
    ("развёрнутый ответ", re.compile(r"\bподробно\b|\bразвёрнут|\bразвернут|\bдетально\b", re.I)),
    ("несколько ограничений", re.compile(r"\bпри\s+условии\b|\bучитыва\w+\s+что\b|\bно\s+так,?\s+чтобы\b", re.I)),
)
_CALC_VERBS = re.compile(r"\bпосчита|\bвычисли|\bсколько\s+(будет|получ)|\bрассчита", re.I)
_NUMBER = re.compile(r"\d+")


def _has_multi_step_math(text: str) -> bool:
    """Счётная задача в несколько действий: глагол вычисления плюс три и более числа."""
    return bool(_CALC_VERBS.search(text)) and len(_NUMBER.findall(text)) >= 3


def classify_question(question: str) -> PreRoute:
    """Сложность запроса: `large` — если сигналов набралось на порог, иначе `small`."""
    text = (question or "").strip()
    reasons: list[str] = []

    for name, pattern in _HARD_MARKERS:
        if pattern.search(text):
            reasons.append(name)
    if _has_multi_step_math(text):
        reasons.append("расчёт в несколько действий")
    if len(text) > ROUTING_LONG_QUESTION_CHARS:
        reasons.append(f"длинный запрос ({len(text)} символов)")
    if text.count("?") >= 2:
        reasons.append("несколько вопросов в одном запросе")

    score = len(reasons)
    tier = "large" if score >= ROUTING_PREROUTE_HARD_SCORE else "small"
    return PreRoute(tier=tier, score=score, reasons=reasons)
