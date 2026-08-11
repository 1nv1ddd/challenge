"""Гейт уровня 1: соседи из банка → метка, score и статус OK/UNSURE с причиной эскалации."""

from __future__ import annotations

from ..agent_constants import (
    MICRO_FALLBACK_LABEL,
    MICRO_MIN_CHARS,
    MICRO_THRESHOLDS,
    MICRO_W_MARGIN,
    MICRO_W_SIM,
    MICRO_W_VOTES,
)
from .schema import MicroVerdict, Neighbor

# Причины, по которым micro-model отказывается решать сама (в порядке проверки).
ESCALATE_SHORT = "short_input"
ESCALATE_NO_NEIGHBOR = "no_close_neighbor"
ESCALATE_FALLBACK_LABEL = "fallback_label"
ESCALATE_LOW_MARGIN = "low_margin"
ESCALATE_LOW_SCORE = "low_score"


def thresholds(backend: str) -> dict:
    """Пороги бэкенда: косинус эмбеддингов и косинус tfidf живут в разных шкалах."""
    if backend not in MICRO_THRESHOLDS:
        raise ValueError(f"Нет порогов для бэкенда «{backend}».")
    return MICRO_THRESHOLDS[backend]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _label_weights(neighbors: list[Neighbor]) -> dict[str, float]:
    """Вес метки среди соседей — сумма их близостей: далёкий сосед голосует слабее ближнего."""
    weights: dict[str, float] = {}
    for neighbor in neighbors:
        weights[neighbor.label] = weights.get(neighbor.label, 0.0) + max(0.0, neighbor.similarity)
    return weights


def _margin(label: str, neighbors: list[Neighbor]) -> float:
    """Отрыв: лучший сосед победившей метки минус лучший сосед любой другой."""
    own = max((n.similarity for n in neighbors if n.label == label), default=0.0)
    other = max((n.similarity for n in neighbors if n.label != label), default=0.0)
    return own - other


def judge(
    backend: str,
    text: str,
    neighbors: list[Neighbor],
    *,
    time_ms: int = 0,
    limits: dict | None = None,
) -> MicroVerdict:
    """Собирает вердикт уровня 1: три сигнала складываются в score, пороги дают статус.

    `limits` переопределяет пороги бэкенда — этим пользуется подбор порогов по сетке.
    """
    limits = limits or thresholds(backend)
    if not neighbors:
        return MicroVerdict(
            backend=backend,
            label=MICRO_FALLBACK_LABEL,
            score=0.0,
            status="UNSURE",
            escalate_reason=ESCALATE_NO_NEIGHBOR,
            time_ms=time_ms,
        )

    weights = _label_weights(neighbors)
    total = sum(weights.values())
    label = max(weights, key=lambda key: weights[key])
    votes = weights[label] / total if total > 0 else 0.0
    top_similarity = max(n.similarity for n in neighbors)
    margin = _margin(label, neighbors)

    sim_norm = _clamp01(
        (top_similarity - limits["sim_floor"]) / max(1e-6, 1.0 - limits["sim_floor"])
    )
    margin_norm = _clamp01(margin / limits["margin_full"])
    score = MICRO_W_SIM * sim_norm + MICRO_W_MARGIN * margin_norm + MICRO_W_VOTES * votes

    if len((text or "").strip()) < MICRO_MIN_CHARS:
        reason = ESCALATE_SHORT
    elif top_similarity < limits["sim_floor"]:
        reason = ESCALATE_NO_NEIGHBOR
    elif label == MICRO_FALLBACK_LABEL:
        # «other» — класс-помойка: уверенность здесь ничего не значит, решает большая модель.
        reason = ESCALATE_FALLBACK_LABEL
    elif margin < limits["margin_min"]:
        reason = ESCALATE_LOW_MARGIN
    elif score < limits["accept"]:
        reason = ESCALATE_LOW_SCORE
    else:
        reason = None

    return MicroVerdict(
        backend=backend,
        label=label,
        score=round(score, 4),
        status="OK" if reason is None else "UNSURE",
        escalate_reason=reason,
        top_similarity=top_similarity,
        margin=margin,
        votes=votes,
        neighbors=neighbors,
        time_ms=time_ms,
    )
