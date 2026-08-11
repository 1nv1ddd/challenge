"""Cost tracking шлюза: токены запроса и ответа плюс их цена по прайсу модели."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..agent_constants import GATEWAY_CHARS_PER_TOKEN
from ..routing.pricing import cost_rub_model


@dataclass(frozen=True)
class Usage:
    """Расход одного вызова. estimated=True — провайдер usage не вернул, токены оценены."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_rub: float = 0.0
    estimated: bool = False

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "cost_rub": self.cost_rub,
            "estimated": self.estimated,
        }


def estimate_tokens(text: str, chars_per_token: float = GATEWAY_CHARS_PER_TOKEN) -> int:
    """Грубая оценка токенов по длине текста — когда провайдер не прислал usage.

    Общая для проекта: тем же расчётом агент прикидывает заполнение окна контекста, только с
    другим делителем (см. `AgentProviderUtilsMixin._estimate_tokens_text`).
    """
    return max(1, math.ceil(len(text or "") / chars_per_token)) if text else 0


def usage_of(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    *,
    prompt_text: str = "",
    answer_text: str = "",
) -> Usage:
    """Расход вызова: usage провайдера, а при его отсутствии — оценка по длине текстов.

    Считать всё равно нужно: без цены на каждом запросе шлюз не отвечает на вопрос «кто сжёг бюджет».
    """
    estimated = not (prompt_tokens or completion_tokens)
    if estimated:
        prompt_tokens = estimate_tokens(prompt_text)
        completion_tokens = estimate_tokens(answer_text)
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_rub=cost_rub_model(model, prompt_tokens, completion_tokens),
        estimated=estimated,
    )
