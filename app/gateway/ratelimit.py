"""Rate limit шлюза: скользящее окно запросов на один IP, в памяти процесса."""

from __future__ import annotations

import time
from collections import defaultdict, deque

from ..agent_constants import GATEWAY_RATE_LIMIT_PER_MIN, GATEWAY_RATE_WINDOW_SEC
from .schema import RateDecision


class RateLimiter:
    """Считает запросы клиента за окно. Состояние живёт в процессе — при нескольких воркерах
    лимит фактически умножается на их число; для прода это выносится в Redis, здесь важен
    сам механизм и его наблюдаемость в аудите."""

    def __init__(
        self,
        limit: int = GATEWAY_RATE_LIMIT_PER_MIN,
        window_sec: int = GATEWAY_RATE_WINDOW_SEC,
    ):
        self.limit = limit
        self.window_sec = window_sec
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, client: str, now: float | None = None) -> RateDecision:
        """Отмечает запрос клиента и говорит, пускать ли его. Отклонённый запрос окно не занимает."""
        moment = time.monotonic() if now is None else now
        hits = self._hits[client or "unknown"]
        while hits and moment - hits[0] >= self.window_sec:
            hits.popleft()
        if len(hits) >= self.limit:
            retry_after = max(1, int(self.window_sec - (moment - hits[0])) + 1)
            return RateDecision(
                allowed=False, used=len(hits), limit=self.limit, retry_after_sec=retry_after
            )
        hits.append(moment)
        return RateDecision(allowed=True, used=len(hits), limit=self.limit, retry_after_sec=0)

    def reset(self, client: str | None = None) -> None:
        """Сброс счётчиков: целиком или по одному клиенту (нужно тестам и ручной разблокировке)."""
        if client is None:
            self._hits.clear()
        else:
            self._hits.pop(client, None)


# Общий лимитер приложения: один на процесс, состояние переживает запросы.
limiter = RateLimiter()
