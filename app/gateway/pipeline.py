"""Прокси-проход шлюза: лимит → input guard → модель → output guard → аудит."""

from __future__ import annotations

import uuid

import httpx

from ..agent_constants import (
    GATEWAY_DEFAULT_MODE,
    GATEWAY_MAX_PROMPT_CHARS,
    GATEWAY_MODEL,
    GATEWAY_TEMPERATURE,
)
from ..confidence.inference import complete
from ..providers import AIProvider, Message
from .audit import log_request
from .cost import usage_of
from .detectors import guard_input
from .output_guard import guard_output
from .prompts import gateway_messages, system_message
from .ratelimit import RateLimiter
from .ratelimit import limiter as default_limiter
from .schema import GatewayResult


def _new_request_id() -> str:
    return uuid.uuid4().hex[:12]


async def proxy_chat(
    provider: AIProvider,
    prompt: str,
    *,
    model: str = GATEWAY_MODEL,
    temperature: float = GATEWAY_TEMPERATURE,
    mode: str = GATEWAY_DEFAULT_MODE,
    client_ip: str = "",
    limiter: RateLimiter | None = None,
    audit: bool = True,
    enforce_output: bool = True,
    system: Message | None = None,
    secret: str = "",
) -> GatewayResult:
    """Один запрос через шлюз. Возвращает результат в любом случае: отказ — тоже результат.

    Проверки идут до вызова модели не только ради безопасности: заблокированный запрос не тратит
    ни токенов, ни денег, и это видно в аудите полем `llm_called`.

    `system` и `secret` — для CTF-Стража Дня 15: переопределённый системный промпт с секретом
    внутри и само охраняемое значение, утечку которого output guard блокирует целиком.
    """
    if not (prompt or "").strip():
        raise ValueError("Пустой промпт: нечего проксировать.")
    result = GatewayResult(
        request_id=_new_request_id(),
        client_ip=client_ip,
        model=model,
        mode=mode,
        output_enforced=enforce_output,
    )
    gate = limiter or default_limiter
    result.rate = gate.check(client_ip or "unknown")
    if not result.rate.allowed:
        result.status = "rate_limited"
        result.answer = (
            f"Слишком много запросов: лимит {result.rate.limit} в минуту с одного адреса. "
            f"Повторите через {result.rate.retry_after_sec} с."
        )
        return _finish(result, audit)

    if len(prompt) > GATEWAY_MAX_PROMPT_CHARS:
        result.status = "too_long"
        result.answer = (
            f"Промпт длиннее {GATEWAY_MAX_PROMPT_CHARS} символов — шлюз такой запрос не пропускает."
        )
        return _finish(result, audit)

    result.input = guard_input(prompt, mode)
    if result.input.blocked:
        result.status = "blocked_input"
        result.answer = result.input.warning
        return _finish(result, audit)

    system = system or system_message()
    try:
        call = await complete(
            provider, model, gateway_messages(result.input.prompt, system), temperature
        )
    except (ValueError, LookupError, OSError, httpx.HTTPError) as exc:
        result.status = "error"
        result.error = f"{type(exc).__name__}: {exc}"
        result.answer = f"Ошибка вызова модели: {result.error}"
        return _finish(result, audit)

    result.time_ms = call.time_ms
    usage = usage_of(
        model,
        call.prompt_tokens,
        call.completion_tokens,
        prompt_text=result.input.prompt,
        answer_text=call.text,
    )
    result.prompt_tokens = usage.prompt_tokens
    result.completion_tokens = usage.completion_tokens
    result.tokens_estimated = usage.estimated
    result.cost_rub = usage.cost_rub

    result.output = guard_output(
        call.text, system.content, enforce=enforce_output, secrets=(secret,) if secret else ()
    )
    result.answer = result.output.answer
    result.status = "blocked_output" if result.output.blocked else "ok"
    return _finish(result, audit)


def _finish(result: GatewayResult, audit: bool) -> GatewayResult:
    """Единая точка выхода: что бы ни случилось, запрос попадает в аудит."""
    if audit:
        log_request(result)
    return result
