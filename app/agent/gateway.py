"""Миксин агента: прокси-вызов через LLM Gateway и офлайн-прогон корпуса кейсов (День 13)."""

from __future__ import annotations

from ..agent_constants import (
    CTF_AUDIT_PATH,
    CTF_MODE,
    CTF_MODEL,
    CTF_RATE_LIMIT_PER_MIN,
    CTF_RATE_WINDOW_SEC,
    GATEWAY_DEFAULT_MODE,
    GATEWAY_MODEL,
    GATEWAY_TEMPERATURE,
)
from ..ctf import load_secret, oracle_system
from ..gateway import (
    CorpusRun,
    GatewayResult,
    load_cases,
    proxy_chat,
    run_corpus,
    select_cases,
)
from ..gateway.audit import log_request
from ..gateway.ratelimit import RateLimiter

# Отдельный лимитер CTF-стенда: свой счётчик, чтобы атака не выедала лимит боевого шлюза.
_ctf_limiter = RateLimiter(limit=CTF_RATE_LIMIT_PER_MIN, window_sec=CTF_RATE_WINDOW_SEC)


class AgentGatewayMixin:
    async def gateway_request(
        self,
        provider_name: str,
        prompt: str,
        *,
        mode: str = GATEWAY_DEFAULT_MODE,
        model: str = GATEWAY_MODEL,
        temperature: float = GATEWAY_TEMPERATURE,
        client_ip: str = "",
    ) -> GatewayResult:
        """Пропускает промпт через шлюз: guard'ы, лимит, вызов модели и аудит."""
        provider = self._validate_provider(provider_name)
        await self._validate_model(provider, provider_name, model)
        return await proxy_chat(
            provider,
            prompt,
            model=model,
            temperature=self._normalize_temperature(temperature),
            mode=mode,
            client_ip=client_ip,
        )

    async def arena_request(
        self,
        provider_name: str,
        prompt: str,
        *,
        model: str = CTF_MODEL,
        client_ip: str = "",
    ) -> GatewayResult:
        """Запрос к Оракулу арены: тот же проход шлюза, но секрет в промпте, а его утечка — блок.

        Боевой аудит шлюза выключен: атаки на Арену пишем отдельно (`data/ctf_audit.jsonl`),
        чтобы не мешать основному логу. Лимитер тоже свой.
        """
        provider = self._validate_provider(provider_name)
        await self._validate_model(provider, provider_name, model)
        secret = load_secret()
        result = await proxy_chat(
            provider,
            prompt,
            model=model,
            temperature=GATEWAY_TEMPERATURE,
            mode=CTF_MODE,
            client_ip=client_ip,
            limiter=_ctf_limiter,
            audit=False,
            system=oracle_system(secret),
            secret=secret,
        )
        log_request(result, path=CTF_AUDIT_PATH)
        return result

    @staticmethod
    def gateway_selftest(ids: tuple[str, ...] = ()) -> CorpusRun:
        """Прогон корпуса кейсов по детекторам — офлайн, без провайдера и без денег."""
        cases = select_cases(load_cases(), ids)
        if not cases:
            raise ValueError("Под фильтр не попал ни один кейс корпуса.")
        return run_corpus(cases)
