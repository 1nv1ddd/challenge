"""Self-check: отдельный вызов модели, который ревьюит уже принятое решение триажа."""

from __future__ import annotations

import json

from ..providers import AIProvider
from .constraints import extract_json_object
from .inference import complete
from .prompts import selfcheck_messages
from .schema import Decision, SelfCheck


def _parse_selfcheck(raw: str) -> tuple[str, str]:
    """(verdict, reason); verdict='unknown', если модель ответила не по контракту."""
    candidate = extract_json_object(raw)
    if candidate is None:
        return "unknown", "ревьюер ответил не JSON-ом"
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return "unknown", "ревьюер вернул неразбираемый JSON"
    if not isinstance(data, dict):
        return "unknown", "ревьюер вернул не объект"
    verdict = str(data.get("verdict") or "").strip().lower()
    reason = str(data.get("reason") or "").strip()
    if verdict not in ("confirm", "reject"):
        return "unknown", reason or "ревьюер не дал confirm/reject"
    return verdict, reason


async def run_selfcheck(
    provider: AIProvider, model: str, text: str, decision: Decision, *, temperature: float = 0.0
) -> SelfCheck:
    """Просит модель подтвердить или отклонить решение по тому же обращению."""
    call = await complete(provider, model, selfcheck_messages(text, decision), temperature)
    verdict, reason = _parse_selfcheck(call.text)
    return SelfCheck(
        verdict=verdict,
        reason=reason,
        time_ms=call.time_ms,
        prompt_tokens=call.prompt_tokens,
        completion_tokens=call.completion_tokens,
    )
