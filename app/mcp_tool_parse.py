"""Разбор ```mcp ...``` из ответа модели и слияние метрик провайдера."""

from __future__ import annotations

import json
import re

_MCP_FENCE_RE = re.compile(r"```mcp\s*([\s\S]*?)```", re.IGNORECASE)
_JSON_FENCE_RE = re.compile(r"```json\s*([\s\S]*?)```", re.IGNORECASE)


def _extract_json_candidate(text: str) -> str | None:
    """Принимаем три формы: ```mcp {...} ```, ```json {...} ```, или голый JSON
    в качестве полного ответа модели (часто эмитят OpenAI-tuned LLM)."""
    s = text.strip()
    if not s:
        return None
    if m := _MCP_FENCE_RE.search(s):
        return m.group(1).strip()
    if m := _JSON_FENCE_RE.search(s):
        return m.group(1).strip()
    if s.startswith("{") and s.endswith("}"):
        return s
    return None


def _parse_mcp_tool_call(text: str) -> dict | None:
    candidate = _extract_json_candidate(text)
    if candidate is None:
        return None
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or "name" not in data:
        return None
    args = data.get("arguments")
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return None
    name = str(data["name"]).strip()
    # Снимаем неймспейсный префикс типа `filemcp.search_in_files` —
    # модели регулярно его додумывают по аналогии с OpenAI function calling.
    if "." in name:
        name = name.rsplit(".", 1)[-1]
    if not name:
        return None
    out: dict = {"name": name, "arguments": args}
    srv = data.get("server")
    if srv is not None and str(srv).strip():
        out["server"] = str(srv).strip().lower()
    return out


def _merge_provider_meta(prev: dict | None, new: dict | None) -> dict | None:
    if not new:
        return prev
    if not prev:
        return dict(new)
    return {
        **new,
        "time_ms": int(prev.get("time_ms", 0)) + int(new.get("time_ms", 0)),
        "prompt_tokens": int(prev.get("prompt_tokens", 0)) + int(new.get("prompt_tokens", 0)),
        "completion_tokens": int(prev.get("completion_tokens", 0))
        + int(new.get("completion_tokens", 0)),
        "total_tokens": int(prev.get("total_tokens", 0)) + int(new.get("total_tokens", 0)),
    }
