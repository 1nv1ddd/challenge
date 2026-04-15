"""Чат-агент: память, FSM задачи, RAG, MCP — см. миксины в подмодулях."""

from __future__ import annotations

from ..mcp_tool_parse import _parse_mcp_tool_call

from .core import SimpleChatAgent

__all__ = ["SimpleChatAgent", "_parse_mcp_tool_call"]
