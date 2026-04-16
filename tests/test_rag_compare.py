"""День 22: сравнение ответов без RAG и с RAG (без реального LLM)."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

from app.agent import SimpleChatAgent
from app.providers import Message, StreamResult


class _FakeProvider:
    models = [{"id": "openai/gpt-4o-mini"}]

    async def stream_chat(self, messages, model, temperature=0.7):
        tag = "plain" if len(messages) == 1 else "rag"
        yield StreamResult(text=f"out-{tag}-msgs={len(messages)}")


class TestCompareRagAnswers(unittest.TestCase):
    def test_two_llm_paths(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            mem = Path(f.name)
        try:

            async def _run():
                agent = SimpleChatAgent({"routerai": _FakeProvider()}, memory_path=mem)
                agent._rag_context_message = AsyncMock(
                    return_value=(
                        Message(role="system", content="[RAG ctx]"),
                        {"rag_mode": "fixed", "rag_top_k": 3},
                        None,
                    )
                )
                out = await agent.compare_rag_answers(
                    "routerai",
                    "openai/gpt-4o-mini",
                    "Какой SLA у PL-203-RET?",
                    temperature=0.2,
                    rag_strategy="fixed",
                    top_k=3,
                )
                self.assertEqual(out["without_rag"], "out-plain-msgs=1")
                self.assertEqual(out["with_rag"], "out-rag-msgs=2")
                self.assertEqual(out["rag"].get("rag_mode"), "fixed")

            asyncio.run(_run())
        finally:
            mem.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
