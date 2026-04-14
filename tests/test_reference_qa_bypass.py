"""Справочные вопросы по корпусу не должны включать FSM задачи (День 22 / честное сравнение без RAG)."""

from __future__ import annotations

import unittest

from app.agent import SimpleChatAgent


class TestReferenceQaBypass(unittest.TestCase):
    def test_golden_block_is_reference_qa(self):
        block = """По справочнику PolarLine: для кода инцидента PL-203-RET укажи SLA.
Что запрещено делать с таблицей shard_map при PL-101-RET?
Где по README лежит SQLite-индекс RAG?"""
        self.assertTrue(SimpleChatAgent._is_reference_or_doc_qa_message(block))

    def test_plain_task_request_not_reference_qa(self):
        self.assertFalse(
            SimpleChatAgent._is_reference_or_doc_qa_message(
                "Нужно сделать REST API на FastAPI с JWT, к завтрашнему дню."
            )
        )

    def test_short_question_not_reference_qa(self):
        self.assertFalse(
            SimpleChatAgent._is_reference_or_doc_qa_message("Что такое PL-203-RET?")
        )


if __name__ == "__main__":
    unittest.main()
