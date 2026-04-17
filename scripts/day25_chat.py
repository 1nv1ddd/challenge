"""День 25. Мини-чат с RAG + память задачи (CLI).

Production-like сборка из готовых компонентов проекта:
    - `SimpleChatAgent` хранит историю диалога (per conversation_id).
    - На каждый вопрос подмешивается RAG-контекст (pipeline Дня 21–24).
    - Сервер автоматически добавляет разделы `## Источники` и `## Цитаты` (День 24).
    - Task memory: phase/step/expected_action (FSM) + facts + working_memory.

Запуск:
    python scripts/day25_chat.py
    python scripts/day25_chat.py --top-k 6 --strategy structural --session my-run

Команды внутри чата:
    /state   — показать текущую память задачи и FSM
    /reset   — начать новый диалог (новый conversation_id)
    /help    — список команд
    /quit    — выход (также Ctrl+D)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.bootstrap import agent  # noqa: E402
from app.rag import pipeline as rag_pipeline  # noqa: E402


DEFAULT_PROVIDER = "routerai"
DEFAULT_MODEL = "openai/gpt-4o-mini"


def _format_task_memory(state: dict) -> str:
    ts = state.get("task_state", {}) or {}
    facts = state.get("facts", {}) or {}
    working = state.get("working_memory", {}) or {}

    lines: list[str] = []
    lines.append("─── Память задачи ───")
    lines.append(
        "FSM: phase={phase} · active={active} · status={status}".format(
            phase=ts.get("phase", "—"),
            active=ts.get("task_active", False),
            status=ts.get("status", "—"),
        )
    )
    step = (ts.get("current_step") or "").strip()
    exp = (ts.get("expected_action") or "").strip()
    if step:
        lines.append(f"  шаг: {step}")
    if exp:
        lines.append(f"  ожидание: {exp}")

    goal = working.get("task_goal") or facts.get("goal")
    if goal:
        lines.append(f"Цель: {goal}")
    scope = working.get("task_scope") or facts.get("scope")
    if scope:
        lines.append(f"Scope: {scope}")
    constraints = working.get("constraints") or facts.get("constraints")
    if constraints:
        lines.append(f"Ограничения: {constraints}")
    deadline = working.get("deadline") or facts.get("deadline")
    if deadline:
        lines.append(f"Дедлайн: {deadline}")

    extra_facts = {
        k: v
        for k, v in facts.items()
        if k not in {"goal", "scope", "constraints", "deadline", "preferences"}
    }
    if extra_facts:
        lines.append("Факты:")
        for k, v in list(extra_facts.items())[:8]:
            lines.append(f"  - {k}: {v}")
    return "\n".join(lines)


def _print_welcome(conversation_id: str, rag_cfg: dict) -> None:
    print("╔══ День 25 · Мини-чат с RAG + памятью задачи ══╗")
    print(f"║ conversation_id: {conversation_id}")
    print(
        "║ RAG: strategy={s} · top_k={k} · index={i}".format(
            s=rag_cfg["strategy"], k=rag_cfg["top_k"], i=rag_cfg["index_path"]
        )
    )
    print("║ Команды: /state /reset /help /quit")
    print("╚═══════════════════════════════════════════════╝")


async def _run_turn(
    provider: str,
    model: str,
    conversation_id: str,
    user_text: str,
    rag_cfg: dict,
) -> dict | None:
    """Стримит ответ в stdout, возвращает enriched_meta последнего StreamResult."""
    last_meta: dict | None = None
    first_chunk = True
    async for result in agent.stream_reply(
        provider_name=provider,
        model=model,
        conversation_id=conversation_id,
        raw_messages=[{"role": "user", "content": user_text}],
        rag=rag_cfg,
    ):
        if result.text:
            if first_chunk:
                print("\n🤖 Ассистент:\n", flush=True)
                first_chunk = False
            print(result.text, end="", flush=True)
        if result.meta is not None:
            last_meta = result.meta
    print()  # newline after stream
    return last_meta


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Day 25 mini-chat (RAG + task memory)")
    p.add_argument("--provider", default=DEFAULT_PROVIDER)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument(
        "--strategy",
        default="structural",
        choices=["fixed", "structural"],
        help="RAG chunking strategy",
    )
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument(
        "--index-path",
        default=str(rag_pipeline.default_index_path()),
        help="Путь к chunks.sqlite (по умолчанию data/rag_index/chunks.sqlite)",
    )
    p.add_argument(
        "--session",
        default=f"day25-{uuid.uuid4().hex[:8]}",
        help="conversation_id; используйте одинаковый, чтобы продолжить прошлую сессию",
    )
    p.add_argument(
        "--min-similarity",
        type=float,
        default=0.28,
        help="Фильтр low-relevance чанков (День 23)",
    )
    p.add_argument(
        "--answer-min-score",
        type=float,
        default=0.25,
        help="Порог отказа от ответа (День 24)",
    )
    return p.parse_args()


async def main_async(args: argparse.Namespace) -> int:
    if args.provider not in agent.providers:
        print(
            f"[день 25] провайдер {args.provider!r} не настроен. "
            "Проверьте ROUTERAI_API_KEY в .env."
        )
        return 2
    idx = Path(args.index_path)
    if not idx.is_file():
        print(
            f"[день 25] RAG-индекс не найден: {idx}\n"
            "Соберите его: python scripts/build_rag_index.py"
        )
        return 2

    rag_cfg: dict = {
        "enabled": True,
        "strategy": args.strategy,
        "top_k": args.top_k,
        "index_path": str(idx),
        "query_rewrite": True,
        "min_similarity": args.min_similarity,
        "rerank": "lexical",
        "top_k_fetch": max(args.top_k * 3, 18),
        "answer_min_score": args.answer_min_score,
    }

    conversation_id = args.session
    _print_welcome(conversation_id, rag_cfg)

    loop = asyncio.get_event_loop()
    while True:
        try:
            user_text = await loop.run_in_executor(None, lambda: input("\n👤 Вы: "))
        except (EOFError, KeyboardInterrupt):
            print("\n[день 25] выход.")
            return 0
        user_text = user_text.strip()
        if not user_text:
            continue

        low = user_text.lower()
        if low in {"/quit", "/exit", ":q"}:
            print("[день 25] выход.")
            return 0
        if low == "/help":
            print("Команды: /state · /reset · /quit")
            continue
        if low == "/state":
            state = agent._get_conversation_state(conversation_id)
            print(_format_task_memory(state))
            continue
        if low == "/reset":
            conversation_id = f"day25-{uuid.uuid4().hex[:8]}"
            print(f"[день 25] новый conversation_id: {conversation_id}")
            continue

        try:
            meta = await _run_turn(
                args.provider,
                args.model,
                conversation_id,
                user_text,
                rag_cfg,
            )
        except Exception as e:  # noqa: BLE001
            print(f"\n[день 25] ошибка: {e}")
            continue

        state = agent._get_conversation_state(conversation_id)
        print("\n" + _format_task_memory(state))
        if meta is not None:
            rag_meta = meta.get("rag") or {}
            hits_fields = [
                k for k in rag_meta if k.startswith("rag_hits_") and rag_meta[k]
            ]
            hit_count = sum(len(rag_meta[k]) for k in hits_fields)
            refused = rag_meta.get("rag_day24_insufficient", False)
            print(
                "RAG: hits={h} · max_score={s} · refused={r} · turn#{t}".format(
                    h=hit_count,
                    s=rag_meta.get("rag_day24_max_score", "—"),
                    r=refused,
                    t=meta.get("conversation_turns", "?"),
                )
            )


def main() -> int:
    return asyncio.run(main_async(_parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
