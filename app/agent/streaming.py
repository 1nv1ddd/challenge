from __future__ import annotations

import html
from collections.abc import AsyncIterator

from .. import mcp_panel
from ..agent_constants import (
    ALLOWED_STRATEGIES,
    INPUT_PRICE_RUB_PER_MILLION,
    MODEL_CONTEXT_LIMITS,
    OUTPUT_PRICE_RUB_PER_MILLION,
    TASK_ALLOWED_EDGES,
    TASK_EVENT_ASSISTANT_TURN_COMPLETED,
    TASK_EVENT_NEW_TASK,
    TASK_EVENT_PAUSE,
    TASK_EVENT_RESUME,
    WINDOW_SIZE_MESSAGES,
)
from ..mcp_tool_parse import _merge_provider_meta, _parse_mcp_tool_call
from ..providers import Message, StreamResult
from ..rag.day24 import build_day24_appendix_markdown, splice_day24_appendix_before_compare


class AgentStreamingMixin:
    async def compare_rag_answers(
        self,
        provider_name: str,
        model: str,
        user_text: str,
        *,
        temperature: float = 0.35,
        rag_strategy: str = "fixed",
        top_k: int = 8,
        index_path: str | None = None,
    ) -> dict:
        """День 22: вопрос → два полных ответа LLM (без RAG и с подмешанным контекстом из индекса)."""
        provider = self._validate_provider(provider_name)
        await self._validate_model(provider, provider_name, model)
        t = self._normalize_temperature(temperature)
        user_msg = Message(role="user", content=user_text)

        async def _collect(msgs: list[Message]) -> str:
            parts: list[str] = []
            async for result in provider.stream_chat(msgs, model, t):
                if result.text:
                    parts.append(result.text)
            return "".join(parts).strip()

        without_rag = await _collect([user_msg])
        rag_cfg: dict = {"enabled": True, "strategy": rag_strategy, "top_k": top_k}
        if index_path:
            rag_cfg["index_path"] = index_path
        rag_sys, rag_meta, rag_apx = await self._rag_context_message(user_text, rag_cfg)
        with_msgs = [rag_sys, user_msg] if rag_sys is not None else [user_msg]
        with_rag = await _collect(with_msgs)
        if rag_apx:
            apx_md = build_day24_appendix_markdown(rag_apx)
            if apx_md:
                with_rag = f"{with_rag.rstrip()}{apx_md}"
        return {
            "without_rag": without_rag,
            "with_rag": with_rag,
            "rag": rag_meta,
        }

    async def compare_rag_modes(
        self,
        provider_name: str,
        model: str,
        user_text: str,
        *,
        temperature: float = 0.35,
        rag_strategy: str = "fixed",
        top_k: int = 8,
        index_path: str | None = None,
        min_similarity: float = 0.28,
    ) -> dict:
        """День 23: два ответа с RAG — базовый пайплайн и с фильтром/реранком/rewrite."""
        provider = self._validate_provider(provider_name)
        await self._validate_model(provider, provider_name, model)
        t = self._normalize_temperature(temperature)
        user_msg = Message(role="user", content=user_text)

        async def _collect(msgs: list[Message]) -> str:
            parts: list[str] = []
            async for result in provider.stream_chat(msgs, model, t):
                if result.text:
                    parts.append(result.text)
            return "".join(parts).strip()

        base: dict = {"enabled": True, "strategy": rag_strategy, "top_k": top_k}
        if index_path:
            base["index_path"] = index_path

        baseline_cfg = {
            **base,
            "query_rewrite": False,
            "min_similarity": 0,
            "rerank": "none",
            "top_k_fetch": None,
        }
        enhanced_cfg = {
            **base,
            "query_rewrite": True,
            "min_similarity": min_similarity,
            "rerank": "lexical",
            "top_k_fetch": max(top_k * 3, 24),
        }

        sys_b, meta_b, apx_b = await self._rag_context_message(user_text, baseline_cfg)
        sys_e, meta_e, apx_e = await self._rag_context_message(user_text, enhanced_cfg)
        msgs_b = [sys_b, user_msg] if sys_b is not None else [user_msg]
        msgs_e = [sys_e, user_msg] if sys_e is not None else [user_msg]
        ans_b = await _collect(msgs_b)
        ans_e = await _collect(msgs_e)
        if apx_b:
            mdb = build_day24_appendix_markdown(apx_b)
            if mdb:
                ans_b = f"{ans_b.rstrip()}{mdb}"
        if apx_e:
            mde = build_day24_appendix_markdown(apx_e)
            if mde:
                ans_e = f"{ans_e.rstrip()}{mde}"
        return {
            "with_rag_baseline": ans_b,
            "with_rag_enhanced": ans_e,
            "rag_baseline_meta": meta_b,
            "rag_enhanced_meta": meta_e,
        }

    async def stream_reply(
        self,
        provider_name: str,
        model: str,
        conversation_id: str,
        raw_messages: list[dict],
        temperature: float = 0.7,
        context_strategy: str = "sliding",
        branch_id: str = "main",
        profile_id: str | None = None,
        resume: bool = False,
        rag: dict | None = None,
        task_workflow: bool | None = None,
    ) -> AsyncIterator[StreamResult]:
        provider = self._validate_provider(provider_name)
        await self._validate_model(provider, provider_name, model)
        active_profile_id, profile = self._resolve_profile(profile_id)
        strategy = context_strategy if context_strategy in ALLOWED_STRATEGIES else "sliding"
        normalized_temperature = self._normalize_temperature(temperature)
        incoming = self._normalize_messages(raw_messages)
        if not incoming:
            raise ValueError("No valid messages to send")

        state = self._get_conversation_state(conversation_id)
        # None = старые клиенты: фазы задачи как раньше. False = только обычный чат (UI «План» выкл).
        task_workflow_enabled = True if task_workflow is None else bool(task_workflow)
        if len(incoming) != 1 or incoming[0].role != "user":
            inv_msg = self._invariants_system_message(state.get("invariants", {}))
            profile_msg = self._profile_system_message(profile)
            prefix = [m for m in [profile_msg, inv_msg] if m is not None]
            request_messages = [*prefix, *incoming]
            inv_norm = self._normalize_invariants(state.get("invariants", {}))
            memory_tok = self._estimate_tokens_messages(prefix)
            all_tok = self._estimate_tokens_messages(request_messages)
            context_meta = {
                "context_strategy": strategy,
                "branch_id": branch_id,
                "history_tokens": self._estimate_tokens_messages(incoming),
                "recent_history_tokens": self._estimate_tokens_messages(incoming),
                "memory_tokens": memory_tok,
                "facts_count": 0,
                "prompt_tokens_with_compression_est": all_tok,
                "prompt_tokens_no_compression_est": all_tok,
                "saved_tokens_est": 0,
                "compressed_messages_count": 0,
                "summary_used": False,
                "summary_tokens": 0,
                "compression_enabled": False,
                "current_request_tokens": self._estimate_tokens_messages(incoming),
                "memory_auto_working_keys": [],
                "memory_auto_keys": [],
                "short_term_count": len(incoming),
                "active_profile_id": active_profile_id,
                "profile_applied": profile_msg is not None,
                "task_phase": "planning",
                "task_current_step": "",
                "task_expected_action": "",
                "task_is_paused": False,
                "task_allowed_next_phases": [],
                "invariants_count": len(inv_norm),
                "invariants_applied": inv_msg is not None,
                "task_workflow_enabled": task_workflow_enabled,
            }
            user_msg = incoming[-1]
        else:
            user_msg = incoming[0]
            ref_qa = self._is_reference_or_doc_qa_message(user_msg.content)
            current_task_state = self._normalize_task_state(state.get("task_state", {}))
            if current_task_state.get("status") == "paused" and not resume:
                raise ValueError("Task is paused. Resume generation to continue.")
            if resume and current_task_state.get("status") == "paused":
                state["task_state"] = self._transition_task_state(
                    task_state=current_task_state,
                    event=TASK_EVENT_RESUME,
                )
                extra = user_msg.content.strip()
                resume_text = (
                    "Continue from the paused point without repeating previous explanation."
                )
                if extra:
                    resume_text += f"\nAdditional instruction from user:\n{extra}"
                user_msg = Message(role="user", content=resume_text)
            elif (
                task_workflow_enabled
                and self._is_task_intent_message(user_msg.content)
                and not ref_qa
                and not current_task_state.get("task_active", False)
            ):
                # Do not reset FSM on follow-ups that still match markers (e.g. "план" + "реализ" in approval).
                state["task_state"] = self._transition_task_state(
                    task_state=current_task_state,
                    event=TASK_EVENT_NEW_TASK,
                )
            if task_workflow_enabled:
                self._promote_to_plan_approved_if_user_approved(state, user_msg.content)
                self._promote_validation_to_done_if_user_confirms(state, user_msg.content)
            current_task_state = self._normalize_task_state(state.get("task_state", {}))
            include_task_state = resume or (
                task_workflow_enabled
                and (
                    (
                        bool(current_task_state.get("task_active", False))
                        or self._is_task_intent_message(user_msg.content)
                    )
                    and not ref_qa
                )
            )
            self._update_facts(state, user_msg.content)
            working_keys = self._auto_update_working_memory(state, user_msg.content)
            auto_keys = self._auto_update_long_term(user_msg.content)
            context_limit = MODEL_CONTEXT_LIMITS.get(model, 128000)
            request_messages, context_meta = self._build_context(
                state=state,
                incoming_user=user_msg,
                profile=profile,
                active_profile_id=active_profile_id,
                include_task_state=include_task_state,
                strategy=strategy,
                branch_id=branch_id,
                context_limit=context_limit,
            )
            context_meta["memory_auto_working_keys"] = working_keys
            context_meta["memory_auto_keys"] = auto_keys
            context_meta["task_workflow_enabled"] = task_workflow_enabled

        bridge = mcp_panel.get_mcp_bridge()
        mcp_instr = self._mcp_system_message_from_bridge(bridge) if bridge else None
        if mcp_instr is not None:
            request_messages = [*request_messages[:-1], mcp_instr, request_messages[-1]]

        rag_msg, rag_meta, rag_appendix_hits = await self._rag_context_message(
            user_msg.content, rag
        )
        if rag_msg is not None:
            request_messages = [*request_messages[:-1], rag_msg, request_messages[-1]]

        compare_rag_buffer = (rag_meta or {}).get("rag_mode") == "compare" and bool(
            rag_appendix_hits
        )

        assistant_chunks: list[str] = []
        provider_meta: dict | None = None
        paused_during_stream = False
        mcp_tool_used: str | None = None

        if bridge is not None:
            _MCP_MAX_STEPS = 8
            working_msgs: list[Message] = list(request_messages)
            mcp_chain_labels: list[str] = []
            mcp_streamed_blocks: list[str] = []
            assistant_text = ""
            for _mcp_step in range(_MCP_MAX_STEPS):
                step_chunks: list[str] = []
                async for result in provider.stream_chat(
                    working_msgs, model, normalized_temperature
                ):
                    if result.text:
                        step_chunks.append(result.text)
                    if result.meta is not None:
                        provider_meta = _merge_provider_meta(provider_meta, result.meta)
                    if (
                        self._normalize_task_state(state.get("task_state", {})).get("status")
                        == "paused"
                    ):
                        paused_during_stream = True
                        break
                step_text = "".join(step_chunks)
                if paused_during_stream:
                    if step_text:
                        yield StreamResult(text=step_text)
                    _parts = [*mcp_streamed_blocks]
                    if step_text.strip():
                        _parts.append(step_text.strip())
                    assistant_text = "\n\n".join(_parts) if _parts else step_text
                    break
                call = _parse_mcp_tool_call(step_text)
                srv = call.get("server") if call else None
                if call and mcp_panel.mcp_call_allowed(srv, call["name"]):
                    try:
                        _, _, resolved_sid = mcp_panel.resolve_invocation(srv, call["name"])
                    except ValueError:
                        resolved_sid = (srv or "default").strip() or "default"
                    label = f"{resolved_sid}:{call['name']}"
                    mcp_chain_labels.append(label)
                    mcp_tool_used = " → ".join(mcp_chain_labels)
                    try:
                        tool_payload = await mcp_panel.invoke_mcp_tool(
                            call["name"],
                            call.get("arguments") or {},
                            server_id=srv,
                        )
                    except Exception as exc:  # noqa: BLE001
                        tool_payload = f"Ошибка выполнения: {exc}"
                    # HTML, не markdown-огороды: иначе marked «съедает» блок при кривых ``` в ответе модели.
                    esc_l = html.escape(label, quote=True)
                    esc_p = html.escape(tool_payload.rstrip(), quote=True)
                    out_vis = (
                        f'<div class="mcp-inline-result"><p><strong>MCP</strong> '
                        f"<code>{esc_l}</code> — вывод инструмента:</p>"
                        f"<pre>{esc_p}</pre></div>\n\n"
                    )
                    yield StreamResult(text=out_vis)
                    mcp_streamed_blocks.append(out_vis.strip())
                    follow_user = (
                        f"Шаг {_mcp_step + 1}. Инструмент `{call['name']}` "
                        f"(сервер `{srv or 'единственный'}`).\n\n{tool_payload}\n\n"
                        "Если нужен ещё один инструмент (в том числе на **другом** MCP-сервере) — "
                        "ответь одним блоком ```mcp с полями server (если несколько серверов), name, arguments. "
                        "Иначе дай **полный** ответ пользователю на русском **без** нового блока ```mcp — "
                        "обязательно **кратко упомяни результаты всех уже выполненных шагов** (включая этот), "
                        "чтобы пользователь видел вывод каждого инструмента."
                    )
                    working_msgs = [
                        *working_msgs,
                        Message(role="assistant", content=step_text.strip()),
                        Message(role="user", content=follow_user),
                    ]
                    continue
                if step_text.strip():
                    yield StreamResult(text=step_text)
                parts = [*mcp_streamed_blocks]
                if step_text.strip():
                    parts.append(step_text.strip())
                assistant_text = "\n\n".join(parts) if parts else ""
                break
            if not (assistant_text or "").strip() and mcp_streamed_blocks:
                note = (
                    "\n\n*Достигнут лимит цепочки MCP за один ответ; при необходимости продолжите новым сообщением.*"
                )
                assistant_text = "\n\n".join(mcp_streamed_blocks) + note
                yield StreamResult(text=note)
        else:
            async for result in provider.stream_chat(
                request_messages, model, normalized_temperature
            ):
                if result.text:
                    assistant_chunks.append(result.text)
                    if not compare_rag_buffer:
                        yield result
                    if (
                        self._normalize_task_state(state.get("task_state", {})).get("status")
                        == "paused"
                    ):
                        paused_during_stream = True
                        break
                if result.meta is not None:
                    provider_meta = _merge_provider_meta(provider_meta, result.meta)

            assistant_text = "".join(assistant_chunks)

        if rag_appendix_hits:
            apx_md = build_day24_appendix_markdown(rag_appendix_hits)
            if apx_md:
                base = (assistant_text or "").rstrip()
                if (rag_meta or {}).get("rag_mode") == "compare":
                    assistant_text = splice_day24_appendix_before_compare(base, apx_md)
                else:
                    assistant_text = f"{base}{apx_md}"
                if compare_rag_buffer and bridge is None:
                    yield StreamResult(text=assistant_text)
                elif not compare_rag_buffer:
                    yield StreamResult(text=apx_md)
                else:
                    yield StreamResult(text=apx_md)

        if not (assistant_text or "").strip():
            assistant_text = (
                "Пустой ответ модели. Проверьте ROUTERAI_API_KEY, перезапустите сервер после смены .env. "
                "Если провайдер вернул ошибку, она должна была отобразиться выше."
            )
            yield StreamResult(text=assistant_text)

        assistant_msg = Message(role="assistant", content=assistant_text)
        self._append_turn(state, strategy, context_meta["branch_id"], user_msg, assistant_msg)
        current_task_state = self._normalize_task_state(state.get("task_state", {}))
        single_user_turn = len(incoming) == 1 and incoming[0].role == "user"
        auto_advance_phase = (
            task_workflow_enabled
            and single_user_turn
            and not paused_during_stream
            and not resume
            and not self._is_smalltalk_message(user_msg.content)
            and current_task_state.get("task_active", False)
            and current_task_state.get("status") == "running"
        )
        if paused_during_stream:
            task_state_after = self._transition_task_state(
                task_state=current_task_state,
                event=TASK_EVENT_PAUSE,
            )
        else:
            task_state_after = self._transition_task_state(
                task_state=current_task_state,
                event=TASK_EVENT_ASSISTANT_TURN_COMPLETED,
                advance_phase=auto_advance_phase,
            )
        state["task_state"] = task_state_after

        prompt_tokens = int((provider_meta or {}).get("prompt_tokens", 0))
        completion_tokens = int((provider_meta or {}).get("completion_tokens", 0))
        total_tokens = int((provider_meta or {}).get("total_tokens", 0))
        response_tokens_est = self._estimate_tokens_text(assistant_text)
        context_limit = MODEL_CONTEXT_LIMITS.get(model, 128000)

        request_cost_rub = (
            prompt_tokens * INPUT_PRICE_RUB_PER_MILLION
            + completion_tokens * OUTPUT_PRICE_RUB_PER_MILLION
        ) / 1_000_000

        stats = state["stats"]
        stats["turns"] += 1
        stats["prompt_tokens_total"] += prompt_tokens
        stats["completion_tokens_total"] += completion_tokens
        stats["total_tokens_total"] += total_tokens
        stats["cost_rub_total"] += request_cost_rub
        self._save_history()

        prompt_for_limit = (
            prompt_tokens
            if prompt_tokens > 0
            else int(context_meta["prompt_tokens_with_compression_est"])
        )
        prompt_pct = round((prompt_for_limit / context_limit) * 100, 2)

        working_memory = self._normalize_kv_dict(state.get("working_memory", {}), max_items=40)
        long_term = self._normalize_kv_dict(self.global_memory.get("long_term", {}), max_items=120)
        facts = state.get("facts", {}) or {}
        if not isinstance(facts, dict):
            facts = {}
        _reserved_fact_keys = {
            "goal",
            "constraints",
            "scope",
            "deadline",
            "preferences",
            "decisions",
            "agreements",
        }
        task_memory = {
            "goal": (working_memory.get("task_goal") or facts.get("goal") or "").strip(),
            "constraints": (
                working_memory.get("constraints") or facts.get("constraints") or ""
            ).strip(),
            "scope": (working_memory.get("task_scope") or facts.get("scope") or "").strip(),
            "deadline": (working_memory.get("deadline") or facts.get("deadline") or "").strip(),
            "terms": {
                k: v
                for k, v in facts.items()
                if k not in _reserved_fact_keys and str(v).strip()
            },
        }

        enriched_meta = {
            "time_ms": int((provider_meta or {}).get("time_ms", 0)),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost_rub": round(request_cost_rub, 6),
            "current_request_tokens": int(context_meta["current_request_tokens"]),
            "history_tokens": int(context_meta["history_tokens"]),
            "recent_history_tokens": int(context_meta["recent_history_tokens"]),
            "memory_tokens": int(context_meta.get("memory_tokens", 0)),
            "facts_count": int(context_meta["facts_count"]),
            "response_tokens": completion_tokens or response_tokens_est,
            "conversation_total_tokens": int(stats["total_tokens_total"]),
            "conversation_total_cost_rub": round(float(stats["cost_rub_total"]), 6),
            "conversation_turns": int(stats["turns"]),
            "model_context_limit_tokens": context_limit,
            "prompt_usage_percent": prompt_pct,
            "summary_used": False,
            "summary_tokens": 0,
            "prompt_tokens_no_compression_est": int(context_meta["prompt_tokens_no_compression_est"]),
            "prompt_tokens_with_compression_est": int(context_meta["prompt_tokens_with_compression_est"]),
            "saved_tokens_est": int(context_meta["saved_tokens_est"]),
            "compressed_messages_count": int(context_meta["compressed_messages_count"]),
            "context_strategy": context_meta["context_strategy"],
            "branch_id": context_meta["branch_id"],
            "active_profile_id": context_meta.get("active_profile_id", active_profile_id),
            "profile_applied": bool(context_meta.get("profile_applied", False)),
            "task_phase": task_state_after.get("phase", context_meta.get("task_phase", "planning")),
            "task_current_step": task_state_after.get(
                "current_step", context_meta.get("task_current_step", "")
            ),
            "task_expected_action": task_state_after.get(
                "expected_action", context_meta.get("task_expected_action", "")
            ),
            "task_is_paused": bool(task_state_after.get("is_paused", False)),
            "task_allowed_next_phases": list(
                TASK_ALLOWED_EDGES.get(task_state_after.get("phase", "planning"), ())
            ),
            "task_workflow_enabled": task_workflow_enabled,
            "paused_during_stream": paused_during_stream,
            "memory_short_notes_count": int(context_meta.get("short_term_count", WINDOW_SIZE_MESSAGES)),
            "memory_working_count": len(working_memory),
            "memory_long_count": len(long_term),
            "memory_auto_working_count": len(context_meta.get("memory_auto_working_keys", [])),
            "memory_auto_working_keys": context_meta.get("memory_auto_working_keys", []),
            "memory_auto_saved_count": len(context_meta.get("memory_auto_keys", [])),
            "memory_auto_saved_keys": context_meta.get("memory_auto_keys", []),
            "compression_enabled": False,
            "estimated": prompt_tokens == 0,
            "invariants_count": int(context_meta.get("invariants_count", 0)),
            "invariants_applied": bool(context_meta.get("invariants_applied", False)),
            "workflow_guard_applied": bool(context_meta.get("workflow_guard_applied", False)),
            "mcp_tool": mcp_tool_used,
            "rag": rag_meta if rag_meta else None,
            "task_memory": task_memory,
        }
        yield StreamResult(meta=enriched_meta)
