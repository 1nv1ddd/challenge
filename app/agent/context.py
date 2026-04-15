from __future__ import annotations

from ..agent_constants import (
    ALLOWED_STRATEGIES,
    MODEL_CONTEXT_LIMITS,
    TASK_ALLOWED_EDGES,
    WINDOW_SIZE_MESSAGES,
)
from ..providers import Message


class AgentContextMixin:
    def _build_context(
        self,
        state: dict,
        incoming_user: Message,
        profile: dict,
        active_profile_id: str,
        include_task_state: bool,
        strategy: str,
        branch_id: str,
        context_limit: int,
    ) -> tuple[list[Message], dict]:
        full_history = self._normalize_messages(state.get("full_messages", []))
        user_tokens = self._estimate_tokens_messages([incoming_user])

        working_msg = self._working_memory_system_message(state.get("working_memory", {}))
        long_term_msg = self._long_term_system_message()
        profile_msg = self._profile_system_message(profile)
        inv_msg = self._invariants_system_message(state.get("invariants", {}))
        task_state = self._normalize_task_state(state.get("task_state", {}))
        task_msg = self._task_state_system_message(task_state) if include_task_state else None
        guard_phases = ("planning", "plan_approved", "execution")
        workflow_guard = None
        if (
            include_task_state
            and task_state.get("phase") in guard_phases
            and self._workflow_bypass_user_intent(incoming_user.content)
        ):
            workflow_guard = self._workflow_bypass_guard_system_message(task_state["phase"])
        memory_messages = [
            m
            for m in [profile_msg, inv_msg, task_msg, workflow_guard, working_msg, long_term_msg]
            if m is not None
        ]
        memory_tokens = self._estimate_tokens_messages(memory_messages)

        if strategy == "branching":
            branches = state.get("branches", {})
            branch = branches.get(branch_id)
            if branch is None:
                branch = branches.get("main")
                branch_id = "main"
            branch_history = self._normalize_messages(branch.get("messages", []))
            short_history = branch_history[-WINDOW_SIZE_MESSAGES:]
            request_messages = [*memory_messages, *short_history, incoming_user]
            no_tokens = self._estimate_tokens_messages([*memory_messages, *branch_history, incoming_user])
            dropped = max(0, len(branch_history) - len(short_history))
            history_tokens = self._estimate_tokens_messages(short_history)
        elif strategy == "facts":
            short_history = full_history[-WINDOW_SIZE_MESSAGES:]
            facts_msg = self._facts_system_message(state.get("facts", {}))
            fact_list = [facts_msg] if facts_msg else []
            request_messages = [*memory_messages, *fact_list, *short_history, incoming_user]
            no_tokens = self._estimate_tokens_messages([*memory_messages, *fact_list, *full_history, incoming_user])
            dropped = max(0, len(full_history) - len(short_history))
            history_tokens = self._estimate_tokens_messages(short_history)
            branch_id = "main"
        else:
            short_history = full_history[-WINDOW_SIZE_MESSAGES:]
            request_messages = [*memory_messages, *short_history, incoming_user]
            no_tokens = self._estimate_tokens_messages([*memory_messages, *full_history, incoming_user])
            dropped = max(0, len(full_history) - len(short_history))
            history_tokens = self._estimate_tokens_messages(short_history)
            branch_id = "main"

        with_tokens = self._estimate_tokens_messages(request_messages)
        if with_tokens > context_limit:
            raise ValueError(
                f"Context overflow: estimated {with_tokens} tokens exceeds limit {context_limit}"
            )

        meta = {
            "context_strategy": strategy,
            "branch_id": branch_id,
            "history_tokens": history_tokens,
            "recent_history_tokens": history_tokens,
            "memory_tokens": memory_tokens,
            "facts_count": len(state.get("facts", {})),
            "prompt_tokens_with_compression_est": with_tokens,
            "prompt_tokens_no_compression_est": no_tokens,
            "saved_tokens_est": max(0, no_tokens - with_tokens),
            "compressed_messages_count": dropped,
            "summary_used": False,
            "summary_tokens": 0,
            "compression_enabled": False,
            "current_request_tokens": user_tokens,
            "short_term_count": len(short_history),
            "active_profile_id": active_profile_id,
            "profile_applied": profile_msg is not None,
            "task_phase": task_state["phase"],
            "task_current_step": task_state["current_step"],
            "task_expected_action": task_state["expected_action"],
            "task_is_paused": task_state["is_paused"],
            "task_allowed_next_phases": list(TASK_ALLOWED_EDGES.get(task_state["phase"], ())),
            "invariants_count": len(self._normalize_invariants(state.get("invariants", {}))),
            "invariants_applied": inv_msg is not None,
            "workflow_guard_applied": workflow_guard is not None,
        }
        return request_messages, meta

    def _append_turn(
        self,
        state: dict,
        strategy: str,
        branch_id: str,
        user_msg: Message,
        assistant_msg: Message,
    ) -> None:
        if strategy == "branching":
            branch = state["branches"].setdefault(
                branch_id,
                {"name": branch_id, "from_checkpoint": None, "messages": []},
            )
            branch_messages = self._normalize_messages(branch.get("messages", []))
            branch_messages.extend([user_msg, assistant_msg])
            branch["messages"] = [{"role": m.role, "content": m.content} for m in branch_messages]
            state["short_term_messages"] = branch["messages"][-WINDOW_SIZE_MESSAGES:]
            if branch_id == "main":
                state["full_messages"] = branch["messages"]
        else:
            history = self._normalize_messages(state.get("full_messages", []))
            history.extend([user_msg, assistant_msg])
            serialized = [{"role": m.role, "content": m.content} for m in history]
            state["full_messages"] = serialized
            state["short_term_messages"] = serialized[-WINDOW_SIZE_MESSAGES:]
            state["branches"]["main"] = {
                "name": "main",
                "from_checkpoint": None,
                "messages": serialized,
            }
