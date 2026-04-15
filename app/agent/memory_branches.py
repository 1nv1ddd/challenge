from __future__ import annotations

import time

from ..providers import Message


class AgentMemoryBranchesMixin:
    def list_memory_layers(self, conversation_id: str, branch_id: str = "main") -> dict:
        state = self._get_conversation_state(conversation_id)
        branches = state.get("branches", {})
        branch = branches.get(branch_id) or branches.get("main", {"messages": []})
        source_msgs = self._normalize_messages(branch.get("messages", []))
        short_msgs = source_msgs[-WINDOW_SIZE_MESSAGES:]
        return {
            "short_term": {
                "window_size": WINDOW_SIZE_MESSAGES,
                "messages": [{"role": m.role, "content": m.content} for m in short_msgs],
            },
            "working_memory": state.get("working_memory", {}),
            "long_term": self.global_memory.get("long_term", {}),
            "invariants": self._normalize_invariants(state.get("invariants", {})),
        }

    def list_profiles(self) -> dict:
        profiles = self.global_memory.get("profiles", {})
        items = []
        for profile_id, profile in profiles.items():
            items.append(
                {
                    "id": profile_id,
                    "name": profile.get("name", profile_id),
                    "style": profile.get("style", ""),
                    "format": profile.get("format", ""),
                    "constraints": profile.get("constraints", ""),
                }
            )
        items.sort(key=lambda x: (x["id"] != "default", x["id"]))
        return {
            "profiles": items,
            "default_profile_id": self.global_memory.get("default_profile_id", "default"),
        }
    def list_invariants(self, conversation_id: str) -> dict:
        state = self._get_conversation_state(conversation_id)
        inv = self._normalize_invariants(state.get("invariants", {}))
        return {"invariants": inv, "count": len(inv)}

    def set_invariants(
        self,
        conversation_id: str,
        invariants: dict | None = None,
        replace: bool = True,
    ) -> dict:
        state = self._get_conversation_state(conversation_id)
        patch = self._normalize_invariants(invariants if isinstance(invariants, dict) else {})
        if replace:
            state["invariants"] = patch
        else:
            merged = dict(self._normalize_invariants(state.get("invariants", {})))
            merged.update(patch)
            state["invariants"] = self._normalize_invariants(merged)
        self._save_history()
        inv = state["invariants"]
        return {"invariants": inv, "count": len(inv)}
    def upsert_profile(
        self,
        profile_id: str,
        name: str,
        style: str,
        format_pref: str,
        constraints: str,
    ) -> dict:
        pid = str(profile_id or "").strip()[:64]
        if not pid:
            raise ValueError("profile_id is required")
        profile = {
            "name": (str(name).strip()[:80] or pid),
            "style": str(style).strip()[:500],
            "format": str(format_pref).strip()[:500],
            "constraints": str(constraints).strip()[:500],
        }
        profiles = dict(self.global_memory.get("profiles", {}))
        profiles[pid] = profile
        self.global_memory["profiles"] = profiles
        if not self.global_memory.get("default_profile_id"):
            self.global_memory["default_profile_id"] = "default"
        self._save_history()
        return {"ok": True, "profile_id": pid}

    def _resolve_profile(self, profile_id: str | None) -> tuple[str, dict]:
        profiles = self.global_memory.get("profiles", {})
        default_id = self.global_memory.get("default_profile_id", "default")
        requested = str(profile_id or "").strip()
        pid = requested if requested in profiles else default_id
        profile = profiles.get(pid, profiles.get("default", {}))
        return pid, profile

    def list_branches(self, conversation_id: str) -> dict:
        state = self._get_conversation_state(conversation_id)
        branches = state.get("branches", {})
        out = []
        for bid, b in branches.items():
            out.append(
                {
                    "id": bid,
                    "name": b.get("name", bid),
                    "message_count": len(b.get("messages", [])),
                    "from_checkpoint": b.get("from_checkpoint"),
                }
            )
        out.sort(key=lambda x: (x["id"] != "main", x["id"]))
        return {"branches": out}

    def create_checkpoint(self, conversation_id: str, branch_id: str = "main") -> dict:
        state = self._get_conversation_state(conversation_id)
        branch = state.get("branches", {}).get(branch_id)
        if branch is None:
            raise ValueError(f"Branch '{branch_id}' not found")
        checkpoint_id = f"cp_{int(time.time() * 1000)}"
        state["checkpoints"][checkpoint_id] = {
            "branch_id": branch_id,
            "full_messages": branch.get("messages", []),
            "facts": dict(state.get("facts", {})),
            "working_memory": dict(state.get("working_memory", {})),
            "invariants": dict(self._normalize_invariants(state.get("invariants", {}))),
            "created_at": int(time.time()),
        }
        self._save_history()
        return {"checkpoint_id": checkpoint_id}

    def create_branch(
        self,
        conversation_id: str,
        checkpoint_id: str,
        branch_name: str | None = None,
    ) -> dict:
        state = self._get_conversation_state(conversation_id)
        checkpoint = state.get("checkpoints", {}).get(checkpoint_id)
        if checkpoint is None:
            raise ValueError(f"Checkpoint '{checkpoint_id}' not found")

        n = 1
        while True:
            bid = f"branch-{n}"
            if bid not in state["branches"]:
                break
            n += 1

        state["branches"][bid] = {
            "name": (branch_name or bid),
            "from_checkpoint": checkpoint_id,
            "messages": checkpoint.get("full_messages", []),
        }
        self._save_history()
        return {"branch_id": bid, "name": state["branches"][bid]["name"]}
