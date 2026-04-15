from __future__ import annotations

import re

from ..agent_constants import LONG_TERM_ALLOWED_KEYS


class AgentFactsMixin:
    def _update_facts(self, state: dict, user_text: str) -> None:
        facts = state.get("facts", {})
        if not isinstance(facts, dict):
            facts = {}

        lines = [x.strip() for x in user_text.splitlines() if x.strip()]
        for line in lines:
            if ":" in line:
                k, v = line.split(":", 1)
                key = k.strip().lower()[:40]
                val = v.strip()[:240]
                if key and val:
                    facts[key] = val

            lower = line.lower()
            if "цель" in lower:
                facts["goal"] = line[:240]
            if "огранич" in lower:
                facts["constraints"] = line[:240]
            if "предпочт" in lower:
                facts["preferences"] = line[:240]
            if "решени" in lower:
                facts["decisions"] = line[:240]
            if "договор" in lower:
                facts["agreements"] = line[:240]

        trimmed = {}
        for k in list(facts.keys())[-20:]:
            trimmed[k] = facts[k]
        state["facts"] = trimmed

    def _auto_update_long_term(self, user_text: str) -> list[str]:
        long_term = dict(self._sanitize_long_term(self.global_memory.get("long_term", {})))
        updated_keys: list[str] = []
        for line in [x.strip() for x in user_text.splitlines() if x.strip()]:
            if ":" in line:
                k, v = line.split(":", 1)
                key = k.strip().lower().replace(" ", "_")[:64]
                val = v.strip()[:320]
                if key in LONG_TERM_ALLOWED_KEYS and val and len(val) >= 3:
                    long_term[key] = val
                    updated_keys.append(key)
                continue

            lower = line.lower()
            if "предпочт" in lower and len(line) <= 220:
                long_term["preferences"] = line[:320]
                updated_keys.append("preferences")
            if "решен" in lower and len(line) <= 220:
                long_term["decisions"] = line[:320]
                updated_keys.append("decisions")
            if "профил" in lower and len(line) <= 220:
                long_term["profile"] = line[:320]
                updated_keys.append("profile")
            if "бюджет" in lower and len(line) <= 120:
                match = re.search(r"(\d[\d\s]{1,20})(?:\s*[₽$€]| ?руб| ?rub)?", line, flags=re.I)
                if match:
                    budget_val = " ".join(match.group(1).split())
                    if "₽" in line or "руб" in lower or "rub" in lower:
                        budget_val = f"{budget_val} ₽"
                    long_term["budget"] = budget_val[:48]
                    updated_keys.append("budget")
            if "дедлайн" in lower and len(line) <= 120:
                long_term["deadline"] = line[:64]
                updated_keys.append("deadline")

        self.global_memory["long_term"] = self._sanitize_long_term(
            self._normalize_kv_dict(long_term, max_items=120)
        )
        uniq: list[str] = []
        for key in updated_keys:
            if key not in uniq:
                uniq.append(key)
        return uniq
    def _auto_update_working_memory(self, state: dict, user_text: str) -> list[str]:
        working = dict(state.get("working_memory", {}))
        updated: list[str] = []
        lines = [x.strip() for x in user_text.splitlines() if x.strip()]

        for line in lines:
            if ":" in line:
                k, v = line.split(":", 1)
                key = k.strip().lower().replace(" ", "_")[:64]
                val = v.strip()[:320]
                if key and val:
                    working[key] = val
                    updated.append(key)
                continue

            lower = line.lower()
            if "задач" in lower or "scope" in lower:
                working["task_scope"] = line[:320]
                updated.append("task_scope")
            if "цель" in lower:
                working["task_goal"] = line[:320]
                updated.append("task_goal")
            if "дедлайн" in lower:
                working["deadline"] = line[:320]
                updated.append("deadline")
            if "бюджет" in lower:
                working["budget"] = line[:320]
                updated.append("budget")
            if "огранич" in lower:
                working["constraints"] = line[:320]
                updated.append("constraints")

        state["working_memory"] = self._normalize_kv_dict(working, max_items=40)
        uniq: list[str] = []
        for key in updated:
            if key not in uniq:
                uniq.append(key)
        return uniq
