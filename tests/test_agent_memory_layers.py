"""Слои памяти агента: инварианты, профили, чекпойнты/ветки и восстановление битого файла."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent import SimpleChatAgent  # noqa: E402
from app.agent_constants import (  # noqa: E402
    GLOBAL_KEY,
    INVARIANT_KEY_MAX_LEN,
    INVARIANT_VAL_MAX_LEN,
    INVARIANTS_MAX_ITEMS,
    WINDOW_SIZE_MESSAGES,
)

_CONV = "test-memory-layers"


class _AgentOnTempFile(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.memory_path = Path(self._tmp.name) / "agent_memory.json"
        self.agent = SimpleChatAgent({}, memory_path=self.memory_path)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _reload(self) -> SimpleChatAgent:
        return SimpleChatAgent({}, memory_path=self.memory_path)


class TestInvariants(_AgentOnTempFile):
    def test_replace_drops_previous_keys(self) -> None:
        self.agent.set_invariants(_CONV, {"tone": "строго"}, replace=True)
        out = self.agent.set_invariants(_CONV, {"lang": "ru"}, replace=True)
        self.assertEqual(out["invariants"], {"lang": "ru"})
        self.assertEqual(out["count"], 1)

    def test_merge_keeps_previous_and_overrides_same_key(self) -> None:
        self.agent.set_invariants(_CONV, {"tone": "строго", "lang": "ru"}, replace=True)
        out = self.agent.set_invariants(_CONV, {"lang": "en"}, replace=False)
        self.assertEqual(out["invariants"], {"tone": "строго", "lang": "en"})
        self.assertEqual(out["count"], 2)

    def test_blank_key_or_value_is_dropped(self) -> None:
        out = self.agent.set_invariants(_CONV, {"  ": "x", "tone": "   ", "lang": "ru"}, replace=True)
        self.assertEqual(out["invariants"], {"lang": "ru"})

    def test_long_key_and_value_are_truncated(self) -> None:
        out = self.agent.set_invariants(_CONV, {"k" * 200: "v" * 900}, replace=True)
        (key, val), = out["invariants"].items()
        self.assertEqual(len(key), INVARIANT_KEY_MAX_LEN)
        self.assertEqual(len(val), INVARIANT_VAL_MAX_LEN)

    def test_item_limit_keeps_last_items(self) -> None:
        raw = {f"k{i:03d}": f"v{i}" for i in range(INVARIANTS_MAX_ITEMS + 5)}
        out = self.agent.set_invariants(_CONV, raw, replace=True)
        self.assertEqual(out["count"], INVARIANTS_MAX_ITEMS)
        self.assertNotIn("k000", out["invariants"])
        self.assertIn(f"k{INVARIANTS_MAX_ITEMS + 4:03d}", out["invariants"])

    def test_invariants_survive_reload(self) -> None:
        self.agent.set_invariants(_CONV, {"lang": "ru"}, replace=True)
        self.assertEqual(self._reload().list_invariants(_CONV)["invariants"], {"lang": "ru"})

    def test_non_dict_payload_clears_invariants(self) -> None:
        self.agent.set_invariants(_CONV, {"lang": "ru"}, replace=True)
        out = self.agent.set_invariants(_CONV, None, replace=True)
        self.assertEqual(out["invariants"], {})
        self.assertEqual(out["count"], 0)


class TestProfiles(_AgentOnTempFile):
    def test_empty_profile_id_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.agent.upsert_profile(profile_id="  ", name="X", style="", format_pref="", constraints="")

    def test_upsert_persists_and_truncates_fields(self) -> None:
        self.agent.upsert_profile(
            profile_id="qa",
            name="Q" * 200,
            style="s" * 900,
            format_pref="f",
            constraints="c",
        )
        profiles = {p["id"]: p for p in self._reload().list_profiles()["profiles"]}
        self.assertIn("qa", profiles)
        self.assertEqual(len(profiles["qa"]["name"]), 80)
        self.assertEqual(len(profiles["qa"]["style"]), 500)
        self.assertEqual(profiles["qa"]["format"], "f")

    def test_blank_name_falls_back_to_profile_id(self) -> None:
        self.agent.upsert_profile(profile_id="qa", name="   ", style="", format_pref="", constraints="")
        profiles = {p["id"]: p for p in self.agent.list_profiles()["profiles"]}
        self.assertEqual(profiles["qa"]["name"], "qa")

    def test_default_profile_is_listed_first(self) -> None:
        self.agent.upsert_profile(profile_id="alpha", name="A", style="", format_pref="", constraints="")
        listed = self.agent.list_profiles()
        self.assertEqual(listed["profiles"][0]["id"], "default")
        self.assertEqual(listed["default_profile_id"], "default")

    def test_resolve_unknown_profile_falls_back_to_default(self) -> None:
        pid, profile = self.agent._resolve_profile("no-such-profile")
        self.assertEqual(pid, "default")
        self.assertEqual(profile["name"], "Default")


class TestCheckpointsAndBranches(_AgentOnTempFile):
    def _seed_main(self, count: int) -> None:
        state = self.agent._get_conversation_state(_CONV)
        msgs = [{"role": "user", "content": f"m{i}"} for i in range(count)]
        state["full_messages"] = msgs
        state["branches"]["main"]["messages"] = msgs

    def test_checkpoint_of_unknown_branch_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self.agent.create_checkpoint(_CONV, branch_id="no-such-branch")
        self.assertIn("no-such-branch", str(ctx.exception))

    def test_branch_from_unknown_checkpoint_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self.agent.create_branch(_CONV, checkpoint_id="cp_missing")
        self.assertIn("cp_missing", str(ctx.exception))

    def test_branch_inherits_checkpoint_messages(self) -> None:
        self._seed_main(3)
        cp = self.agent.create_checkpoint(_CONV)["checkpoint_id"]
        created = self.agent.create_branch(_CONV, checkpoint_id=cp, branch_name="Идея Б")
        branches = {b["id"]: b for b in self.agent.list_branches(_CONV)["branches"]}
        self.assertIn(created["branch_id"], branches)
        self.assertEqual(branches[created["branch_id"]]["message_count"], 3)
        self.assertEqual(branches[created["branch_id"]]["from_checkpoint"], cp)
        self.assertEqual(branches[created["branch_id"]]["name"], "Идея Б")

    def test_second_branch_gets_free_id(self) -> None:
        cp = self.agent.create_checkpoint(_CONV)["checkpoint_id"]
        first = self.agent.create_branch(_CONV, checkpoint_id=cp)["branch_id"]
        second = self.agent.create_branch(_CONV, checkpoint_id=cp)["branch_id"]
        self.assertNotEqual(first, second)
        self.assertEqual({first, second}, {"branch-1", "branch-2"})

    def test_branches_survive_reload_with_main_first(self) -> None:
        cp = self.agent.create_checkpoint(_CONV)["checkpoint_id"]
        self.agent.create_branch(_CONV, checkpoint_id=cp)
        listed = self._reload().list_branches(_CONV)["branches"]
        self.assertEqual([b["id"] for b in listed], ["main", "branch-1"])

    def test_memory_layers_show_only_short_term_window(self) -> None:
        self._seed_main(WINDOW_SIZE_MESSAGES + 4)
        layers = self.agent.list_memory_layers(_CONV)
        self.assertEqual(layers["short_term"]["window_size"], WINDOW_SIZE_MESSAGES)
        self.assertEqual(len(layers["short_term"]["messages"]), WINDOW_SIZE_MESSAGES)
        self.assertEqual(layers["short_term"]["messages"][-1]["content"], f"m{WINDOW_SIZE_MESSAGES + 3}")

    def test_memory_layers_of_unknown_branch_fall_back_to_main(self) -> None:
        self._seed_main(2)
        layers = self.agent.list_memory_layers(_CONV, branch_id="no-such-branch")
        self.assertEqual(len(layers["short_term"]["messages"]), 2)


class TestMemoryFileRecovery(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.memory_path = Path(self._tmp.name) / "agent_memory.json"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_broken_json_does_not_break_startup(self) -> None:
        self.memory_path.write_text("{ это не json", encoding="utf-8")
        agent = SimpleChatAgent({}, memory_path=self.memory_path)
        self.assertEqual(agent.state_by_conversation, {})
        self.assertIn("default", agent.global_memory["profiles"])

    def test_top_level_list_is_ignored(self) -> None:
        self.memory_path.write_text("[1, 2, 3]", encoding="utf-8")
        agent = SimpleChatAgent({}, memory_path=self.memory_path)
        self.assertEqual(agent.state_by_conversation, {})

    def test_legacy_list_conversation_is_upgraded_to_branches(self) -> None:
        legacy = {_CONV: [{"role": "user", "content": "привет"}]}
        self.memory_path.write_text(json.dumps(legacy, ensure_ascii=False), encoding="utf-8")
        agent = SimpleChatAgent({}, memory_path=self.memory_path)
        listed = agent.list_branches(_CONV)["branches"]
        self.assertEqual([b["id"] for b in listed], ["main"])
        self.assertEqual(listed[0]["message_count"], 1)

    def test_long_term_memory_is_sanitized_on_load(self) -> None:
        raw = {
            GLOBAL_KEY: {
                "long_term": {
                    "language": "ru",
                    "budget": "без цифр",
                    "deadline": "2026-01-01",
                    "api_key": "секрет-который-нельзя-хранить",
                    "profile": "x" * 500,
                }
            }
        }
        self.memory_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        long_term = SimpleChatAgent({}, memory_path=self.memory_path).global_memory["long_term"]
        self.assertEqual(long_term.get("language"), "ru")
        self.assertEqual(long_term.get("deadline"), "2026-01-01")
        self.assertNotIn("api_key", long_term)
        self.assertNotIn("budget", long_term)
        self.assertNotIn("profile", long_term)


if __name__ == "__main__":
    unittest.main()
