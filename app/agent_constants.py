"""Константы FSM задачи, памяти и лимитов модели — вынесены из agent для читаемости."""

from __future__ import annotations

MODEL_CONTEXT_LIMITS = {
    "openai/gpt-4o-mini": 128000,
}
INPUT_PRICE_RUB_PER_MILLION = 15.0
OUTPUT_PRICE_RUB_PER_MILLION = 63.0
WINDOW_SIZE_MESSAGES = 10
INVARIANTS_MAX_ITEMS = 30
INVARIANT_KEY_MAX_LEN = 80
INVARIANT_VAL_MAX_LEN = 600
ALLOWED_STRATEGIES = {"sliding", "facts", "branching"}
GLOBAL_KEY = "__global__"
TASK_PHASES = ("planning", "plan_approved", "execution", "validation", "done")
# Controlled transitions (Day 15): no skips (e.g. execution only after plan_approved).
TASK_ALLOWED_EDGES: dict[str, tuple[str, ...]] = {
    "planning": ("plan_approved",),
    "plan_approved": ("execution",),
    "execution": ("validation",),
    "validation": ("done",),
    "done": (),
}
TASK_EVENT_NEW_TASK = "new_task"
TASK_EVENT_ASSISTANT_TURN_COMPLETED = "assistant_turn_completed"
TASK_EVENT_PAUSE = "pause"
TASK_EVENT_RESUME = "resume"
LONG_TERM_ALLOWED_KEYS = {
    "profile",
    "preferences",
    "decisions",
    "budget",
    "deadline",
    "style",
    "format",
    "language",
    "tone",
}
TASK_PHASE_TO_DEFAULTS = {
    "planning": {
        "current_step": "Define scope and acceptance criteria",
        "expected_action": "Provide goal, constraints, and desired result",
    },
    "plan_approved": {
        "current_step": "Plan approved — implementation only",
        "expected_action": (
            "Implement strictly per approved plan; do not restart planning unless user asks"
        ),
    },
    "execution": {
        "current_step": "Implement the agreed plan",
        "expected_action": "Proceed with implementation and share progress",
    },
    "validation": {
        "current_step": "Verify behavior and quality",
        "expected_action": "Run checks/tests and confirm requirements",
    },
    "done": {
        "current_step": "Task completed",
        "expected_action": "No action required",
    },
}
# Short, phase-specific instructions so the model cannot confuse id "plan_approved" with "still planning".
TASK_PHASE_MODEL_GUIDANCE: dict[str, str] = {
    "planning": (
        "Planning only: scope, risks, acceptance criteria, questions. "
        "No full implementation yet. The plan is NOT approved until the user clearly confirms; "
        "after your reply you remain in planning until they approve (or use Next in UI)."
    ),
    "plan_approved": (
        "CRITICAL: phase plan_approved means the plan is ALREADY APPROVED by the workflow "
        "(not a request to approve again). The user may now ask for module layout, pseudocode, "
        "or code — you MUST produce that. "
        "It is an ERROR to say you are still in 'планирование'/planning or that you cannot write code. "
        "Proceed with implementation-aligned output for this turn."
    ),
    "execution": (
        "Implementation: concrete code, files, steps. No fake task closure or customer sign-off."
    ),
    "validation": (
        "Validation: tests, checklists, evidence. Phase 'done' is only after user confirms closure "
        "(or manual Next); a plain 'continue' does not finish the task."
    ),
    "done": "Done: short wrap-up only.",
}
