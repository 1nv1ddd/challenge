"""Константы FSM задачи, памяти и лимитов модели — вынесены из agent для читаемости."""

from __future__ import annotations

MODEL_CONTEXT_LIMITS = {
    "openai/gpt-4o-mini": 128000,
}
INPUT_PRICE_RUB_PER_MILLION = 15.0
OUTPUT_PRICE_RUB_PER_MILLION = 63.0
WINDOW_SIZE_MESSAGES = 10
# Day 7 (advance): триаж обращений поддержки с оценкой уверенности инференса.
TRIAGE_CATEGORIES = ("billing", "technical", "account", "data_loss", "feedback", "other")
TRIAGE_PRIORITIES = ("low", "normal", "high", "critical")
TRIAGE_ACTIONS = ("auto_reply", "request_info", "escalate", "close")
# Действия с необратимым эффектом: их гейт пускает только со статусом OK.
TRIAGE_RISKY_ACTIONS = ("auto_reply", "close")
# Порядок «безопасности» действия — используется при равенстве голосов (чем меньше, тем безопаснее).
TRIAGE_ACTION_SAFETY_RANK = {"escalate": 0, "request_info": 1, "auto_reply": 2, "close": 3}
TRIAGE_SAMPLES = 3
TRIAGE_TEMPERATURE = 0.5
TRIAGE_MAX_REPAIRS = 1
TRIAGE_REASON_MAX_LEN = 400
TRIAGE_SELFCHECK_BELOW = 0.9
TRIAGE_CONFIDENCE_OK = 0.75
TRIAGE_CONFIDENCE_UNSURE = 0.45
TRIAGE_SELFCHECK_BONUS = 0.1
TRIAGE_SELFCHECK_PENALTY = 0.35
# Day 8 (advance): routing между моделями — дешёвый тир с эскалацией на сильный.
ROUTING_SMALL_MODEL = "google/gemma-3n-e4b-it"
ROUTING_LARGE_MODEL = "openai/gpt-4.1"
# Прайс RouterAI, ₽ за миллион токенов: (prompt, completion). Тиры отличаются в десятки раз.
ROUTING_MODEL_PRICES_RUB = {
    "google/gemma-3n-e4b-it": (6.1, 12.2),
    "openai/gpt-4.1-nano": (10.1, 40.6),
    "openai/gpt-4.1-mini": (40.6, 162.3),
    "openai/gpt-4.1": (202.8, 811.4),
    "openai/gpt-4o-mini": (15.2, 60.9),
}
ROUTING_TEMPERATURE = 0.2
# Ниже этого confidence ответ дешёвой модели не принимается — идём на сильную.
ROUTING_ESCALATE_BELOW = 0.7
# Сколько маркеров сложности в запросе, чтобы не тратить вызов на дешёвую модель.
ROUTING_PREROUTE_HARD_SCORE = 2
ROUTING_LONG_QUESTION_CHARS = 400
ROUTING_MIN_ANSWER_CHARS = 40
# Уверенность, когда модель не выдала строку CONFIDENCE (контракт нарушен — доверия меньше).
ROUTING_NO_CONFIDENCE_BASE = 0.5
# Самосогласованность: сколько раз спросить дешёвую модель (1 — проверка выключена).
# Нужна потому, что самооценка мелких моделей насыщена: они пишут CONFIDENCE 1.0 и на ошибках.
ROUTING_CONSISTENCY_SAMPLES = 2
# Дубль берём при повышенной температуре: на температуре основного ответа выборки почти
# совпадают, и расхождение — сигнал, которого просто нет.
ROUTING_CONSISTENCY_TEMPERATURE = 0.8
ROUTING_CONSISTENCY_JACCARD = 0.5
ROUTING_DISAGREE_PENALTY = 0.4
ROUTING_HEDGE_PENALTY = 0.25
ROUTING_SHORT_PENALTY = 0.2
ROUTING_TRUNCATED_PENALTY = 0.4
ROUTING_REFUSAL_PENALTY = 0.6
# Day 9 (advance): декомпозиция инференса — разбор письма-заявки одним запросом или по этапам.
# Значение "unknown" во всех перечислениях — единственный способ сказать «в письме этого нет».
INTAKE_PRODUCTS = ("pipe_steel", "sheet_steel", "rebar", "wire_rope", "fittings", "other", "unknown")
INTAKE_REGIONS = ("moscow", "spb", "ural", "siberia", "south", "abroad", "unknown")
INTAKE_PAYMENTS = ("prepay", "postpay_30", "postpay_60", "unknown")
INTAKE_DECISIONS = ("accept", "clarify", "reject")
INTAKE_REASONS = (
    "ok",
    "missing_fields",
    "below_min_order",
    "deadline_unrealistic",
    "region_not_served",
    "product_not_in_catalog",
    "payment_terms_review",
)
# Порядок полей в компактном формате этапа 1 — он же порядок строк на вход этапа 2.
INTAKE_FIELD_KEYS = (
    "company",
    "product",
    "qty_kg",
    "budget_rub",
    "deadline",
    "region",
    "contact",
    "payment",
)
INTAKE_NUMERIC_FIELDS = ("qty_kg", "budget_rub")
# Без этих полей заявку нельзя ни принять, ни отклонить — только уточнять.
INTAKE_REQUIRED_FIELDS = ("product", "qty_kg", "budget_rub", "deadline", "region", "contact")
INTAKE_MIN_ORDER_RUB = 100_000
# Минимальный срок поставки по регионам (календарные дни от даты обращения).
INTAKE_REGION_SLA_DAYS = {"moscow": 3, "spb": 4, "ural": 7, "siberia": 10, "south": 7}
# Разные модели по этапам: извлечение полей требует аккуратности, enum-решение и письмо — нет.
INTAKE_STAGE_MODELS = {
    "normalize": "openai/gpt-4o-mini",
    "decide": "google/gemma-3n-e4b-it",
    "compose": "google/gemma-3n-e4b-it",
}
INTAKE_MONO_MODEL = "openai/gpt-4.1"
INTAKE_MODES = ("mono", "staged", "staged_rules")
INTAKE_TEMPERATURE = 0.1
INTAKE_MAX_REPAIRS = 1
INTAKE_REPLY_MAX_WORDS = 70
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
