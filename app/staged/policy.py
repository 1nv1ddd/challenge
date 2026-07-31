"""Политика приёма заявки: текст правил для промптов и та же политика кодом по нормализованным полям."""

from __future__ import annotations

from datetime import date

from ..agent_constants import (
    INTAKE_MIN_ORDER_RUB,
    INTAKE_REGION_SLA_DAYS,
)
from .schema import IntakeFields, StageDecision

# Правила пронумерованы: порядок проверки — часть политики, а не деталь реализации.
# Один и тот же текст уходит и в монолитный промпт, и в промпт этапа 2 — иначе сравнение нечестное.
_POLICY = """Политика приёма заявки (проверять строго в этом порядке, первое сработавшее правило и есть решение):
1. product = other → reject / product_not_in_catalog
2. region = abroad → reject / region_not_served
3. budget_rub известен и меньше {min_order} ₽ → reject / below_min_order
4. хотя бы одно обязательное поле = unknown → clarify / missing_fields
   (обязательные: product, qty_kg, budget_rub, deadline, region, contact)
5. до deadline осталось меньше срока поставки по региону → reject / deadline_unrealistic
   срок поставки, календарных дней от сегодня: {sla}
6. payment = postpay_60 → clarify / payment_terms_review
7. иначе → accept / ok"""


def policy_text() -> str:
    """Текст политики для промптов — собран из констант, чтобы правила жили в одном месте."""
    sla = ", ".join(f"{region} {days}" for region, days in INTAKE_REGION_SLA_DAYS.items())
    return _POLICY.format(min_order=f"{INTAKE_MIN_ORDER_RUB:,}".replace(",", " "), sla=sla)


def _deadline_days(deadline: str, today: date) -> int | None:
    """Сколько календарных дней от today до deadline; None — если дата не разобралась."""
    try:
        return (date.fromisoformat(deadline) - today).days
    except ValueError:
        return None


def decide_by_rules(fields: IntakeFields, today: date) -> StageDecision:
    """Та же политика, что в промпте, но кодом: на нормализованных полях решение детерминировано."""
    if fields.product == "other":
        return StageDecision("reject", "product_not_in_catalog", source="rules")
    if fields.region == "abroad":
        return StageDecision("reject", "region_not_served", source="rules")
    if fields.budget_rub is not None and fields.budget_rub < INTAKE_MIN_ORDER_RUB:
        return StageDecision("reject", "below_min_order", source="rules")

    missing = fields.missing()
    if missing:
        return StageDecision("clarify", "missing_fields", missing=missing, source="rules")

    days = _deadline_days(fields.deadline, today)
    sla = INTAKE_REGION_SLA_DAYS.get(fields.region)
    if days is not None and sla is not None and days < sla:
        return StageDecision("reject", "deadline_unrealistic", source="rules")
    if fields.payment == "postpay_60":
        return StageDecision("clarify", "payment_terms_review", source="rules")
    return StageDecision("accept", "ok", source="rules")
