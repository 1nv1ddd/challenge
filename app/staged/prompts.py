"""Промпты трёх этапов и монолитного варианта: правила нормализации и политика — общий текст."""

from __future__ import annotations

from ..agent_constants import INTAKE_REPLY_MAX_WORDS
from ..providers import Message
from .policy import policy_text
from .schema import IntakeFields, StageDecision

# Правила нормализации живут в одной строке и уходят и в этап 1, и в монолитный промпт:
# иначе разница между вариантами объяснялась бы разными инструкциями, а не декомпозицией.
_NORMALIZE_RULES = """Правила нормализации:
- company — организация в виде «ООО Ромашка» (форма + название, без кавычек), иначе unknown.
- product — один из: pipe_steel (трубы), sheet_steel (лист, листовой прокат), rebar (арматура),
  wire_rope (тросы, канаты), fittings (фитинги, отводы, фланцы). Запрошено то, чего нет в этом
  каталоге (швеллер, уголок, профнастил, метизы и прочее) → other. Товар не назван → unknown.
- qty_kg — объём в килограммах, целым числом: «1,5 т» → 1500, «полторы тонны» → 1500,
  «800 кг» → 800, «2 тонны» → 2000. Объёма нет → unknown.
- budget_rub — бюджет в рублях, целым числом: «300к» → 300000, «1,2 млн» → 1200000,
  «до 500 тыс» → 500000. Диапазон «от X до Y» → Y. Бюджета нет → unknown.
- deadline — дата в формате YYYY-MM-DD, считая от даты обращения: «завтра» → +1 день,
  «через N дней» → +N, «через две недели» → +14, «в течение недели» → +7,
  «до конца месяца» → последний день месяца обращения, «в начале августа» → 1-е августа,
  «в середине августа» → 15-е августа. Прямая дата берётся как есть. Срока нет → unknown.
- region — по городу доставки: moscow (Москва и область), spb (Санкт-Петербург, Ленобласть),
  ural (Екатеринбург, Челябинск, Пермь, Тюмень, Уфа), siberia (Новосибирск, Красноярск, Омск,
  Иркутск), south (Ростов-на-Дону, Краснодар, Сочи, Волгоград, Ставрополь), abroad (город вне
  России). Город не назван или не подпадает ни под одну группу → unknown. Опечатки в названии
  города игнорируй, ориентируйся на смысл.
- contact — телефон в виде +7XXXXXXXXXX (8 и +7 приводи к +7, скобки и дефисы убирай) либо email
  в нижнем регистре. Если есть и то и другое — телефон. Контакта нет → unknown.
- payment — prepay (предоплата), postpay_30 (отсрочка до 30 дней включительно),
  postpay_60 (отсрочка больше 30 дней), условия не названы → unknown."""

_FIELDS_FORMAT = """Формат ответа — ровно 8 строк «ключ: значение», в этом порядке, без пояснений,
без markdown и без пустых строк:
company: <текст|unknown>
product: <enum|unknown>
qty_kg: <целое|unknown>
budget_rub: <целое|unknown>
deadline: <YYYY-MM-DD|unknown>
region: <enum|unknown>
contact: <+7XXXXXXXXXX|email|unknown>
payment: <enum|unknown>"""

_DECISION_FORMAT = """Формат ответа — ровно 3 строки, без пояснений и без markdown:
DECISION: <accept|clarify|reject>
REASON: <ok|missing_fields|below_min_order|deadline_unrealistic|region_not_served|product_not_in_catalog|payment_terms_review>
MISSING: перечисление полей через запятую, если REASON = missing_fields; иначе один дефис

В строке DECISION допустимы только accept, clarify, reject; код причины идёт в REASON.
Пример правильного ответа:
DECISION: clarify
REASON: missing_fields
MISSING: budget_rub, contact"""

_REPLY_RULES = f"""Письмо клиенту: по-деловому, на русском, без приветственной воды,
не длиннее {INTAKE_REPLY_MAX_WORDS} слов. Цены, сроки и скидки не выдумывай — опирайся только на
поля заявки. accept — подтверждаем приём заявки в работу; clarify — просим ровно те данные,
которых не хватает; reject — вежливо отказываем и называем причину человеческими словами."""

_REPLY_FORMAT = """Формат ответа: первая строка «SUBJECT: <тема>», дальше текст письма.
Ничего кроме темы и текста."""

SYSTEM_STAGE_NORMALIZE = (
    "Ты — парсер входящих заявок. Твоя единственная задача — вытащить поля из письма и привести "
    "их к канону. Решения по заявке не принимай, письмо клиенту не пиши, лишнего не добавляй.\n\n"
    f"{_NORMALIZE_RULES}\n\n{_FIELDS_FORMAT}"
)

SYSTEM_STAGE_DECIDE = (
    "Ты — правило приёма заявок. На входе уже нормализованные поля, извлекать ничего не нужно. "
    "Применяй политику буквально и по порядку.\n\n"
    f"{policy_text()}\n\n{_DECISION_FORMAT}"
)

SYSTEM_STAGE_COMPOSE = (
    "Ты — менеджер отдела продаж. Решение по заявке уже принято, оспаривать его нельзя: "
    "твоя задача — сообщить его клиенту.\n\n"
    f"{_REPLY_RULES}\n\n{_REPLY_FORMAT}"
)

SYSTEM_MONOLITHIC = (
    "Ты — обработчик входящих заявок. По письму клиента сделай всё сразу: извлеки и нормализуй "
    "поля, примени политику приёма и напиши ответ клиенту.\n\n"
    f"{_NORMALIZE_RULES}\n\n{policy_text()}\n\n{_REPLY_RULES}\n\n"
    "Формат ответа — три блока подряд, без markdown и без пояснений между ними.\n\n"
    # Шаблоны те же, что у отдельных этапов: разница между вариантами должна быть
    # в декомпозиции, а не в том, что монолиту хуже объяснили формат.
    f"Блок 1. {_FIELDS_FORMAT}\n\n"
    f"Блок 2. {_DECISION_FORMAT}\n\n"
    f"Блок 3. {_REPLY_FORMAT}"
)


def _letter_block(letter: str, today: str) -> str:
    return f"Дата обращения: {today}\n\nПисьмо клиента:\n{letter.strip()}"


def stage_normalize_messages(letter: str, today: str) -> list[Message]:
    """Этап 1: письмо → нормализованные поля."""
    return [
        Message(role="system", content=SYSTEM_STAGE_NORMALIZE),
        Message(role="user", content=_letter_block(letter, today)),
    ]


def stage_decide_messages(fields: IntakeFields, today: str) -> list[Message]:
    """Этап 2: поля → решение. На вход идут 8 строк, а не письмо — потому запрос дешёвый."""
    return [
        Message(role="system", content=SYSTEM_STAGE_DECIDE),
        Message(
            role="user",
            content=f"Сегодня: {today}\n\nЗаявка:\n{fields.to_compact()}",
        ),
    ]


def stage_compose_messages(fields: IntakeFields, decision: StageDecision) -> list[Message]:
    """Этап 3: поля + решение → письмо клиенту."""
    missing = ", ".join(decision.missing) if decision.missing else "-"
    return [
        Message(role="system", content=SYSTEM_STAGE_COMPOSE),
        Message(
            role="user",
            content=(
                f"Заявка:\n{fields.to_compact()}\n\n"
                f"Решение: {decision.decision}\nПричина: {decision.reason}\n"
                f"Не хватает данных: {missing}"
            ),
        ),
    ]


def monolithic_messages(letter: str, today: str) -> list[Message]:
    """Вариант A: один большой запрос — извлечение, решение и письмо в одном ответе."""
    return [
        Message(role="system", content=SYSTEM_MONOLITHIC),
        Message(role="user", content=_letter_block(letter, today)),
    ]


def repair_messages(base: list[Message], raw: str, error: str) -> list[Message]:
    """Повторный инференс того же этапа: показываем модели её ответ и в чём он нарушил формат."""
    return [
        *base,
        Message(role="assistant", content=raw),
        Message(
            role="user",
            content=(
                f"Ответ не прошёл проверку: {error}\n"
                "Верни тот же результат заново, строго в требуемом формате. "
                "Никаких пояснений, markdown и лишних строк."
            ),
        ),
    ]
