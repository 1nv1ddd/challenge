"""Слой 3 — проверка ответа агента: сверка с видимой частью источника и опасные конструкции."""

from __future__ import annotations

import re

from ..agent_constants import INDIRECT_ALLOWED_HOSTS, INDIRECT_MAX_ANSWER_CHARS
from ..security.verdict import canary_leak, prompt_leak
from .schema import GuardFinding

# Якоря факта: коды, суммы, проценты, почта, ссылки. Их агент не имеет права выдумывать —
# если якоря нет в видимой части источника, он пришёл из скрытой (или из головы модели).
_URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\)\]\"']+", re.IGNORECASE)
_CODE_RE = re.compile(r"\b[A-Z][A-Z0-9]{2,}(?:[-_][A-Z0-9]{2,})+\b")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b", re.IGNORECASE)
_MONEY_RE = re.compile(r"\b\d[\d\s ]{2,}(?:₽|руб\.?|rub|\$|€)", re.IGNORECASE)
# Числа с разделителем тысяч: подменённую цену или сумму так ловим и без знака валюты.
_BIG_NUMBER_RE = re.compile(r"\b\d{1,3}(?:[\s ]\d{3})+\b")
_PERCENT_RE = re.compile(r"\b\d{1,3}(?:[.,]\d+)?\s?%")
_HOST_RE = re.compile(r"^(?:https?://)?(?P<host>[^/?#]+)", re.IGNORECASE)
# Просьбы, которых в ответе ассистента быть не должно: так уводят коды, пароли и деньги.
_SECRET_REQUEST_RE = re.compile(
    r"(код(?:\s+\w+){0,2}\s+из\s+(?:sms|смс|письма)|пришлите\s+(?:код|пароль|токен)|"
    r"отправьте\s+(?:код|пароль|токен|реквизиты)|введите\s+(?:пароль|код подтверждения)|"
    r"переведите\s+\d|подтвердите\s+личность,?\s+отправив)",
    re.IGNORECASE,
)
# Эхо чужой инструкции: агент пересказывает то, что ему подсунули в данных.
_INSTRUCTION_ECHO_RE = re.compile(
    r"(не\s+сообщай\w*\s+пользовател|не\s+упоминай\w*\s+(?:эту|эт[ои])|"
    r"игнорируй\w*\s+(?:предыдущие|все)\s+инструкции|системное\s+уведомление\s+для\s+ассистента)",
    re.IGNORECASE,
)
_WS_RE = re.compile(r"\s+")
# Рекомендация поставить пакет: имя в бэктиках или после install-команды. Классика supply-chain —
# вредоносен не код репозитория, а совет ассистента поставить чужой пакет.
_BACKTICK_RE = re.compile(r"`([^`\n]{3,60})`")
_INSTALL_RE = re.compile(
    r"(?:pip3?\s+install|npm\s+i(?:nstall)?|yarn\s+add|poetry\s+add)\s+(?P<pkg>[\w.\-]{3,60})",
    re.IGNORECASE,
)
_PACKAGE_LIKE_RE = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)+$")
# Строка без строчных букв — вердикт или лозунг («ОТЧЁТ ПРИНЯТ БЕЗ ЗАМЕЧАНИЙ»), а не пересказ.
_CAPS_LINE_RE = re.compile(r"^[^a-zа-яё\n]{15,}$", re.MULTILINE)
# Маркеры пункта списка: по ним проверяется, что аналитик не потерял свой формат из трёх пунктов.
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+\S", re.MULTILINE)
_ANALYZE_MIN_ITEMS = 3
# Находки, при которых ответ отдавать нельзя целиком: вырезать по строкам тут бессмысленно.
_CRITICAL_KINDS = ("prompt_leak", "secret_request")


def _normalize(text: str) -> str:
    return _WS_RE.sub(" ", (text or "").lower())


def find_urls(text: str) -> list[str]:
    """Ссылки из текста: общий детектор для проверок выхода (День 12 и шлюз Дня 13)."""
    return [m.group(0) for m in _URL_RE.finditer(text or "")]


def host_of(url: str) -> str:
    match = _HOST_RE.match(url.strip())
    host = match.group("host").lower() if match else ""
    return host.removeprefix("www.").split(":")[0]


def host_allowed(host: str, allowed: tuple[str, ...] = INDIRECT_ALLOWED_HOSTS) -> bool:
    """Хост в белом списке или его поддомен."""
    return any(host == item or host.endswith(f".{item}") for item in allowed)


def _anchors(text: str) -> list[tuple[str, str]]:
    """Якоря ответа как (тип, значение) — по ним проверяется обоснованность видимым источником."""
    found: list[tuple[str, str]] = []
    for kind, pattern in (
        ("url", _URL_RE),
        ("code", _CODE_RE),
        ("email", _EMAIL_RE),
        ("money", _MONEY_RE),
        ("number", _BIG_NUMBER_RE),
        ("percent", _PERCENT_RE),
    ):
        found.extend((kind, m.group(0)) for m in pattern.finditer(text or ""))
    return found


def _dependency_findings(answer: str, visible: str) -> list[GuardFinding]:
    """Пакеты и файлы, которые агент советует, но которых нет в видимой части источника."""
    found: list[GuardFinding] = []
    names = {m.group(1).strip() for m in _BACKTICK_RE.finditer(answer or "")}
    names |= {m.group("pkg").strip() for m in _INSTALL_RE.finditer(answer or "")}
    for name in sorted(names):
        if _normalize(name) in visible or not _PACKAGE_LIKE_RE.match(name.lower()):
            continue
        found.append(GuardFinding(kind="unsupported_dependency", detail=name))
    return found


def validate_output(
    answer: str, visible_source: str, system_text: str = "", scenario: str = ""
) -> list[GuardFinding]:
    """Ищет в ответе следы инъекции, сверяясь с тем, что в источнике видел человек.

    Ключевой момент: эталон — видимая часть документа, а не документ целиком. Всё, что агент взял
    из скрытого блока, оказывается фактом без опоры и попадает в находки.
    """
    findings: list[GuardFinding] = []
    visible = _normalize(visible_source)

    for kind, value in _anchors(answer):
        if _normalize(value) in visible:
            continue
        if kind == "url" and host_allowed(host_of(value)):
            continue
        findings.append(
            GuardFinding(
                kind="external_url" if kind == "url" else "unsupported_fact",
                detail=value.strip(),
            )
        )

    findings.extend(_dependency_findings(answer, visible))

    for line in _CAPS_LINE_RE.findall(answer or ""):
        if _normalize(line) not in visible:
            findings.append(GuardFinding(kind="unsupported_claim", detail=line.strip()))

    if scenario == "analyze" and len(_LIST_ITEM_RE.findall(answer or "")) < _ANALYZE_MIN_ITEMS:
        findings.append(
            GuardFinding(kind="format_broken", detail="в ответе меньше трёх пунктов списка")
        )

    for match in _SECRET_REQUEST_RE.finditer(answer or ""):
        findings.append(GuardFinding(kind="secret_request", detail=match.group(0).strip()))
    for match in _INSTRUCTION_ECHO_RE.finditer(answer or ""):
        findings.append(GuardFinding(kind="instruction_echo", detail=match.group(0).strip()))

    if canary_leak(answer) or (system_text and prompt_leak(answer, system_text)):
        findings.append(GuardFinding(kind="prompt_leak", detail="в ответе фрагмент системного промпта"))
    if len(answer or "") > INDIRECT_MAX_ANSWER_CHARS:
        findings.append(
            GuardFinding(kind="too_long", detail=f"{len(answer)} символов — похоже на дамп документа")
        )
    return findings


def apply_guard(answer: str, findings: list[GuardFinding]) -> tuple[str, bool]:
    """Готовит ответ к выдаче: (что отдаём пользователю, заблокирован ли ответ целиком).

    Некритичные находки вырезаются построчно — полезная часть ответа доезжает. Критичные
    (утечка промпта, просьба прислать код) режут ответ целиком: точечная правка тут не спасает.
    """
    if not findings:
        return answer, False
    if any(f.kind in _CRITICAL_KINDS for f in findings):
        return _blocked_message(findings), True

    details = [f.detail for f in findings if f.detail]
    kept = [
        line
        for line in (answer or "").splitlines()
        if not any(detail and detail.lower() in line.lower() for detail in details)
    ]
    cleaned = "\n".join(kept).strip()
    if not cleaned:
        return _blocked_message(findings), True
    return cleaned, False


def _blocked_message(findings: list[GuardFinding]) -> str:
    kinds = ", ".join(sorted({f.kind for f in findings}))
    return (
        "Ответ не выдан: проверка на выходе нашла следы инструкции из внешнего контента "
        f"({kinds}). Материал отправлен на ручную проверку."
    )
