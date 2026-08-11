"""Input guard: поиск секретов в промпте (в том числе спрятанных), маскирование и вердикт."""

from __future__ import annotations

import base64
import binascii
import re

from ..agent_constants import (
    GATEWAY_BASE64_MASK,
    GATEWAY_BASE64_MIN_CHARS,
    GATEWAY_CREDENTIAL_KINDS,
    GATEWAY_DEFAULT_MODE,
    GATEWAY_MASKABLE_VARIANTS,
    GATEWAY_MODES,
    GATEWAY_REDACTIONS,
    INDIRECT_ZERO_WIDTH,
)
from .schema import InputVerdict, SecretFinding

# Порядок паттернов = приоритет: конкретный вид секрета забирает участок текста раньше общего.
# Каждый элемент — (вид, регулярка, имя группы со значением или "" для всего совпадения).
_PATTERNS: tuple[tuple[str, re.Pattern, str], ...] = (
    ("private_key", re.compile(r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----"), ""),
    (
        "jwt",
        re.compile(r"\beyJ[A-Za-z0-9_\-]{6,}\.eyJ[A-Za-z0-9_\-]{6,}\.[A-Za-z0-9_\-]{6,}"),
        "",
    ),
    # Строка подключения с паролем внутри: найдено живым прогоном, когда модель выдала
    # postgres://user:password@localhost/db — ни один паттерн ключа такое не покрывает.
    (
        "connection_string",
        re.compile(r"\b[a-z][a-z0-9+.\-]*://[^\s:@/]+:[^\s:@/]+@[^\s\"'<>]+"),
        "",
    ),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{8,}"), ""),
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}"), ""),
    (
        "github_token",
        re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{10,}|github_pat_[A-Za-z0-9_]{20,})"),
        "",
    ),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA|AIDA|AROA|AGPA)[0-9A-Z]{12,20}\b"), ""),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{20,}\b"), ""),
    ("slack_token", re.compile(r"\bxox[baprse]-[A-Za-z0-9\-]{10,}"), ""),
    # Карту ищем отдельной функцией (нужна проверка Луна), здесь только PII без арифметики.
    (
        "phone",
        re.compile(r"(?<![\d\-])(?:\+7|8)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}(?!\d)"),
        "",
    ),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b"), ""),
    # Присваивание секрета: ловит то, у чего нет узнаваемого префикса («password: hunter2»).
    (
        "credential_assignment",
        re.compile(
            r"(?i)\b(?:api[_\- ]?key|secret[_\- ]?key|access[_\- ]?token|auth[_\- ]?token"
            r"|client[_\- ]?secret|password|passwd|пароль|токен)\b\s*[:=]\s*"
            r"[\"']?(?P<value>[^\s\"',;]{6,})",
        ),
        "value",
    ),
)
# Номер карты: 13–19 цифр, допускаются пробелы и дефисы между ними.
_CARD_RE = re.compile(r"(?<!\d)(?:\d[ \-]?){12,18}\d(?!\d)")
# Склейка разорванного литерала: `"sk-" + "proj-abc"`, `'ghp' 'xxx'`, перенос строки между кусками.
_CONCAT_RE = re.compile(r"[\"'`]\s*(?:\+|\.|,|&|\|\||concat|\bи\b)?\s*[\"'`]")
# Base64-блоб: без разделителей ключей внутри, поэтому сам ключ этой регуляркой не «съедается».
_BASE64_RE = re.compile(rf"[A-Za-z0-9+/]{{{GATEWAY_BASE64_MIN_CHARS},}}={{0,2}}")
_ZERO_WIDTH_RE = re.compile(f"[{''.join(INDIRECT_ZERO_WIDTH)}]")
# Доля печатных символов, ниже которой декодированный base64 считаем бинарным мусором.
_PRINTABLE_RATIO = 0.9


def _luhn_ok(digits: str) -> bool:
    """Проверка Луна: без неё любой длинный номер заказа улетал бы в находки как карта."""
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def _card_matches(text: str) -> list[tuple[str, str, tuple[int, int]]]:
    """Кандидаты в номера карт, прошедшие Луна — (вид, значение, границы)."""
    out: list[tuple[str, str, tuple[int, int]]] = []
    for match in _CARD_RE.finditer(text or ""):
        digits = re.sub(r"\D", "", match.group(0))
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            out.append(("card", match.group(0), match.span()))
    return out


def _raw_matches(text: str) -> list[tuple[str, str, tuple[int, int]]]:
    """Все совпадения детекторов без пересечений: приоритет у более конкретного паттерна."""
    kept: list[tuple[str, str, tuple[int, int]]] = []
    taken: list[tuple[int, int]] = []

    def _free(span: tuple[int, int]) -> bool:
        return not any(span[0] < end and start < span[1] for start, end in taken)

    for kind, pattern, group in _PATTERNS:
        for match in pattern.finditer(text or ""):
            span = match.span(group) if group else match.span()
            if span[0] < 0 or not _free(span):
                continue
            kept.append((kind, match.group(group) if group else match.group(0), span))
            taken.append(span)
    for kind, value, span in _card_matches(text):
        if _free(span):
            kept.append((kind, value, span))
            taken.append(span)
    return sorted(kept, key=lambda item: item[2][0])


def _joined_text(text: str) -> str:
    """Текст со склеенными литералами и без невидимых символов — так виден разорванный ключ."""
    return _CONCAT_RE.sub("", _ZERO_WIDTH_RE.sub("", text or ""))


def _decoded(blob: str) -> str:
    """Расшифрованный base64 или пустая строка, если это не текст (или вообще не base64)."""
    padded = blob + "=" * (-len(blob) % 4)
    try:
        data = base64.b64decode(padded, validate=True)
    except (binascii.Error, ValueError):
        return ""
    try:
        decoded = data.decode("utf-8")
    except UnicodeDecodeError:
        return ""
    if not decoded:
        return ""
    printable = sum(1 for ch in decoded if ch.isprintable() or ch in "\n\r\t")
    return decoded if printable / len(decoded) >= _PRINTABLE_RATIO else ""


def scan_text(text: str) -> list[SecretFinding]:
    """Секреты в промпте: прямо в тексте, внутри base64 и в разорванном на куски виде.

    Один и тот же секрет не дублируется: находка из преобразованного текста отбрасывается,
    если тот же самый секрет уже найден в исходном (сравнение по отпечатку значения).
    """
    findings = [
        SecretFinding.make(kind, value, span=span) for kind, value, span in _raw_matches(text)
    ]
    seen = {f.digest for f in findings}

    for match in _BASE64_RE.finditer(text or ""):
        decoded = _decoded(match.group(0))
        if not decoded:
            continue
        for kind, value, _span in _raw_matches(decoded):
            finding = SecretFinding.make(kind, value, variant="base64", span=match.span())
            if finding.digest not in seen:
                findings.append(finding)
                seen.add(finding.digest)

    joined = _joined_text(text)
    if joined != (text or ""):
        for kind, value, _span in _raw_matches(joined):
            finding = SecretFinding.make(kind, value, variant="joined")
            if finding.digest not in seen:
                findings.append(finding)
                seen.add(finding.digest)
    return findings


def mask_text(text: str, findings: list[SecretFinding]) -> tuple[str, int]:
    """Заменяет находки с известными границами на плейсхолдеры; возвращает (текст, сколько скрыл).

    Идём справа налево, чтобы границы предыдущих находок не съезжали после подстановки.
    """
    out = text or ""
    maskable = [
        f
        for f in findings
        if f.variant in GATEWAY_MASKABLE_VARIANTS and f.start >= 0 and f.end > f.start
    ]
    for finding in sorted(maskable, key=lambda f: f.start, reverse=True):
        mask = (
            GATEWAY_BASE64_MASK
            if finding.variant == "base64"
            else GATEWAY_REDACTIONS.get(finding.kind, "[REDACTED]")
        )
        out = out[: finding.start] + mask + out[finding.end :]
    return out, len(maskable)


def _warning(findings: list[SecretFinding], action: str, hidden_only: bool) -> str:
    """Человекочитаемое предупреждение: что нашли и почему запрос так обработан."""
    kinds = ", ".join(sorted({f.kind for f in findings}))
    if action == "block":
        reason = (
            "секрет виден только после склейки кусков — маскировать нечего, точных границ в "
            "исходном тексте нет"
            if hidden_only
            else "учётные данные в промпт не пропускаются: ключ уже скомпрометирован, его нужно "
            "отозвать, а не прятать"
        )
        return f"Запрос заблокирован input guard. Найдено: {kinds}. Причина: {reason}."
    return f"Секреты замаскированы перед отправкой в модель. Найдено: {kinds}."


def guard_input(text: str, mode: str = GATEWAY_DEFAULT_MODE) -> InputVerdict:
    """Вердикт по промпту: пропустить как есть, замаскировать или заблокировать.

    Режим `hybrid` — рабочий по умолчанию: ПДн маскируются (запрос всё ещё полезен), а учётные
    данные блокируются, потому что замаскировать утёкший ключ — не то же самое, что не потерять его.
    """
    if mode not in GATEWAY_MODES:
        raise ValueError(f"Неизвестный режим guard: {mode!r}; доступны: {', '.join(GATEWAY_MODES)}.")
    prompt = text or ""
    if mode == "off":
        return InputVerdict(action="pass", prompt=prompt)

    findings = scan_text(prompt)
    if not findings:
        return InputVerdict(action="pass", prompt=prompt)

    hidden_only = any(f.variant not in GATEWAY_MASKABLE_VARIANTS for f in findings)
    has_credentials = any(f.kind in GATEWAY_CREDENTIAL_KINDS for f in findings)
    block = mode == "block" or hidden_only or (mode == "hybrid" and has_credentials)
    masked_prompt, masked = mask_text(prompt, findings)
    if block:
        # В block-режиме в модель не уходит ничего, но маскированный текст нужен аудиту.
        return InputVerdict(
            action="block",
            prompt=masked_prompt,
            findings=findings,
            masked=masked,
            warning=_warning(findings, "block", hidden_only),
        )
    return InputVerdict(
        action="mask",
        prompt=masked_prompt,
        findings=findings,
        masked=masked,
        warning=_warning(findings, "mask", hidden_only),
    )
