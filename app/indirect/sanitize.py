"""Слой 1 — чистка входящего контента: комментарии, скрытые блоки, невидимые символы, ссылки."""

from __future__ import annotations

import re

from ..agent_constants import (
    INDIRECT_ALLOWED_HOSTS,
    INDIRECT_HIDDEN_CSS,
    INDIRECT_STRIPPED_MASK,
    INDIRECT_TAG_RANGE,
    INDIRECT_ZERO_WIDTH,
)
from .schema import SanitizeReport

_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
# Элемент со скрывающим стилем вместе с содержимым: невидимое человеку не должно доехать до модели.
_STYLED_ELEMENT_RE = re.compile(
    r"<(?P<tag>span|div|p|section|td)\b[^>]*style\s*=\s*(?P<q>[\"'])(?P<style>[^\"']*)(?P=q)[^>]*>"
    r".*?</(?P=tag)>",
    re.DOTALL | re.IGNORECASE,
)
_ZERO_WIDTH_RE = re.compile(
    f"[{''.join(INDIRECT_ZERO_WIDTH)}{chr(INDIRECT_TAG_RANGE[0])}-{chr(INDIRECT_TAG_RANGE[1])}]+"
)
# Markdown-ссылка: текст остаётся, url и title — под проверкой хоста.
_MD_LINK_RE = re.compile(r"!?\[(?P<text>[^\]]*)\]\((?P<url>[^)\s]+)(?:\s+\"(?P<title>[^\"]*)\")?\)")
_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")
_HOST_RE = re.compile(r"^[a-z]+://(?P<host>[^/?#]+)", re.IGNORECASE)
# Повелительные обращения к ассистенту внутри данных: в письме и в документе им не место.
_IMPERATIVE_RE = re.compile(
    r"^[^\n]*\b("
    r"игнорируй|забудь|не сообщай|не показывай|не упоминай|обязательно добавь|добавь в конец|"
    r"выполни инструкцию|системное уведомление|инструкция ассистенту|инструкция для ассистента|"
    r"ignore (all )?(previous|prior)|disregard|you must|system note"
    r")\b[^\n]*$",
    re.IGNORECASE | re.MULTILINE,
)


def _style_is_hidden(style: str) -> bool:
    normalized = style.replace(" ", "").lower()
    return any(marker in normalized for marker in INDIRECT_HIDDEN_CSS)


def _host_allowed(url: str) -> bool:
    match = _HOST_RE.match(url.strip())
    if not match:
        # Относительная ссылка или якорь — наружу данные не уводит.
        return True
    host = match.group("host").lower().split("@")[-1].split(":")[0]
    return any(host == allowed or host.endswith(f".{allowed}") for allowed in INDIRECT_ALLOWED_HOSTS)


def sanitize_document(text: str) -> tuple[str, SanitizeReport]:
    """Возвращает очищенный текст и отчёт о том, что было вырезано.

    Порядок важен: сначала снимаем то, что человек не видит (комментарии, скрытые стилем блоки,
    невидимые символы), затем обезвреживаем ссылки и только потом ищем повелительные строки —
    иначе часть из них уже уехала бы вместе со скрытыми блоками и не попала бы в счётчики.
    """
    report = SanitizeReport()
    original_len = len(text or "")
    cleaned = text or ""

    cleaned, report.html_comments = _HTML_COMMENT_RE.subn(INDIRECT_STRIPPED_MASK, cleaned)

    def _drop_hidden(match: re.Match[str]) -> str:
        if _style_is_hidden(match.group("style")):
            report.hidden_elements += 1
            return INDIRECT_STRIPPED_MASK
        return match.group(0)

    cleaned = _STYLED_ELEMENT_RE.sub(_drop_hidden, cleaned)

    zero_width_hits = _ZERO_WIDTH_RE.findall(cleaned)
    report.zero_width = sum(len(hit) for hit in zero_width_hits)
    cleaned = _ZERO_WIDTH_RE.sub("", cleaned)

    def _defang_link(match: re.Match[str]) -> str:
        url = match.group("url")
        title = match.group("title") or ""
        if _host_allowed(url) and not title:
            return match.group(0)
        report.suspicious_links += 1
        label = match.group("text").strip() or "ссылка"
        return f"{label} {INDIRECT_STRIPPED_MASK}"

    cleaned = _MD_LINK_RE.sub(_defang_link, cleaned)
    cleaned = _HTML_TAG_RE.sub("", cleaned)
    cleaned, report.imperative_lines = _IMPERATIVE_RE.subn(INDIRECT_STRIPPED_MASK, cleaned)

    cleaned = "\n".join(line.rstrip() for line in cleaned.splitlines()).strip()
    report.removed_chars = max(0, original_len - len(cleaned))
    return cleaned, report
