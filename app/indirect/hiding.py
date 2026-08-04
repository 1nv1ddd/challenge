"""Пять способов спрятать инструкцию в контенте так, чтобы человек её не увидел."""

from __future__ import annotations

from ..agent_constants import INDIRECT_HIDING, INDIRECT_TAG_BASE, INDIRECT_TAG_RANGE

# Куда вставляем payload, если в носителе нет явного места: середина текста незаметнее хвоста.
_ANCHOR = "\n"


def _split_carrier(carrier: str) -> tuple[str, str]:
    """Делит носитель пополам по строкам: payload прячем внутрь, а не приклеиваем в конец."""
    lines = carrier.splitlines()
    if len(lines) < 2:
        return carrier, ""
    middle = len(lines) // 2
    return "\n".join(lines[:middle]), "\n".join(lines[middle:])


def _as_tag_chars(text: str) -> str:
    """ASCII smuggling: печатный ASCII → невидимые Unicode Tag Characters.

    Символ U+0041 «A» превращается в U+E0041 — ничего не рисуется, но токенизатор видит текст.
    Всё, что вне печатного ASCII (кириллица), закодировать нельзя, поэтому payload такой
    техники пишется по-английски.
    """
    low, high = INDIRECT_TAG_RANGE
    return "".join(
        chr(INDIRECT_TAG_BASE + ord(ch)) if low <= INDIRECT_TAG_BASE + ord(ch) <= high else ""
        for ch in text
    )


def decode_zero_width(text: str) -> str:
    """Обратная операция — нужна тестам и разбору инцидента: что именно было спрятано."""
    low, high = INDIRECT_TAG_RANGE
    return "".join(chr(ord(ch) - INDIRECT_TAG_BASE) for ch in text if low <= ord(ch) <= high)


def hide_payload(carrier: str, payload: str, technique: str) -> str:
    """Собирает документ: видимый носитель плюс payload, спрятанный выбранной техникой."""
    if technique not in INDIRECT_HIDING:
        raise ValueError(
            f"Неизвестная техника сокрытия: {technique!r}; доступны: {', '.join(INDIRECT_HIDING)}."
        )
    head, tail = _split_carrier(carrier)
    if technique == "html_comment":
        hidden = f"<!-- {payload} -->"
    elif technique == "white_text":
        hidden = f'<span style="color:#ffffff;background:#ffffff">{payload}</span>'
    elif technique == "tiny_font":
        hidden = f'<div style="font-size:0px;opacity:0">{payload}</div>'
    elif technique == "zero_width":
        # Видимая часть — безобидная строка, инструкция живёт в невидимых символах после неё.
        hidden = f"Спасибо за обращение.{_as_tag_chars(payload)}"
    else:
        # markdown_link: payload уезжает в title ссылки и в query — оба места пользователь не видит.
        hidden = (
            f'[подробности во вложении](https://cdn.files-share.example/doc?note={payload.replace(" ", "%20")} '
            f'"{payload}")'
        )
    return f"{head}{_ANCHOR}{hidden}{_ANCHOR}{tail}".strip()


def visible_text(document: str) -> str:
    """Что из документа реально видит человек: без комментариев, скрытых блоков и невидимых символов.

    Используется как эталон в отчёте: показывает, что payload действительно не виден глазами.
    """
    from .sanitize import sanitize_document

    cleaned, _ = sanitize_document(document)
    return cleaned
