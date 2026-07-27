"""Валидация датасета в формате OpenAI chat fine-tuning (JSONL)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REQUIRED_ROLES = ("system", "user", "assistant")
_MIN_CONTENT_CHARS = 3
_MAX_CONTENT_CHARS = 12000


def _check_line(lineno: int, raw: str, seen_users: set[str]) -> list[str]:
    """Проверить одну строку JSONL; вернуть список ошибок (пустой — всё ок)."""
    errors: list[str] = []
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        return [f"строка {lineno}: невалидный JSON ({exc})"]

    if not isinstance(obj, dict) or "messages" not in obj:
        return [f"строка {lineno}: нет ключа 'messages'"]

    messages = obj["messages"]
    if not isinstance(messages, list) or not messages:
        return [f"строка {lineno}: 'messages' должен быть непустым списком"]

    roles: list[str] = []
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            errors.append(f"строка {lineno}: сообщение #{i} не объект")
            continue
        role = msg.get("role")
        content = msg.get("content")
        if role not in _REQUIRED_ROLES:
            errors.append(f"строка {lineno}: сообщение #{i} — недопустимая роль {role!r}")
        if not isinstance(content, str) or not content.strip():
            errors.append(f"строка {lineno}: сообщение #{i} — пустой или нестроковый content")
        elif len(content) > _MAX_CONTENT_CHARS:
            errors.append(f"строка {lineno}: сообщение #{i} — content длиннее {_MAX_CONTENT_CHARS}")
        elif len(content.strip()) < _MIN_CONTENT_CHARS:
            errors.append(f"строка {lineno}: сообщение #{i} — content короче {_MIN_CONTENT_CHARS} символов")
        if isinstance(role, str):
            roles.append(role)

    for required in _REQUIRED_ROLES:
        if required not in roles:
            errors.append(f"строка {lineno}: отсутствует роль '{required}'")

    # Дубли по тексту user-запроса.
    user_texts = [
        m.get("content", "") for m in messages
        if isinstance(m, dict) and m.get("role") == "user"
    ]
    for ut in user_texts:
        key = ut.strip().lower()
        if key and key in seen_users:
            errors.append(f"строка {lineno}: дубль user-запроса")
        seen_users.add(key)

    return errors


def validate_file(path: Path) -> tuple[int, list[str]]:
    """Проверить JSONL-файл; вернуть (число_строк, список_ошибок)."""
    seen_users: set[str] = set()
    all_errors: list[str] = []
    count = 0
    with path.open(encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            if not raw.strip():
                continue
            count += 1
            all_errors.extend(_check_line(lineno, raw, seen_users))
    return count, all_errors


def main() -> int:
    args = sys.argv[1:]
    if not args:
        here = Path(__file__).resolve().parent / "data"
        args = [str(here / "train.jsonl"), str(here / "eval.jsonl")]

    exit_code = 0
    for arg in args:
        path = Path(arg)
        if not path.exists():
            print(f"НЕТ ФАЙЛА: {path}", file=sys.stderr)
            exit_code = 1
            continue
        count, errors = validate_file(path)
        if errors:
            exit_code = 1
            print(f"✗ {path.name}: {count} строк, ошибок {len(errors)}:", file=sys.stderr)
            for e in errors[:50]:
                print(f"  - {e}", file=sys.stderr)
        else:
            print(f"✓ {path.name}: {count} строк, ошибок нет.", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
