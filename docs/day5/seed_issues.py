"""Завести GitHub Issues из backlog.json и вписать их номера обратно (идемпотентно)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_BACKLOG = Path(__file__).with_name("backlog.json")
_LABEL_COLORS = {
    "test": "0e8a16",
    "bug": "d73a4a",
    "refactor": "5319e7",
    "feature": "1d76db",
    "docs": "0075ca",
    "research": "fbca04",
}


def _run(args: list[str]) -> str:
    """Запустить gh и вернуть stdout; при ошибке — пробросить с текстом stderr."""
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(args)} -> {proc.stderr.strip()}")
    return proc.stdout.strip()


def _ensure_labels(repo: str) -> None:
    for name, color in _LABEL_COLORS.items():
        # --force делает создание идемпотентным (обновляет цвет, если лейбл уже есть).
        _run(["gh", "label", "create", f"type:{name}", "--color", color,
              "--repo", repo, "--force"])


def main() -> int:
    data = json.loads(_BACKLOG.read_text(encoding="utf-8"))
    repo = data["meta"]["repo"]
    _ensure_labels(repo)

    created = 0
    for task in data["tasks"]:
        if task.get("issue"):
            continue
        body = (
            f"**Тип:** {task['type']}  ·  **Профиль:** {task['profile']}\n\n"
            f"{task['detail']}\n\n"
            f"**Критерий done:** {task['done']}\n\n"
            f"_Заведено автоматически (Day 5 execution loop, {task['id']})._"
        )
        url = _run([
            "gh", "issue", "create", "--repo", repo,
            "--title", f"[{task['id']}] {task['title']}",
            "--body", body,
            "--label", f"type:{task['type']}",
        ])
        number = int(url.rstrip("/").rsplit("/", 1)[-1])
        task["issue"] = number
        created += 1
        print(f"{task['id']} -> #{number}  {url}", file=sys.stderr)

    _BACKLOG.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Создано issue: {created}; всего задач: {len(data['tasks'])}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
