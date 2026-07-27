"""Baseline: прогон 10 примеров из eval через базовый gpt-4o-mini без файнтюна."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import httpx

# Провайдеры baseline. openai — official flow (нужен OPENAI_API_KEY);
# routerai — прокси к тому же gpt-4o-mini (ключ уже есть в .env проекта).
_PROVIDERS = {
    "openai": {
        "url": "https://api.openai.com/v1/chat/completions",
        "key_env": "OPENAI_API_KEY",
        "model": "gpt-4o-mini",
    },
    "routerai": {
        "url": "https://routerai.ru/api/v1/chat/completions",
        "key_env": "ROUTERAI_API_KEY",
        "model": "openai/gpt-4o-mini",
    },
}

_N_SAMPLES = 10


def _load_env_file(path: Path) -> None:
    """Подтянуть KEY=VALUE из .env в окружение, не перетирая уже заданные."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")


def _read_eval(path: Path, n: int) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for raw in f:
            if raw.strip():
                records.append(json.loads(raw))
            if len(records) >= n:
                break
    return records


def _call(url: str, api_key: str, model: str, system: str, user: str) -> tuple[str, int]:
    """Один запрос к chat/completions; вернуть (ответ, время_мс)."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "stream": False,
    }
    t0 = time.monotonic()
    with httpx.Client(timeout=120) as client:
        r = client.post(url, json=body, headers=headers)
        r.raise_for_status()
        payload = r.json()
    elapsed_ms = round((time.monotonic() - t0) * 1000)
    choice = (payload.get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content") or ""
    return content, elapsed_ms


def main() -> int:
    here = Path(__file__).resolve().parent
    _load_env_file(here.parent / ".env")

    provider_name = os.environ.get("BASELINE_PROVIDER", "openai").strip().lower()
    for arg in sys.argv[1:]:
        if arg.startswith("--provider="):
            provider_name = arg.split("=", 1)[1].strip().lower()

    provider = _PROVIDERS.get(provider_name)
    if provider is None:
        print(f"Неизвестный провайдер: {provider_name!r}. Доступны: {', '.join(_PROVIDERS)}",
              file=sys.stderr)
        return 2

    api_key = os.environ.get(provider["key_env"], "").strip()
    if not api_key:
        print(
            f"Нет ключа {provider['key_env']} для baseline через '{provider_name}'.\n"
            f"Задайте его в окружении или .env, либо смените провайдера:\n"
            f"  python finetune/baseline.py --provider=routerai",
            file=sys.stderr,
        )
        return 1

    eval_path = here / "data" / "eval.jsonl"
    records = _read_eval(eval_path, _N_SAMPLES)
    if len(records) < _N_SAMPLES:
        print(f"В eval только {len(records)} примеров, нужно {_N_SAMPLES}.", file=sys.stderr)
        return 1

    results: list[dict] = []
    for i, rec in enumerate(records, start=1):
        msgs = {m["role"]: m["content"] for m in rec["messages"]}
        system, user, reference = msgs["system"], msgs["user"], msgs["assistant"]
        try:
            answer, ms = _call(provider["url"], api_key, provider["model"], system, user)
        except httpx.HTTPError as exc:
            print(f"Пример {i}: ошибка запроса — {exc}", file=sys.stderr)
            return 1
        results.append({"n": i, "user": user, "reference": reference,
                        "baseline_answer": answer, "time_ms": ms})
        print(f"Пример {i}/{_N_SAMPLES}: {ms} мс", file=sys.stderr)

    # Сырые данные (JSON) + человекочитаемый отчёт (Markdown).
    (here / "baseline_outputs.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_markdown(here / "baseline_outputs.md", provider_name, provider["model"], results)
    print(f"Готово: {len(results)} baseline-ответов "
          f"(провайдер {provider_name}, модель {provider['model']}).", file=sys.stderr)
    return 0


def _write_markdown(path: Path, provider: str, model: str, results: list[dict]) -> None:
    lines = [
        f"# Baseline ({model}, без файнтюна)",
        "",
        f"Провайдер: **{provider}**. Температура 0.0. Примеров: {len(results)} (из eval).",
        "",
        "Точка отсчёта: как базовая модель отвечает на те же запросы ДО дообучения. "
        "После файнтюна сравниваем по критериям из `criteria.md`.",
        "",
    ]
    for r in results:
        lines += [
            f"## Пример {r['n']} ({r['time_ms']} мс)",
            "",
            "**User:**",
            "",
            f"> {r['user']}",
            "",
            "**Baseline-ответ:**",
            "",
            "````python",
            r["baseline_answer"].strip(),
            "````",
            "",
            "**Эталон (reference):**",
            "",
            "````python",
            r["reference"].strip(),
            "````",
            "",
            "---",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
