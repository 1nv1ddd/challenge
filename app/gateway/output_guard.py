"""Output guard: проверка ответа модели на секреты, утечку промпта, чужие ссылки и команды."""

from __future__ import annotations

import re

from ..agent_constants import GATEWAY_ALLOWED_HOSTS
from ..indirect.guard import find_urls, host_allowed, host_of
from ..security.verdict import canary_leak, prompt_leak
from .detectors import mask_text, scan_text
from .schema import OutputFinding, OutputVerdict

# Команды, которые ассистент не должен предлагать выполнить: удаление, запуск скачанного кода,
# раздача прав, слив данных наружу. Ловим формулировку, а не факт запуска — запускает человек.
_DANGEROUS_COMMANDS: tuple[tuple[str, re.Pattern], ...] = (
    ("rm_rf", re.compile(r"\brm\s+-[a-z]*r[a-z]*f?\s+(?:/|~|\$HOME|\*)", re.IGNORECASE)),
    ("pipe_to_shell", re.compile(r"\b(?:curl|wget)\b[^\n|]{0,200}\|\s*(?:sudo\s+)?(?:ba|z|)sh")),
    ("decode_to_shell", re.compile(r"\bbase64\s+(?:-d|--decode)[^\n|]{0,80}\|\s*(?:ba|z|)sh")),
    ("chmod_777", re.compile(r"\bchmod\s+(?:-R\s+)?777\b")),
    ("dd_disk", re.compile(r"\bdd\s+if=[^\s]+\s+of=/dev/[a-z]+", re.IGNORECASE)),
    ("fork_bomb", re.compile(r":\(\)\s*\{\s*:\|:&\s*\}\s*;\s*:")),
    ("eval_download", re.compile(r"\b(?:eval|exec|Invoke-Expression|iex)\b[^\n]{0,60}(?:http|curl|wget)", re.IGNORECASE)),
    ("history_wipe", re.compile(r"\bhistory\s+-c\b|\bshred\s+-[a-z]*u\b", re.IGNORECASE)),
)
# Ответ, в котором модель пересказывает свои инструкции. Канарейку ловит security.verdict,
# но промпт можно вытащить и без дословного совпадения — по этим оборотам.
_PROMPT_DUMP_RE = re.compile(
    r"(мой\s+системный\s+промпт|системн\w+\s+(?:промпт|инструкци\w+)\s*[:—-]|"
    r"вот\s+(?:мои|полные)\s+инструкции|system\s+prompt\s*[:—-]|"
    r"my\s+(?:system\s+prompt|instructions)\s+(?:is|are)\s*[:—-]?)",
    re.IGNORECASE,
)
_URL_MASK = "[REDACTED_URL]"
# Находки, при которых ответ не отдаётся целиком: вырезать по кускам тут бессмысленно.
_CRITICAL_KINDS = ("prompt_leak", "dangerous_command", "secret_leak")
# Всё, кроме букв и цифр: по этому нормализуем и ответ, и охраняемый секрет, чтобы поймать флаг,
# разбитый пробелами, дефисами или переносами («F L A G-1 2 3» → «flag123»).
_ALNUM_RE = re.compile(r"[^0-9a-zа-яё]+", re.IGNORECASE)


def _normalize(text: str) -> str:
    return _ALNUM_RE.sub("", (text or "").lower())


def _secret_leaked(answer: str, secret: str) -> bool:
    """Флаг утёк, если он есть в ответе как есть или после снятия любых разделителей."""
    if not secret:
        return False
    low = (answer or "").lower()
    if secret.lower() in low:
        return True
    return _normalize(secret) in _normalize(answer)


def validate_output(
    answer: str, system_text: str = "", secrets: tuple[str, ...] = ()
) -> list[OutputFinding]:
    """Всё подозрительное в ответе модели: секреты, следы промпта, чужие хосты, опасные команды.

    `secrets` — конкретные охраняемые значения (флаг CTF-стенда): их утечку блокируем целиком,
    даже если общие детекторы такой формат не ловят.
    """
    findings: list[OutputFinding] = []

    for secret in secrets:
        if _secret_leaked(answer, secret):
            findings.append(OutputFinding(kind="secret_leak", detail="охраняемый флаг в ответе"))

    for finding in scan_text(answer):
        findings.append(
            OutputFinding(kind="secret", detail=f"{finding.kind}: {finding.preview}")
        )

    if canary_leak(answer) or (system_text and prompt_leak(answer, system_text)):
        findings.append(OutputFinding(kind="prompt_leak", detail="дословный фрагмент системного промпта"))
    elif match := _PROMPT_DUMP_RE.search(answer or ""):
        findings.append(OutputFinding(kind="prompt_leak", detail=match.group(0).strip()))

    for url in find_urls(answer):
        if not host_allowed(host_of(url), GATEWAY_ALLOWED_HOSTS):
            findings.append(OutputFinding(kind="suspicious_url", detail=url))

    for name, pattern in _DANGEROUS_COMMANDS:
        for match in pattern.finditer(answer or ""):
            findings.append(
                OutputFinding(kind="dangerous_command", detail=f"{name}: {match.group(0).strip()}")
            )
    return findings


def guard_output(
    answer: str, system_text: str = "", enforce: bool = True, secrets: tuple[str, ...] = ()
) -> OutputVerdict:
    """Готовит ответ к выдаче: секреты и чужие ссылки маскируются, критичное режется целиком.

    `enforce=False` — режим «только отчёт» для машинных потребителей: находки считаются и попадают
    в аудит, но текст не правится. Нужен там, где маскирование ломает сам артефакт: в коде
    подменённый URL превращается в `[REDACTED_URL]` и модуль перестаёт работать. Критичные находки
    блокируют ответ в обоих режимах — вырезать из них нечего.
    """
    findings = validate_output(answer, system_text, secrets)
    verdict = OutputVerdict(action="pass", answer=answer or "", raw_answer=answer or "", findings=findings)
    if not findings:
        return verdict

    if any(f.kind in _CRITICAL_KINDS for f in findings):
        verdict.action = "block"
        verdict.answer = _blocked_message(findings)
        return verdict

    if not enforce:
        # Действие в отчёте честное («нашли и замаскировали бы»), но ответ отдаём как есть.
        verdict.action = "mask_reported"
        return verdict

    cleaned, masked = mask_text(answer or "", scan_text(answer))
    for url in find_urls(cleaned):
        if not host_allowed(host_of(url), GATEWAY_ALLOWED_HOSTS):
            cleaned = cleaned.replace(url, _URL_MASK)
            masked += 1
    verdict.action = "mask" if masked else "pass"
    verdict.answer = cleaned
    return verdict


def _blocked_message(findings: list[OutputFinding]) -> str:
    kinds = ", ".join(sorted({f.kind for f in findings if f.kind in _CRITICAL_KINDS}))
    return (
        "Ответ модели не выдан: output guard нашёл в нём "
        f"{kinds}. Запрос записан в аудит, обратитесь к администратору шлюза."
    )
