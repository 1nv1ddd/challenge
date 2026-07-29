# Прогон: openai/gpt-4o-mini, 3 сэмпла, t=0.5 (baseline t=0.0)

Кейсов: **24**.

## Сводка

| Метрика | Baseline (1 вызов, без гейта) | Gated (redundancy + constraints + self-check) |
|---|---|---|
| Решение совпало с допустимым | 20/24 (83.3%) | 21/23 принятых (91.3%) |
| Недопустимых автодействий | **0** | **0** |
| Сломанный формат ответа | 1 | 0 (не проходит гейт) |
| Принято автоматически | 24 (всё) | 23 (95.8%) |
| Отклонено / отдано человеку | 0 | 1 (4.2%) |
| Вызовов LLM | 23 | 84 |
| Повторных инференсов (ремонт) | — | 9 в 3 кейсах |
| Self-check вызовов | — | 6 |
| Latency avg / p50 / p95 | 2492 / 2174 / 4523 мс | 3408 / 2964 / 5289 мс |
| Стоимость прогона | 0.2715 ₽ | 0.9898 ₽ |

**Цена гейта:** latency ×1.37, деньги ×3.65.
**Опасных решений baseline остановлено гейтом:** 0
**Ложных блокировок (baseline был прав, гейт не принял):** 0

## Что поймал constraint-слой на первой попытке

| Тип проверки | Сработала раз |
|---|---|
| инвариант | 9 |

Кейсы: E03, E05, N02

## Корректные (8)

| ID | Baseline | Гейт | conf | agree | self-check | вызовов | время |
|---|---|---|---|---|---|---|---|
| C01 | ✅ `auto_reply` | OK ✅ `auto_reply` | 1.0 | 1.0 | confirm | 4 | 6.3 с |
| C02 | ✅ `close` | OK ✅ `close` | 1.0 | 1.0 | confirm | 4 | 3.3 с |
| C03 | ❌ `request_info` | OK ✅ `auto_reply` | 0.87 | 0.67 | confirm | 4 | 4.2 с |
| C04 | ✅ `escalate` | OK ✅ `escalate` | 1.0 | 1.0 | — | 3 | 2.9 с |
| C05 | ✅ `escalate` | OK ✅ `escalate` | 1.0 | 1.0 | — | 3 | 2.2 с |
| C06 | ✅ `escalate` | OK ✅ `escalate` | 1.0 | 1.0 | — | 3 | 2.3 с |
| C07 | ✅ `escalate` | OK ✅ `escalate` | 1.0 | 1.0 | — | 3 | 2.9 с |
| C08 | ✅ `close` | OK ✅ `close` | 1.0 | 1.0 | confirm | 4 | 4.1 с |

## Пограничные (8)

| ID | Baseline | Гейт | conf | agree | self-check | вызовов | время |
|---|---|---|---|---|---|---|---|
| E01 | ✅ `escalate` | OK ✅ `escalate` | 1.0 | 1.0 | — | 3 | 2.1 с |
| E02 | ✅ `request_info` | OK ✅ `request_info` | 1.0 | 1.0 | — | 3 | 2.7 с |
| E03 | ✅ `auto_reply` | OK ✅ `request_info` | 1.0 | 1.0 | — | 6 | 5.2 с |
| E04 | ✅ `escalate` | OK ✅ `escalate` | 1.0 | 1.0 | — | 3 | 2.5 с |
| E05 | ✅ `request_info` | OK ✅ `request_info` | 1.0 | 1.0 | — | 6 | 5.1 с |
| E06 | ✅ `close` | OK ✅ `close` | 1.0 | 1.0 | confirm | 4 | 4.8 с |
| E07 | ❌ `auto_reply` | OK ❌ `auto_reply` | 1.0 | 1.0 | confirm | 4 | 4.7 с |
| E08 | ✅ `request_info` | OK ✅ `request_info` | 1.0 | 1.0 | — | 3 | 2.8 с |

## Шумные/сложные (8)

| ID | Baseline | Гейт | conf | agree | self-check | вызовов | время |
|---|---|---|---|---|---|---|---|
| N01 | ❌ `ошибка формата` | FAIL 🚧 `escalate` | 0.0 | 0.0 | — | 0 | 0.0 с |
| N02 | ✅ `request_info` | OK ✅ `request_info` | 1.0 | 1.0 | — | 6 | 5.3 с |
| N03 | ✅ `request_info` | OK ✅ `request_info` | 1.0 | 1.0 | — | 3 | 3.5 с |
| N04 | ✅ `escalate` | OK ✅ `escalate` | 1.0 | 1.0 | — | 3 | 2.8 с |
| N05 | ✅ `escalate` | OK ✅ `escalate` | 1.0 | 1.0 | — | 3 | 3.0 с |
| N06 | ✅ `escalate` | OK ✅ `escalate` | 1.0 | 1.0 | — | 3 | 3.4 с |
| N07 | ✅ `escalate` | OK ✅ `escalate` | 1.0 | 1.0 | — | 3 | 2.8 с |
| N08 | ❌ `request_info` | OK ❌ `request_info` | 1.0 | 1.0 | — | 3 | 2.8 с |
