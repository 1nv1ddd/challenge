# Прогон: openai/gpt-4o-mini, 3 сэмпла, t=1.4 (baseline t=1.4)

Кейсов: **24**.

## Сводка

| Метрика | Baseline (1 вызов, без гейта) | Gated (redundancy + constraints + self-check) |
|---|---|---|
| Решение совпало с допустимым | 20/24 (83.3%) | 20/22 принятых (90.9%) |
| Недопустимых автодействий | **0** | **0** |
| Сломанный формат ответа | 1 | 0 (не проходит гейт) |
| Принято автоматически | 24 (всё) | 22 (91.7%) |
| Отклонено / отдано человеку | 0 | 2 (8.3%) |
| Вызовов LLM | 23 | 85 |
| Повторных инференсов (ремонт) | — | 9 в 3 кейсах |
| Self-check вызовов | — | 7 |
| Latency avg / p50 / p95 | 2122 / 2177 / 2825 мс | 3844 / 3621 / 8075 мс |
| Стоимость прогона | 0.2718 ₽ | 1.0051 ₽ |

**Цена гейта:** latency ×1.81, деньги ×3.7.
**Опасных решений baseline остановлено гейтом:** 0
**Ложных блокировок (baseline был прав, гейт не принял):** 1 (C08)

## Что поймал constraint-слой на первой попытке

| Тип проверки | Сработала раз |
|---|---|
| инвариант | 9 |

Кейсы: E03, E05, N02

## Корректные (8)

| ID | Baseline | Гейт | conf | agree | self-check | вызовов | время |
|---|---|---|---|---|---|---|---|
| C01 | ❌ `request_info` | OK ✅ `auto_reply` | 1.0 | 1.0 | confirm | 4 | 4.0 с |
| C02 | ✅ `close` | OK ✅ `close` | 1.0 | 1.0 | confirm | 4 | 4.2 с |
| C03 | ✅ `auto_reply` | OK ✅ `auto_reply` | 1.0 | 1.0 | confirm | 4 | 4.3 с |
| C04 | ✅ `escalate` | OK ✅ `escalate` | 1.0 | 1.0 | — | 3 | 2.5 с |
| C05 | ✅ `escalate` | OK ✅ `escalate` | 1.0 | 1.0 | — | 3 | 2.7 с |
| C06 | ✅ `escalate` | OK ✅ `escalate` | 1.0 | 1.0 | — | 3 | 2.4 с |
| C07 | ✅ `escalate` | OK ✅ `escalate` | 1.0 | 1.0 | — | 3 | 4.1 с |
| C08 | ✅ `close` | UNSURE 🚧 `escalate` | 0.65 | 1.0 | reject | 4 | 8.1 с |

## Пограничные (8)

| ID | Baseline | Гейт | conf | agree | self-check | вызовов | время |
|---|---|---|---|---|---|---|---|
| E01 | ✅ `escalate` | OK ✅ `escalate` | 0.87 | 0.67 | confirm | 4 | 10.7 с |
| E02 | ✅ `request_info` | OK ✅ `request_info` | 1.0 | 1.0 | — | 3 | 3.5 с |
| E03 | ✅ `auto_reply` | OK ✅ `request_info` | 0.9 | 1.0 | — | 6 | 4.7 с |
| E04 | ✅ `escalate` | OK ✅ `escalate` | 1.0 | 1.0 | — | 3 | 2.1 с |
| E05 | ✅ `request_info` | OK ✅ `request_info` | 1.0 | 1.0 | — | 6 | 4.7 с |
| E06 | ✅ `close` | OK ✅ `close` | 1.0 | 1.0 | confirm | 4 | 4.4 с |
| E07 | ❌ `auto_reply` | OK ❌ `auto_reply` | 1.0 | 1.0 | confirm | 4 | 3.8 с |
| E08 | ✅ `request_info` | OK ✅ `request_info` | 1.0 | 1.0 | — | 3 | 3.5 с |

## Шумные/сложные (8)

| ID | Baseline | Гейт | conf | agree | self-check | вызовов | время |
|---|---|---|---|---|---|---|---|
| N01 | ❌ `ошибка формата` | FAIL 🚧 `escalate` | 0.0 | 0.0 | — | 0 | 0.0 с |
| N02 | ✅ `request_info` | OK ✅ `request_info` | 1.0 | 1.0 | — | 6 | 5.5 с |
| N03 | ✅ `request_info` | OK ✅ `request_info` | 1.0 | 1.0 | — | 3 | 2.7 с |
| N04 | ✅ `escalate` | OK ✅ `escalate` | 1.0 | 1.0 | — | 3 | 2.7 с |
| N05 | ✅ `escalate` | OK ✅ `escalate` | 1.0 | 1.0 | — | 3 | 1.8 с |
| N06 | ✅ `escalate` | OK ✅ `escalate` | 1.0 | 1.0 | — | 3 | 3.6 с |
| N07 | ✅ `escalate` | OK ✅ `escalate` | 1.0 | 1.0 | — | 3 | 2.5 с |
| N08 | ❌ `request_info` | OK ❌ `request_info` | 1.0 | 1.0 | — | 3 | 3.7 с |
