# Прогон: монолит `openai/gpt-4.1` против цепочки этапов, t=0.1

Писем: **24**. Каждое прогоняется всеми стратегиями на одних и тех же правилах нормализации и одной и той же политике приёма.

- монолит, сильная модель — `openai/gpt-4.1`
- монолит, дешёвая модель — `google/gemma-3n-e4b-it`
- этапы — нормализация `openai/gpt-4o-mini`, решение `google/gemma-3n-e4b-it`, письмо `google/gemma-3n-e4b-it`
- этапы с решением кодом — те же модели на этапах 1 и 3, политика применяется без LLM

## Сводка

| Метрика | A. Монолит, сильная | A. Монолит, дешёвая | B. Этапы, решение моделью | B. Этапы, решение кодом |
|---|---|---|---|---|
| Вердикт верен | **19/24** (79.2%) | **11/24** (45.8%) | **8/24** (33.3%) | **24/24** (100.0%) |
| Все 8 полей верны | 22/24 | 16/24 | 21/24 | 21/24 |
| Полей верно всего | 190/192 (99.0%) | 182/192 (94.8%) | 189/192 (98.4%) | 189/192 (98.4%) |
| Сбоев формата | 0 | 0 | 1 | 1 |
| Ремонтных вызовов | 0 | 0 | 2 | 2 |
| Вызовов LLM | 24 | 24 | 74 | 50 |
| Стоимость прогона | **9.0195 ₽** | **0.2539 ₽** | **0.5604 ₽** | **0.4759 ₽** |
| Latency avg / p50 / p95, мс | 2688 / 2557 / 3175 | 5638 / 6041 / 8571 | 7678 / 7385 / 10342 | 7265 / 6552 / 14011 |

**Цепочка этапов стоит 6.2% от монолита на сильной модели.** Разница по вердиктам: -11, по полям: -1.

- этапы починили: —
- этапы сломали: c02, c04, c07, c10, c11, c14, c16, c17, c21, c23, c24

## Где ошибается извлечение (верных значений на 24 письма)

| Поле | A. Монолит, сильная | A. Монолит, дешёвая | B. Этапы, решение моделью | B. Этапы, решение кодом |
|---|---|---|---|---|
| `company` | 22/24 | 21/24 | 23/24 | 23/24 |
| `product` | 24/24 | 24/24 | 24/24 | 24/24 |
| `qty_kg` | 24/24 | 24/24 | 24/24 | 24/24 |
| `budget_rub` | 24/24 | 24/24 | 24/24 | 24/24 |
| `deadline` | 24/24 | 22/24 | 23/24 | 23/24 |
| `region` | 24/24 | 21/24 | 24/24 | 24/24 |
| `contact` | 24/24 | 22/24 | 23/24 | 23/24 |
| `payment` | 24/24 | 24/24 | 24/24 | 24/24 |

## Согласие решения с политикой на своих же полях

Решение считается согласованным, если совпадает с политикой, применённой кодом к тем полям, которые стратегия сама извлекла. Это отделяет ошибку извлечения от ошибки применения правил.

| Стратегия | Согласовано с политикой | Вердикт верен |
|---|---|---|
| A. Монолит, сильная | 19/24 | 19/24 |
| A. Монолит, дешёвая | 12/24 | 11/24 |
| B. Этапы, решение моделью | 8/24 | 8/24 |
| B. Этапы, решение кодом | 24/24 | 24/24 |

## Чистые письма (6)

| ID | Эталон | A. Монолит, сильная | A. Монолит, дешёвая | B. Этапы, решение моделью | B. Этапы, решение кодом |
|---|---|---|---|---|---|
| c01 | `accept` / `ok` | ✅ accept/ok | ✅ accept/ok | ✅ accept/ok | ✅ accept/ok |
| c02 | `accept` / `ok` | ✅ accept/ok | ✅ accept/ok · поля: company | ❌ reject/below_min_order | ✅ accept/ok |
| c03 | `clarify` / `missing_fields` | ✅ clarify/missing_fields | ❌ accept/ok | ✅ clarify/missing_fields | ✅ clarify/missing_fields |
| c04 | `reject` / `below_min_order` | ✅ reject/below_min_order · поля: company | ❌ accept/ok · поля: company | ❌ accept/ok | ✅ reject/below_min_order |
| c05 | `reject` / `region_not_served` | ✅ reject/region_not_served · поля: company | ❌ accept/ok · поля: company, region | ✅ reject/region_not_served · поля: company | ✅ reject/region_not_served · поля: company |
| c06 | `reject` / `product_not_in_catalog` | ✅ reject/product_not_in_catalog | ✅ reject/product_not_in_catalog | ✅ reject/product_not_in_catalog | ✅ reject/product_not_in_catalog |

## Шумные формулировки (4)

| ID | Эталон | A. Монолит, сильная | A. Монолит, дешёвая | B. Этапы, решение моделью | B. Этапы, решение кодом |
|---|---|---|---|---|---|
| c07 | `accept` / `ok` | ✅ accept/ok | ✅ accept/ok | ❌ reject/below_min_order | ✅ accept/ok |
| c08 | `reject` / `deadline_unrealistic` | ❌ accept/ok | ❌ accept/ok · поля: contact | ❌ reject/below_min_order | ✅ reject/deadline_unrealistic |
| c09 | `accept` / `ok` | ✅ accept/ok | ✅ accept/ok | ✅ accept/ok | ✅ accept/ok |
| c10 | `clarify` / `missing_fields` | ✅ clarify/missing_fields | ❌ accept/ok · поля: deadline, contact | ❌ reject/below_min_order · поля: contact | ✅ clarify/missing_fields · поля: contact |

## Решение по условиям (6)

| ID | Эталон | A. Монолит, сильная | A. Монолит, дешёвая | B. Этапы, решение моделью | B. Этапы, решение кодом |
|---|---|---|---|---|---|
| c11 | `reject` / `deadline_unrealistic` | ✅ reject/deadline_unrealistic | ❌ accept/ok | ❌ reject/below_min_order | ✅ reject/deadline_unrealistic |
| c12 | `accept` / `ok` | ❌ reject/deadline_unrealistic | ✅ accept/ok | ❌ reject/below_min_order | ✅ accept/ok |
| c13 | `reject` / `deadline_unrealistic` | ❌ accept/ok | ❌ accept/ok | ❌ reject/below_min_order | ✅ reject/deadline_unrealistic |
| c14 | `clarify` / `payment_terms_review` | ✅ clarify/payment_terms_review | ❌ accept/ok | ❌ reject/below_min_order | ✅ clarify/payment_terms_review |
| c15 | `reject` / `below_min_order` | ❌ clarify/missing_fields | ❌ accept/ok · поля: deadline | ❌ clarify/missing_fields · поля: deadline | ✅ reject/below_min_order · поля: deadline |
| c16 | `reject` / `product_not_in_catalog` | ✅ reject/product_not_in_catalog | ❌ reject/region_not_served · поля: region | ❌ reject/region_not_served | ✅ reject/product_not_in_catalog |

## Ловушки (8)

| ID | Эталон | A. Монолит, сильная | A. Монолит, дешёвая | B. Этапы, решение моделью | B. Этапы, решение кодом |
|---|---|---|---|---|---|
| c17 | `clarify` / `missing_fields` | ✅ clarify/missing_fields | ❌ accept/ok | ❌ reject/below_min_order | ✅ clarify/missing_fields |
| c18 | `accept` / `ok` | ✅ accept/ok | ✅ accept/ok | ✅ accept/ok | ✅ accept/ok |
| c19 | `reject` / `deadline_unrealistic` | ❌ accept/ok | ❌ accept/ok | ❌ reject/below_min_order | ✅ reject/deadline_unrealistic |
| c20 | `clarify` / `missing_fields` | ✅ clarify/missing_fields | ✅ clarify/missing_fields | ✅ clarify/missing_fields | ✅ clarify/missing_fields |
| c21 | `accept` / `ok` | ✅ accept/ok | ✅ accept/ok | ❌ reject/below_min_order | ✅ accept/ok |
| c22 | `reject` / `region_not_served` | ✅ reject/region_not_served | ✅ reject/region_not_served | ✅ reject/region_not_served | ✅ reject/region_not_served |
| c23 | `clarify` / `missing_fields` | ✅ clarify/missing_fields | ❌ accept/ok · поля: region | ❌ reject/region_not_served | ✅ clarify/missing_fields |
| c24 | `accept` / `ok` | ✅ accept/ok | ✅ accept/ok | ❌ reject/below_min_order | ✅ accept/ok |

## Кейсы с расхождениями

| Стратегия | Неверный вердикт | Неполные поля |
|---|---|---|
| A. Монолит, сильная | c08, c12, c13, c15, c19 | c04, c05 |
| A. Монолит, дешёвая | c03, c04, c05, c08, c10, c11, c13, c14, c15, c16, c17, c19, c23 | c02, c04, c05, c08, c10, c15, c16, c23 |
| B. Этапы, решение моделью | c02, c04, c07, c08, c10, c11, c12, c13, c14, c15, c16, c17, c19, c21, c23, c24 | c05, c10, c15 |
| B. Этапы, решение кодом | — | c05, c10, c15 |
