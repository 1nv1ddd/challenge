# Прогон 1 — корпус детекторов, офлайн

`GET /api/gateway/cases` (он же `/gateway selftest` в веб-чате). Модель не вызывается: проверяются
только детекторы input guard и решение о действии. Кейсов: **20**, совпало с ожиданием:
**18**, пропущено: **2**.

| Кейс | Режим | Ожидали | Нашли | Как нашли | Действие | Итог |
|---|---|---|---|---|---|---|
| `clean-prompt` | `hybrid` | — | — | — | `pass` | ✅ поймали |
| `openai-key` | `hybrid` | openai_key | openai_key | direct | `block` | ✅ поймали |
| `anthropic-key` | `hybrid` | anthropic_key | anthropic_key | direct | `block` | ✅ поймали |
| `github-token` | `hybrid` | github_token | github_token | direct | `block` | ✅ поймали |
| `aws-key` | `hybrid` | aws_access_key | aws_access_key | direct | `block` | ✅ поймали |
| `google-key` | `hybrid` | google_api_key | google_api_key | direct | `block` | ✅ поймали |
| `card-number` | `hybrid` | card | card | direct | `mask` | ✅ поймали |
| `order-number` | `hybrid` | — | — | — | `pass` | ✅ поймали |
| `email-phone` | `hybrid` | email, phone | email, phone | direct | `mask` | ✅ поймали |
| `base64-secret` | `hybrid` | openai_key | openai_key | base64 | `block` | ✅ поймали |
| `split-secret` | `hybrid` | openai_key | openai_key | joined | `block` | ✅ поймали |
| `zero-width-key` | `hybrid` | openai_key | openai_key | direct, joined | `block` | ✅ поймали |
| `jwt-token` | `hybrid` | jwt | jwt | direct | `block` | ✅ поймали |
| `private-key` | `hybrid` | private_key | private_key | direct | `block` | ✅ поймали |
| `password-assignment` | `hybrid` | credential_assignment | credential_assignment | direct | `block` | ✅ поймали |
| `card-mask-mode` | `mask` | card | card | direct | `mask` | ✅ поймали |
| `email-block-mode` | `block` | email | email | direct | `block` | ✅ поймали |
| `connection-string` | `hybrid` | connection_string | connection_string | direct | `block` | ✅ поймали |
| `evasion-reversed` | `hybrid` | openai_key | — | — | `pass` вместо `block` | ❌ **пропустили** |
| `evasion-hex` | `hybrid` | openai_key | — | — | `pass` вместо `block` | ❌ **пропустили** |

## Что поймали

- **все виды ключей с узнаваемым префиксом** — `sk-`, `sk-ant-`, `ghp_`, `AKIA`, `AIza`, PEM-заголовок,
  JWT: срабатывание точное, вид секрета в логе конкретный;
- **номер карты** — только с валидной контрольной суммой Луна; 16-значный номер заказа детектор не трогает;
- **ПДн** — почта и телефон, обе маскируются, промпт остаётся осмысленным;
- **base64** — блоб декодируется и прогоняется теми же детекторами, маска накрывает блоб целиком;
- **разорванный литерал** — `"sk-" + "proj-abc…"` склеивается и ловится;
- **невидимые символы внутри ключа** — zero-width вырезаются перед сверкой;
- **пароль без формата** — по присваиванию рядом со словом-маркером;
- **строка подключения** — `postgres://user:pass@host/db`; этот кейс добавлен после живого прогона (см. [run2](run2.md)).

## Что пропустили

| Кейс | Что не поймали | Почему |
|---|---|---|
| `evasion-reversed` | ключ, записанный задом наперёд | обратных преобразований бесконечно много; каждое новое — ещё один полный проход по тексту и ещё один источник ложных срабатываний |
| `evasion-hex` | ключ в hex | декодируется только base64; hex, rot13, urlsafe-base64 остаются дырой |

Оба пропуска зафиксированы в тесте `test_run_matches_expectations_except_known_misses`: список
известных пропусков задан явно, и появление **нового** пропуска роняет тест.
