# Метаданные прогона локальной модели

- **Модель:** `huggingface.co/bartowski/Vikhr-Nemo-12B-Instruct-R-21-09-24-GGUF:Q4_K_M`
- **Runtime:** Ollama 0.21.0, `POST /api/chat`, `stream=false`
- **Параметры:** `temperature=0.2`, `num_ctx=8192`
- **Системный промпт:** `docs/day4/rules-local.md` (компактные правила)
- **Латентность:** 30.7 с
- **Промпт:** 1081 токен (`prompt_eval_count`)
- **Ответ:** 166 токенов (`eval_count`)
- **Скорость генерации:** 14.8 ток/с
- **Harness:** `scratchpad/run_local.py` (urllib → Ollama, без внешних зависимостей)
- **Сырой ответ:** `docs/day4/runs/local-vikhr.out.md`

Воспроизвести: убедиться, что `ollama serve` слушает `:11434`, затем прогнать harness
с системным промптом = `rules-local.md` и user-промптом = задача + контекст из `task.md`.
