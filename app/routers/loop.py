"""API цикла: прогон задач через генерацию, проверки и security review (День 14)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ..agent_constants import (
    LOOP_ARTIFACTS_DIR,
    LOOP_BLOCKING_SEVERITIES,
    LOOP_GATEWAY_MODE,
    LOOP_GEN_MODEL,
    LOOP_MAX_ATTEMPTS,
    LOOP_REVIEW_MODEL,
    LOOP_SECURITY_RULES,
    LOOP_SEVERITIES,
    LOOP_TASKS_PATH,
)
from ..bootstrap import agent, providers
from ..loop import load_tasks, loop_summary
from ..payloads import LoopPayload

router = APIRouter(prefix="/api/loop", tags=["loop"])


@router.post("/run")
async def loop_run(request: Request):
    """Прогон задач: генерация → проверки → security review → «коммит». Вызовы идут через шлюз."""
    if not providers:
        raise HTTPException(
            status_code=503, detail="Нет настроенного провайдера (нужен ROUTERAI_API_KEY)."
        )
    body = await request.json()
    p = LoopPayload.from_body(body if isinstance(body, dict) else {})
    try:
        runs = await agent.run_execution_loop(
            p.provider_name,
            ids=p.ids,
            gen_model=p.gen_model,
            review_model=p.review_model,
            max_attempts=p.max_attempts,
        )
    except LookupError as exc:
        raise HTTPException(status_code=400, detail=str(exc).strip() or "LookupError") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc).strip() or "ValueError") from exc
    return {"runs": [run.to_dict() for run in runs], "summary": loop_summary(runs)}


@router.get("/tasks")
async def loop_tasks():
    """Корпус задач цикла: промпт, тесты и ловушки, которые задача провоцирует."""
    try:
        tasks = load_tasks()
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "path": LOOP_TASKS_PATH,
        "total": len(tasks),
        "tasks": [task.to_dict() for task in tasks],
    }


@router.get("/config")
async def loop_config():
    """Настройки цикла: модели этапов, лимит попыток, уровни находок и правила security-промпта."""
    return {
        "gen_model": LOOP_GEN_MODEL,
        "review_model": LOOP_REVIEW_MODEL,
        "max_attempts": LOOP_MAX_ATTEMPTS,
        "gateway_mode": LOOP_GATEWAY_MODE,
        "severities": list(LOOP_SEVERITIES),
        "blocking_severities": list(LOOP_BLOCKING_SEVERITIES),
        "security_rules": list(LOOP_SECURITY_RULES),
        "artifacts_dir": LOOP_ARTIFACTS_DIR,
    }
