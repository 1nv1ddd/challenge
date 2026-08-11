```python
from fastapi import APIRouter, HTTPException
from .scheduler_store import _connect

router = APIRouter(prefix="/api/scheduler", tags=["scheduler"])

# ... (остальной код роутера)

@router.get("/jobs/{task_id}")
async def get_job(task_id: str):
    init_schema()
    with _connect() as conn:
        row = conn.execute(
            "SELECT task_id, interval_seconds, task_type, payload, next_run, last_run, created_at FROM jobs WHERE task_id = ?",
            (task_id,)
        ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Job not found")

    return dict(row)

# ... (остальной код роутера)
```