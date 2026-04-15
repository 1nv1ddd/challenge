from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .bootstrap import STATIC_DIR
from .lifespan import lifespan
from .mcp_panel import router as mcp_router
from .routers.hub import router as hub_router
from .scheduler_routes import router as scheduler_router

app = FastAPI(title="AI Chat Hub", lifespan=lifespan)
app.include_router(mcp_router)
app.include_router(scheduler_router)
app.include_router(hub_router)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
