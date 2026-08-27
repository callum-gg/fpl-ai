"""FastAPI app factory. docs/01 + docs/09."""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import get_settings
from .db.engine import init_db

log = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    for noisy in ("httpx", "httpcore", "apscheduler.executors"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    setup_logging(s.log_level)
    init_db()
    log.info("database ready at %s", s.db_path)

    from .scheduler.jobs import shutdown, start

    start()
    if s.ingest_on_startup:
        from .scheduler.jobs import run_named

        await run_named("fpl_bootstrap")
        await run_named("fpl_fixtures")
    try:
        yield
    finally:
        shutdown()
        from .connectors.base import close_http

        await close_http()


def create_app() -> FastAPI:
    s = get_settings()
    app = FastAPI(
        title="FPL AI",
        version="0.1.0",
        description="AI-powered Fantasy Premier League squad builder and manager.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=s.allowed_origins or ["http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def auth(request: Request, call_next):
        """Off by default. `token` mode checks X-App-Token, for when this is exposed."""
        settings = get_settings()
        if settings.app_auth_mode == "token" and request.url.path.startswith("/api"):
            if request.headers.get("X-App-Token") != settings.app_token:
                return JSONResponse(
                    {"error": {"code": "unauthorised", "message": "bad or missing X-App-Token"}},
                    status_code=401,
                )
        return await call_next(request)

    from .api.routers import admin, content, players, squads

    app.include_router(squads.router)
    app.include_router(players.router)
    app.include_router(admin.router)
    app.include_router(content.router)

    @app.get("/api/health")
    def health() -> dict:
        from .connectors import registry
        from .db.engine import scalar, vec_available
        from .llm.client import available as llm_available

        return {
            "ok": True,
            "season": s.current_season,
            "players": scalar("SELECT COUNT(*) FROM players", default=0),
            "fixtures": scalar("SELECT COUNT(*) FROM fixtures", default=0),
            "raw_documents": scalar("SELECT COUNT(*) FROM raw_documents", default=0),
            "predictions": scalar("SELECT COUNT(*) FROM predictions", default=0),
            "connectors": len(registry.CONNECTORS),
            "connectors_available": len(registry.available()),
            "sqlite_vec": vec_available(),
            "llm_configured": llm_available(),
            "scrape_enabled": s.scrape_enabled,
            "fpl_write_enabled": s.fpl_write_enabled,
        }

    return app


app = create_app()
