"""FastAPI application factory. Run with: uvicorn app.main:create_app --factory"""

from __future__ import annotations

import logging

from fastapi import FastAPI

from app.api import routes_health, routes_tasks
from app.config import Settings, get_settings
from app.database.database import Database
from app.database.seed import seed_database
from app.logging_config import configure_logging
from app.services.llm import LLMClient
from app.services.task_service import TaskService

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, llm: LLMClient | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    db = Database(settings.database_path)
    db.init_schema()
    if not db.is_seeded():
        logger.info("Empty database detected; seeding the simulated company environment")
        seed_database(db, settings.invoices_dir, reset=False)

    app = FastAPI(
        title="CentrAlign AI - Autonomous Invoice Processing Worker",
        version="1.0.0",
        description=(
            "Submit a natural-language invoice task. A LangGraph agent plans, calls tools against a "
            "controlled local company environment, recovers from failures, asks for human approval "
            "when needed, and only reports success after independent verification."
        ),
    )
    app.state.settings = settings
    app.state.task_service = TaskService(db, settings, llm)
    app.include_router(routes_health.router)
    app.include_router(routes_tasks.router)
    return app
