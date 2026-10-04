from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health", summary="Service health and active model")
def health(request: Request) -> dict[str, Any]:
    service = request.app.state.task_service
    settings = request.app.state.settings
    return {
        "status": "ok" if service.db.ping() else "degraded",
        "database": "ok" if service.db.ping() else "unavailable",
        "llm": service.llm.info(),
        "approval_threshold": settings.approval_threshold,
        "max_retries": settings.max_retries,
        "fault_injection_enabled": settings.enable_fault_injection,
    }
