"""API request/response schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

FaultScenario = Literal["update_transient", "update_persistent", "corrupt_record", "db_transient"]


class TaskCreateRequest(BaseModel):
    task: str = Field(
        min_length=5,
        max_length=500,
        examples=["Find the latest invoice from Acme Corp, enter it into the company system and verify it."],
    )
    simulate_fault: FaultScenario | None = Field(
        default=None,
        description="Demo only: arm a controlled failure inside the simulated company system.",
    )


class ApprovalDecisionRequest(BaseModel):
    note: str = Field(default="", max_length=300)


class TaskResponse(BaseModel):
    task_id: str
    status: str = Field(description="QUEUED | RUNNING | AWAITING_APPROVAL | or a terminal status")
    task: str
    result: str | None = Field(default=None, description="Final status once the task has finished")
    summary: str = ""
    evidence: dict[str, Any] = Field(default_factory=dict)
    plan: list[dict[str, Any]] = Field(default_factory=list)
    current_step: str = ""
    approval: dict[str, Any] | None = None
    verification: dict[str, Any] | None = None
    errors: list[dict[str, Any]] = Field(default_factory=list)
    retry_count: int = 0
    llm: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str


class HistoryResponse(BaseModel):
    task_id: str
    events: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
