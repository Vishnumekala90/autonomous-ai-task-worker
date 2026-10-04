"""Typed, JSON-serialisable agent state plus helpers for structured execution events."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field

event_logger = logging.getLogger("worker.events")

TERMINAL_STATUSES = {
    "COMPLETED",
    "NEEDS_CLARIFICATION",
    "ESCALATED_TO_HUMAN",
    "VERIFICATION_FAILED",
    "REJECTED_BY_HUMAN",
    "FAILED",
}


class InterpretedGoal(BaseModel):
    """What the user actually wants, as understood by the LLM."""

    intent: str = "record_invoice"
    company_name: str | None = None
    invoice_id: str | None = None
    selection: Literal["latest", "specific"] = "latest"
    required_fields: list[str] = Field(
        default_factory=lambda: ["invoice_id", "company", "amount", "due_date"]
    )
    needs_clarification: bool = False
    clarification_question: str | None = None
    summary: str = ""


class PlanStep(BaseModel):
    id: int
    description: str
    tool: str | None = None  # hint used only to tick the plan off in the UI
    status: Literal["pending", "in_progress", "done", "failed"] = "pending"


class AgentState(TypedDict, total=False):
    task_id: str
    original_task: str
    interpreted_goal: dict[str, Any]
    plan: list[dict[str, Any]]
    current_step: str
    messages: list[dict[str, Any]]            # LLM conversation (OpenAI message format)
    pending_tool_call: dict[str, Any] | None  # {id, name, arguments}
    last_result: dict[str, Any] | None        # most recent ToolResult as dict
    last_failure: dict[str, Any] | None
    discovered_information: dict[str, Any]    # search_results, invoice, record, last_check
    tool_results: list[dict[str, Any]]
    execution_history: list[dict[str, Any]]
    errors: list[dict[str, Any]]
    retry_count: int          # same-call retries for the current action
    adaptation_count: int     # LLM-driven corrections (bad args, failed verification)
    step_count: int           # decide-loop iterations, hard-capped by MAX_STEPS
    policy_decision: dict[str, Any] | None
    approval_required: bool
    approval_status: str      # none | pending | approved | rejected
    approval_request: dict[str, Any] | None
    approval_note: str
    approved_fingerprints: list[str]
    approvals: list[dict[str, Any]]
    resume_pending: bool
    next_route: str           # set by routing-relevant nodes; read by conditional edges
    verification_result: dict[str, Any] | None
    final_status: str | None
    final_reason: str
    summary: str
    evidence: dict[str, Any]


def new_state(task_id: str, task: str) -> AgentState:
    return AgentState(
        task_id=task_id,
        original_task=task,
        interpreted_goal={},
        plan=[],
        current_step="Queued",
        messages=[],
        pending_tool_call=None,
        last_result=None,
        last_failure=None,
        discovered_information={},
        tool_results=[],
        execution_history=[],
        errors=[],
        retry_count=0,
        adaptation_count=0,
        step_count=0,
        policy_decision=None,
        approval_required=False,
        approval_status="none",
        approval_request=None,
        approval_note="",
        approved_fingerprints=[],
        approvals=[],
        resume_pending=False,
        next_route="",
        verification_result=None,
        final_status=None,
        final_reason="",
        summary="",
        evidence={},
    )


def derive_status(state: AgentState) -> str:
    if state.get("final_status"):
        return state["final_status"]  # type: ignore[return-value]
    if state.get("approval_status") == "pending":
        return "AWAITING_APPROVAL"
    return "RUNNING"


# ---------- structured execution events ----------

_SECRET_KEY = re.compile(r"(api[_-]?key|secret|token|password|authorization)", re.I)


def _redact(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: "[REDACTED]" if _SECRET_KEY.search(str(k)) else _redact(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_redact(v) for v in obj]
    return obj


def summarize(obj: Any, limit: int = 240) -> str:
    text = obj if isinstance(obj, str) else json.dumps(_redact(obj), default=str, ensure_ascii=False)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def make_event(
    node: str,
    action: str,
    status: str,
    *,
    tool: str | None = None,
    input_summary: Any = "",
    output_summary: Any = "",
    error: str | None = None,
) -> dict[str, Any]:
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "node": node,
        "action": action,
        "tool": tool,
        "input_summary": summarize(input_summary),
        "output_summary": summarize(output_summary),
        "status": status,
        "error": error,
    }
    event_logger.info("event %s", json.dumps(event, ensure_ascii=False))
    return event


def with_event(state: AgentState, node: str, action: str, status: str, **kwargs: Any) -> list[dict[str, Any]]:
    """Return a new execution_history list including one more event."""
    return [*state.get("execution_history", []), make_event(node, action, status, **kwargs)]
