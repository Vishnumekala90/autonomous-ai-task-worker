"""Human approval checkpoint tool."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from app.database.repository import create_approval
from app.tools.base import StrictArgs, ToolContext, ToolResult


class RequestHumanApprovalArgs(StrictArgs):
    reason: str = Field(min_length=3, max_length=500)
    proposed_action: str = Field(min_length=3, max_length=500)
    relevant_context: dict[str, Any] = Field(default_factory=dict)


def request_human_approval(args: RequestHumanApprovalArgs, ctx: ToolContext) -> ToolResult:
    """Record a pending approval. The graph then pauses until a human decides."""
    approval_id = create_approval(
        ctx.db, ctx.task_id, args.reason, args.proposed_action, args.relevant_context
    )
    return ToolResult.ok(
        "Approval requested; waiting for a human decision.",
        approval_id=approval_id, status="pending",
    )
