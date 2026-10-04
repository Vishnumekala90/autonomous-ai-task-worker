"""Planning helpers: parse LLM output into plan steps and keep plan status up to date."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from app.agent.state import InterpretedGoal, PlanStep
from app.tools.registry import TOOLS

PLAN_TOOL_HINTS = set(TOOLS) | {"verify_result"}


class PlanParseError(ValueError):
    pass


def parse_goal(raw: dict[str, Any]) -> InterpretedGoal:
    try:
        return InterpretedGoal.model_validate(raw)
    except ValidationError as exc:
        raise PlanParseError(f"LLM returned an invalid goal: {exc}") from exc


def parse_plan(raw: dict[str, Any]) -> list[PlanStep]:
    steps_raw = raw.get("steps") if isinstance(raw, dict) else None
    if not isinstance(steps_raw, list) or not steps_raw:
        raise PlanParseError("LLM returned no plan steps")
    steps: list[PlanStep] = []
    for i, item in enumerate(steps_raw, start=1):
        if not isinstance(item, dict) or not item.get("description"):
            raise PlanParseError("plan step missing description")
        tool = item.get("tool")
        steps.append(
            PlanStep(
                id=i,
                description=str(item["description"])[:200],
                tool=tool if tool in PLAN_TOOL_HINTS else None,
            )
        )
    return steps


def update_plan(plan: list[dict[str, Any]], tool: str, status: str) -> list[dict[str, Any]]:
    """Mark the first not-yet-done plan step that uses `tool`."""
    updated = [dict(step) for step in plan]
    for step in updated:
        if step.get("tool") == tool and step["status"] != "done":
            step["status"] = status
            break
    return updated
