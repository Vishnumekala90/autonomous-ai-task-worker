"""Shared tool types: result envelope, execution context and tool specification."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.database.database import Database


class StrictArgs(BaseModel):
    """Base for tool arguments: unknown/hallucinated parameters are rejected."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ToolResult(BaseModel):
    success: bool
    data: dict[str, Any] = Field(default_factory=dict)
    message: str = ""
    error_code: str | None = None
    retryable: bool = False

    @classmethod
    def ok(cls, message: str = "", **data: Any) -> "ToolResult":
        return cls(success=True, data=data, message=message)

    @classmethod
    def fail(
        cls, code: str, message: str, retryable: bool = False, **data: Any
    ) -> "ToolResult":
        return cls(success=False, error_code=code, message=message, retryable=retryable, data=data)


@dataclass
class ToolContext:
    db: Database
    task_id: str
    invoices_dir: Path
    # Set only by the graph after a human approved an overwrite. The LLM cannot set this.
    overwrite_approved: bool = False


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[StrictArgs]
    handler: Callable[[Any, ToolContext], ToolResult] = field(repr=False)
