"""Shared test helpers: isolated environments and deliberately misbehaving LLM stubs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config import Settings
from app.database.database import Database
from app.database.seed import seed_database
from app.services.llm import AgentTurn, LLMClient, LLMError, ScriptedLLM
from app.services.task_service import TaskService

ACME_TASK = "Find the latest invoice from Acme Corp, record it in the internal system, verify it, and tell me when it is done."


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = dict(
        database_path=tmp_path / "test.db",
        invoices_dir=tmp_path / "invoices",
        llm_provider="scripted",
        retry_backoff_seconds=0,
        approval_threshold=100_000,
        max_retries=3,
        max_steps=30,
    )
    values.update(overrides)
    return Settings(_env_file=None, **values)


def make_env(
    tmp_path: Path, llm: LLMClient | None = None, **overrides: Any
) -> tuple[TaskService, Database, Settings]:
    settings = make_settings(tmp_path, **overrides)
    db = Database(settings.database_path)
    seed_database(db, settings.invoices_dir)
    return TaskService(db, settings, llm), db, settings


def run(service: TaskService, task: str = ACME_TASK, fault: str | None = None):
    task_id = service.create_task(task, fault)
    return service.run_task(task_id), task_id


def events(service: TaskService, task_id: str, action: str) -> list[dict]:
    return [e for e in service.get_history(task_id).events if e["action"] == action]


def tool_attempts(service: TaskService, task_id: str, tool: str) -> list[int]:
    return [t["attempt"] for t in service.get_history(task_id).tool_calls if t["tool"] == tool]


# ---- misbehaving "LLMs": the graph must stay safe whatever the model does ----


class PrematureUpdateLLM(ScriptedLLM):
    """Tries to update the company system before reading the invoice (with a made-up amount)."""

    def next_turn(self, messages, tools):
        if not any(m["role"] == "tool" for m in messages):
            return self._call(
                messages, "update_company_system",
                invoice_id="INV-1003", company="Acme Corp", amount=99999, due_date="2026-10-20",
            )
        return super().next_turn(messages, tools)


class WrongAmountOnceLLM(ScriptedLLM):
    """Hallucinates the amount on its first update attempt."""

    def __init__(self) -> None:
        self.tampered = False

    def next_turn(self, messages, tools):
        turn = super().next_turn(messages, tools)
        if turn.tool_call and turn.tool_call.name == "update_company_system" and not self.tampered:
            self.tampered = True
            turn.tool_call.arguments["amount"] += 1000
        return turn


class BadArgsOnceLLM(ScriptedLLM):
    """Sends an invalid (negative) amount on its first update attempt."""

    def __init__(self) -> None:
        self.tampered = False

    def next_turn(self, messages, tools):
        turn = super().next_turn(messages, tools)
        if turn.tool_call and turn.tool_call.name == "update_company_system" and not self.tampered:
            self.tampered = True
            turn.tool_call.arguments["amount"] = -5
        return turn


class LoopingLLM(ScriptedLLM):
    """Never makes progress: searches forever."""

    def next_turn(self, messages, tools):
        return self._call(messages, "search_invoices", company_name="Acme")


class ImmediateFinishLLM(ScriptedLLM):
    """Claims it is done without doing anything."""

    def next_turn(self, messages, tools):
        return AgentTurn(content="All done! The invoice has been recorded.")


class BrokenLLM(ScriptedLLM):
    def next_turn(self, messages, tools):
        raise LLMError("provider is down")


class ApprovalFirstLLM(ScriptedLLM):
    """Asks a human for approval on its own initiative before doing any work."""

    def next_turn(self, messages, tools):
        if not any(m["role"] == "tool" for m in messages):
            return self._call(
                messages, "request_human_approval",
                reason="Please confirm I should process Acme invoices",
                proposed_action="Record the latest Acme invoice",
                relevant_context={"company": "Acme Corp"},
            )
        return super().next_turn(messages, tools)
