"""Task lifecycle: create, run the graph, persist state after every node, resume after approval."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from app.agent.graph import build_graph
from app.agent.state import AgentState, derive_status, new_state, with_event
from app.config import Settings
from app.database.database import Database
from app.database.repository import decide_approval, utc_now
from app.schemas.task import HistoryResponse, TaskResponse
from app.services import faults
from app.services.llm import LLMClient, get_llm
from app.services.reporting import build_evidence, build_summary

logger = logging.getLogger(__name__)


class TaskNotFound(Exception):
    pass


class InvalidTaskState(Exception):
    pass


def sanitize_task(text: str) -> str:
    """Strip control characters and collapse whitespace in user-provided task text."""
    cleaned = "".join(ch if ch.isprintable() or ch in "\n\t" else " " for ch in text)
    return " ".join(cleaned.split())[:500]


class TaskService:
    def __init__(self, db: Database, settings: Settings, llm: LLMClient | None = None) -> None:
        self.db = db
        self.settings = settings
        self.llm = llm or get_llm(settings)
        self.graph = build_graph(db, self.llm, settings)

    # ------------------------------------------------------------ persistence

    def _save(self, task_id: str, state: AgentState, status: str | None = None) -> None:
        with self.db.connect() as conn:
            conn.execute(
                "UPDATE tasks SET state_json = ?, status = ?, updated_at = ? WHERE task_id = ?",
                (json.dumps(state, default=str), status or derive_status(state), utc_now(), task_id),
            )

    def _load(self, task_id: str) -> tuple[AgentState, dict[str, Any]]:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
        if row is None:
            raise TaskNotFound(task_id)
        return json.loads(row["state_json"]), dict(row)

    # ------------------------------------------------------------ lifecycle

    def create_task(self, task_text: str, simulate_fault: str | None = None) -> str:
        task_text = sanitize_task(task_text)
        task_id = f"TASK-{uuid.uuid4().hex[:8].upper()}"
        state = new_state(task_id, task_text)
        state["execution_history"] = with_event(
            state, "api", "task_created", "ok", input_summary=task_text,
            output_summary=f"fault injection: {simulate_fault}" if simulate_fault else "",
        )
        now = utc_now()
        with self.db.connect() as conn:
            conn.execute(
                "INSERT INTO tasks (task_id, task_text, status, state_json, created_at, updated_at) "
                "VALUES (?, ?, 'QUEUED', ?, ?, ?)",
                (task_id, task_text, json.dumps(state, default=str), now, now),
            )
        if simulate_fault:
            faults.arm(self.db, task_id, simulate_fault)
        return task_id

    def run_task(self, task_id: str) -> TaskResponse:
        """Run (or resume) the graph, saving state after every node so the UI can follow along."""
        state, _ = self._load(task_id)
        config = {"recursion_limit": self.settings.max_steps * 10 + 30}
        latest: AgentState = state
        try:
            for snapshot in self.graph.stream(state, config=config, stream_mode="values"):
                latest = snapshot
                self._save(task_id, latest)
        except Exception as exc:  # never let a background run crash silently
            logger.exception("task %s crashed", task_id)
            latest = self._crash_state(latest, exc)
            self._save(task_id, latest)
        return self.get_task(task_id)

    def _crash_state(self, state: AgentState, exc: Exception) -> AgentState:
        crashed: AgentState = {
            **state,
            "final_status": "FAILED",
            "final_reason": f"Internal error: {type(exc).__name__}",
            "pending_tool_call": None,
            "execution_history": with_event(
                state, "task_service", "crash", "failed", error=f"{type(exc).__name__}: {exc}"),
        }
        evidence = build_evidence(crashed, self.llm.info())
        crashed["evidence"] = evidence
        crashed["summary"] = build_summary(crashed, evidence)
        return crashed

    def decide_approval(self, task_id: str, approved: bool, note: str = "") -> TaskResponse:
        """Record the human decision and mark the task ready to resume (call run_task next)."""
        state, _ = self._load(task_id)
        request = state.get("approval_request")
        if state.get("approval_status") != "pending" or not request or state.get("final_status"):
            raise InvalidTaskState("This task has no pending approval.")
        if not decide_approval(self.db, request["approval_id"], approved, note):
            raise InvalidTaskState("This approval has already been decided.")
        state["approval_status"] = "approved" if approved else "rejected"
        state["approval_note"] = note
        state["resume_pending"] = True
        self._save(task_id, state, status="RUNNING")
        return self.get_task(task_id)

    # ------------------------------------------------------------ views

    def get_task(self, task_id: str) -> TaskResponse:
        state, row = self._load(task_id)
        status = row["status"] if row["status"] == "QUEUED" and not state.get("plan") else derive_status(state)
        pending = state.get("approval_status") == "pending" and not state.get("final_status")
        return TaskResponse(
            task_id=task_id,
            status=status,
            task=state["original_task"],
            result=state.get("final_status"),
            summary=state.get("summary", ""),
            evidence=state.get("evidence", {}),
            plan=state.get("plan", []),
            current_step=state.get("current_step", ""),
            approval=state.get("approval_request") if pending else None,
            verification=state.get("verification_result"),
            errors=state.get("errors", []),
            retry_count=state.get("retry_count", 0),
            llm=self.llm.info(),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def get_history(self, task_id: str) -> HistoryResponse:
        state, _ = self._load(task_id)
        return HistoryResponse(
            task_id=task_id,
            events=state.get("execution_history", []),
            tool_calls=state.get("tool_results", []),
        )
