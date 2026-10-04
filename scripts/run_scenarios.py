"""Run the five demo scenarios against a throwaway database and print what happened.

    python scripts/run_scenarios.py

Uses whichever LLM provider is configured (.env). With no API key it falls back to the
scripted offline policy and says so.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.database.database import Database  # noqa: E402
from app.database.seed import seed_database  # noqa: E402
from app.logging_config import configure_logging  # noqa: E402
from app.services.task_service import TaskService  # noqa: E402

SCENARIOS = [
    ("1. SUCCESS", "Find the latest invoice from Acme Corp, record it in the internal system, verify it, and tell me when it is done.", None, None),
    ("2. NOT FOUND", "Find the latest invoice from UnknownCompany and record it.", None, None),
    ("3a. UPDATE FAILURE (transient: recovers after retries)", "Find the latest invoice from Acme Corp and record it.", "update_transient", None),
    ("3b. UPDATE FAILURE (persistent: escalates to a human)", "Find the latest invoice from Acme Corp and record it.", "update_persistent", None),
    ("4. VERIFICATION FAILURE (record written with wrong data)", "Find the latest invoice from Acme Corp and record it.", "corrupt_record", None),
    ("5. HUMAN APPROVAL (amount above threshold) - approve", "Find the latest invoice from Globex and record it.", None, True),
    ("5b. HUMAN APPROVAL - reject", "Find the latest invoice from Globex and record it.", None, False),
]


def show(title: str, service: TaskService, task_id: str) -> None:
    view = service.get_task(task_id)
    history = service.get_history(task_id)
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    print(f"task: {view.task}")
    print(f"status: {view.status}   verification: {view.evidence.get('verification', 'n/a')}")
    print(f"summary: {view.summary or view.current_step}")
    notable = [e for e in history.events if e["action"] in {"retry", "adapt", "approval_required", "approval_decision", "blocked"} or e["status"] == "failed"]
    for e in notable:
        print(f"   - [{e['node']}] {e['action']} ({e['status']}) {e['output_summary'][:110]}")
    if view.evidence.get("problems_encountered"):
        print(f"problems: {len(view.evidence['problems_encountered'])} tool failure(s) observed")


def main() -> None:
    configure_logging("WARNING")
    tmp = Path(tempfile.mkdtemp())
    settings = Settings(database_path=tmp / "demo.db", invoices_dir=tmp / "invoices", retry_backoff_seconds=0)
    db = Database(settings.database_path)
    seed_database(db, settings.invoices_dir)
    service = TaskService(db, settings)
    print(f"LLM in use: {service.llm.info()}")
    for title, task, fault, approve in SCENARIOS:
        seed_database(db, settings.invoices_dir)  # each scenario starts from a clean environment
        task_id = service.create_task(task, fault)
        service.run_task(task_id)
        if approve is not None:
            show(title + " [paused]", service, task_id)
            service.decide_approval(task_id, approve, "demo decision")
            service.run_task(task_id)
        show(title, service, task_id)


if __name__ == "__main__":
    main()
