"""Controlled fault injection for demos and tests.

CONTROLLED SIMULATION: nothing here fires unless a fault was explicitly armed for a task
id. It lets us prove the agent's recovery paths against failures that really occur inside
the tools, rather than faking the outcome in the agent.
"""

from __future__ import annotations

from app.database.database import Database

# user-facing scenario -> {internal fault name: number of times it fires}
SCENARIOS: dict[str, dict[str, int]] = {
    "update_transient": {"update_failure": 2},          # fails twice, then recovers
    "update_persistent": {"update_failure": 10_000},    # never recovers -> escalation
    "corrupt_record": {"corrupt_write": 10_000},        # write "succeeds" with wrong data
    "db_transient": {"search_db_failure": 1},           # one database error on search
}


def arm(db: Database, task_id: str, scenario: str) -> None:
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown fault scenario: {scenario}")
    with db.connect() as conn:
        for name, remaining in SCENARIOS[scenario].items():
            conn.execute(
                "INSERT OR REPLACE INTO faults (task_id, name, remaining) VALUES (?, ?, ?)",
                (task_id, name, remaining),
            )


def consume(db: Database, task_id: str, name: str) -> bool:
    """Return True (and decrement) if this fault should fire now."""
    with db.connect() as conn:
        row = conn.execute(
            "SELECT remaining FROM faults WHERE task_id = ? AND name = ?", (task_id, name)
        ).fetchone()
        if not row or row["remaining"] <= 0:
            return False
        conn.execute(
            "UPDATE faults SET remaining = remaining - 1 WHERE task_id = ? AND name = ?",
            (task_id, name),
        )
        return True
