"""Small query helpers shared by tools, verification and the task service."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from app.database.database import Database


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_company_record(db: Database, invoice_id: str) -> dict | None:
    with db.connect() as conn:
        row = conn.execute(
            "SELECT * FROM company_records WHERE invoice_id = ?", (invoice_id,)
        ).fetchone()
    return dict(row) if row else None


def list_company_names(db: Database) -> list[str]:
    with db.connect() as conn:
        rows = conn.execute("SELECT DISTINCT company FROM invoices ORDER BY company").fetchall()
    return [r["company"] for r in rows]


def create_approval(
    db: Database, task_id: str, reason: str, proposed_action: str, context: dict
) -> int:
    with db.connect() as conn:
        cur = conn.execute(
            "INSERT INTO approvals (task_id, reason, proposed_action, context_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (task_id, reason, proposed_action, json.dumps(context, default=str), utc_now()),
        )
        return int(cur.lastrowid)


def decide_approval(db: Database, approval_id: int, approved: bool, note: str = "") -> bool:
    """Atomically settle a pending approval. Returns False if it was not pending."""
    with db.connect() as conn:
        cur = conn.execute(
            "UPDATE approvals SET status = ?, note = ?, decided_at = ? "
            "WHERE approval_id = ? AND status = 'pending'",
            ("approved" if approved else "rejected", note, utc_now(), approval_id),
        )
        return cur.rowcount == 1


def get_approval(db: Database, approval_id: int) -> dict | None:
    with db.connect() as conn:
        row = conn.execute(
            "SELECT * FROM approvals WHERE approval_id = ?", (approval_id,)
        ).fetchone()
    return dict(row) if row else None
