"""SQLite access. Plain sqlite3 keeps the simulated company environment easy to inspect."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS invoices (
    invoice_id     TEXT PRIMARY KEY,
    company        TEXT NOT NULL,
    invoice_date   TEXT NOT NULL,
    file_reference TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'received'
);
CREATE INDEX IF NOT EXISTS idx_invoices_company ON invoices (company COLLATE NOCASE);

-- The simulated "internal company system" the agent writes into.
CREATE TABLE IF NOT EXISTS company_records (
    record_id       TEXT PRIMARY KEY,
    invoice_id      TEXT NOT NULL UNIQUE,
    company         TEXT NOT NULL,
    amount          REAL NOT NULL,
    due_date        TEXT NOT NULL,
    created_by_task TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS approvals (
    approval_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id         TEXT NOT NULL,
    reason          TEXT NOT NULL,
    proposed_action TEXT NOT NULL,
    context_json    TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',
    note            TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL,
    decided_at      TEXT
);

-- Serialised agent state, so tasks survive restarts and can resume after approval.
CREATE TABLE IF NOT EXISTS tasks (
    task_id    TEXT PRIMARY KEY,
    task_text  TEXT NOT NULL,
    status     TEXT NOT NULL,
    state_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Armed fault injections (controlled simulation, see app/services/faults.py).
CREATE TABLE IF NOT EXISTS faults (
    task_id   TEXT NOT NULL,
    name      TEXT NOT NULL,
    remaining INTEGER NOT NULL,
    PRIMARY KEY (task_id, name)
);
"""


class Database:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Commit on success, roll back on error, always close."""
        conn = sqlite3.connect(str(self.path), timeout=5)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_schema(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    def is_seeded(self) -> bool:
        with self.connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM invoices").fetchone()[0] > 0

    def ping(self) -> bool:
        try:
            with self.connect() as conn:
                conn.execute("SELECT 1")
            return True
        except sqlite3.Error:
            return False
