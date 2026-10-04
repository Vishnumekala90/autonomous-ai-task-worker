"""Seed data for the controlled environment: invoice registry rows, invoice documents, one stale record."""

from __future__ import annotations

from pathlib import Path

from app.database.database import Database
from app.database.repository import utc_now

# (invoice_id, company, invoice_date, due_date, amount)
# Note TechNova: INV-3002 and INV-3003 share the newest date, so "latest" is genuinely ambiguous.
SEED_INVOICES: list[tuple[str, str, str, str, float]] = [
    ("INV-1001", "Acme Corp", "2026-07-05", "2026-08-04", 42000.00),
    ("INV-1002", "Acme Corp", "2026-08-10", "2026-09-09", 56750.00),
    ("INV-1003", "Acme Corp", "2026-09-15", "2026-10-20", 84500.00),
    ("INV-2001", "Globex", "2026-08-20", "2026-09-19", 85000.00),
    ("INV-2002", "Globex", "2026-09-25", "2026-10-25", 310000.00),
    ("INV-3001", "TechNova", "2026-09-01", "2026-10-01", 31200.00),
    ("INV-3002", "TechNova", "2026-09-28", "2026-10-28", 28400.00),
    ("INV-3003", "TechNova", "2026-09-28", "2026-10-28", 29100.00),
    ("INV-4001", "Contoso", "2026-08-12", "2026-09-11", 27300.00),
    ("INV-4002", "Contoso", "2026-09-30", "2026-10-30", 72300.00),
]

# Already present in the internal system. INV-4002 is stored with a STALE amount (70,000 vs 72,300),
# which exercises duplicate-record handling.
SEED_RECORDS: list[tuple[str, str, str, float, str]] = [
    ("REC-10001", "INV-4002", "Contoso", 70000.00, "2026-10-30"),
]


def render_invoice_document(invoice_id: str, company: str, invoice_date: str, due_date: str, amount: float) -> str:
    return (
        "INVOICE\n"
        f"Invoice Number: {invoice_id}\n"
        f"Vendor: {company}\n"
        f"Invoice Date: {invoice_date}\n"
        f"Due Date: {due_date}\n"
        "Currency: INR\n"
        f"Amount Due: {amount:,.2f}\n"
    )


def seed_database(db: Database, invoices_dir: Path, reset: bool = True) -> None:
    """Create the schema and load the sample company environment.

    reset=True wipes every table first (used by the seed script and tests);
    reset=False only seeds an empty database.
    """
    db.init_schema()
    invoices_dir.mkdir(parents=True, exist_ok=True)
    now = utc_now()
    with db.connect() as conn:
        if reset:
            for table in ("invoices", "company_records", "approvals", "faults", "tasks"):
                conn.execute(f"DELETE FROM {table}")
            conn.execute("DELETE FROM sqlite_sequence")
        for invoice_id, company, invoice_date, due_date, amount in SEED_INVOICES:
            conn.execute(
                "INSERT OR REPLACE INTO invoices VALUES (?, ?, ?, ?, ?)",
                (invoice_id, company, invoice_date, f"invoices/{invoice_id}.txt", "received"),
            )
            (invoices_dir / f"{invoice_id}.txt").write_text(
                render_invoice_document(invoice_id, company, invoice_date, due_date, amount),
                encoding="utf-8",
            )
        for record_id, invoice_id, company, amount, due_date in SEED_RECORDS:
            conn.execute(
                "INSERT OR REPLACE INTO company_records VALUES (?, ?, ?, ?, ?, 'seed', ?, ?)",
                (record_id, invoice_id, company, amount, due_date, now, now),
            )
