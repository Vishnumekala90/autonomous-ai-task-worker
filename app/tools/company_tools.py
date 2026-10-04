"""Tools for the simulated internal company system."""

from __future__ import annotations

from datetime import date

from pydantic import Field, field_validator

from app.database.repository import get_company_record, utc_now
from app.services import faults
from app.services.verification import compare_fields
from app.tools.base import StrictArgs, ToolContext, ToolResult
from app.tools.invoice_tools import normalize_invoice_id


def _valid_iso_date(value: str) -> str:
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise ValueError("date must be ISO format YYYY-MM-DD") from exc


class UpdateCompanySystemArgs(StrictArgs):
    invoice_id: str
    company: str = Field(min_length=2, max_length=100)
    amount: float = Field(gt=0, lt=1_000_000_000)
    due_date: str

    @field_validator("invoice_id")
    @classmethod
    def _check_id(cls, v: str) -> str:
        return normalize_invoice_id(v)

    @field_validator("due_date")
    @classmethod
    def _check_due(cls, v: str) -> str:
        return _valid_iso_date(v)


class CheckCompanyRecordArgs(StrictArgs):
    invoice_id: str
    expected_company: str | None = None
    expected_amount: float | None = Field(default=None, gt=0)
    expected_due_date: str | None = None

    @field_validator("invoice_id")
    @classmethod
    def _check_id(cls, v: str) -> str:
        return normalize_invoice_id(v)

    @field_validator("expected_due_date")
    @classmethod
    def _check_due(cls, v: str | None) -> str | None:
        return _valid_iso_date(v) if v is not None else v


def _next_record_id(conn) -> str:
    row = conn.execute(
        "SELECT COALESCE(MAX(CAST(SUBSTR(record_id, 5) AS INTEGER)), 10000) + 1 AS n "
        "FROM company_records"
    ).fetchone()
    return f"REC-{row['n']}"


def update_company_system(args: UpdateCompanySystemArgs, ctx: ToolContext) -> ToolResult:
    if faults.consume(ctx.db, ctx.task_id, "update_failure"):
        return ToolResult.fail(
            "SYSTEM_UNAVAILABLE",
            "Internal company system is temporarily unavailable (simulated fault).",
            retryable=True,
        )
    # Corrupt-write fault: the write "succeeds" but stores a wrong amount.
    stored_amount = args.amount * 1.1 if faults.consume(ctx.db, ctx.task_id, "corrupt_write") else args.amount
    now = utc_now()
    existing = get_company_record(ctx.db, args.invoice_id)
    with ctx.db.connect() as conn:
        if existing is None:
            record_id = _next_record_id(conn)
            conn.execute(
                "INSERT INTO company_records VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (record_id, args.invoice_id, args.company, stored_amount, args.due_date,
                 ctx.task_id, now, now),
            )
            return ToolResult.ok(
                f"Created record {record_id} for {args.invoice_id}.",
                record_id=record_id, action="created",
            )
        same = compare_fields(args.model_dump(), existing).passed
        if same:
            return ToolResult.ok(
                f"{args.invoice_id} is already recorded as {existing['record_id']}; no change.",
                record_id=existing["record_id"], action="unchanged",
            )
        may_overwrite = ctx.overwrite_approved or existing["created_by_task"] == ctx.task_id
        if not may_overwrite:
            return ToolResult.fail(
                "DUPLICATE_RECORD",
                f"A different record {existing['record_id']} already exists for {args.invoice_id}.",
                existing_record=existing,
            )
        conn.execute(
            "UPDATE company_records SET company = ?, amount = ?, due_date = ?, "
            "created_by_task = ?, updated_at = ? WHERE record_id = ?",
            (args.company, stored_amount, args.due_date, ctx.task_id, now, existing["record_id"]),
        )
    return ToolResult.ok(
        f"Updated record {existing['record_id']} for {args.invoice_id}.",
        record_id=existing["record_id"], action="updated",
    )


def check_company_record(args: CheckCompanyRecordArgs, ctx: ToolContext) -> ToolResult:
    record = get_company_record(ctx.db, args.invoice_id)
    if record is None:
        return ToolResult.fail(
            "RECORD_NOT_FOUND", f"No company record exists for {args.invoice_id}."
        )
    expected = {"invoice_id": args.invoice_id}
    if args.expected_company is not None:
        expected["company"] = args.expected_company
    if args.expected_amount is not None:
        expected["amount"] = args.expected_amount
    if args.expected_due_date is not None:
        expected["due_date"] = args.expected_due_date
    if len(expected) == 1:
        return ToolResult.ok("Record found (no expected values supplied to compare).", record=record)
    verification = compare_fields(expected, record)
    data = {"record": record, "verification": verification.model_dump()}
    if not verification.passed:
        return ToolResult.fail("VERIFICATION_FAILED", verification.reason, **data)
    return ToolResult.ok("Record matches all expected values.", **data)
