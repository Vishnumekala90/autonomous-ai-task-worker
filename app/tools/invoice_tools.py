"""Invoice tools: search the invoice registry and extract data from invoice documents."""

from __future__ import annotations

import difflib
import re
import sqlite3
from pathlib import Path

from pydantic import Field, field_validator

from app.database.repository import list_company_names
from app.services import faults
from app.tools.base import StrictArgs, ToolContext, ToolResult

INVOICE_ID_RE = re.compile(r"^INV-\d{3,8}$")
COMPANY_RE = re.compile(r"^[\w .,&'\-]+$")

_DOC_PATTERNS = {
    "invoice_id": r"^Invoice Number:\s*(INV-\d{3,8})\s*$",
    "company": r"^Vendor:\s*(.+?)\s*$",
    "invoice_date": r"^Invoice Date:\s*(\d{4}-\d{2}-\d{2})\s*$",
    "due_date": r"^Due Date:\s*(\d{4}-\d{2}-\d{2})\s*$",
    "currency": r"^Currency:\s*([A-Z]{3})\s*$",
    "amount": r"^Amount Due:\s*([\d,]+(?:\.\d{1,2})?)\s*$",
}


def normalize_invoice_id(value: str) -> str:
    value = value.strip().upper()
    if not INVOICE_ID_RE.match(value):
        raise ValueError("invoice_id must look like INV-1234")
    return value


class SearchInvoicesArgs(StrictArgs):
    company_name: str = Field(min_length=2, max_length=100, description="Company to search for")

    @field_validator("company_name")
    @classmethod
    def _safe_chars(cls, v: str) -> str:
        if not COMPANY_RE.match(v):
            raise ValueError("company_name contains unsupported characters")
        return v


class ReadInvoiceArgs(StrictArgs):
    invoice_id: str = Field(description="Invoice id such as INV-1003")

    @field_validator("invoice_id")
    @classmethod
    def _valid_id(cls, v: str) -> str:
        return normalize_invoice_id(v)


def search_invoices(args: SearchInvoicesArgs, ctx: ToolContext) -> ToolResult:
    if faults.consume(ctx.db, ctx.task_id, "search_db_failure"):
        raise sqlite3.OperationalError("database is locked (simulated fault)")
    needle = args.company_name.casefold()
    with ctx.db.connect() as conn:
        rows = conn.execute(
            "SELECT invoice_id, company, invoice_date, file_reference, status FROM invoices"
        ).fetchall()
    matches = [dict(r) for r in rows if r["company"].casefold().startswith(needle)]
    if not matches:
        suggestions = difflib.get_close_matches(
            args.company_name, list_company_names(ctx.db), n=3, cutoff=0.5
        )
        return ToolResult.fail(
            "NOT_FOUND",
            f"No invoices found for company '{args.company_name}'.",
            suggestions=suggestions,
            known_companies=list_company_names(ctx.db),
        )
    companies = sorted({m["company"] for m in matches})
    if len(companies) > 1:
        return ToolResult.fail(
            "AMBIGUOUS_COMPANY",
            f"'{args.company_name}' matches several companies: {', '.join(companies)}.",
            companies=companies,
        )
    matches.sort(key=lambda m: (m["invoice_date"], m["invoice_id"]), reverse=True)
    return ToolResult.ok(
        f"Found {len(matches)} invoice(s) for {companies[0]}, newest first.",
        invoices=matches,
    )


def _parse_document(text: str) -> dict:
    found: dict = {}
    for name, pattern in _DOC_PATTERNS.items():
        m = re.search(pattern, text, re.MULTILINE)
        if m:
            found[name] = m.group(1)
    missing = [n for n in _DOC_PATTERNS if n not in found]
    if missing:
        raise ValueError(f"missing fields in document: {', '.join(missing)}")
    found["amount"] = float(found["amount"].replace(",", ""))
    return found


def _safe_document_path(ctx: ToolContext, file_reference: str) -> Path:
    base = ctx.invoices_dir.resolve()
    path = (base / Path(file_reference).name).resolve()
    if not path.is_relative_to(base):
        raise PermissionError("document path escapes the invoices directory")
    return path


def read_invoice(args: ReadInvoiceArgs, ctx: ToolContext) -> ToolResult:
    with ctx.db.connect() as conn:
        row = conn.execute(
            "SELECT * FROM invoices WHERE invoice_id = ?", (args.invoice_id,)
        ).fetchone()
    if row is None:
        return ToolResult.fail("NOT_FOUND", f"Invoice {args.invoice_id} does not exist.")
    try:
        text = _safe_document_path(ctx, row["file_reference"]).read_text(encoding="utf-8")
        doc = _parse_document(text)
    except (OSError, ValueError, PermissionError) as exc:
        return ToolResult.fail(
            "EXTRACTION_FAILED", f"Could not extract {args.invoice_id}: {exc}"
        )
    if doc["invoice_id"] != args.invoice_id or doc["company"] != row["company"]:
        return ToolResult.fail(
            "EXTRACTION_FAILED",
            f"Document contents for {args.invoice_id} do not match the invoice registry.",
        )
    return ToolResult.ok(
        f"Read invoice {args.invoice_id}.",
        invoice_id=doc["invoice_id"],
        company=doc["company"],
        amount=doc["amount"],
        currency=doc["currency"],
        due_date=doc["due_date"],
        invoice_date=doc["invoice_date"],
        status=row["status"],
    )
