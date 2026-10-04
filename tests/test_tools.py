"""Tool-level tests against a real seeded SQLite database (no mocks)."""

from __future__ import annotations

import pytest

from app.database.database import Database
from app.database.repository import get_approval, get_company_record
from app.database.seed import seed_database
from app.services import faults
from app.tools.base import ToolContext
from app.tools.registry import run_tool, tool_definitions
from tests.helpers import make_settings


@pytest.fixture
def ctx(tmp_path):
    settings = make_settings(tmp_path)
    db = Database(settings.database_path)
    seed_database(db, settings.invoices_dir)
    return ToolContext(db=db, task_id="TEST-1", invoices_dir=settings.invoices_dir)


ACME_UPDATE = {"invoice_id": "INV-1003", "company": "Acme Corp", "amount": 84500, "due_date": "2026-10-20"}


# ---- search_invoices ----

def test_search_returns_newest_first(ctx):
    r = run_tool("search_invoices", {"company_name": "Acme Corp"}, ctx)
    assert r.success
    ids = [i["invoice_id"] for i in r.data["invoices"]]
    assert ids == ["INV-1003", "INV-1002", "INV-1001"]
    assert {"invoice_id", "company", "invoice_date", "file_reference", "status"} <= set(r.data["invoices"][0])


def test_search_is_case_insensitive_prefix(ctx):
    r = run_tool("search_invoices", {"company_name": "acme"}, ctx)
    assert r.success and r.data["invoices"][0]["company"] == "Acme Corp"


def test_search_unknown_company_fails_with_suggestion(ctx):
    r = run_tool("search_invoices", {"company_name": "Acmee Corp"}, ctx)
    assert not r.success and r.error_code == "NOT_FOUND"
    assert "Acme Corp" in r.data["suggestions"]


def test_search_ambiguous_company(ctx):
    with ctx.db.connect() as conn:
        conn.execute(
            "INSERT INTO invoices VALUES ('INV-9001', 'Acme Industries', '2026-01-01', 'invoices/INV-9001.txt', 'received')"
        )
    r = run_tool("search_invoices", {"company_name": "Acme"}, ctx)
    assert r.error_code == "AMBIGUOUS_COMPANY" and len(r.data["companies"]) == 2


@pytest.mark.parametrize("bad", ["A", "", "Acme'; DROP TABLE invoices;--", "x" * 101])
def test_search_rejects_invalid_input(ctx, bad):
    r = run_tool("search_invoices", {"company_name": bad}, ctx)
    assert not r.success and r.error_code == "INVALID_ARGUMENTS"
    assert get_company_record(ctx.db, "INV-4002") is not None  # database untouched


# ---- read_invoice ----

def test_read_invoice_extracts_fields_from_document(ctx):
    r = run_tool("read_invoice", {"invoice_id": "inv-1003"}, ctx)  # normalised to upper-case
    assert r.success
    assert r.data == {
        "invoice_id": "INV-1003", "company": "Acme Corp", "amount": 84500.0, "currency": "INR",
        "due_date": "2026-10-20", "invoice_date": "2026-09-15", "status": "received",
    }


@pytest.mark.parametrize("bad", ["../../etc/passwd", "INV-12", "1003", "INV-1003; rm -rf /"])
def test_read_invoice_rejects_malformed_ids(ctx, bad):
    assert run_tool("read_invoice", {"invoice_id": bad}, ctx).error_code == "INVALID_ARGUMENTS"


def test_read_invoice_unknown_id(ctx):
    assert run_tool("read_invoice", {"invoice_id": "INV-9999"}, ctx).error_code == "NOT_FOUND"


def test_read_invoice_unparseable_document(ctx):
    (ctx.invoices_dir / "INV-1003.txt").write_text("garbage", encoding="utf-8")
    r = run_tool("read_invoice", {"invoice_id": "INV-1003"}, ctx)
    assert r.error_code == "EXTRACTION_FAILED"


def test_read_invoice_document_mismatching_registry(ctx):
    doc = (ctx.invoices_dir / "INV-1003.txt").read_text().replace("Acme Corp", "Somebody Else")
    (ctx.invoices_dir / "INV-1003.txt").write_text(doc, encoding="utf-8")
    assert run_tool("read_invoice", {"invoice_id": "INV-1003"}, ctx).error_code == "EXTRACTION_FAILED"


# ---- update_company_system / check_company_record ----

def test_update_creates_record(ctx):
    r = run_tool("update_company_system", ACME_UPDATE, ctx)
    assert r.success and r.data["action"] == "created" and r.data["record_id"] == "REC-10002"
    rec = get_company_record(ctx.db, "INV-1003")
    assert (rec["company"], rec["amount"], rec["due_date"]) == ("Acme Corp", 84500.0, "2026-10-20")


def test_check_record_passes_when_values_match(ctx):
    run_tool("update_company_system", ACME_UPDATE, ctx)
    r = run_tool("check_company_record", {
        "invoice_id": "INV-1003", "expected_company": "Acme Corp",
        "expected_amount": 84500, "expected_due_date": "2026-10-20"}, ctx)
    assert r.success and r.data["verification"]["passed"]
    assert r.data["record"]["record_id"] == "REC-10002"


def test_check_record_reports_mismatch(ctx):
    run_tool("update_company_system", ACME_UPDATE, ctx)
    r = run_tool("check_company_record", {"invoice_id": "INV-1003", "expected_amount": 1}, ctx)
    assert r.error_code == "VERIFICATION_FAILED"
    amount = next(c for c in r.data["verification"]["checks"] if c["field"] == "amount")
    assert amount["match"] is False and amount["actual"] == 84500.0


def test_check_record_missing(ctx):
    assert run_tool("check_company_record", {"invoice_id": "INV-1003"}, ctx).error_code == "RECORD_NOT_FOUND"


def test_update_is_idempotent(ctx):
    first = run_tool("update_company_system", ACME_UPDATE, ctx)
    second = run_tool("update_company_system", ACME_UPDATE, ctx)
    assert second.success and second.data["action"] == "unchanged"
    assert second.data["record_id"] == first.data["record_id"]


def test_duplicate_with_different_values_is_refused(ctx):
    # seed contains a stale REC-10001 for INV-4002 with a wrong amount
    r = run_tool("update_company_system", {
        "invoice_id": "INV-4002", "company": "Contoso", "amount": 72300, "due_date": "2026-10-30"}, ctx)
    assert r.error_code == "DUPLICATE_RECORD" and r.data["existing_record"]["amount"] == 70000.0
    assert get_company_record(ctx.db, "INV-4002")["amount"] == 70000.0


def test_overwrite_only_with_approval_flag(ctx):
    args = {"invoice_id": "INV-4002", "company": "Contoso", "amount": 72300, "due_date": "2026-10-30"}
    ctx.overwrite_approved = True
    r = run_tool("update_company_system", args, ctx)
    assert r.success and r.data["action"] == "updated" and r.data["record_id"] == "REC-10001"
    assert get_company_record(ctx.db, "INV-4002")["amount"] == 72300.0


@pytest.mark.parametrize("patch", [
    {"amount": -1}, {"amount": 0}, {"amount": "84,500"}, {"due_date": "20-10-2026"},
    {"due_date": "2026-13-45"}, {"invoice_id": "ACME-1"}, {"company": ""}, {"unexpected": "field"},
])
def test_update_rejects_invalid_arguments(ctx, patch):
    r = run_tool("update_company_system", {**ACME_UPDATE, **patch}, ctx)
    assert r.error_code == "INVALID_ARGUMENTS"
    assert get_company_record(ctx.db, "INV-1003") is None


# ---- approvals, registry, faults ----

def test_request_human_approval_creates_pending_row(ctx):
    r = run_tool("request_human_approval", {
        "reason": "Amount is large", "proposed_action": "update INV-2002",
        "relevant_context": {"amount": 310000}}, ctx)
    assert r.success and r.data["status"] == "pending"
    row = get_approval(ctx.db, r.data["approval_id"])
    assert row["status"] == "pending" and row["task_id"] == "TEST-1"


def test_unknown_tool_is_reported_not_raised(ctx):
    r = run_tool("delete_everything", {}, ctx)
    assert r.error_code == "UNKNOWN_TOOL"


def test_database_exception_is_normalised_and_retryable(ctx):
    faults.arm(ctx.db, ctx.task_id, "db_transient")
    first = run_tool("search_invoices", {"company_name": "Acme"}, ctx)
    assert first.error_code == "DB_ERROR" and first.retryable
    assert run_tool("search_invoices", {"company_name": "Acme"}, ctx).success  # fault fired once


def test_tool_definitions_expose_five_tools_with_schemas():
    defs = tool_definitions()
    assert {d["function"]["name"] for d in defs} == {
        "search_invoices", "read_invoice", "update_company_system",
        "check_company_record", "request_human_approval"}
    assert all(d["function"]["parameters"]["type"] == "object" for d in defs)
