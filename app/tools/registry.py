"""Tool registry: the only way the agent can act. Validates input and normalises failures."""

from __future__ import annotations

import logging
import sqlite3
from typing import Any

from pydantic import ValidationError

from app.tools.approval_tools import RequestHumanApprovalArgs, request_human_approval
from app.tools.base import ToolContext, ToolResult, ToolSpec
from app.tools.company_tools import (
    CheckCompanyRecordArgs,
    UpdateCompanySystemArgs,
    check_company_record,
    update_company_system,
)
from app.tools.invoice_tools import (
    ReadInvoiceArgs,
    SearchInvoicesArgs,
    read_invoice,
    search_invoices,
)

logger = logging.getLogger(__name__)

TOOLS: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in [
        ToolSpec(
            "search_invoices",
            "Search the invoice registry for a company (case-insensitive prefix match). "
            "Returns invoices newest first with invoice_id, company, invoice_date, "
            "file_reference and status.",
            SearchInvoicesArgs, search_invoices,
        ),
        ToolSpec(
            "read_invoice",
            "Read one invoice document and extract invoice_id, company, amount, currency, "
            "due_date and invoice_date.",
            ReadInvoiceArgs, read_invoice,
        ),
        ToolSpec(
            "update_company_system",
            "Create or update the invoice record in the internal company system. Use only "
            "values extracted by read_invoice.",
            UpdateCompanySystemArgs, update_company_system,
        ),
        ToolSpec(
            "check_company_record",
            "Read the internal company system record for an invoice and compare it with the "
            "expected company, amount and due date.",
            CheckCompanyRecordArgs, check_company_record,
        ),
        ToolSpec(
            "request_human_approval",
            "Ask a human to approve a sensitive or uncertain action. Use when you cannot "
            "safely proceed on your own.",
            RequestHumanApprovalArgs, request_human_approval,
        ),
    ]
}


def tool_definitions() -> list[dict[str, Any]]:
    """Tool schemas in the OpenAI function-calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.args_model.model_json_schema(),
            },
        }
        for spec in TOOLS.values()
    ]


def _format_validation_error(exc: ValidationError) -> str:
    parts = [f"{'.'.join(str(p) for p in e['loc']) or 'arguments'}: {e['msg']}" for e in exc.errors()]
    return "; ".join(parts)


def run_tool(name: str, raw_args: Any, ctx: ToolContext) -> ToolResult:
    spec = TOOLS.get(name)
    if spec is None:
        return ToolResult.fail(
            "UNKNOWN_TOOL", f"Unknown tool '{name}'. Available tools: {', '.join(TOOLS)}."
        )
    try:
        args = spec.args_model.model_validate(raw_args)
    except ValidationError as exc:
        return ToolResult.fail("INVALID_ARGUMENTS", _format_validation_error(exc))
    try:
        return spec.handler(args, ctx)
    except sqlite3.Error as exc:
        logger.warning("database error in %s: %s", name, exc)
        return ToolResult.fail("DB_ERROR", f"Database error: {exc}", retryable=True)
    except Exception as exc:  # tools must never crash the agent loop
        logger.exception("unexpected error in tool %s", name)
        return ToolResult.fail("TOOL_EXCEPTION", f"Unexpected tool error: {exc}", retryable=True)
