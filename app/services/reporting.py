"""Deterministic final summary and evidence.

The summary is built from recorded facts (tool results, verification), not written by the
LLM, so it cannot claim success that did not happen.
"""

from __future__ import annotations

from typing import Any

from app.agent.state import AgentState


def format_money(amount: float, currency: str = "INR") -> str:
    whole, _, frac = f"{amount:.2f}".partition(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        groups: list[str] = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        whole = ",".join([*groups, tail])
    number = whole if frac == "00" else f"{whole}.{frac}"
    return f"₹{number}" if currency == "INR" else f"{currency} {number}"


def _describe_action(entry: dict[str, Any]) -> str | None:
    tool, args, result = entry["tool"], entry["arguments"], entry["result"]
    data = result.get("data", {})
    if not result.get("success"):
        return None
    if tool == "search_invoices":
        invoices = data.get("invoices", [])
        latest = invoices[0] if invoices else {}
        return (
            f"Searched invoices for '{args.get('company_name')}': {len(invoices)} found "
            f"(newest {latest.get('invoice_id')}, dated {latest.get('invoice_date')})"
        )
    if tool == "read_invoice":
        return (
            f"Read {data.get('invoice_id')}: {format_money(data.get('amount', 0), data.get('currency', 'INR'))}, "
            f"due {data.get('due_date')}"
        )
    if tool == "update_company_system":
        verb = {"created": "Created", "updated": "Updated", "unchanged": "Found existing"}.get(
            data.get("action", ""), "Wrote"
        )
        return f"{verb} internal record {data.get('record_id')} for {args.get('invoice_id')}"
    if tool == "check_company_record":
        return f"Checked internal record for {args.get('invoice_id')}"
    return None


def build_evidence(state: AgentState, llm_info: dict[str, Any]) -> dict[str, Any]:
    info = state.get("discovered_information", {})
    invoice = info.get("invoice") or {}
    record = info.get("record") or {}
    verification = state.get("verification_result")
    if verification is None:
        verdict = "NOT_RUN"
    else:
        verdict = "PASSED" if verification["passed"] else "FAILED"

    actions = [a for e in state.get("tool_results", []) if (a := _describe_action(e))]
    if verification is not None:
        actions.append(f"Independently verified the record: {verdict} ({verification['reason']})")
    problems = [
        f"{e['tool']} failed ({e['error_code']}): {e['message']}" for e in state.get("errors", [])
    ]
    return {
        "invoice_id": invoice.get("invoice_id"),
        "company": invoice.get("company"),
        "amount": invoice.get("amount"),
        "currency": invoice.get("currency"),
        "amount_display": (
            format_money(invoice["amount"], invoice.get("currency", "INR")) if "amount" in invoice else None
        ),
        "due_date": invoice.get("due_date"),
        "internal_record_id": record.get("record_id"),
        "verification": verdict,
        "verification_checks": (verification or {}).get("checks", []),
        "actions_performed": actions,
        "problems_encountered": problems,
        "human_approvals": state.get("approvals", []),
        "retries_used": sum(1 for e in state.get("execution_history", []) if e["action"] == "retry"),
        "model": llm_info,
    }


def build_summary(state: AgentState, evidence: dict[str, Any]) -> str:
    status = state.get("final_status")
    reason = state.get("final_reason", "")
    inv, comp = evidence["invoice_id"], evidence["company"]
    if status == "COMPLETED":
        return (
            f"Task completed successfully. Invoice {inv} from {comp} "
            f"({evidence['amount_display']}, due {evidence['due_date']}) is recorded as "
            f"{evidence['internal_record_id']} and verification PASSED."
        )
    if status == "NEEDS_CLARIFICATION":
        return f"I could not complete this task and need more information. {reason}"
    if status == "REJECTED_BY_HUMAN":
        return f"Task stopped: a human rejected the proposed action. Nothing was changed. {reason}".strip()
    if status == "VERIFICATION_FAILED":
        return (
            "Task NOT completed: the internal record could not be verified against the invoice. "
            f"{reason.rstrip('.')}. A human needs to review it."
        )
    if status == "ESCALATED_TO_HUMAN":
        return f"Task NOT completed: I could not recover automatically and need human help. {reason}"
    return f"Task failed. {reason}"
