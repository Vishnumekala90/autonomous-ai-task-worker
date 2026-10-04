"""Agent behaviour: end-to-end runs through the real LangGraph with real tools and SQLite."""

from __future__ import annotations

from app.database.repository import get_company_record
from tests.helpers import (
    ACME_TASK,
    BadArgsOnceLLM,
    BrokenLLM,
    ImmediateFinishLLM,
    LoopingLLM,
    PrematureUpdateLLM,
    WrongAmountOnceLLM,
    events,
    make_env,
    run,
    tool_attempts,
)


def test_successful_end_to_end_task(tmp_path):
    service, db, _ = make_env(tmp_path)
    view, task_id = run(service)

    assert view.status == view.result == "COMPLETED"
    ev = view.evidence
    assert (ev["invoice_id"], ev["company"], ev["amount"], ev["due_date"]) == (
        "INV-1003", "Acme Corp", 84500.0, "2026-10-20")
    assert ev["verification"] == "PASSED" and ev["internal_record_id"] == "REC-10002"
    assert all(c["match"] for c in ev["verification_checks"])
    assert "Task completed successfully" in view.summary and "₹84,500" in view.summary
    assert get_company_record(db, "INV-1003")["amount"] == 84500.0  # the work really happened
    assert [s["status"] for s in view.plan] == ["done"] * 5

    tools_called = [t["tool"] for t in service.get_history(task_id).tool_calls]
    assert tools_called == ["search_invoices", "read_invoice", "update_company_system", "check_company_record"]
    nodes = {e["node"] for e in service.get_history(task_id).events}
    assert {"understand_task", "create_plan", "decide_next_action", "execute_tool",
            "observe_result", "verify_result", "finalize"} <= nodes


def test_history_events_have_required_fields(tmp_path):
    service, _, _ = make_env(tmp_path)
    _, task_id = run(service)
    for e in service.get_history(task_id).events:
        assert {"timestamp", "node", "action", "tool", "input_summary", "output_summary", "status", "error"} <= set(e)


def test_different_company_needs_no_code_changes(tmp_path):
    service, db, _ = make_env(tmp_path)
    view, _ = run(service, "Please record the newest Acme invoice, thanks")  # different phrasing
    assert view.result == "COMPLETED" and view.evidence["invoice_id"] == "INV-1003"
    view, _ = run(service, "Find the latest invoice from TechNova and enter it", None)
    assert view.status == "AWAITING_APPROVAL"  # TechNova's latest date is tied -> ambiguity


def test_specific_invoice_id_skips_search(tmp_path):
    service, db, _ = make_env(tmp_path)
    view, task_id = run(service, "Record invoice INV-1002 from Acme Corp in the company system")
    assert view.result == "COMPLETED" and view.evidence["invoice_id"] == "INV-1002"
    assert "search_invoices" not in [t["tool"] for t in service.get_history(task_id).tool_calls]
    assert get_company_record(db, "INV-1003") is None


def test_invoice_not_found_asks_for_clarification(tmp_path):
    service, db, _ = make_env(tmp_path)
    view, task_id = run(service, "Find the latest invoice from Acmee Corp and record it.")
    assert view.result == "NEEDS_CLARIFICATION"
    assert "Acme Corp" in view.summary  # offers a close match
    assert view.evidence["verification"] == "NOT_RUN"
    assert tool_attempts(service, task_id, "update_company_system") == []
    assert get_company_record(db, "INV-1003") is None


def test_unknown_company_does_not_write_anything(tmp_path):
    service, db, _ = make_env(tmp_path)
    view, task_id = run(service, "Find the latest invoice from UnknownCompany and record it.")
    assert view.result == "NEEDS_CLARIFICATION" and "Task completed" not in view.summary
    assert tool_attempts(service, task_id, "update_company_system") == []


def test_unsupported_task_is_declined(tmp_path):
    service, _, _ = make_env(tmp_path)
    view, task_id = run(service, "What is the weather like in Vijayawada today?")
    assert view.result == "NEEDS_CLARIFICATION"
    assert service.get_history(task_id).tool_calls == []


def test_missing_company_asks_which_one(tmp_path):
    service, _, _ = make_env(tmp_path)
    view, _ = run(service, "Record the latest invoice")
    assert view.result == "NEEDS_CLARIFICATION" and "Which company" in view.summary


def test_task_text_is_sanitised(tmp_path):
    service, _, _ = make_env(tmp_path)
    task_id = service.create_task("Find the latest invoice\x00 from\x07   Acme Corp\n\n and record it")
    assert service.get_task(task_id).task == "Find the latest invoice from Acme Corp and record it"


# ---- the graph, not the model, enforces safety ----

def test_update_before_read_is_blocked_then_model_adapts(tmp_path):
    service, db, _ = make_env(tmp_path, llm=PrematureUpdateLLM())
    view, task_id = run(service)
    blocked = events(service, task_id, "blocked")
    assert len(blocked) == 1 and blocked[0]["error"] == "PRECONDITION_FAILED"
    assert view.result == "COMPLETED"
    assert get_company_record(db, "INV-1003")["amount"] == 84500.0  # never the made-up 99999


def test_hallucinated_amount_is_blocked(tmp_path):
    service, db, _ = make_env(tmp_path, llm=WrongAmountOnceLLM())
    view, task_id = run(service)
    blocked = events(service, task_id, "blocked")
    assert [b["error"] for b in blocked] == ["ARGUMENT_MISMATCH"]
    assert view.result == "COMPLETED" and get_company_record(db, "INV-1003")["amount"] == 84500.0


def test_invalid_tool_arguments_are_fed_back_to_the_model(tmp_path):
    service, db, _ = make_env(tmp_path, llm=BadArgsOnceLLM())
    view, task_id = run(service)
    assert view.result == "COMPLETED"
    assert [e["error_code"] for e in view.errors] == ["INVALID_ARGUMENTS"]
    assert len(events(service, task_id, "adapt")) == 1


def test_infinite_loop_is_stopped_by_step_budget(tmp_path):
    service, _, _ = make_env(tmp_path, llm=LoopingLLM(), max_steps=5)
    view, _ = run(service)
    assert view.result == "FAILED" and "Step budget" in view.summary


def test_claiming_done_without_work_is_not_success(tmp_path):
    service, db, _ = make_env(tmp_path, llm=ImmediateFinishLLM())
    view, _ = run(service)
    assert view.result == "VERIFICATION_FAILED"
    assert "completed successfully" not in view.summary.lower()
    assert get_company_record(db, "INV-1003") is None


def test_llm_outage_fails_gracefully(tmp_path):
    service, _, _ = make_env(tmp_path, llm=BrokenLLM())
    view, _ = run(service)
    assert view.result == "FAILED" and "provider is down" in view.summary


def test_evidence_documents_which_model_was_used(tmp_path):
    service, _, _ = make_env(tmp_path)
    view, _ = run(service)
    assert view.evidence["model"] == {"provider": "scripted", "model": "rule-based-policy-v1", "is_real_llm": False}
    assert view.llm["is_real_llm"] is False
