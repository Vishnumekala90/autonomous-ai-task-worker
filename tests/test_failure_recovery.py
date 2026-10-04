"""Failure recovery and human-in-the-loop behaviour."""

from __future__ import annotations

import pytest

from app.database.repository import get_approval, get_company_record
from app.services.task_service import InvalidTaskState, TaskService
from tests.helpers import ApprovalFirstLLM, events, make_env, run, tool_attempts

GLOBEX = "Find the latest invoice from Globex and record it."
CONTOSO = "Find the latest invoice from Contoso and record it."


# ---------------------------------------------------------------- retries

def test_transient_update_failure_is_retried_and_verified(tmp_path):
    service, db, _ = make_env(tmp_path)
    view, task_id = run(service, fault="update_transient")
    assert view.result == "COMPLETED"
    assert tool_attempts(service, task_id, "update_company_system") == [1, 2, 3]
    assert len(events(service, task_id, "retry")) == 2
    assert [e["error_code"] for e in view.errors] == ["SYSTEM_UNAVAILABLE"] * 2
    assert view.evidence["verification"] == "PASSED" and view.evidence["retries_used"] == 2
    assert get_company_record(db, "INV-1003")["amount"] == 84500.0


def test_persistent_update_failure_escalates_after_bounded_retries(tmp_path):
    service, db, _ = make_env(tmp_path)
    view, task_id = run(service, fault="update_persistent")
    assert view.result == "ESCALATED_TO_HUMAN"
    assert tool_attempts(service, task_id, "update_company_system") == [1, 2, 3, 4]  # 1 try + 3 retries
    assert "human" in view.summary.lower() and "completed successfully" not in view.summary.lower()
    assert view.evidence["verification"] == "NOT_RUN"
    assert get_company_record(db, "INV-1003") is None


def test_retry_limit_is_configurable(tmp_path):
    service, _, _ = make_env(tmp_path, max_retries=1)
    view, task_id = run(service, fault="update_persistent")
    assert view.result == "ESCALATED_TO_HUMAN"
    assert tool_attempts(service, task_id, "update_company_system") == [1, 2]


def test_database_error_on_search_is_retried(tmp_path):
    service, _, _ = make_env(tmp_path)
    view, task_id = run(service, fault="db_transient")
    assert view.result == "COMPLETED"
    assert [e["error_code"] for e in view.errors] == ["DB_ERROR"]
    assert tool_attempts(service, task_id, "search_invoices") == [1, 2]


# ---------------------------------------------------------------- verification

def test_verification_failure_is_detected_and_never_reported_as_success(tmp_path):
    service, db, _ = make_env(tmp_path)
    view, task_id = run(service, fault="corrupt_record")
    assert view.result == "VERIFICATION_FAILED"
    assert view.evidence["verification"] == "FAILED"
    amount = next(c for c in view.evidence["verification_checks"] if c["field"] == "amount")
    assert amount["expected"] == 84500.0 and amount["actual"] == pytest.approx(92950.0)
    assert "NOT completed" in view.summary and "completed successfully" not in view.summary.lower()
    assert len(tool_attempts(service, task_id, "update_company_system")) == 4  # bounded corrective attempts
    assert get_company_record(db, "INV-1003")["amount"] != 84500.0


# ---------------------------------------------------------------- human approval

def test_high_amount_pauses_then_resumes_after_approval(tmp_path):
    service, db, _ = make_env(tmp_path)
    view, task_id = run(service, GLOBEX)

    assert view.status == "AWAITING_APPROVAL" and view.result is None
    assert "exceeds the approval threshold" in view.approval["reason"]
    assert "INV-2002" in view.approval["proposed_action"]
    assert get_company_record(db, "INV-2002") is None  # nothing written while waiting
    assert get_approval(db, view.approval["approval_id"])["status"] == "pending"

    service.decide_approval(task_id, True, "ok by finance")
    view = service.run_task(task_id)
    assert view.status == "COMPLETED" and view.evidence["verification"] == "PASSED"
    assert get_company_record(db, "INV-2002")["amount"] == 310000.0
    assert get_approval(db, 1)["status"] == "approved"
    assert view.evidence["human_approvals"][0]["status"] == "approved"


def test_rejection_makes_no_change(tmp_path):
    service, db, _ = make_env(tmp_path)
    _, task_id = run(service, GLOBEX)
    service.decide_approval(task_id, False, "wrong vendor")
    view = service.run_task(task_id)
    assert view.result == "REJECTED_BY_HUMAN" and "wrong vendor" in view.summary
    assert get_company_record(db, "INV-2002") is None
    assert tool_attempts(service, task_id, "update_company_system") == []


def test_approval_survives_a_service_restart(tmp_path):
    service, db, settings = make_env(tmp_path)
    _, task_id = run(service, GLOBEX)
    restarted = TaskService(db, settings)  # new process, same database
    assert restarted.get_task(task_id).status == "AWAITING_APPROVAL"
    restarted.decide_approval(task_id, True)
    assert restarted.run_task(task_id).result == "COMPLETED"


def test_threshold_is_configurable(tmp_path):
    service, _, _ = make_env(tmp_path, approval_threshold=1_000_000)
    view, _ = run(service, GLOBEX)
    assert view.result == "COMPLETED"


def test_cannot_decide_twice_or_without_a_pending_approval(tmp_path):
    service, _, _ = make_env(tmp_path)
    done, done_id = run(service)
    with pytest.raises(InvalidTaskState):
        service.decide_approval(done_id, True)
    _, task_id = run(service, GLOBEX)
    service.decide_approval(task_id, True)
    with pytest.raises(InvalidTaskState):
        service.decide_approval(task_id, True)


def test_model_can_request_approval_on_its_own(tmp_path):
    service, _, _ = make_env(tmp_path, llm=ApprovalFirstLLM())
    view, task_id = run(service)
    assert view.status == "AWAITING_APPROVAL" and view.approval["gated"] is False
    service.decide_approval(task_id, True)
    assert service.run_task(task_id).result == "COMPLETED"


def test_ambiguous_latest_invoice_requires_approval(tmp_path):
    service, db, _ = make_env(tmp_path)
    view, _ = run(service, "Find the latest invoice from TechNova and record it.")
    assert view.status == "AWAITING_APPROVAL"
    assert "ambiguous" in view.approval["reason"] and "INV-3002" in view.approval["reason"]
    assert get_company_record(db, "INV-3003") is None


# ---------------------------------------------------------------- duplicates

def test_duplicate_record_requires_approval_then_overwrites(tmp_path):
    service, db, _ = make_env(tmp_path)
    view, task_id = run(service, CONTOSO)
    assert view.status == "AWAITING_APPROVAL"
    assert "REC-10001" in view.approval["reason"] and "duplicate_record" in view.approval["reasons"]
    assert get_company_record(db, "INV-4002")["amount"] == 70000.0  # untouched while waiting

    service.decide_approval(task_id, True)
    view = service.run_task(task_id)
    assert view.result == "COMPLETED" and view.evidence["internal_record_id"] == "REC-10001"
    assert get_company_record(db, "INV-4002")["amount"] == 72300.0


def test_duplicate_record_rejection_keeps_existing_record(tmp_path):
    service, db, _ = make_env(tmp_path)
    _, task_id = run(service, CONTOSO)
    service.decide_approval(task_id, False)
    view = service.run_task(task_id)
    assert view.result == "REJECTED_BY_HUMAN"
    assert get_company_record(db, "INV-4002")["amount"] == 70000.0


def test_identical_existing_record_is_not_a_duplicate_problem(tmp_path):
    service, db, _ = make_env(tmp_path)
    first, _ = run(service)
    second, task_id = run(service)  # same invoice again
    assert second.result == "COMPLETED"
    assert second.evidence["internal_record_id"] == first.evidence["internal_record_id"]
    assert "Found existing" in " ".join(second.evidence["actions_performed"])
