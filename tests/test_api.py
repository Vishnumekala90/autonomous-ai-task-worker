"""API tests through FastAPI's TestClient (real app, real graph, real SQLite)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from tests.helpers import ACME_TASK, make_settings

GLOBEX = {"task": "Find the latest invoice from Globex and record it."}


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(make_settings(tmp_path)))


def test_health_reports_model_and_database(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["database"] == "ok"
    assert body["llm"] == {"provider": "scripted", "model": "rule-based-policy-v1", "is_real_llm": False}


def test_post_task_wait_returns_result_summary_and_evidence(client):
    r = client.post("/tasks?wait=true", json={"task": ACME_TASK})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == body["result"] == "COMPLETED"
    assert body["evidence"]["internal_record_id"] == "REC-10002"
    assert body["evidence"]["verification"] == "PASSED"
    assert body["task_id"].startswith("TASK-") and body["summary"].startswith("Task completed successfully")


def test_post_task_is_asynchronous_by_default(client):
    r = client.post("/tasks", json={"task": ACME_TASK})
    assert r.status_code == 202
    body = client.get(f"/tasks/{r.json()['task_id']}").json()
    assert body["status"] == "COMPLETED"  # background run finished


def test_get_task_and_history(client):
    task_id = client.post("/tasks?wait=true", json={"task": ACME_TASK}).json()["task_id"]
    assert client.get(f"/tasks/{task_id}").json()["task"].startswith("Find the latest invoice")
    history = client.get(f"/tasks/{task_id}/history").json()
    assert [t["tool"] for t in history["tool_calls"]][:2] == ["search_invoices", "read_invoice"]
    assert any(e["node"] == "verify_result" for e in history["events"])


def test_unknown_task_returns_404(client):
    assert client.get("/tasks/TASK-NOPE").status_code == 404
    assert client.get("/tasks/TASK-NOPE/history").status_code == 404
    assert client.post("/tasks/TASK-NOPE/approve").status_code == 404


@pytest.mark.parametrize("payload", [{"task": "hi"}, {"task": "x" * 501}, {}, {"task": ACME_TASK, "simulate_fault": "explode"}])
def test_invalid_requests_are_rejected(client, payload):
    assert client.post("/tasks", json=payload).status_code == 422


def test_not_found_company_over_api(client):
    body = client.post("/tasks?wait=true", json={"task": "Find the latest invoice from UnknownCompany and record it."}).json()
    assert body["result"] == "NEEDS_CLARIFICATION"


def test_fault_injection_over_api(client):
    body = client.post("/tasks?wait=true", json={"task": ACME_TASK, "simulate_fault": "update_persistent"}).json()
    assert body["result"] == "ESCALATED_TO_HUMAN" and body["evidence"]["verification"] == "NOT_RUN"


def test_fault_injection_can_be_disabled(tmp_path):
    client = TestClient(create_app(make_settings(tmp_path, enable_fault_injection=False)))
    r = client.post("/tasks?wait=true", json={"task": ACME_TASK, "simulate_fault": "update_persistent"})
    assert r.status_code == 403


def test_approval_flow_over_api(client):
    paused = client.post("/tasks?wait=true", json=GLOBEX).json()
    assert paused["status"] == "AWAITING_APPROVAL" and paused["approval"]["reason"]
    task_id = paused["task_id"]

    done = client.post(f"/tasks/{task_id}/approve?wait=true", json={"note": "approved by test"}).json()
    assert done["status"] == "COMPLETED" and done["evidence"]["human_approvals"][0]["note"] == "approved by test"
    assert client.post(f"/tasks/{task_id}/approve").status_code == 409  # already decided


def test_rejection_flow_over_api(client):
    task_id = client.post("/tasks?wait=true", json=GLOBEX).json()["task_id"]
    body = client.post(f"/tasks/{task_id}/reject?wait=true").json()
    assert body["result"] == "REJECTED_BY_HUMAN"
    assert client.post(f"/tasks/{task_id}/reject").status_code == 409


def test_approving_a_task_without_pending_approval_is_a_conflict(client):
    task_id = client.post("/tasks?wait=true", json={"task": ACME_TASK}).json()["task_id"]
    assert client.post(f"/tasks/{task_id}/approve").status_code == 409


def test_approval_in_background_mode(client):
    task_id = client.post("/tasks", json=GLOBEX).json()["task_id"]
    assert client.get(f"/tasks/{task_id}").json()["status"] == "AWAITING_APPROVAL"
    assert client.post(f"/tasks/{task_id}/approve").status_code == 202
    assert client.get(f"/tasks/{task_id}").json()["status"] == "COMPLETED"


def test_openapi_documents_all_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/health", "/tasks", "/tasks/{task_id}", "/tasks/{task_id}/history",
            "/tasks/{task_id}/approve", "/tasks/{task_id}/reject"} <= set(paths)
