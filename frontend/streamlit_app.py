"""Streamlit UI for the Autonomous Invoice Processing Worker.

Talks to the FastAPI backend only (default http://localhost:8000, override with API_URL).
"""

from __future__ import annotations

import os
import time
from typing import Any

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
DEFAULT_TASK = (
    "Find the latest invoice from Acme Corp, extract the invoice amount and due date, enter it into "
    "our internal company system, verify that it was entered correctly, and tell me once it is done."
)
EXAMPLES = {
    "Success - Acme": DEFAULT_TASK,
    "Not found - UnknownCompany": "Find the latest invoice from UnknownCompany and record it.",
    "Needs approval - Globex (large amount)": "Find the latest invoice from Globex and record it.",
    "Duplicate record - Contoso": "Find the latest invoice from Contoso and record it.",
    "Ambiguous latest - TechNova": "Find the latest invoice from TechNova and record it.",
}
FAULTS = {
    "None": None,
    "Update fails twice, then recovers": "update_transient",
    "Update keeps failing (escalates)": "update_persistent",
    "Update writes wrong data (verification fails)": "corrupt_record",
    "One database error on search": "db_transient",
}
PLAN_ICON = {"done": "✓", "in_progress": "→", "pending": "○", "failed": "✗"}
RUNNING = {"QUEUED", "RUNNING"}


class ApiError(Exception):
    pass


def api(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
    try:
        resp = requests.request(method, f"{API_URL}{path}", timeout=30, **kwargs)
    except requests.RequestException as exc:
        raise ApiError(f"Cannot reach the backend at {API_URL}. Is it running? ({type(exc).__name__})") from exc
    if resp.status_code >= 400:
        raise ApiError(f"{resp.status_code}: {resp.json().get('detail', resp.text)}")
    return resp.json()


def status_banner(status: str) -> None:
    text = f"Status: **{status}**"
    if status == "COMPLETED":
        st.success(text)
    elif status in RUNNING:
        st.info(text)
    elif status == "AWAITING_APPROVAL":
        st.warning(text)
    else:
        st.error(text)


def render_plan(plan: list[dict[str, Any]]) -> None:
    if not plan:
        st.caption("Plan not created yet.")
    for step in plan:
        st.markdown(f"{PLAN_ICON.get(step['status'], '○')} {step['description']}")


def render_approval(task_id: str, approval: dict[str, Any]) -> None:
    st.subheader("Approval required")
    st.markdown(f"**Reason:** {approval['reason']}")
    st.markdown(f"**Proposed action:** `{approval['proposed_action']}`")
    note = st.text_input("Note (optional)", key=f"note_{task_id}")
    approve_col, reject_col, _ = st.columns([1, 1, 4])
    if approve_col.button("Approve", type="primary", key=f"approve_{task_id}"):
        api("POST", f"/tasks/{task_id}/approve", json={"note": note})
        st.rerun()
    if reject_col.button("Reject", key=f"reject_{task_id}"):
        api("POST", f"/tasks/{task_id}/reject", json={"note": note})
        st.rerun()


def render_verification(verification: dict[str, Any] | None) -> None:
    st.subheader("Verification")
    if not verification:
        st.caption("Not run yet.")
        return
    (st.success if verification["passed"] else st.error)(
        ("PASSED" if verification["passed"] else "FAILED") + f" - {verification['reason']}"
    )
    if verification["checks"]:
        st.dataframe(pd.DataFrame(verification["checks"]), hide_index=True, use_container_width=True)


def render_result(view: dict[str, Any]) -> None:
    st.subheader("Final result")
    if not view["result"]:
        st.caption("Task has not finished.")
        return
    (st.success if view["result"] == "COMPLETED" else st.error)(view["summary"])
    ev = view["evidence"]
    if ev.get("invoice_id"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Invoice", ev["invoice_id"])
        c2.metric("Amount", ev["amount_display"])
        c3.metric("Due date", ev["due_date"])
        c4.metric("Internal record", ev.get("internal_record_id") or "-")
    st.markdown(f"**Verification:** {ev.get('verification')}  ·  **Model:** {ev['model']['provider']} / {ev['model']['model']}"
                + ("" if ev["model"]["is_real_llm"] else " *(scripted offline policy, not a real LLM)*"))
    if ev.get("actions_performed"):
        st.markdown("**Actions performed**")
        for i, action in enumerate(ev["actions_performed"], start=1):
            st.markdown(f"{i}. {action}")
    if ev.get("problems_encountered"):
        with st.expander(f"Problems encountered ({len(ev['problems_encountered'])})"):
            for problem in ev["problems_encountered"]:
                st.markdown(f"- {problem}")


def render_history(task_id: str) -> None:
    history = api("GET", f"/tasks/{task_id}/history")
    st.subheader("Execution history")
    events = pd.DataFrame(history["events"])
    if not events.empty:
        st.dataframe(
            events[["timestamp", "node", "action", "tool", "status", "input_summary", "output_summary", "error"]],
            hide_index=True, use_container_width=True,
        )
    with st.expander(f"Tool calls ({len(history['tool_calls'])})"):
        for call in history["tool_calls"]:
            ok = call["result"]["success"]
            st.markdown(f"{'✅' if ok else '❌'} `{call['tool']}` attempt {call['attempt']}")
            st.json({"arguments": call["arguments"], "result": call["result"]}, expanded=False)


def main() -> None:
    st.set_page_config(page_title="Autonomous AI Worker", page_icon="🧾", layout="wide")
    st.title("Autonomous AI Worker")
    st.caption("CentrAlign AI · invoice processing against a controlled local company environment")

    try:
        health = api("GET", "/health")
    except ApiError as exc:
        st.error(str(exc))
        st.stop()

    with st.sidebar:
        st.header("Environment")
        llm = health["llm"]
        st.markdown(f"**Model:** {llm['provider']} / `{llm['model']}`")
        if not llm["is_real_llm"]:
            st.warning("No API key configured: using the scripted offline policy, not a real LLM.")
        st.markdown(f"**Approval threshold:** {health['approval_threshold']:,.0f}  \n**Max retries:** {health['max_retries']}")
        st.divider()
        st.header("Demo helpers")
        example = st.selectbox("Example task", list(EXAMPLES))
        if st.button("Use example"):
            st.session_state["task_text"] = EXAMPLES[example]
        fault = None
        if health["fault_injection_enabled"]:
            fault = FAULTS[st.selectbox("Simulate a failure (controlled simulation)", list(FAULTS))]

    task_text = st.text_area("Task", value=st.session_state.get("task_text", DEFAULT_TASK), key="task_text", height=100)
    if st.button("RUN TASK", type="primary"):
        try:
            created = api("POST", "/tasks", json={"task": task_text, "simulate_fault": fault})
            st.session_state["task_id"] = created["task_id"]
            st.rerun()
        except ApiError as exc:
            st.error(str(exc))

    task_id = st.session_state.get("task_id")
    if not task_id:
        return
    try:
        view = api("GET", f"/tasks/{task_id}")
    except ApiError as exc:
        st.error(str(exc))
        return

    st.divider()
    st.caption(f"Task ID: {task_id}")
    status_banner(view["status"])
    left, right = st.columns(2)
    with left:
        st.subheader("Plan")
        render_plan(view["plan"])
        st.markdown(f"**Current action:** {view['current_step']}")
        if view["errors"] or view["retry_count"]:
            st.subheader("Errors / retries")
            st.markdown(f"Retries used on current action: **{view['retry_count']}**")
            st.dataframe(pd.DataFrame(view["errors"]), hide_index=True, use_container_width=True)
    with right:
        render_verification(view["verification"])
    if view["approval"]:
        render_approval(task_id, view["approval"])
    render_result(view)
    render_history(task_id)

    if view["status"] in RUNNING:
        time.sleep(1)
        st.rerun()


main()
