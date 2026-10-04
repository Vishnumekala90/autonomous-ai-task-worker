# Demo Script (5-7 minutes)

Before recording: `python scripts/seed_database.py`, then start the backend and UI in two terminals (commands below). Use a real key in `.env` if you have one; otherwise say clearly that the offline scripted policy is in use.

## 0:00 - 0:40  Introduction
**Say:** "This is an autonomous invoice worker. I give it a sentence; it finds the invoice, reads the amount and due date, enters them into a simulated internal company system, and independently verifies the record. The company system is a controlled local sandbox - no real credentials or third-party systems."
**Show:** README title and the sentence from the assessment.

## 0:40 - 1:30  Architecture
**Show:** the two Mermaid diagrams in the README.
**Say:** "LangGraph state machine. The LLM chooses the next tool from what it has observed - the graph does not hard-code search, read, update. Code around the LLM enforces safety: a policy gate before writes, bounded retries, human approval, and a separate verification node. State is saved to SQLite after every node, so approvals can pause and resume."

## 1:30 - 2:00  Start backend and frontend
```bash
uvicorn app.main:create_app --factory --port 8000
streamlit run frontend/streamlit_app.py
```
**Show:** `http://localhost:8000/health` (active model is visible) and the UI sidebar (model, threshold, retries). If running offline, point at the "scripted offline policy, not a real LLM" label.

## 2:00 - 3:15  Successful task, history, verification
**Do:** Task = "Find the latest invoice from Acme Corp, record it in the internal system, verify it, and tell me when it is done." -> RUN TASK.
**Show:** plan ticking off, status `COMPLETED`, Verification `PASSED`, evidence (INV-1003, ₹84,500, 2026-10-20, REC-10002).
**Open Execution history:** point at `select_tool` (search_invoices -> read_invoice -> update_company_system -> check_company_record), `check_policy: allowed`, and the final `verify_result`.
**Say:** "Nothing here is an LLM claiming success. The `verify_result` node re-read the company record and compared four fields to the invoice."

## 3:15 - 4:00  Failure scenario, retry and escalation
**Do:** Sidebar -> simulate `update_transient`, run the Acme task.
**Show:** history: `execute (failed) SYSTEM_UNAVAILABLE` -> `retry 1/3` -> `retry 2/3` -> success -> verification PASSED.
**Do:** Simulate `update_persistent`.
**Show:** three retries with backoff, then `ESCALATED_TO_HUMAN` - "needs human intervention" - and it never claims success.
**Do (30 s):** Simulate `corrupt_record`: the write returns success but stores a wrong amount. **Show** `VERIFICATION_FAILED` with the per-field mismatch. "A successful tool response is not proof; verification catches it."
**Do (10 s):** Task for `UnknownCompany` -> `NEEDS_CLARIFICATION` with suggestions.

## 4:00 - 5:00  Human approval
**Do:** Task = "Find the latest invoice from Globex and record it."
**Show:** "Approval required" panel: reason (₹3,10,000 above ₹1,00,000), proposed action, Approve / Reject. Check that nothing was written yet.
**Do:** Approve -> the agent resumes, writes, verifies, `COMPLETED`.
**Optional:** run Globex again and Reject -> `REJECTED_BY_HUMAN`, no record written. Mention duplicate (Contoso) and ambiguous-latest (TechNova) approvals.

## 5:00 - 6:15  Code walkthrough
- `app/agent/graph.py`: the conditional edges (each decision depends on state).
- `app/agent/nodes.py`: `decide_next_action` (LLM picks the tool), `check_policy` (precondition + approval rules), `handle_failure` (retry vs adapt vs escalate), `verify_result`.
- `app/tools/registry.py`: validation + error normalisation. `app/services/verification.py`: the field comparison.
- `tests/test_failure_recovery.py`: misbehaving fake LLMs (hallucinated amounts, bad args, infinite loop, false "I'm done") and the graph staying safe.
- `python -m pytest` -> 101 passed.

## 6:15 - 7:00  Limitations and next steps
**Say honestly:** simulated company system; template-based invoice parsing; without a key the offline scripted policy is used; escalations are terminal; single process.
**Next:** real connectors (browser/desktop, email, ERP), OCR/vision extraction, durable queue, company memory, per-role permissions and audit log, and an evaluation harness across models.
