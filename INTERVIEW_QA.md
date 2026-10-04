# Interview Q&A (about this exact implementation)

**1. Why did you choose LangGraph?**
The task is a loop with branches (act, observe, retry, adapt, ask a human, verify). LangGraph gives me nodes and conditional edges, so I can place hard rules (policy gate, retry budget, verification) between the LLM and the tools and see the control flow in `graph.py`. It also lets a run end at `request_approval` and re-enter later at `resume_after_approval`.

**2. Why not use a simple LangChain agent?**
A prebuilt agent loop is "LLM calls tools until it stops". I needed: a pre-write safety check, classified failures with different responses, a pause/resume for approvals, and a verification step the LLM cannot skip. Those are explicit nodes here. A simple agent would hide them inside one loop.

**3. How does the agent decide which tool to call?**
`decide_next_action` sends the conversation so far (task, goal, plan, every tool result) plus the tool schemas to the LLM using native function calling. The model returns one tool call or plain text. The graph does not contain "search then read then update"; the order comes from what the model has observed. The graph only checks the choice afterwards (`check_policy`).

**4. How is state maintained?**
A typed `AgentState` dict (goal, plan, messages, `discovered_information`, tool results, history, errors, counters, approval and verification fields). Each node returns a partial update. `TaskService.run_task` streams the graph and saves the whole state as JSON in the `tasks` table after every node.

**5. What happens if a tool fails?**
The tool returns (or the registry converts an exception into) `ToolResult(success=False, error_code, retryable)`. `observe_result` records the error and routes to `handle_failure`, which retries, hands the error to the LLM, asks a human, or terminates gracefully, depending on the error.

**6. How does retry work?**
Only for `retryable` errors (system unavailable, `sqlite3.Error`). `handle_failure` sleeps `backoff * 2^n`, increments `retry_count` and re-runs the same call. After `MAX_RETRIES` (3) it ends with `ESCALATED_TO_HUMAN`. `retry_count` resets after a success. Demo: `update_transient` fails twice, then succeeds.

**7. How do you prevent infinite loops?**
Several limits: retry budget (3), adaptation budget (3 LLM corrections), `MAX_STEPS` (30 decisions) -> `FAILED`, and LangGraph's recursion limit set from `MAX_STEPS`. A test uses an LLM that searches forever and asserts the run ends as `FAILED`.

**8. How do you verify task completion?**
`verify_result` re-reads the company record and compares `invoice_id`, `company`, `amount`, `due_date` with the values parsed from the invoice document. All four must match. Only then does `finalize` set `COMPLETED`.

**9. Why is verification separate from execution?**
Because a tool can report success and still be wrong (simulated by `corrupt_record`: the write succeeds but stores a wrong amount), and because an LLM can claim it is done when it is not. Verification reads the real system state independently and is triggered by the platform, not the model.

**10. How does human-in-the-loop work?**
`check_policy` decides (in code) that approval is needed: amount above threshold, a different existing record (duplicate), or a tie for "latest". `request_approval` writes an `approvals` row, sets `AWAITING_APPROVAL` and ends the run. Approve/reject endpoints save the decision and `run_task` resumes at `resume_after_approval`. The LLM can also call `request_human_approval` itself.

**11. How would you add browser automation?**
As new tools behind the same registry (e.g. `open_page`, `click`, `read_page_text`) with Pydantic argument models, domain allow-lists and screenshots saved as evidence. The graph, retries, approval and verification stay the same; verification would re-read the page or the target system.

**12. How would you add persistent memory?**
Add a `memory` table (per company: vendor aliases, approval rules, past decisions/outcomes) and retrieve relevant rows into the prompt at `understand_task`/`create_plan`. Today only per-task state persists.

**13. How would you support multiple companies (tenants)?**
Add a `tenant_id` to every table and to `ToolContext`, per-tenant settings (threshold, allowed tools), and scope every query by it. Credentials per tenant in a secrets manager.

**14. How would you secure the system?**
Already: fixed tool set, no shell/filesystem access, regex-validated ids and names, parameterised SQL, path-traversal protection, `extra="forbid"` on tool args, key redaction in logs, secrets only via env. Add: authentication, per-role permissions, signed approvals, network egress allow-lists, least-privilege service accounts.

**15. How would you deploy this?**
Containerise (Dockerfile and compose are included), run API + UI behind a reverse proxy with TLS, Postgres instead of SQLite, secrets from a vault, workers for graph runs, CI that runs the test suite and a scenario evaluation.

**16. How would you evaluate agent reliability?**
Build a scenario suite (like `scripts/run_scenarios.py` and the failure tests) with many invoices and injected faults, run each against each model/prompt version, and track completion rate, false-success rate (must be 0), retries, escalations, unsafe-action attempts, latency and cost.

**17. What are the current limitations?**
Simulated company system; template-based invoice parsing; the live LLM path was not run against a real model in my environment (tested with a fake client); offline mode is a rule-based stand-in; escalations are terminal; single process, no auth, no queue.

**18. How would you make this production-ready?**
Real connectors, Postgres, queue + workers, auth and audit log, OCR/vision extraction with confidence, observability (traces/metrics), evaluation in CI, rate limits and timeouts on approvals, resumable escalations.

**19. How would you handle hallucinations?**
Don't trust the model for facts: values come from `read_invoice`. The policy gate rejects `update_company_system` if company/amount/due date differ from what was read (`ARGUMENT_MISMATCH`) or if the invoice was never read (`PRECONDITION_FAILED`). Verification compares against the document, not the model's claim. Tests use an LLM that invents amounts.

**20. How would you handle incorrect tool arguments?**
Pydantic validates every call (`extra="forbid"`, ranges, date format, id regex). Failures become `INVALID_ARGUMENTS` results returned to the LLM as the tool output so it can fix them, up to 3 corrections. Nothing invalid reaches the database.

**21. How would you scale the architecture?**
Move runs to a queue with workers, store state in Postgres, make the API stateless, add idempotency keys on writes (identical record already returns `unchanged`), and shard by tenant.

**22. How would you handle concurrent tasks?**
Each task has its own state row and `task_id`, so graph runs are independent. For shared resources I'd use DB transactions and a unique constraint (already `UNIQUE(invoice_id)`), plus per-invoice locks so two tasks do not write the same record at once.

**23. How would you implement task queues?**
`POST /tasks` would insert the task and enqueue its id (Redis/RQ, Celery or Temporal). Workers load state, run the graph, and save state. Today FastAPI `BackgroundTasks` does this in-process.

**24. How would you implement audit logs?**
Today: structured events (timestamp, node, action, tool, input/output summary, status, error) are stored in the task state and logged. For audit I'd write them to an append-only table or log store with actor identity, and never mutate them.

**25. How would you handle permissions?**
A permission check in the registry before each tool: (user/role, tool, resource) -> allow/deny/needs approval, configured per tenant. Today the "permission" is the fixed tool list plus the policy gate for writes.

**26. How would you prevent an agent from performing unauthorized actions?**
It can only call registered tools; each call is validated; writes pass the policy gate; risky writes require human approval for that exact action (fingerprint); overwrite of a differing record is only permitted for an approved duplicate; step and retry limits bound what it can do.

**27. What happens if verification fails?**
The mismatch (per-field expected vs actual) goes back to the LLM so it can correct the record, up to 3 times. If it is still wrong the task ends as `VERIFICATION_FAILED`. It is never reported as completed, and the summary says what mismatched.

**28. How would you add browser/computer use?**
Same pattern: expose perception/action tools (screenshot, click, type) through the registry with allow-listed apps/sites, a policy gate for sensitive actions, and verification that re-observes the final state (page content or the target system) rather than trusting the click succeeded.

**29. What would you build next?**
A resumable escalation flow, OCR/vision-LLM invoice extraction with confidence, a scenario-based evaluation harness across models, company memory, and one real connector (e.g. email inbox + a real accounting API sandbox).

**30. Explain the complete execution flow.**
`POST /tasks` creates a task row and runs the graph. `understand_task` -> structured goal. `create_plan` -> advisory plan. `decide_next_action`: the LLM picks a tool call. `check_policy`: preconditions and approval rules. `execute_tool`: registry validates and runs it. `observe_result`: success updates `discovered_information` and the plan; failure -> `handle_failure`. For approvals the run pauses and resumes after a human decision. When the LLM stops calling tools, `verify_result` re-reads the company record and compares fields. `finalize` sets the status (`COMPLETED` only if verified) and builds summary, evidence and history.

**31. Where does the "latest invoice" decision happen, and what if two invoices share the date?**
The search tool returns invoices newest first; the LLM picks one. The policy gate recomputes the max date from the search results and, if several invoices tie, raises `ambiguous_invoice` and asks a human (TechNova INV-3002 / INV-3003 in the seed data).

**32. What is the ScriptedLLM and is it cheating?**
It implements the same `LLMClient` interface with hand-written rules and is labelled everywhere (`is_real_llm: false`, UI notice, README). It exists so tests and demos run offline and deterministically. It is not presented as an LLM. The graph, tools, policy, retries and verification are identical with a real model; I tested the real-model client against a fake SDK client, and I would run the scenarios with a real key before claiming model-specific behaviour.

**33. How do you protect against prompt injection?**
Task text and tool outputs are treated as data in the prompts. `read_invoice` returns only parsed fields, never raw document text. More importantly, even a manipulated model cannot do much: it can only call the five tools, arguments must match the invoice, and risky writes need human approval.

**34. How do you test the failure paths without mocking everything?**
Faults are armed per task in the real database (`faults` table) and fire *inside* the real tools (e.g. `update_company_system` returns `SYSTEM_UNAVAILABLE`, or stores a wrong amount). Misbehaving fake LLMs (invented amounts, bad args, endless loop, false "done", provider down) drive the real graph. Tests check database contents, not only return values.
