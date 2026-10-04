# Technical Walkthrough

## Why LangGraph?
The task is a loop with branches: decide -> check -> act -> observe -> (retry | adapt | ask a human | verify). LangGraph lets me write that as nodes plus conditional edges, so the control flow is visible and testable (`app/agent/graph.py`). A plain "agent executor" hides the loop; I need to put hard rules (policy, retry budget, verification) *between* the LLM and the world, and to stop and resume for approvals.

## Why SQLite?
Zero setup, real SQL, a single inspectable file. It holds the simulated company system (`company_records`), the invoice registry, approvals, serialised task state and armed faults. For production I would use Postgres; the code only touches the DB through `Database.connect()` and `repository.py`, so the swap is contained.

## Why tool calling?
The LLM must produce structured, validated actions rather than prose. Each tool has a Pydantic argument model; the JSON schema of that model is sent to the LLM as the function definition, and the same model validates the arguments that come back (`extra="forbid"` rejects invented parameters). The LLM decides *which* tool; the registry decides whether the call is *valid*.

## How planning works
1. `understand_task`: the LLM returns an `InterpretedGoal` (company, invoice id, latest/specific, needs_clarification). Bad or non-invoice tasks stop with `NEEDS_CLARIFICATION`.
2. `create_plan`: the LLM returns ordered steps with a tool hint. `parse_plan` validates it (a final verification step is part of the prompt contract).
3. The plan is **advisory**. It is shown in the UI and given to the LLM as context, and `update_plan` ticks steps when the matching tool succeeds/fails. The actual next action is chosen every turn by `decide_next_action`, so the agent can deviate (e.g. read a different invoice, ask a human).

## How state is maintained
`AgentState` (`app/agent/state.py`) is a typed, JSON-serialisable dict: `original_task`, `interpreted_goal`, `plan`, `current_step`, `messages` (LLM conversation), `discovered_information` (search results, the invoice read, the record written), `tool_results`, `execution_history`, `errors`, `retry_count`, `adaptation_count`, `step_count`, `approval_*`, `verification_result`, `final_status`, `evidence`. Each node returns a partial update. `TaskService.run_task` streams the graph and **saves the full state to SQLite after every node**, so the UI can poll progress, the process can restart, and an approval can resume later.

## How failures are detected
Every tool returns `ToolResult(success, data, message, error_code, retryable)`. The registry (`app/tools/registry.py`) turns validation errors, `sqlite3.Error`, and unexpected exceptions into failed `ToolResult`s (it never lets one escape). `observe_result` sees `success=False` and routes to `handle_failure`. Separate detection exists for: policy violations (`check_policy` -> `PRECONDITION_FAILED`/`ARGUMENT_MISMATCH`), LLM errors (`LLMError` -> `FAILED`), loops (`MAX_STEPS`), and wrong data (verification).

## How retries work
`handle_failure` classifies by `error_code`/`retryable`:
- **Retryable** (system unavailable, DB error): sleep `backoff * 2^n`, increment `retry_count`, re-run the *same* call. After `MAX_RETRIES` (3) -> `ESCALATED_TO_HUMAN`. `retry_count` resets after a success.
- **Fixable by the model** (invalid args, argument mismatch, verification mismatch): the error is appended to the conversation as the tool result (with a hint) and the LLM decides again. Bounded by `adaptation_count` <= `MAX_RETRIES`.
- **Needs a human**: not found/ambiguous company -> `NEEDS_CLARIFICATION`; duplicate record -> approval; unreadable document -> escalate.
- Global backstops: `MAX_STEPS` decisions and a LangGraph recursion limit.

## How verification works
When the LLM stops calling tools, the graph routes to `verify_result` instead of finishing. It calls `verify_record(db, expected)` which reads the stored record and compares `invoice_id`, `company`, `amount` (to the cent), `due_date` against the values extracted from the invoice document (the source of truth, held in `discovered_information`). A pass -> `finalize` -> `COMPLETED`. A mismatch -> `handle_failure` (the model gets the mismatch and may correct it; still wrong -> `VERIFICATION_FAILED`). `finalize` independently refuses `COMPLETED` without a passing verification. The per-field expected/actual table is stored as evidence.

## How human approval works
`check_policy` computes reasons (`high_amount`, `duplicate_record`, `ambiguous_invoice`) from code and DB state, not from the LLM. If any apply and this exact action (fingerprint of invoice id + amount + due date) has not been approved, `request_approval` calls the `request_human_approval` tool (row in `approvals`), sets `AWAITING_APPROVAL`, and the graph run ends. `POST /approve|/reject` marks the decision and `run_task` re-enters the graph at `resume_after_approval` (conditional entry from `START`): approve -> record the fingerprint and execute the paused tool call (the tool permits overwriting a differing record only for an approved duplicate); reject -> `REJECTED_BY_HUMAN`. The LLM can also request approval itself; after approval it continues deciding.

## How the system could scale
- Run graphs in workers (Celery/RQ/Temporal) fed by a queue; the API only enqueues and reads state.
- Postgres instead of SQLite; row-level locking per task; idempotency keys on writes (already idempotent: identical record -> `unchanged`).
- Stateless API/workers behind a load balancer; a LangGraph checkpointer for mid-run recovery.
- Tool/connector interface stays the same, so new systems (ERP, email, browser) are new tools plus argument models.
- Per-tenant config (thresholds, policies), rate limiting and model routing.

## What would change for production
- Real connectors, secrets manager, least-privilege service accounts per tool.
- AuthN/AuthZ for users and approvers; approvals tied to identity, with audit trail (immutable event log, not just JSON in a task row).
- OCR/vision-LLM extraction with confidence → approval policy.
- Observability: OpenTelemetry traces, metrics for success/retry/escalation rates, cost tracking.
- Evaluation: scenario suite run against each model/prompt version in CI; red-team prompt-injection cases in invoice text.
- Resumable escalations, timeouts on approvals, and notifications (Slack/email) for approvers.
