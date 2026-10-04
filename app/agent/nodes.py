"""LangGraph nodes.

Each node reads the state and returns a partial update. Nodes never decide the tool
sequence themselves: `decide_next_action` asks the LLM, and the other nodes only enforce
safety boundaries (policy gate, retry budget, approvals, independent verification).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from pydantic import ValidationError

from app.agent import prompts
from app.agent.planner import update_plan
from app.agent.state import AgentState, InterpretedGoal, with_event
from app.config import Settings
from app.database.database import Database
from app.database.repository import get_company_record
from app.services.llm import LLMClient, LLMError
from app.services.reporting import build_evidence, build_summary, format_money
from app.services.verification import VerificationResult, compare_fields, verify_record
from app.tools.base import ToolContext
from app.tools.company_tools import UpdateCompanySystemArgs
from app.tools.registry import run_tool, tool_definitions
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

VERIFICATION_CODES = {"VERIFICATION_FAILED", "RECORD_NOT_FOUND"}


def _fingerprint(args: UpdateCompanySystemArgs) -> str:
    return f"update_company_system|{args.invoice_id}|{args.amount:.2f}|{args.due_date}"


class AgentNodes:
    def __init__(self, db: Database, llm: LLMClient, settings: Settings) -> None:
        self.db = db
        self.llm = llm
        self.settings = settings

    # ------------------------------------------------------------------ helpers

    def _ctx(self, state: AgentState, overwrite_approved: bool = False) -> ToolContext:
        return ToolContext(
            db=self.db,
            task_id=state["task_id"],
            invoices_dir=self.settings.invoices_dir,
            overwrite_approved=overwrite_approved,
        )

    def _terminate(
        self, state: AgentState, node: str, final_status: str, reason: str, error: str | None = None
    ) -> dict[str, Any]:
        return {
            "final_status": final_status,
            "final_reason": reason,
            "pending_tool_call": None,
            "next_route": "finalize",
            "execution_history": with_event(
                state, node, "terminate", "failed", output_summary=f"{final_status}: {reason}", error=error
            ),
        }

    @staticmethod
    def _tool_message(call: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
        return {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(payload, default=str)}

    # ------------------------------------------------------------------ nodes

    def understand_task(self, state: AgentState) -> dict[str, Any]:
        try:
            goal = self.llm.interpret_task(state["original_task"])
        except LLMError as exc:
            return self._terminate(
                state, "understand_task", "FAILED", f"The language model failed: {exc}", error=str(exc)
            )
        update: dict[str, Any] = {"interpreted_goal": goal.model_dump(), "current_step": "Understand task"}
        unsupported = goal.intent != "record_invoice"
        if unsupported or goal.needs_clarification or not (goal.company_name or goal.invoice_id):
            question = goal.clarification_question or (
                "I can only process invoice tasks. Which company's invoice should I handle?"
            )
            term = self._terminate(
                {**state, "execution_history": with_event(
                    state, "understand_task", "interpret", "needs_clarification", input_summary=state["original_task"],
                    output_summary=goal.model_dump(exclude_none=True))},
                "understand_task", "NEEDS_CLARIFICATION", question,
            )
            return {**update, **term}
        update["execution_history"] = with_event(
            state, "understand_task", "interpret", "ok",
            input_summary=state["original_task"], output_summary=goal.model_dump(exclude_none=True),
        )
        update["next_route"] = "plan"
        return update

    def create_plan(self, state: AgentState) -> dict[str, Any]:
        goal = InterpretedGoal(**state["interpreted_goal"])
        try:
            plan = [step.model_dump() for step in self.llm.create_plan(goal)]
        except LLMError as exc:
            return self._terminate(state, "create_plan", "FAILED", f"Planning failed: {exc}", error=str(exc))
        first = {
            "role": "user",
            "content": prompts.task_message(state["original_task"], goal.model_dump(), plan),
        }
        return {
            "plan": plan,
            "messages": [first],
            "current_step": "Create plan",
            "next_route": "decide",
            "execution_history": with_event(
                state, "create_plan", "plan", "ok", output_summary=[p["description"] for p in plan]
            ),
        }

    def decide_next_action(self, state: AgentState) -> dict[str, Any]:
        step = state.get("step_count", 0) + 1
        if step > self.settings.max_steps:
            return self._terminate(
                state, "decide_next_action", "FAILED",
                f"Step budget of {self.settings.max_steps} decisions exceeded; stopping to avoid a loop.",
            )
        try:
            turn = self.llm.next_turn(state["messages"], tool_definitions())
        except LLMError as exc:
            return self._terminate(
                state, "decide_next_action", "FAILED", f"The language model failed: {exc}", error=str(exc)
            )
        message: dict[str, Any] = {"role": "assistant", "content": turn.content or None}
        if turn.tool_call is None:
            return {
                "messages": [*state["messages"], message],
                "pending_tool_call": None,
                "step_count": step,
                "current_step": "Finish requested - verifying outcome",
                "next_route": "verify",
                "execution_history": with_event(
                    state, "decide_next_action", "finish_requested", "ok", output_summary=turn.content
                ),
            }
        call = turn.tool_call
        message["tool_calls"] = [{
            "id": call.id, "type": "function",
            "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
        }]
        return {
            "messages": [*state["messages"], message],
            "pending_tool_call": {"id": call.id, "name": call.name, "arguments": call.arguments},
            "step_count": step,
            "plan": update_plan(state["plan"], call.name, "in_progress"),
            "current_step": f"Action: {call.name}",
            "next_route": "policy",
            "execution_history": with_event(
                state, "decide_next_action", "select_tool", "ok", tool=call.name, input_summary=call.arguments
            ),
        }

    # ---- safety boundary --------------------------------------------------

    def check_policy(self, state: AgentState) -> dict[str, Any]:
        call = state["pending_tool_call"]
        assert call is not None
        if call["name"] == "request_human_approval":
            return {"policy_decision": {"gated": False}, "next_route": "approval"}
        if call["name"] != "update_company_system":
            return {"policy_decision": None, "next_route": "execute"}
        try:
            args = UpdateCompanySystemArgs.model_validate(call["arguments"])
        except ValidationError:
            return {"policy_decision": None, "next_route": "execute"}  # tool reports INVALID_ARGUMENTS

        invoice = state["discovered_information"].get("invoice")
        blocked = self._precondition_failure(args, invoice)
        if blocked:
            code, message = blocked
            result = {"success": False, "data": {}, "message": message, "error_code": code, "retryable": False}
            return {
                "last_result": result,
                "last_failure": {"tool": call["name"], "error_code": code, "message": message, "retryable": False, "data": {}},
                "errors": [*state["errors"], {"tool": call["name"], "error_code": code, "message": message, "retryable": False}],
                "tool_results": [*state["tool_results"], {
                    "tool": call["name"], "arguments": call["arguments"], "attempt": 0,
                    "result": result, "blocked_by_policy": True}],
                "plan": update_plan(state["plan"], call["name"], "failed"),
                "next_route": "failure",
                "execution_history": with_event(
                    state, "check_policy", "blocked", "failed", tool=call["name"],
                    input_summary=call["arguments"], output_summary=message, error=code),
            }

        fingerprint = _fingerprint(args)
        reasons, messages = self._approval_reasons(state, args)
        if reasons and fingerprint not in state.get("approved_fingerprints", []):
            decision = self._gate_decision(args, invoice, reasons, messages)
            return {
                "policy_decision": decision,
                "next_route": "approval",
                "execution_history": with_event(
                    state, "check_policy", "approval_required", "paused", tool=call["name"],
                    output_summary=decision["messages"]),
            }
        return {
            "policy_decision": None,
            "next_route": "execute",
            "execution_history": with_event(
                state, "check_policy", "allowed", "ok", tool=call["name"], output_summary="policy checks passed"),
        }

    @staticmethod
    def _precondition_failure(args: UpdateCompanySystemArgs, invoice: dict | None) -> tuple[str, str] | None:
        if not invoice or invoice.get("invoice_id") != args.invoice_id:
            return "PRECONDITION_FAILED", (
                f"update_company_system requires a successful read_invoice for {args.invoice_id} first."
            )
        diffs = []
        if args.company.casefold() != str(invoice["company"]).casefold():
            diffs.append(f"company: you sent '{args.company}' but the invoice says '{invoice['company']}'")
        if abs(args.amount - float(invoice["amount"])) > 0.005:
            diffs.append(f"amount: you sent {args.amount} but the invoice says {invoice['amount']}")
        if args.due_date != invoice["due_date"]:
            diffs.append(f"due_date: you sent {args.due_date} but the invoice says {invoice['due_date']}")
        if diffs:
            return "ARGUMENT_MISMATCH", (
                "Arguments do not match the invoice document (" + "; ".join(diffs)
                + "). Use exactly the values returned by read_invoice."
            )
        return None

    def _approval_reasons(self, state: AgentState, args: UpdateCompanySystemArgs) -> tuple[list[str], list[str]]:
        reasons: list[str] = []
        messages: list[str] = []
        invoice = state["discovered_information"].get("invoice") or {}
        currency = invoice.get("currency", "INR")
        threshold = self.settings.approval_threshold
        if args.amount > threshold:
            reasons.append("high_amount")
            messages.append(
                f"Amount {format_money(args.amount, currency)} exceeds the approval threshold of "
                f"{format_money(threshold, currency)}."
            )
        existing = get_company_record(self.db, args.invoice_id)
        if (
            existing
            and existing["created_by_task"] != state["task_id"]
            and not compare_fields(args.model_dump(), existing).passed
        ):
            reasons.append("duplicate_record")
            messages.append(
                f"A different record {existing['record_id']} already exists for {args.invoice_id} "
                f"({format_money(existing['amount'], currency)}, due {existing['due_date']}); updating would overwrite it."
            )
        goal = state.get("interpreted_goal", {})
        results = state["discovered_information"].get("search_results") or []
        if results and not goal.get("invoice_id"):
            top_date = max(r["invoice_date"] for r in results)
            tied = [r["invoice_id"] for r in results if r["invoice_date"] == top_date]
            if len(tied) > 1 and args.invoice_id in tied:
                reasons.append("ambiguous_invoice")
                messages.append(
                    f"{len(tied)} invoices share the latest date {top_date} ({', '.join(tied)}), "
                    f"so 'latest' is ambiguous. I picked {args.invoice_id}."
                )
        return reasons, messages

    @staticmethod
    def _gate_decision(
        args: UpdateCompanySystemArgs, invoice: dict | None, reasons: list[str], messages: list[str]
    ) -> dict[str, Any]:
        return {
            "gated": True,
            "reasons": reasons,
            "messages": messages,
            "fingerprint": _fingerprint(args),
            "proposed_action": (
                f"update_company_system(invoice_id={args.invoice_id}, company={args.company}, "
                f"amount={args.amount}, due_date={args.due_date})"
            ),
            "context": {"invoice": invoice or {}, "reasons": reasons},
        }

    # ---- approvals --------------------------------------------------------

    def request_approval(self, state: AgentState) -> dict[str, Any]:
        call = state["pending_tool_call"]
        assert call is not None
        decision = state.get("policy_decision") or {}
        gated = bool(decision.get("gated"))
        args = (
            {
                "reason": " ".join(decision["messages"]),
                "proposed_action": decision["proposed_action"],
                "relevant_context": decision["context"],
            }
            if gated else call["arguments"]
        )
        result = run_tool("request_human_approval", args, self._ctx(state))
        result_dict = result.model_dump()
        tool_results = [*state["tool_results"], {
            "tool": "request_human_approval", "arguments": args, "attempt": 1, "result": result_dict}]
        if not result.success:
            if gated:
                return {
                    **self._terminate(state, "request_approval", "ESCALATED_TO_HUMAN",
                                      f"Approval was required but could not be requested: {result.message}",
                                      error=result.error_code),
                    "tool_results": tool_results,
                }
            failure = {"tool": call["name"], "error_code": result.error_code, "message": result.message,
                       "retryable": result.retryable, "data": result.data}
            return {
                "tool_results": tool_results, "last_result": result_dict, "last_failure": failure,
                "errors": [*state["errors"], {k: failure[k] for k in ("tool", "error_code", "message", "retryable")}],
                "next_route": "failure",
                "execution_history": with_event(
                    state, "request_approval", "request_approval", "failed", tool=call["name"],
                    input_summary=args, error=result.error_code),
            }
        approval_id = result.data["approval_id"]
        request = {
            "approval_id": approval_id,
            "reason": args["reason"],
            "proposed_action": args["proposed_action"],
            "context": args.get("relevant_context", {}),
            "gated": gated,
            "reasons": decision.get("reasons", []),
            "fingerprint": decision.get("fingerprint"),
        }
        return {
            "tool_results": tool_results,
            "approval_required": True,
            "approval_status": "pending",
            "approval_request": request,
            "approvals": [*state["approvals"], {
                "approval_id": approval_id, "reason": request["reason"], "status": "pending", "note": ""}],
            "current_step": "Waiting for human approval",
            "next_route": "wait",
            "execution_history": with_event(
                state, "request_approval", "request_approval", "paused", tool="request_human_approval",
                input_summary=args, output_summary=f"approval #{approval_id} pending"),
        }

    def resume_after_approval(self, state: AgentState) -> dict[str, Any]:
        request = state["approval_request"] or {}
        status = state["approval_status"]
        note = state.get("approval_note", "")
        approvals = [dict(a) for a in state["approvals"]]
        for entry in approvals:
            if entry["approval_id"] == request.get("approval_id"):
                entry["status"], entry["note"] = status, note
        history_state: AgentState = {**state, "execution_history": with_event(
            state, "resume_after_approval", "approval_decision", status,
            output_summary=f"approval #{request.get('approval_id')} {status}" + (f": {note}" if note else ""))}
        base = {"resume_pending": False, "approval_required": False, "approvals": approvals}
        if status == "rejected":
            return {**base, **self._terminate(
                history_state, "resume_after_approval", "REJECTED_BY_HUMAN",
                f"Reason for approval request: {request.get('reason', '')}" + (f" Human note: {note}" if note else ""))}
        if request.get("gated"):
            return {
                **base,
                "approved_fingerprints": [*state["approved_fingerprints"], request["fingerprint"]],
                "current_step": "Approved - resuming action",
                "next_route": "execute",
                "execution_history": history_state["execution_history"],
            }
        call = state["pending_tool_call"]
        assert call is not None
        tool_message = self._tool_message(call, {
            "success": True, "message": "A human approved. Continue with the task.",
            "data": {"approval_id": request.get("approval_id"), "status": "approved", "note": note}})
        return {
            **base,
            "messages": [*state["messages"], tool_message],
            "pending_tool_call": None,
            "next_route": "decide",
            "execution_history": history_state["execution_history"],
        }

    # ---- execute / observe ------------------------------------------------

    def execute_tool(self, state: AgentState) -> dict[str, Any]:
        call = state["pending_tool_call"]
        assert call is not None
        overwrite = False
        if call["name"] == "update_company_system":
            try:
                fp = _fingerprint(UpdateCompanySystemArgs.model_validate(call["arguments"]))
                overwrite = (
                    fp in state.get("approved_fingerprints", [])
                    and "duplicate_record" in (state.get("approval_request") or {}).get("reasons", [])
                )
            except ValidationError:
                pass
        result = run_tool(call["name"], call["arguments"], self._ctx(state, overwrite_approved=overwrite)).model_dump()
        attempt = state.get("retry_count", 0) + 1
        return {
            "last_result": result,
            "tool_results": [*state["tool_results"], {
                "tool": call["name"], "arguments": call["arguments"], "attempt": attempt, "result": result,
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds")}],
            "current_step": f"Executed {call['name']} (attempt {attempt})",
            "execution_history": with_event(
                state, "execute_tool", "execute", "ok" if result["success"] else "failed", tool=call["name"],
                input_summary=call["arguments"], output_summary=result["message"] or result["data"],
                error=result["error_code"]),
        }

    def observe_result(self, state: AgentState) -> dict[str, Any]:
        call = state["pending_tool_call"]
        result = state["last_result"]
        assert call is not None and result is not None
        name = call["name"]
        if result["success"]:
            info = dict(state["discovered_information"])
            data = result["data"]
            if name == "search_invoices":
                info["search_results"] = data["invoices"]
            elif name == "read_invoice":
                info["invoice"] = data
            elif name == "update_company_system":
                info["record"] = {
                    "record_id": data["record_id"], "action": data["action"],
                    "invoice_id": (info.get("invoice") or {}).get("invoice_id"),
                }
            elif name == "check_company_record":
                info["last_check"] = data
            return {
                "discovered_information": info,
                "messages": [*state["messages"], self._tool_message(call, result)],
                "pending_tool_call": None,
                "retry_count": 0,
                "last_failure": None,
                "plan": update_plan(state["plan"], name, "done"),
                "next_route": "decide",
                "execution_history": with_event(
                    state, "observe_result", "observe", "ok", tool=name, output_summary="tool succeeded; state updated"),
            }
        failure = {
            "tool": name, "error_code": result["error_code"], "message": result["message"],
            "retryable": result["retryable"], "data": result["data"],
        }
        return {
            "last_failure": failure,
            "errors": [*state["errors"], {k: failure[k] for k in ("tool", "error_code", "message", "retryable")}],
            "plan": update_plan(state["plan"], name, "failed"),
            "next_route": "failure",
            "execution_history": with_event(
                state, "observe_result", "observe", "failed", tool=name,
                output_summary=failure["message"], error=failure["error_code"]),
        }

    # ---- failure handling -------------------------------------------------

    def handle_failure(self, state: AgentState) -> dict[str, Any]:
        failure = state["last_failure"]
        assert failure is not None
        call = state.get("pending_tool_call")
        code, tool, message = failure["error_code"], failure["tool"], failure["message"]
        max_retries = self.settings.max_retries

        if code == "AMBIGUOUS_COMPANY" or (code == "NOT_FOUND" and tool == "search_invoices"):
            suggestions = failure.get("data", {}).get("suggestions") or failure.get("data", {}).get("companies")
            hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
            return self._terminate(state, "handle_failure", "NEEDS_CLARIFICATION", message + hint)

        if code == "DUPLICATE_RECORD" and call and call["name"] == "update_company_system":
            args = UpdateCompanySystemArgs.model_validate(call["arguments"])
            existing = failure.get("data", {}).get("existing_record", {})
            decision = self._gate_decision(
                args, state["discovered_information"].get("invoice"), ["duplicate_record"],
                [f"A different record {existing.get('record_id')} already exists for {args.invoice_id}; updating would overwrite it."],
            )
            return {
                "policy_decision": decision, "next_route": "approval",
                "execution_history": with_event(
                    state, "handle_failure", "approval_required", "paused", tool=tool, output_summary=decision["messages"]),
            }

        if failure["retryable"] and call and call["name"] != "request_human_approval":
            retries = state.get("retry_count", 0)
            if retries < max_retries:
                if self.settings.retry_backoff_seconds > 0:
                    time.sleep(self.settings.retry_backoff_seconds * (2 ** retries))
                return {
                    "retry_count": retries + 1,
                    "next_route": "retry",
                    "current_step": f"Retrying {tool} ({retries + 1}/{max_retries})",
                    "execution_history": with_event(
                        state, "handle_failure", "retry", "retrying", tool=tool,
                        input_summary=call["arguments"],
                        output_summary=f"retry {retries + 1}/{max_retries} after {code}", error=code),
                }
            return self._terminate(
                state, "handle_failure", "ESCALATED_TO_HUMAN",
                f"{tool} still failing after {max_retries} retries ({code}: {message}). "
                "Human intervention is needed.", error=code,
            )

        if code == "EXTRACTION_FAILED":
            return self._terminate(
                state, "handle_failure", "ESCALATED_TO_HUMAN",
                f"{message} The invoice document needs a human to review it.", error=code,
            )

        if state.get("adaptation_count", 0) < max_retries:
            hint = "Fix the problem or choose a different action." if code != "VERIFICATION_FAILED" else (
                "The record in the company system does not match the invoice. Correct it.")
            if call:
                feedback = self._tool_message(call, {**(state["last_result"] or {}), "hint": hint})
            else:
                feedback = {"role": "user", "content": (
                    f"Independent verification FAILED: {message} The task is not complete. {hint}")}
            return {
                "adaptation_count": state.get("adaptation_count", 0) + 1,
                "messages": [*state["messages"], feedback],
                "pending_tool_call": None,
                "next_route": "adapt",
                "current_step": "Adapting after failure",
                "execution_history": with_event(
                    state, "handle_failure", "adapt", "adapting", tool=tool,
                    output_summary=f"returning error to the model ({state.get('adaptation_count', 0) + 1}/{max_retries})",
                    error=code),
            }
        final = "VERIFICATION_FAILED" if (code in VERIFICATION_CODES or tool == "verify_result") else "ESCALATED_TO_HUMAN"
        return self._terminate(
            state, "handle_failure", final,
            f"Could not resolve '{code}' after {max_retries} corrective attempts: {message}", error=code,
        )

    # ---- verification & completion ---------------------------------------

    def verify_result(self, state: AgentState) -> dict[str, Any]:
        info = dict(state["discovered_information"])
        invoice = info.get("invoice")
        if not invoice:
            result = VerificationResult(
                passed=False, checks=[], reason="No invoice data was read, so there is nothing to verify against."
            )
        else:
            expected = {k: invoice[k] for k in ("invoice_id", "company", "amount", "due_date")}
            result = verify_record(self.db, expected)
        update: dict[str, Any] = {
            "verification_result": result.model_dump(),
            "plan": update_plan(state["plan"], "verify_result", "done" if result.passed else "failed"),
            "current_step": "Independent verification",
            "execution_history": with_event(
                state, "verify_result", "verify", "ok" if result.passed else "failed",
                tool="check_company_record" if invoice else None,
                output_summary=result.model_dump(include={"passed", "reason", "record_id"}),
                error=None if result.passed else "VERIFICATION_FAILED"),
        }
        if result.passed:
            info["record"] = {**(info.get("record") or {}), "record_id": result.record_id, "invoice_id": invoice["invoice_id"]}
            update.update(discovered_information=info, next_route="finalize", last_failure=None)
            return update
        failure = {"tool": "verify_result", "error_code": "VERIFICATION_FAILED", "message": result.reason,
                   "retryable": False, "data": {}}
        update.update(
            last_failure=failure,
            last_result=None,
            errors=[*state["errors"], {k: failure[k] for k in ("tool", "error_code", "message", "retryable")}],
            pending_tool_call=None,
            next_route="failure",
        )
        return update

    def finalize(self, state: AgentState) -> dict[str, Any]:
        verification = state.get("verification_result")
        invoice = state["discovered_information"].get("invoice")
        if verification is None and state.get("final_status") == "VERIFICATION_FAILED" and invoice:
            # Record the actual mismatch as evidence even though the agent never asked to finish.
            expected = {k: invoice[k] for k in ("invoice_id", "company", "amount", "due_date")}
            verification = verify_record(self.db, expected).model_dump()
            state = {**state, "verification_result": verification}
        verified = bool(verification and verification["passed"])
        status = state.get("final_status") or ("COMPLETED" if verified else "FAILED")
        reason = state.get("final_reason", "")
        if status == "COMPLETED" and not verified:  # never claim success without verification
            status, reason = "VERIFICATION_FAILED", "Completion was claimed without a passing verification."
        final_state: AgentState = {**state, "final_status": status, "final_reason": reason}
        evidence = build_evidence(final_state, self.llm.info())
        return {
            "final_status": status,
            "final_reason": reason,
            "verification_result": verification,
            "evidence": evidence,
            "summary": build_summary(final_state, evidence),
            "pending_tool_call": None,
            "current_step": "Done",
            "execution_history": with_event(
                state, "finalize", "finalize", "ok" if status == "COMPLETED" else "failed",
                output_summary=f"final status {status}"),
        }
