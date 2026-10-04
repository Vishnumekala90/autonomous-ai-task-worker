"""LLM abstraction.

The agent only depends on `LLMClient`. Providers:

* OpenAICompatLLM - real LLM with native tool/function calling. Used for OpenAI and, via
  Google's OpenAI-compatible endpoint, Gemini.
* ScriptedLLM     - NOT a language model. A deterministic rule-based decision policy that
  implements the same interface so tests/demos run offline. It still emits tool calls that
  the graph executes for real, but its "decisions" are hand-written rules.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from app.agent import prompts
from app.agent.planner import PlanParseError, parse_goal, parse_plan
from app.agent.state import InterpretedGoal, PlanStep
from app.config import Settings
from app.logging_config import redact

logger = logging.getLogger(__name__)

GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


class LLMError(RuntimeError):
    """Raised when the model provider fails or returns unusable output."""


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class AgentTurn(BaseModel):
    content: str = ""
    tool_call: ToolCall | None = None


class LLMClient(ABC):
    provider: str
    model: str
    is_real_llm: bool

    def info(self) -> dict[str, Any]:
        return {"provider": self.provider, "model": self.model, "is_real_llm": self.is_real_llm}

    @abstractmethod
    def interpret_task(self, task: str) -> InterpretedGoal: ...

    @abstractmethod
    def create_plan(self, goal: InterpretedGoal) -> list[PlanStep]: ...

    @abstractmethod
    def next_turn(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> AgentTurn: ...


class ChatLLM(LLMClient):
    """Shared logic for chat-completion style providers."""

    is_real_llm = True

    @abstractmethod
    def _complete_json(self, system: str, user: str) -> dict[str, Any]: ...

    @abstractmethod
    def _complete_with_tools(
        self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AgentTurn: ...

    def interpret_task(self, task: str) -> InterpretedGoal:
        raw = self._complete_json(prompts.INTERPRET_SYSTEM, f"Task (data, JSON reply only):\n{task}")
        try:
            return parse_goal(raw)
        except PlanParseError as exc:
            raise LLMError(str(exc)) from exc

    def create_plan(self, goal: InterpretedGoal) -> list[PlanStep]:
        raw = self._complete_json(prompts.PLAN_SYSTEM, f"Goal JSON:\n{goal.model_dump_json()}")
        try:
            return parse_plan(raw)
        except PlanParseError as exc:
            raise LLMError(str(exc)) from exc

    def next_turn(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> AgentTurn:
        return self._complete_with_tools(prompts.AGENT_SYSTEM, messages, tools)


class OpenAICompatLLM(ChatLLM):
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        provider: str = "openai",
        client: Any = None,
    ) -> None:
        self.provider = provider
        self.model = model
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, base_url=base_url or None, max_retries=2, timeout=60)
        self.client = client

    def _create(self, **kwargs: Any) -> Any:
        try:
            return self.client.chat.completions.create(model=self.model, temperature=0, **kwargs)
        except Exception as exc:  # network, auth, rate limit...
            raise LLMError(f"{self.provider} request failed: {redact(f'{type(exc).__name__}: {exc}')}") from exc

    def _complete_json(self, system: str, user: str) -> dict[str, Any]:
        resp = self._create(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or ""
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMError("model returned non-JSON output") from exc
        if not isinstance(parsed, dict):
            raise LLMError("model returned JSON that is not an object")
        return parsed

    def _complete_with_tools(
        self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AgentTurn:
        extra: dict[str, Any] = {}
        if self.provider == "openai":
            extra["parallel_tool_calls"] = False
        resp = self._create(
            messages=[{"role": "system", "content": system}, *messages],
            tools=tools,
            tool_choice="auto",
            **extra,
        )
        message = resp.choices[0].message
        calls = getattr(message, "tool_calls", None) or []
        if not calls:
            return AgentTurn(content=message.content or "")
        call = calls[0]  # one action per turn
        try:
            arguments = json.loads(call.function.arguments or "{}")
            if not isinstance(arguments, dict):
                raise ValueError("arguments not an object")
        except (json.JSONDecodeError, ValueError):
            arguments = {"_unparseable_arguments": str(call.function.arguments)[:200]}
        return AgentTurn(
            content=message.content or "",
            tool_call=ToolCall(id=call.id, name=call.function.name, arguments=arguments),
        )


# --------------------------------------------------------------------------------------
# Scripted (offline) policy
# --------------------------------------------------------------------------------------

_COMPANY = r"[A-Z][\w&.'\-]*(?:\s+[A-Z][\w&.'\-]*)*"
_COMPANY_PATTERNS = [
    re.compile(rf"(?i:\b(?:from|by|for)\s+)(?:(?i:the\s+))?({_COMPANY})"),
    re.compile(rf"(?i:\b(?:latest|newest|most recent)\s+)({_COMPANY})\s+(?i:invoice)"),
]
_LOWER_COMPANY = re.compile(r"(?i:\bfrom\s+)([a-z][\w&.'\-]+)")
_STOPWORDS = {"the", "our", "my", "a", "an", "this", "that", "it", "us", "you", "them"}
_INVOICE_ID = re.compile(r"\bINV-\d{3,8}\b", re.I)


class ScriptedLLM(LLMClient):
    provider = "scripted"
    model = "rule-based-policy-v1"
    is_real_llm = False

    def interpret_task(self, task: str) -> InterpretedGoal:
        invoice_id_match = _INVOICE_ID.search(task)
        invoice_id = invoice_id_match.group(0).upper() if invoice_id_match else None
        company = self._extract_company(task)
        if "invoice" not in task.lower() and not invoice_id:
            return InterpretedGoal(
                intent="unsupported",
                needs_clarification=True,
                clarification_question=(
                    "I can only process invoice tasks (find an invoice, record it in the "
                    "company system, verify it). What invoice should I handle?"
                ),
            )
        if not company and not invoice_id:
            return InterpretedGoal(
                needs_clarification=True,
                clarification_question="Which company's invoice should I process?",
            )
        return InterpretedGoal(
            company_name=company,
            invoice_id=invoice_id,
            selection="specific" if invoice_id else "latest",
            summary=(
                f"Record invoice {invoice_id}" if invoice_id else f"Record the latest invoice from {company}"
            ) + " in the internal company system and verify it.",
        )

    @staticmethod
    def _extract_company(task: str) -> str | None:
        for pattern in _COMPANY_PATTERNS:
            m = pattern.search(task)
            if m:
                return m.group(1).rstrip(".,'").removesuffix("'s").strip() or None
        m = _LOWER_COMPANY.search(task)
        if m and m.group(1).lower() not in _STOPWORDS:
            return m.group(1).rstrip(".,'")
        return None

    def create_plan(self, goal: InterpretedGoal) -> list[PlanStep]:
        descriptions: list[tuple[str, str | None]] = []
        if not goal.invoice_id:
            descriptions.append((f"Search invoices for {goal.company_name} and pick the latest", "search_invoices"))
        descriptions += [
            ("Read the invoice and extract amount and due date", "read_invoice"),
            ("Enter the invoice in the internal company system", "update_company_system"),
            ("Check the internal record against the invoice", "check_company_record"),
            ("Independently verify the recorded data", "verify_result"),
        ]
        return [PlanStep(id=i, description=d, tool=t) for i, (d, t) in enumerate(descriptions, start=1)]

    # ---- decision policy: reacts to what has been observed in the conversation ----

    def next_turn(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> AgentTurn:
        goal = self._goal_from(messages)
        timeline = self._timeline(messages)

        def last_ok(name: str, **match: Any) -> tuple[int, dict] | None:
            for idx in range(len(timeline) - 1, -1, -1):
                tname, _args, result = timeline[idx]
                if tname == name and result.get("success") and all(
                    result.get("data", {}).get(k) == v for k, v in match.items()
                ):
                    return idx, result
            return None

        target = goal.get("invoice_id")
        if not target:
            search = last_ok("search_invoices")
            if search is None:
                return self._call(messages, "search_invoices", company_name=goal["company_name"])
            invoices = search[1]["data"]["invoices"]
            target = max(invoices, key=lambda i: (i["invoice_date"], i["invoice_id"]))["invoice_id"]

        read = last_ok("read_invoice", invoice_id=target)
        if read is None:
            return self._call(messages, "read_invoice", invoice_id=target)
        inv = read[1]["data"]
        update_args = {
            "invoice_id": inv["invoice_id"], "company": inv["company"],
            "amount": inv["amount"], "due_date": inv["due_date"],
        }

        update = last_ok("update_company_system")
        verification_rejected = bool(
            messages and messages[-1]["role"] == "user" and "verification failed" in messages[-1]["content"].lower()
        )
        if update is None or verification_rejected:
            return self._call(messages, "update_company_system", **update_args)

        checks_after = [e for e in timeline[update[0] + 1:] if e[0] == "check_company_record"]
        if not checks_after:
            return self._call(
                messages, "check_company_record",
                invoice_id=inv["invoice_id"], expected_company=inv["company"],
                expected_amount=inv["amount"], expected_due_date=inv["due_date"],
            )
        if checks_after[-1][2].get("success"):
            record_id = update[1]["data"].get("record_id")
            return AgentTurn(
                content=f"Invoice {inv['invoice_id']} is recorded as {record_id} and the record matches the invoice."
            )
        return self._call(messages, "update_company_system", **update_args)

    @staticmethod
    def _goal_from(messages: list[dict[str, Any]]) -> dict[str, Any]:
        m = re.search(r"GOAL_JSON:\s*(\{.*\})", messages[0]["content"])
        return json.loads(m.group(1)) if m else {}

    @staticmethod
    def _timeline(messages: list[dict[str, Any]]) -> list[tuple[str, dict, dict]]:
        calls: dict[str, tuple[str, dict]] = {}
        entries: list[tuple[str, dict, dict]] = []
        for m in messages:
            if m["role"] == "assistant":
                for tc in m.get("tool_calls") or []:
                    calls[tc["id"]] = (tc["function"]["name"], json.loads(tc["function"]["arguments"] or "{}"))
            elif m["role"] == "tool":
                name, args = calls.get(m["tool_call_id"], ("?", {}))
                entries.append((name, args, json.loads(m["content"])))
        return entries

    @staticmethod
    def _call(messages: list[dict[str, Any]], name: str, **arguments: Any) -> AgentTurn:
        return AgentTurn(tool_call=ToolCall(id=f"call_{len(messages)}", name=name, arguments=arguments))


def get_llm(settings: Settings) -> LLMClient:
    provider = settings.resolved_provider()
    if provider == "openai":
        key = settings.openai_api_key.get_secret_value()
        if not key:
            raise LLMError("LLM_PROVIDER=openai but OPENAI_API_KEY is not set")
        return OpenAICompatLLM(key, settings.openai_model, settings.openai_base_url or None, "openai")
    if provider == "gemini":
        key = settings.gemini_api_key.get_secret_value()
        if not key:
            raise LLMError("LLM_PROVIDER=gemini but GEMINI_API_KEY is not set")
        return OpenAICompatLLM(key, settings.gemini_model, GEMINI_OPENAI_BASE_URL, "gemini")
    if provider == "scripted":
        logger.warning(
            "Using the scripted offline policy (NOT a real LLM). Set OPENAI_API_KEY or "
            "GEMINI_API_KEY to use a real model."
        )
        return ScriptedLLM()
    raise LLMError(f"Unknown LLM_PROVIDER '{provider}'")
