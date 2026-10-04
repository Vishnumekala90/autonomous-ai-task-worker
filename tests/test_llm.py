"""LLM abstraction: provider selection, OpenAI-compatible message translation, secret hygiene.

The OpenAI-compatible client is exercised with a stub transport object (no network, no key),
so this verifies our translation/parsing logic, not a live model.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from app.logging_config import RedactingFilter, redact
from app.services.llm import (
    GEMINI_OPENAI_BASE_URL,
    LLMError,
    OpenAICompatLLM,
    ScriptedLLM,
    get_llm,
)
from tests.helpers import make_settings


class FakeClient:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def reply(content=None, tool_calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))])


def tool_call(call_id, name, arguments):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


TOOLS = [{"type": "function", "function": {"name": "read_invoice", "description": "", "parameters": {}}}]
MESSAGES = [{"role": "user", "content": "Task: ..."}]


def test_tool_call_is_translated_to_agent_turn():
    client = FakeClient(reply(tool_calls=[tool_call("c1", "read_invoice", '{"invoice_id": "INV-1003"}')]))
    llm = OpenAICompatLLM("sk-test", "gpt-test", client=client)
    turn = llm.next_turn(MESSAGES, TOOLS)
    assert turn.tool_call.name == "read_invoice" and turn.tool_call.arguments == {"invoice_id": "INV-1003"}
    sent = client.calls[0]
    assert sent["messages"][0]["role"] == "system" and sent["tools"] == TOOLS
    assert sent["parallel_tool_calls"] is False and sent["model"] == "gpt-test" and sent["temperature"] == 0


def test_plain_text_reply_means_finish():
    turn = OpenAICompatLLM("sk-test", "m", client=FakeClient(reply("All done."))).next_turn(MESSAGES, TOOLS)
    assert turn.tool_call is None and turn.content == "All done."


def test_unparseable_tool_arguments_do_not_crash():
    client = FakeClient(reply(tool_calls=[tool_call("c1", "read_invoice", "{not json")]))
    turn = OpenAICompatLLM("sk-test", "m", client=client).next_turn(MESSAGES, TOOLS)
    assert turn.tool_call.name == "read_invoice" and "_unparseable_arguments" in turn.tool_call.arguments


def test_only_the_first_tool_call_is_used():
    client = FakeClient(reply(tool_calls=[tool_call("a", "read_invoice", "{}"), tool_call("b", "search_invoices", "{}")]))
    assert OpenAICompatLLM("k", "m", client=client).next_turn(MESSAGES, TOOLS).tool_call.id == "a"


def test_gemini_provider_does_not_send_openai_only_parameters():
    client = FakeClient(reply("ok"))
    OpenAICompatLLM("k", "gemini-x", provider="gemini", client=client).next_turn(MESSAGES, TOOLS)
    assert "parallel_tool_calls" not in client.calls[0]


def test_provider_failure_becomes_llm_error_without_leaking_key():
    client = FakeClient(RuntimeError("Incorrect API key provided: sk-abcdefghijklmnop1234"))
    with pytest.raises(LLMError) as exc:
        OpenAICompatLLM("sk-abcdefghijklmnop1234", "m", client=client).next_turn(MESSAGES, TOOLS)
    assert "sk-abcdefghijklmnop1234" not in str(exc.value)


def test_interpret_task_parses_model_json():
    json_reply = '{"intent": "record_invoice", "company_name": "Acme Corp", "selection": "latest", "summary": "x"}'
    goal = OpenAICompatLLM("k", "m", client=FakeClient(reply(json_reply))).interpret_task("Record Acme's latest invoice")
    assert goal.company_name == "Acme Corp" and not goal.needs_clarification


@pytest.mark.parametrize("bad", ["not json at all", "[1, 2]", '{"selection": "newest-ish"}'])
def test_invalid_model_output_raises_llm_error(bad):
    with pytest.raises(LLMError):
        OpenAICompatLLM("k", "m", client=FakeClient(reply(bad))).interpret_task("task text here")


def test_create_plan_parses_steps_and_drops_unknown_tool_hints():
    plan_json = '{"steps": [{"description": "Find it", "tool": "search_invoices"}, {"description": "Do magic", "tool": "teleport"}]}'
    from app.agent.state import InterpretedGoal

    steps = OpenAICompatLLM("k", "m", client=FakeClient(reply(plan_json))).create_plan(InterpretedGoal(company_name="Acme"))
    assert [s.tool for s in steps] == ["search_invoices", None]


# ---- provider selection ----

def test_auto_without_keys_falls_back_to_scripted(tmp_path):
    llm = get_llm(make_settings(tmp_path, llm_provider="auto"))
    assert isinstance(llm, ScriptedLLM) and llm.info()["is_real_llm"] is False


def test_auto_prefers_openai_then_gemini(tmp_path):
    openai_llm = get_llm(make_settings(tmp_path, llm_provider="auto", openai_api_key="sk-test-123456789", openai_model="gpt-x"))
    assert (openai_llm.provider, openai_llm.model, openai_llm.is_real_llm) == ("openai", "gpt-x", True)
    gemini_llm = get_llm(make_settings(tmp_path, llm_provider="auto", gemini_api_key="AIza-test-key", gemini_model="gem-y"))
    assert (gemini_llm.provider, gemini_llm.model) == ("gemini", "gem-y")
    assert str(gemini_llm.client.base_url).startswith(GEMINI_OPENAI_BASE_URL.rstrip("/"))


def test_explicit_provider_without_key_is_an_error(tmp_path):
    with pytest.raises(LLMError):
        get_llm(make_settings(tmp_path, llm_provider="openai"))
    with pytest.raises(LLMError):
        get_llm(make_settings(tmp_path, llm_provider="nonsense"))


# ---- secret hygiene ----

def test_settings_repr_does_not_expose_keys(tmp_path):
    settings = make_settings(tmp_path, openai_api_key="sk-super-secret-value-123")
    assert "super-secret" not in repr(settings) and "super-secret" not in str(settings.model_dump())


def test_redact_masks_key_shapes():
    assert "sk-abcdefghijkl" not in redact("using key sk-abcdefghijklmnop now")
    assert "AIzaSyA" not in redact("key=AIzaSyA1234567890123456789012")
    assert "hunter2" not in redact('{"api_key": "hunter2"}')


def test_logging_filter_redacts_formatted_messages():
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "token %s", ("sk-abcdefghijklmnop",), None)
    RedactingFilter().filter(record)
    assert "sk-abcdefghijklmnop" not in record.getMessage()


def test_event_summaries_redact_secret_keys():
    from app.agent.state import summarize

    assert "abc123" not in summarize({"api_key": "abc123", "amount": 5})
