"""LangGraph wiring: nodes + conditional edges that react to tool results.

                       +------------------- adapt ------------------+
                       v                                            |
START -> understand -> plan -> decide -> policy -> execute -> observe -+-> decide ...
            |                    |         |                    |
            v                    |         v                    v
        finalize            (finish) -> verify           handle_failure -> retry | approval | finalize
                                         |   |
                                  finalize   handle_failure

request_approval ends the run (state is persisted); resume_after_approval re-enters the graph.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agent.nodes import AgentNodes
from app.agent.state import AgentState
from app.config import Settings
from app.database.database import Database
from app.services.llm import LLMClient


def _entry(state: AgentState) -> str:
    return "resume_after_approval" if state.get("resume_pending") else "understand_task"


def _after_understand(state: AgentState) -> str:
    return "finalize" if state.get("final_status") else "create_plan"


def _after_plan(state: AgentState) -> str:
    return "finalize" if state.get("final_status") else "decide_next_action"


def _after_decide(state: AgentState) -> str:
    if state.get("final_status"):
        return "finalize"
    return "check_policy" if state.get("next_route") == "policy" else "verify_result"


def _after_policy(state: AgentState) -> str:
    return {"approval": "request_approval", "failure": "handle_failure"}.get(state["next_route"], "execute_tool")


def _after_observe(state: AgentState) -> str:
    return "handle_failure" if state.get("next_route") == "failure" else "decide_next_action"


def _after_failure(state: AgentState) -> str:
    if state.get("final_status"):
        return "finalize"
    return {"retry": "execute_tool", "approval": "request_approval"}.get(state["next_route"], "decide_next_action")


def _after_request_approval(state: AgentState) -> str:
    if state.get("final_status"):
        return "finalize"
    return "handle_failure" if state.get("next_route") == "failure" else "wait"


def _after_resume(state: AgentState) -> str:
    if state.get("final_status"):
        return "finalize"
    return "execute_tool" if state.get("next_route") == "execute" else "decide_next_action"


def _after_verify(state: AgentState) -> str:
    return "finalize" if state.get("next_route") == "finalize" else "handle_failure"


def build_graph(db: Database, llm: LLMClient, settings: Settings) -> Any:
    nodes = AgentNodes(db, llm, settings)
    graph = StateGraph(AgentState)
    for name in (
        "understand_task", "create_plan", "decide_next_action", "check_policy", "execute_tool",
        "observe_result", "handle_failure", "request_approval", "resume_after_approval",
        "verify_result", "finalize",
    ):
        graph.add_node(name, getattr(nodes, name))

    graph.add_conditional_edges(START, _entry, ["understand_task", "resume_after_approval"])
    graph.add_conditional_edges("understand_task", _after_understand, ["create_plan", "finalize"])
    graph.add_conditional_edges("create_plan", _after_plan, ["decide_next_action", "finalize"])
    graph.add_conditional_edges(
        "decide_next_action", _after_decide, ["check_policy", "verify_result", "finalize"]
    )
    graph.add_conditional_edges(
        "check_policy", _after_policy, ["request_approval", "handle_failure", "execute_tool"]
    )
    graph.add_edge("execute_tool", "observe_result")
    graph.add_conditional_edges("observe_result", _after_observe, ["handle_failure", "decide_next_action"])
    graph.add_conditional_edges(
        "handle_failure", _after_failure, ["execute_tool", "request_approval", "decide_next_action", "finalize"]
    )
    graph.add_conditional_edges(
        "request_approval", _after_request_approval, {"wait": END, "handle_failure": "handle_failure", "finalize": "finalize"}
    )
    graph.add_conditional_edges(
        "resume_after_approval", _after_resume, ["finalize", "execute_tool", "decide_next_action"]
    )
    graph.add_conditional_edges("verify_result", _after_verify, ["finalize", "handle_failure"])
    graph.add_edge("finalize", END)
    return graph.compile()
