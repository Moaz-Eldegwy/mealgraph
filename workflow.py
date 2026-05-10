"""LangGraph wiring for the Coach <-> action loop.

Phase 1 keeps the same two-node graph (``coach`` -> ``execute_action`` -> loop)
so the public contract is unchanged. Phase 2 will explode this into subgraphs
with parallel branches and a Validator critic loop.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from config import get_settings
from logging_setup import get_logger
from state import NutritionState
from utils import FileCheckpointSaver, set_nested

_logger = get_logger("workflow")


def should_continue(state: NutritionState) -> str:
    """Edge predicate: stop on terminal action or when we hit max_turns."""
    current = state.get("current_action") or {}
    if current.get("action") in {"compose_response", "ask_user"}:
        return "end"
    if state["num_turns"] >= state["max_turns"]:
        return "end"
    return "execute_action"


def coach_node(state: NutritionState, coach_agent) -> NutritionState:
    return coach_agent.handle_task(state)


def execute_action_node(state: NutritionState, agents: Dict[str, Any], tools: Dict[str, Any]) -> NutritionState:
    action = state.get("current_action") or {}
    action_name = action.get("action")
    params = action.get("params", {}) or {}

    if not action_name:
        return state

    settings = get_settings()
    if settings.debug_mode:
        _logger.debug("Executing Action: %s", action_name)
    else:
        if action_name == "ask_user":
            _logger.info("❓ Asking user: %s", params.get("prompt"))
        elif action_name == "write_memory":
            _logger.info("Writing to memory partition: %s", params.get("partition"))

    if action.get("_parse_error"):
        error_message = "I encountered an error processing the request. Let me try a different approach."
        state["conversation_history"].append({"role": "assistant", "content": error_message})
        return {**state, "agent_result": error_message}

    if "previous_actions" not in state:
        state["previous_actions"] = []

    try:
        if action_name == "call_agent":
            agent_name = params["agent_name"]
            task = params["task"]
            agent_result = agents[agent_name].handle_task(task, state["memory"])
            success_message = (
                f"{agent_name} task completed and stored in the memory successfully"
                if agent_result
                else f"{agent_name} task failed"
            )
            state["previous_actions"].append(f"Called agent {agent_name} with task: {task}")
            return {**state, "agent_result": success_message}

        if action_name == "call_tool":
            tool_name = params["tool_name"]
            task = params["task"]
            tool_result = (
                tools[tool_name].handle_task(task) if tool_name in tools else f"Unknown tool: {tool_name}"
            )
            state["previous_actions"].append(f"Called tool {tool_name} with task: {task}")
            return {**state, "agent_result": tool_result}

        if action_name == "write_memory":
            partition = params["partition"]
            data = params["data"]
            updated_data = {**data, "last_updated": datetime.now().isoformat()}
            set_nested(state["memory"], partition, updated_data)
            state["previous_actions"].append(f"Wrote to memory partition: {partition}")
            return {**state, "agent_result": "Memory updated successfully"}

        if action_name == "compose_response":
            response_text = params.get("text") or params.get("response")
            if not response_text:
                raise ValueError("Missing 'text' or 'response' in params for compose_response")
            state["conversation_history"].append({"role": "assistant", "content": response_text})
            state["previous_actions"].append("Composed response to user")
            return {**state, "agent_result": response_text}

        if action_name == "ask_user":
            prompt_text = params["prompt"]
            state["conversation_history"].append({"role": "assistant", "content": prompt_text})
            state["previous_actions"].append(f"Asked user: {prompt_text}")
            return {**state, "agent_result": "User prompted for input"}

        state["previous_actions"].append(f"Executed {action_name} with params: {params}")
        return {**state, "agent_result": f"Unknown action: {action_name}"}

    except Exception as e:  # noqa: BLE001
        _logger.exception("Error executing %s", action_name)
        state["previous_actions"].append(f"Attempted {action_name} with params: {params}")
        return {**state, "agent_result": f"Error executing {action_name}: {str(e)}"}


def setup_workflow(coach_agent, agents: Dict[str, Any], tools: Dict[str, Any], persistence_dir: str | None = None):
    workflow = StateGraph(NutritionState)
    workflow.add_node("coach", lambda state: coach_node(state, coach_agent))
    workflow.add_node("execute_action", lambda state: execute_action_node(state, agents, tools))
    workflow.set_entry_point("coach")
    workflow.add_edge("coach", "execute_action")
    workflow.add_conditional_edges(
        "execute_action",
        should_continue,
        {"execute_action": "coach", "end": END},
    )

    if persistence_dir:
        checkpointer = FileCheckpointSaver(persistence_dir)
        _logger.info("MAS workflow compiled with file-based persistence at %s.", persistence_dir)
    else:
        checkpointer = MemorySaver()
        _logger.info("MAS workflow compiled with in-memory persistence.")

    return workflow.compile(checkpointer=checkpointer)
