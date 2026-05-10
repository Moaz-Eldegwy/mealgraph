from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from state import NutritionState
from utils import extract_and_parse_json, set_nested, FileCheckpointSaver
from datetime import datetime
import json
import config

def should_continue(state: NutritionState) -> str:
    if state["current_action"] and state["current_action"]["action"] in ["compose_response", "ask_user"]:
        return "end"
    if state["num_turns"] >= state["max_turns"]:
        return "end"
    return "execute_action"

def coach_node(state: NutritionState, coach_agent) -> NutritionState:
    return coach_agent.handle_task(state)

def execute_action_node(state: NutritionState, agents, tools) -> NutritionState:
    action = state["current_action"]
    if not action or not action.get("action"):
        return state
    if config.DEBUG_MODE:
        print(f"Executing Action: {action['action']}")

    # Add more specific high-level print for user mode
    if not config.DEBUG_MODE:
        if action["action"] == "call_agent":
            agent_name = action["params"]["agent_name"]
            task = action["params"]["task"]
        elif action["action"] == "call_tool":
            tool_name = action["params"]["tool_name"]
            task = action["params"]["task"]
        elif action["action"] == "ask_user":
            print(f"❓Asking user: {action['params']['prompt']}")
        elif action["action"] == "write_memory":
            print(f"Writing to memory partition: {action['params']['partition']}")

    # Handle JSON parsing errors
    if action.get("_parse_error"):
        error_message = "I encountered an error processing the request. Let me try a different approach."
        state["conversation_history"].append({"role": "assistant", "content": error_message})
        return {**state, "agent_result": error_message}

    # Initialize previous_actions if not present
    if 'previous_actions' not in state:
        state['previous_actions'] = []

    try:
        if action["action"] == "call_agent":
            agent_name = action["params"]["agent_name"]
            task = action["params"]["task"]
            agent_result = agents[agent_name].handle_task(task, state["memory"])
            # Set success message instead of full result
            success_message = f"{agent_name} task completed and stored in the memory successfully" if agent_result else f"{agent_name} task failed"
            action_description = f"Called agent {agent_name} with task: {task}"
            state['previous_actions'].append(action_description)
            return {**state, "agent_result": success_message}

        elif action["action"] == "call_tool":
            tool_name = action["params"]["tool_name"]
            task = action["params"]["task"]
            tool_result = tools[tool_name].handle_task(task) if tool_name in tools else f"Unknown tool: {tool_name}"
            action_description = f"Called tool {tool_name} with task: {task}"
            state['previous_actions'].append(action_description)
            return {**state, "agent_result": tool_result}

        elif action["action"] == "write_memory":
            partition = action["params"]["partition"]
            data = action["params"]["data"]
            updated_data = {**data, "last_updated": datetime.now().isoformat()}
            set_nested(state["memory"], partition, updated_data)
            action_description = f"Wrote to memory partition: {partition}"
            state['previous_actions'].append(action_description)
            return {**state, "agent_result": "Memory updated successfully"}

        elif action["action"] == "compose_response":
            response_text = action["params"].get("text") or action["params"].get("response")
            if not response_text:
                raise ValueError("Missing 'text' or 'response' in params for compose_response")
            state["conversation_history"].append({"role": "assistant", "content": response_text})
            action_description = "Composed response to user"
            state['previous_actions'].append(action_description)
            return {**state, "agent_result": response_text}

        elif action["action"] == "ask_user":
            prompt_text = action["params"]["prompt"]
            state["conversation_history"].append({"role": "assistant", "content": prompt_text})
            action_description = f"Asked user: {prompt_text}"
            state['previous_actions'].append(action_description)
            return {**state, "agent_result": "User prompted for input"}

        else:
            action_description = f"Executed {action['action']} with params: {action.get('params', {})}"
            state['previous_actions'].append(action_description)
            return {**state, "agent_result": f"Unknown action: {action['action']}"}

    except Exception as e:
        error_result = f"Error executing {action['action']}: {str(e)}"
        action_description = f"Attempted {action['action']} with params: {action.get('params', {})}"
        state['previous_actions'].append(action_description)
        return {**state, "agent_result": error_result}

def setup_workflow(coach_agent, agents, tools, persistence_dir=None):
    workflow = StateGraph(NutritionState)
    workflow.add_node("coach", lambda state: coach_node(state, coach_agent))
    workflow.add_node("execute_action", lambda state: execute_action_node(state, agents, tools))
    workflow.set_entry_point("coach")
    workflow.add_edge("coach", "execute_action")
    workflow.add_conditional_edges("execute_action", should_continue, {"execute_action": "coach", "end": END})
    
    if persistence_dir:
        checkpointer = FileCheckpointSaver(persistence_dir)
        print(f"MAS workflow compiled with file-based persistence at {persistence_dir}.")
    else:
        checkpointer = MemorySaver()
        print("MAS workflow compiled with in-memory persistence.")
    
    app = workflow.compile(checkpointer=checkpointer)
    return app

