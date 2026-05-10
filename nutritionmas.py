import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from IPython.display import Markdown, display

from agents import CoachAgent, MedicalAssessmentAgent, PlannerAgent
from config import set_settings
from knowledge import KnowledgeAgent
from logging_setup import get_logger, refresh_level
from memory import LongTermMemory
from state import initialize_empty_memory
from tools import ComputationTool, QuantitiesFinder, WebSearchTool
from utils import APIPoolManager, create_llm
from validation import ValidationAgent
from workflow import setup_workflow as setup_workflow_workflow

_logger = get_logger("nutritionmas")


def debug(level: str = "full", scopes: Optional[Dict[str, List[str]]] = None) -> None:
    """Enable debug mode with the given level and scopes.

    Args:
        level: 'full' (default) to show inputs and outputs, or 'output' to show only outputs.
        scopes: Optional dict like ``{'agents': ['all'], 'tools': ['ComputationTool']}``.
                If None, defaults to all agents and tools.
    """
    if scopes is None:
        scopes = {"agents": ["all"], "tools": ["all"]}
    set_settings(debug_mode=True, debug_level=level, debug_scopes=scopes)
    refresh_level()


def logging(log_dir: Optional[str] = None, persistence_dir: Optional[str] = None) -> None:  # noqa: A001 - public name kept for backwards compat
    """Set directories for log files and LangGraph checkpoint persistence.

    If ``log_dir`` is provided, agent/tool I/O is dumped there as JSON.
    If ``persistence_dir`` is provided, LangGraph checkpoints are persisted to disk.
    If neither is set, logging is disabled and persistence is in-memory.
    """
    updates: Dict[str, Any] = {}
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        updates["log_dir"] = log_dir
    if persistence_dir is not None:
        os.makedirs(persistence_dir, exist_ok=True)
        updates["persistence_dir"] = persistence_dir
    if updates:
        set_settings(**updates)

# Default model configurations (without API keys, as they will be provided by the user)
DEFAULT_MODEL_CONFIGS = {
    "main": {
        "type": "gemini",
        "model_name": "gemini-2.5-pro",
        "structured_output": True,
        "thinking_budget": 600,
        "params": {"max_tokens": 5120, "temperature": 0.3}
    },
    "agents_llm": {
        "type": "gemini",
        "model_name": "gemini-2.5-pro",
        "structured_output": True,
        "thinking_budget": 600,
        "params": {"max_tokens": 5120, "temperature": 0.3}
    },
    "tools_llm": {
        "type": "gemini",
        "model_name": "gemini-2.5-flash",
        "structured_output": False,
        "thinking_budget": 600,
        "params": {"max_tokens": 5120, "temperature": 0.3}
    },
    "planner_agent": {
        "type": "gemini",
        "model_name": "gemini-2.5-pro",
        "structured_output": True,
        "thinking_budget": 600,
        "params": {"max_tokens": 5120, "temperature": 0.3}
    },
    "validation_agent": {
        "type": "gemini",
        "model_name": "gemini-2.5-flash",
        "structured_output": True,
        "thinking_budget": 300,
        "params": {"max_tokens": 3072, "temperature": 0.2}
    },
    "user_simulator": {
        "type": "gemini",
        "model_name": "gemini-2.5-flash",
        "structured_output": False,
        "thinking_budget": 300,
        "params": {"max_tokens": 5120, "temperature": 0.5}
    }
}

    # "planner_agent": {
    #     "type": "gemini",
    #     "model_name": "gemma-3-27b-it",
    #     "params": {"max_tokens": 8192, "temperature": 0.3}
    # },


# Global variables to hold the system components
LLM_INSTANCES = None
TOOLS = None
AGENTS = None
APP = None

def create_llm_instances(api_keys: list[str], model_overrides: Optional[Dict[str, Any]] = None, enable_rate_limiting: bool = True):
    """Create LLM instances using provided API keys list and optional model overrides.
    
    Args:
        api_keys: List of API keys to cycle through.
        model_overrides: Optional overrides for model configs.
        enable_rate_limiting: If True, apply rate limits (default). If False, disable rate limiting and just cycle keys.
    """
    global LLM_INSTANCES
    if not api_keys:
        raise ValueError("At least one API key must be provided.")

    if enable_rate_limiting:
        rate_limits = {
            "gemini-2.5-pro": (5, 100),
            "gemini-2.5-flash": (10, 250),
            "gemma-3-27b-it": (30, 1000),
        }
    else:
        rate_limits = None

    manager = APIPoolManager(api_keys, rate_limits)
    _logger.info(
        "APIPoolManager initialized with %s and %d API keys.",
        "rate limiting enabled" if enable_rate_limiting else "rate limiting disabled",
        len(api_keys),
    )

    # Note: previously this loop used a local variable named ``config`` which
    # shadowed the imported ``config`` module — now ``cfg`` to avoid the trap.
    model_configs: Dict[str, Dict[str, Any]] = {}
    for key in DEFAULT_MODEL_CONFIGS:
        cfg = DEFAULT_MODEL_CONFIGS[key].copy()
        if model_overrides and key in model_overrides:
            override = model_overrides[key]
            if "model_name" in override:
                cfg["model_name"] = override["model_name"]
            if "params" in override:
                cfg["params"] = {**cfg.get("params", {}), **override["params"]}
        model_configs[key] = cfg

    LLM_INSTANCES = {
        "main": create_llm(model_configs["main"], manager),
        "agents_llm": create_llm(model_configs["agents_llm"], manager),
        "tools_llm": create_llm(model_configs["tools_llm"], manager),
        "planner_agent": create_llm(model_configs["planner_agent"], manager),
        "validation_agent": create_llm(model_configs["validation_agent"], manager),
        "user_simulator": create_llm(model_configs["user_simulator"], manager),
    }

def initialize_tools():
    """Initialize tools using the LLM instances."""
    global TOOLS
    if LLM_INSTANCES is None:
        raise RuntimeError("LLM instances must be created before initializing tools.")
    TOOLS_LLM = LLM_INSTANCES["tools_llm"]
    TOOLS = {
        "ComputationTool": ComputationTool(TOOLS_LLM),
        "WebSearchTool": WebSearchTool(TOOLS_LLM),
        "QuantitiesFinder": QuantitiesFinder()
    }

def initialize_agents():
    """Initialize agents using the LLM instances and tools."""
    global AGENTS
    if LLM_INSTANCES is None or TOOLS is None:
        raise RuntimeError("LLM instances and tools must be initialized before agents.")
    MAIN_LLM = LLM_INSTANCES["main"]
    AGENTS_LLM = LLM_INSTANCES["agents_llm"]
    PLANNER_LLM = LLM_INSTANCES["planner_agent"]
    VALIDATION_LLM = LLM_INSTANCES["validation_agent"]
    AGENTS = {
        "CoachAgent": CoachAgent(MAIN_LLM),
        "MedicalAssessmentAgent": MedicalAssessmentAgent(
            AGENTS_LLM, TOOLS["ComputationTool"], TOOLS["WebSearchTool"]
        ),
        "PlannerAgent": PlannerAgent(
            PLANNER_LLM, TOOLS["ComputationTool"], TOOLS["WebSearchTool"], TOOLS["QuantitiesFinder"]
        ),
        "ValidationAgent": ValidationAgent(VALIDATION_LLM),
        # KnowledgeAgent is the citation-first retrieval seam; defaults to
        # WebSearch backing. Phase 3+ can swap in a RAG-backed implementation
        # over USDA / WHO / ADA / EFSA without touching Coach call-sites.
        "KnowledgeAgent": KnowledgeAgent(TOOLS["WebSearchTool"]),
    }


# ---------------------------------------------------------------------------
# Long-term memory singleton (Phase 3)
# ---------------------------------------------------------------------------
LONG_TERM_MEMORY: Optional[LongTermMemory] = None


def initialize_long_term_memory(db_path: Optional[str] = None) -> LongTermMemory:
    """Initialise the SQLite-backed three-tier memory.

    Pass a file path for cross-session persistence, or omit for an in-memory
    DB (default; tests / ephemeral demos).
    """
    global LONG_TERM_MEMORY
    LONG_TERM_MEMORY = LongTermMemory(db_path=db_path)
    _logger.info("Long-term memory initialised at %s", db_path or ":memory:")
    return LONG_TERM_MEMORY

def setup_workflow():
    global APP
    if AGENTS is None or TOOLS is None:
        raise RuntimeError("Agents and tools must be initialized before setting up workflow.")
    APP = setup_workflow_workflow(AGENTS["CoachAgent"], AGENTS, TOOLS)

class UserSimulator:
    def __init__(self, llm, user_profile, medical_history):
        self.llm = llm
        self.user_data = {
            "user_profile": user_profile,
            "medical_history": medical_history
        }

    def get_response(self, assistant_message):
        user_data_str = json.dumps(self.user_data, indent=2)
        sim_prompt = f"""You are a user interacting with a nutrition app. Here is your profile and medical history:
        {user_data_str}

        The app has asked: {assistant_message}

        Provide a realistic answer based on your profile and medical history.
        Your response:
        """
        response = self.llm(sim_prompt)[0]
        return response

def initialize_user_data():
    """Collect user information interactively to initialize memory."""
    print("Let's collect some information about you to personalize your experience.")

    print("\nFirst, your demographics and anthropometrics:")
    name = input("What is your name? ")
    age = float(input("How old are you? (in years) "))
    sex = input("What is your sex? (male/female) ")
    height = float(input("What is your height? (in cm) "))
    weight = float(input("What is your weight? (in kg) "))

    print("\nNext, about your lifestyle and goals:")
    activity_level = input("What is your activity level? (e.g., sedentary, lightly active, moderately active, very active, extra active) ")
    goal = input("What is your primary goal? (e.g., lose weight, maintain weight, gain muscle) ")
    job = input("What is your job or daily routine? ")

    print("\nDietary preferences:")
    dietary_restrictions = input("Any dietary restrictions? (e.g., vegetarian, vegan, keto) ")
    food_likes = input("Favorite foods? ")
    food_dislikes = input("Foods you dislike? ")
    allergies_input = input("Any allergies? (comma-separated) ")
    allergies_list = [a.strip() for a in allergies_input.split(",") if a.strip()] if allergies_input else []

    print("\nLocation and budget:")
    country = input("Which country are you in? ")
    currency = input("Preferred currency? (e.g., USD, EGP) ")

    print("\nMedical history:")
    conditions_input = input("Any medical conditions? (comma-separated) ")
    conditions_list = [c.strip() for c in conditions_input.split(",") if c.strip()] if conditions_input else []
    medications_input = input("Current medications? (comma-separated) ")
    medications_list = [m.strip() for m in medications_input.split(",") if m.strip()] if medications_input else []
    past_issues_input = input("Past health issues? (comma-separated) ")
    past_issues_list = [p.strip() for p in past_issues_input.split(",") if p.strip()] if past_issues_input else []
    lab_results = input("Any recent lab results? (e.g., cholesterol levels) ")

    user_profile = {
        "name": name,
        "age": age,
        "sex": sex,
        "height": height,
        "weight": weight,
        "activity_level": activity_level,
        "goal": goal,
        "job": job,
        "dietary_restrictions": dietary_restrictions,
        "food_likes": food_likes,
        "food_dislikes": food_dislikes,
        "allergies": allergies_list,
        "country": country,
        "currency": currency,
        "last_updated": datetime.now().isoformat()
    }

    medical_history = {
        "conditions": conditions_list,
        "medications": medications_list,
        "past_issues": past_issues_list,
        "lab_results": lab_results,
        "last_updated": datetime.now().isoformat()
    }

    memory = initialize_empty_memory()
    memory["user_profile"] = user_profile
    memory["medical_history"] = medical_history

    return memory

def run(simulate=False, simulated_users=None):
    """Run the system in either simulation or interactive mode."""
    if APP is None:
        raise RuntimeError("Workflow must be set up before running the system.")

    if simulate:
        print("\n" + "="*80)
        print("STARTING SIMULATION MODE")
        print("="*80)
        if not simulated_users:
            raise ValueError("simulated_users must be provided when simulate=True")

        for user in simulated_users:
            print(f"\nProcessing user: {user['user_profile']['name']}")
            user["user_profile"]["last_updated"]  = datetime.now().isoformat()
            user["medical_history"]["last_updated"] = datetime.now().isoformat()
            memory = {
                "user_profile":       user["user_profile"],
                "medical_history":    user["medical_history"],
                "flags_and_assessments": {},
                "plans": {}
            }
            conversation_history = []
            previous_actions = []
            user_simulator = UserSimulator(LLM_INSTANCES["user_simulator"], user["user_profile"], user["medical_history"])
            for question in user["questions"]:
                print(f"\n🙍🏻‍♂️Asking: {question}")
                state = {
                    "memory": memory,
                    "user_question": question,
                    "conversation_history": conversation_history + [{"role": "user", "content": question}],
                    "current_action": None,
                    "agent_result": None,
                    "num_turns": 0,
                    "max_turns": 10,
                    "previous_actions": previous_actions,
                    "response_steps": []
                }
                while True:
                    final_state = APP.invoke(state, config={"configurable": {"thread_id": f"user_{user['user_profile']['name']}"}})
                    if final_state["num_turns"] >= final_state["max_turns"]:
                        print("Max turns reached without composing a response.")
                        break
                    last_action = final_state["current_action"]["action"]
                    if last_action == "compose_response":
                        print(f"\n{'='*60}")
                        # print(f"**Answer:**\n\n{final_state['agent_result']}")
                        display(Markdown(f"**Answer:**\n\n{final_state['agent_result']}"))
                        conversation_history = final_state["conversation_history"]
                        memory = final_state["memory"]
                        previous_actions = final_state["previous_actions"]
                        break
                    elif last_action == "ask_user":
                        prompt = final_state["agent_result"]
                        print(f"System asks: {prompt}")
                        response = user_simulator.get_response(prompt)
                        print(f"Simulated user responds: {response}")
                        state = {
                            "memory": memory,
                            "user_question": question,
                            "conversation_history": conversation_history + [{"role": "user", "content": question}],
                            "current_action": None,
                            "agent_result": None,
                            "num_turns": 0,
                            "max_turns": 10,
                            "previous_actions": previous_actions,
                            "response_steps": [] 
                        }
                    else:
                        print(f"Unexpected action: {last_action}")
                        break
    else:
        print("\n" + "="*80)
        print("STARTING INTERACTIVE MODE")
        print("="*80)
        memory = initialize_user_data()

        print("\n" + "="*80)
        print("WELCOME TO THE NUTRITION APP")
        print("="*80)

        initial_state = {
            "memory": memory,
            "user_question": "welcome",
            "conversation_history": [],
            "current_action": None,
            "agent_result": None,
            "num_turns": 0,
            "max_turns": 10,
            "previous_actions": [],
            "response_steps": []
        }

        final_state = APP.invoke(initial_state, config={"configurable": {"thread_id": "user1"}})

        if final_state["agent_result"]:
            display(Markdown(f"\n🤖 Coach: {final_state['agent_result']}"))
            memory = final_state["memory"]
            conversation_history = final_state["conversation_history"]

        while True:
            q = input("\n❓ Your question: ")
            if q.lower() == 'exit':
                break

            state = {
                "memory": final_state["memory"],
                "user_question": q,
                "conversation_history": final_state["conversation_history"] + [{"role": "user", "content": q}],
                "current_action": None,
                "agent_result": None,
                "num_turns": 0,
                "max_turns": 10,
                "previous_actions": final_state["previous_actions"],
                "response_steps": []  
            }

            final_state = APP.invoke(state, config={"configurable": {"thread_id": "user1"}})

            if final_state["agent_result"]:
                print(f"\n{'='*60}")
                display(Markdown(f"\n🤖 Coach: {final_state['agent_result']}"))
                memory = final_state["memory"]
                conversation_history = final_state["conversation_history"]