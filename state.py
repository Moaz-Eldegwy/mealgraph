from typing import TypedDict, Dict, Any, List, Optional

class NutritionState(TypedDict):
    memory: Dict[str, Any]
    user_question: str
    conversation_history: List[Dict[str, str]]
    current_action: Optional[Dict[str, Any]]
    agent_result: Optional[str]
    num_turns: int
    max_turns: int
    previous_actions: List[str]
    response_steps: List[Dict[str, Any]]


def initialize_empty_memory() -> Dict[str, Any]:
    """Initialize empty memory structure following the design"""
    return {
        "user_profile": {},
        "medical_history": {},
        "flags_and_assessments": {},
        "plans": {}
    }
