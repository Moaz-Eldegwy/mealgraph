"""Agent implementations.

Phase 1: every agent's per-turn output is now a Pydantic model from
``schemas``. The prompts are split so the static system rules sit in a
module-level constant (eligible for Gemini's implicit prompt cache) and only
the dynamic state changes per call.

The action-dispatch loops are still *inside* the agent classes — Phase 2 will
break them into LangGraph subgraphs with parallel tool nodes and the
ValidationAgent critic loop.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from config import get_settings
from logging_setup import get_logger
from schemas import (
    CoachDecision,
    MedicalAssessmentDecision,
    MedicalAssessmentResult,
    PlannerDecision,
)
from tools import ComputationTool, QuantitiesFinder, WebSearchTool
from utils import save_to_json, should_debug, update_memory_partition

_coach_logger = get_logger("agents.coach")
_medical_logger = get_logger("agents.medical")
_planner_logger = get_logger("agents.planner")


# ---------------------------------------------------------------------------
# Coach
# ---------------------------------------------------------------------------
_COACH_SYSTEM_PROMPT = """\
You are the Coach Agent (central orchestrator) of a nutrition Multi-Agent System.

Primary responsibilities:
- Translate user intent into a concrete workflow of response_steps.
- Enforce system rules (MedicalAssessment must complete before Planner runs).
- Decide and perform exactly one action per turn: call_agent, call_tool,
  ask_user, write_memory, or compose_response.

Inputs each turn:
- observation (string built from user query + memory + history)
- memory partitions: user_profile, medical_history, flags_and_assessments, plans
- response_steps (list, may be empty on the first turn)

Behaviour rules (mandatory):
1. If response_steps is empty, generate ordered steps (max 6). Each step
   must include id, actor, prerequisites, and status "pending".
   Typical personal-workflow (when the user asks for a personalised plan):
     1) Validate required user data (height, weight, age, sex, activity_level,
        allergies, goal). If missing -> ask_user.
     2) Update memory if the user provided new data [action: write_memory].
     3) Call MedicalAssessmentAgent with a task to assess the user.
     4) Wait for assessment to be completed and stored in memory.
     5) Call PlannerAgent with the relevant task.
2. When calling any agent, set the called step status to "in_progress" and
   include prerequisites satisfied by your observation.
3. Only call PlannerAgent if memory.flags_and_assessments contains an
   "assessment_status" of "assessment_complete". If missing, call
   MedicalAssessmentAgent first.
4. When new personal data appears in user input, add steps to: propose memory
   update (write_memory), call MedicalAssessmentAgent if needed, re-plan if
   needed.
5. For any write_memory action, provide the full partition contents in
   params.data (not diffs). The Coach is responsible for merging and storing.

Output JSON shape (enforced by schema):
{
  "observation": "...",
  "thought": "...",
  "response_steps": [ ... ],
  "action": "call_agent | call_tool | ask_user | write_memory | compose_response",
  "params": { ... }
}

Required params per action:
- call_agent:       {"agent_name": "...", "task": "..."}
- call_tool:        {"tool_name": "...", "task": "..."}
- ask_user:         {"prompt": "..."}
- write_memory:     {"partition": "...", "data": {...}}
- compose_response: {"text": "...markdown..."}

Composition rules:
- When composing the response, extract relevant information from memory state
  (calorie target, plan details, dietary restrictions, citations) in markdown.
- Always include a "trace" line summarising which agents/tools contributed.
- For high-risk profiles (requires_professional_consultation == true), append
  a bold warning advising professional consultation before implementation.
"""


class CoachAgent:
    def __init__(self, llm_instance):
        self.llm = llm_instance

    def handle_task(self, state: Dict[str, Any]) -> Dict[str, Any]:
        settings = get_settings()
        memory_str = json.dumps(state["memory"], indent=2, default=str)
        response_steps = state.get("response_steps", [])
        response_steps_str = (
            json.dumps(response_steps, indent=2, default=str) if response_steps else "None"
        )

        truncated_history: List[Dict[str, str]] = []
        for msg in state["conversation_history"]:
            if msg["role"] == "assistant" and len(msg["content"]) > 200:
                truncated_history.append(
                    {"role": "assistant", "content": msg["content"][:200] + "... (full response in memory)"}
                )
            else:
                truncated_history.append(msg)
        history_str = "\n".join(f"{m['role']}: {m['content']}" for m in truncated_history)

        observation = (
            f"User query: {state['user_question']}\n"
            f"Memory State: {memory_str}\n"
            f"Current Response Steps: {response_steps_str}\n"
            f"Previous Tool Result: {state.get('agent_result', 'None')}\n"
            f"Conversation history: {history_str}"
        )
        prompt = f"{_COACH_SYSTEM_PROMPT}\n\n--- Current State ---\n{observation}"

        if should_debug("agents", "CoachAgent"):
            _coach_logger.debug("--- Coach Agent Turn %d ---", state["num_turns"] + 1)
            if settings.debug_level == "full":
                _coach_logger.debug("Raw LLM input:\n%s", prompt)

        decision = self.llm.call_typed(prompt, CoachDecision)
        if decision is None:
            return self._fallback_state(state, "Coach decision could not be parsed.")

        if should_debug("agents", "CoachAgent"):
            _coach_logger.debug("Coach decision:\n%s", decision.model_dump_json(indent=2))

        if not settings.debug_mode:
            self._log_user_mode_action(decision)

        current_action = {"action": decision.action, "params": decision.params}
        new_steps = [s.model_dump() for s in decision.response_steps] or state.get("response_steps", [])

        save_to_json(
            {
                "prompt": prompt,
                "decision": decision.model_dump(),
                "timestamp": datetime.now().isoformat(),
            },
            f"coach_agent_{datetime.now().isoformat()}.json",
            subdirectory="CoachAgent",
        )

        return {
            **state,
            "current_action": current_action,
            "response_steps": new_steps,
            "num_turns": state["num_turns"] + 1,
            "agent_result": None,
        }

    @staticmethod
    def _log_user_mode_action(decision: CoachDecision) -> None:
        params = decision.params or {}
        action = decision.action
        if action == "call_agent":
            msg = f"Calling {params.get('agent_name')} with task '{params.get('task')}'"
        elif action == "call_tool":
            msg = f"Using {params.get('tool_name')} with task '{params.get('task')}'"
        elif action == "ask_user":
            msg = f"Asking user: {params.get('prompt')}"
        elif action == "write_memory":
            msg = f"Writing to memory partition '{params.get('partition')}'"
        elif action == "compose_response":
            msg = "Composing final response"
        else:
            msg = f"Unknown action: {action}"
        _coach_logger.info("\n🏋️‍♂️Coach Agent: %s", msg)

    @staticmethod
    def _fallback_state(state: Dict[str, Any], message: str) -> Dict[str, Any]:
        _coach_logger.error(message)
        return {
            **state,
            "current_action": {
                "action": "compose_response",
                "params": {"text": f"Sorry — I hit an internal error while planning. ({message})"},
                "_parse_error": True,
            },
            "num_turns": state["num_turns"] + 1,
            "agent_result": None,
        }


# ---------------------------------------------------------------------------
# Medical Assessment
# ---------------------------------------------------------------------------
_MEDICAL_SYSTEM_PROMPT = """\
You are the Medical Assessment Agent. Produce an evidence-based assessment and
the clinical flags / calculations the Planner and Validation agents need.

Available tools: ComputationTool, WebSearchTool.

Mandatory behaviour (do not skip):
1. Critical data check: confirm presence of age, sex, height, weight,
   activity_level, allergies, medications. If any critical field is missing,
   set action_type="ask_user" and list the missing names in ``fields``.
2. Use ComputationTool for ALL numeric calculations (BMI, BMR, TDEE, calorie
   targets, macro targets). Pass numeric inputs in tool_task.
3. Use WebSearchTool to fetch authoritative guidelines (WHO, USDA, ADA,
   EFSA). Capture source URLs with timestamps.
4. Produce a compact assessment_plan (3-6 steps). Default sequence:
   a) ComputationTool: BMI, BMR, TDEE, daily_target_calories (single int).
   b) ComputationTool: macro_targets (protein_g, fat_g, carbohydrates_g - all
      single ints, no ranges) optimised for the user's goal.
   c) WebSearchTool: dietary guidelines for the user's conditions.
   d-f) Optional follow-ups for specific risks.
5. When complete, set action_type="assessment_complete" and populate
   ``result`` (a MedicalAssessmentResult) with:
     - assessment_summary
     - calculations: { BMI, BMR, TDEE, daily_target_calories,
                       macro_targets: { protein_g, fat_g, carbohydrates_g } }
     - flags_to_set (e.g. ["high_ldl", "diabetes_risk"])
     - recommendations (clinical dietary constraints / urgent issues)
     - requires_professional_consultation (True for medically sensitive cases)
     - evidence_sources (list of URLs)
     - trace (one paragraph summarising agent/tool usage)
6. If any tool call fails, fall back to best-known values, set
   data_confidence below 1.0, and mark requires_tool_retry=true.

Output JSON shape (enforced by schema):
{
  "medical_reasoning": "...",
  "observation": "...",
  "risk_assessment_priorities": [...],
  "assessment_plan": [...],
  "action_type": "call_tool" | "ask_user" | "assessment_complete",
  "tool_name": "ComputationTool" | "WebSearchTool" | null,
  "tool_task": "..." | null,
  "fields": [...],            // only when ask_user
  "result": { ... }           // only when assessment_complete
}
"""


class MedicalAssessmentAgent:
    MAX_ITERATIONS = 15

    def __init__(
        self,
        llm_instance,
        computation_tool: ComputationTool,
        web_search_tool: WebSearchTool,
    ):
        self.llm = llm_instance
        self.computation_tool = computation_tool
        self.web_search_tool = web_search_tool

    def handle_task(self, task: str, memory: Dict[str, Any]) -> str:
        _medical_logger.info("\n👨🏻‍⚕️ MEDICAL ASSESSMENT AGENT STARTED")
        settings = get_settings()

        relevant_memory = {
            "user_profile": memory.get("user_profile", {}),
            "medical_history": memory.get("medical_history", {}),
        }
        memory_str = json.dumps(relevant_memory, indent=2, default=str)
        tool_results: List[str] = []
        assessment_plan: List[dict] = []

        for iteration in range(self.MAX_ITERATIONS):
            tool_results_str = (
                "\n".join(f"Tool Result {i+1}: {r}" for i, r in enumerate(tool_results)) or "None"
            )
            assessment_plan_str = (
                json.dumps(assessment_plan, indent=2, default=str) if assessment_plan else "None"
            )

            prompt = (
                f"{_MEDICAL_SYSTEM_PROMPT}\n\n--- Task & State ---\n"
                f"Task: {task}\n"
                f"Current Memory: {memory_str}\n"
                f"Current Assessment Plan: {assessment_plan_str}\n"
                f"Previous Tool Results: {tool_results_str}\n"
            )

            if should_debug("agents", "MedicalAssessmentAgent"):
                _medical_logger.debug("--- Medical Assessment Iteration %d ---", iteration + 1)
                if settings.debug_level == "full":
                    _medical_logger.debug("Raw LLM input:\n%s", prompt)

            decision = self.llm.call_typed(prompt, MedicalAssessmentDecision)
            if decision is None:
                _medical_logger.error("Medical decision parse failed at iteration %d", iteration + 1)
                return "Medical assessment failed: could not parse LLM decision."

            if should_debug("agents", "MedicalAssessmentAgent"):
                _medical_logger.debug("Medical decision:\n%s", decision.model_dump_json(indent=2))

            if decision.assessment_plan:
                assessment_plan = [s.model_dump() for s in decision.assessment_plan]

            if not settings.debug_mode:
                self._log_user_mode_action(decision)

            if decision.action_type == "call_tool":
                tool_results.append(f"{decision.tool_name}: {self._dispatch_tool(decision)}")

            elif decision.action_type == "ask_user":
                fields = decision.fields or []
                msg = (
                    f"Missing critical fields: {', '.join(fields)}. "
                    "Please provide the following information to continue the assessment."
                )
                _medical_logger.info("👨🏻‍⚕️ MEDICAL ASSESSMENT AGENT: User query needed - %s", msg)
                return msg

            elif decision.action_type == "assessment_complete":
                return self._finalize(task, decision, memory, relevant_memory, tool_results)

            else:
                _medical_logger.error("Unknown action_type: %s", decision.action_type)
                break

        _medical_logger.warning("👨🏻‍⚕️ MEDICAL ASSESSMENT AGENT Stopped (MAX ITERATIONS)")
        return f"Medical assessment stopped after {self.MAX_ITERATIONS} iterations"

    # ------------------------------------------------------------------
    def _dispatch_tool(self, decision: MedicalAssessmentDecision) -> str:
        tool_name = decision.tool_name
        tool_task = decision.tool_task
        if not tool_task:
            return f"Missing 'tool_task' for {tool_name}"
        if tool_name == "ComputationTool":
            return self.computation_tool.handle_task(tool_task)
        if tool_name == "WebSearchTool":
            return self.web_search_tool.handle_task(tool_task)
        return f"Unknown tool: {tool_name}"

    @staticmethod
    def _log_user_mode_action(decision: MedicalAssessmentDecision) -> None:
        if decision.action_type == "call_tool":
            _medical_logger.info(
                "👨🏻‍⚕️ Medical Assessment Agent: Using %s for '%s'",
                decision.tool_name,
                decision.tool_task,
            )
        elif decision.action_type == "ask_user":
            _medical_logger.info(
                "👨🏻‍⚕️ Medical Assessment Agent: Asking user for missing fields: %s",
                ", ".join(decision.fields or []),
            )
        elif decision.action_type == "assessment_complete":
            _medical_logger.info("👨🏻‍⚕️ Medical Assessment Agent: Completing assessment")

    def _finalize(
        self,
        task: str,
        decision: MedicalAssessmentDecision,
        memory: Dict[str, Any],
        relevant_memory: Dict[str, Any],
        tool_results: List[str],
    ) -> str:
        result: Optional[MedicalAssessmentResult] = decision.result
        if result is None:
            _medical_logger.error("assessment_complete decision missing result payload")
            return "Medical assessment failed: completion payload missing."

        if result.requires_tool_retry:
            msg = "Assessment requires tool retry due to tool failures."
            _medical_logger.warning("👨🏻‍⚕️ MEDICAL ASSESSMENT AGENT: Tool retry needed - %s", msg)
            return msg

        update_memory_partition(
            memory,
            "flags_and_assessments",
            {
                "assessment_summary": result.assessment_summary,
                "flags": result.flags_to_set,
                "recommendations": result.recommendations,
                "requires_professional_consultation": result.requires_professional_consultation,
                "calculations": result.calculations.model_dump(),
                "evidence_sources": result.evidence_sources,
                "data_confidence": result.data_confidence,
                "trace": result.trace,
                "assessment_status": "assessment_complete",
                "assessment_timestamp": datetime.now().isoformat(),
            },
        )
        save_to_json(
            {
                "task": task,
                "memory_input": relevant_memory,
                "tool_results": tool_results,
                "result": result.model_dump(),
                "timestamp": datetime.now().isoformat(),
            },
            f"medical_assessment_{datetime.now().isoformat()}.json",
            subdirectory="MedicalAssessment",
        )
        _medical_logger.info("👨🏻‍⚕️ MEDICAL ASSESSMENT AGENT COMPLETED: %s", result.assessment_summary)
        return result.assessment_summary


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------
_PLANNER_SYSTEM_PROMPT = """\
You are the Planner Agent. Create personalised meal plans constrained by the
medical assessment.

Available tools: WebSearchTool, QuantitiesFinder, ComputationTool.

Mandatory behaviour & rules:
1. Precondition: do NOT plan unless flags_and_assessments has an
   "assessment_status" of "assessment_complete". If missing, return
   action_type="provide_plan" with final_plan={"error": "..."} explaining the
   blocker and suggesting MedicalAssessmentAgent.
2. Batch tool calls: fetch nutrition facts for ALL foods in one WebSearchTool
   call rather than one call per item.
3. For each food in the draft, look up per-100g nutrition (calories, protein,
   fat, carbohydrates). If WebSearchTool fails for >2 items, fall back to
   internal knowledge.
4. Tolerances: calories +/- 3%, each macro +/- 5% of target.
5. Exclude allergens and disliked foods. Propose alternatives if necessary
   for balance.
6. Multi-day requests: emit a 1-2 day plan and instruct the user to rotate.
7. QuantitiesFinder format: tool_task MUST be a JSON STRING containing
   {"foods": [...], "targets": {...}}. Each food needs name, calories,
   protein, fat, carbohydrates (per 100g) and estimated_g (your best guess).

Planning Steps Handling:
- If Current Planning Steps is empty/None, adopt this fixed 5-step plan:
  1. Draft a realistic plan; assign a realistic estimated_g per food.
  2. Batch-gather nutrition facts via WebSearchTool.
  3. Call QuantitiesFinder with foods + targets to compute precise grams.
  4. Update the draft with the solver's quantities.
  5. Provide the final plan via action_type="provide_plan".
- If steps are provided, you may iterate within a step until targets are met.

Output JSON shape (enforced by schema):
{
  "observation": "...",
  "thought": "...",
  "planning_steps": [...],
  "action_type": "call_tool" | "draft_plan" | "provide_plan",
  "tool_name": "WebSearchTool" | "QuantitiesFinder" | "ComputationTool" | null,
  "tool_task": "..." | null,
  "drafted_plan": { ... } | null,
  "final_plan": { ... } | null
}

Notes:
- Keep plans realistic and culturally appropriate (regional foods if provided).
- Include a "trace" line in the final plan summarising agents/tools used.
- Always echo the full updated planning_steps so they persist across turns.
"""


class PlannerAgent:
    MAX_ITERATIONS = 15

    def __init__(
        self,
        llm_instance,
        computation_tool: ComputationTool,
        web_search_tool: WebSearchTool,
        quantities_finder: QuantitiesFinder,
    ):
        self.llm = llm_instance
        self.computation_tool = computation_tool
        self.web_search_tool = web_search_tool
        self.quantities_finder = quantities_finder

    def handle_task(self, task: str, memory: Dict[str, Any]) -> str:
        _planner_logger.info("\n📋 PLANNER AGENT STARTED")
        settings = get_settings()

        relevant_memory = {
            "user_profile": memory.get("user_profile", {}),
            "flags_and_assessments": memory.get("flags_and_assessments", {}),
        }
        tool_results: List[str] = []
        planning_steps: List[dict] = []

        for iteration in range(self.MAX_ITERATIONS):
            memory_str = json.dumps(
                {
                    "user_profile": memory.get("user_profile", {}),
                    "flags_and_assessments": memory.get("flags_and_assessments", {}),
                    "plans": memory.get("plans", {}),
                },
                indent=2,
                default=str,
            )
            tool_results_str = (
                "\n".join(f"Tool Result {i+1}: {r}" for i, r in enumerate(tool_results)) or "None"
            )
            planning_steps_str = (
                json.dumps(planning_steps, indent=2, default=str) if planning_steps else "None"
            )

            prompt = (
                f"{_PLANNER_SYSTEM_PROMPT}\n\n--- Task & State ---\n"
                f"Task: {task}\n"
                f"Current Memory: {memory_str}\n"
                f"Current Planning Steps: {planning_steps_str}\n"
                f"Previous Tool Results: {tool_results_str}\n"
            )

            if should_debug("agents", "PlannerAgent"):
                _planner_logger.debug("--- Planner Iteration %d ---", iteration + 1)
                if settings.debug_level == "full":
                    _planner_logger.debug("Raw LLM input:\n%s", prompt)

            decision = self.llm.call_typed(prompt, PlannerDecision)
            if decision is None:
                _planner_logger.error("Planner decision parse failed at iteration %d", iteration + 1)
                return "Planner failed: could not parse LLM decision."

            if should_debug("agents", "PlannerAgent"):
                _planner_logger.debug("Planner decision:\n%s", decision.model_dump_json(indent=2))

            if decision.planning_steps:
                planning_steps = [s.model_dump() for s in decision.planning_steps]

            if not settings.debug_mode:
                self._log_user_mode_action(decision)

            if decision.action_type == "call_tool":
                tool_results.append(f"{decision.tool_name}: {self._dispatch_tool(decision)}")

            elif decision.action_type == "draft_plan":
                if decision.drafted_plan:
                    memory.setdefault("plans", {})["drafted_plan"] = decision.drafted_plan
                    tool_results.append("Plan drafted and stored in memory")
                else:
                    tool_results.append("Drafted plan not provided")

            elif decision.action_type == "provide_plan":
                final = decision.final_plan or memory.get("plans", {}).get("drafted_plan")

                # Error escape hatch (e.g. precondition not met)
                if isinstance(final, dict) and "error" in final:
                    _planner_logger.error("📋 PLANNER AGENT ERROR: %s", final)
                    return json.dumps(final)

                if not final:
                    tool_results.append("Cannot finalize: missing plan")
                    continue  # let the loop try another iteration

                memory.setdefault("plans", {})
                memory["plans"]["current_plan"] = final
                memory["plans"]["plan_timestamp"] = datetime.now().isoformat()
                memory["plans"].pop("drafted_plan", None)

                save_to_json(
                    {
                        "task": task,
                        "memory_input": relevant_memory,
                        "tool_results": tool_results,
                        "final_response": decision.model_dump(),
                        "timestamp": datetime.now().isoformat(),
                    },
                    f"planner_agent_{datetime.now().isoformat()}.json",
                    subdirectory="PlannerAgent",
                )
                _planner_logger.info("\n📋 PLANNER AGENT COMPLETED")
                return json.dumps(final) if isinstance(final, dict) else str(final)

            else:
                _planner_logger.error("Unknown action_type: %s", decision.action_type)
                break

        _planner_logger.warning("📋 PLANNER AGENT Stopped (MAX ITERATIONS)")
        return (
            f"Planning stopped after {self.MAX_ITERATIONS} iterations "
            f"with {len(tool_results)} actions"
        )

    # ------------------------------------------------------------------
    def _dispatch_tool(self, decision: PlannerDecision) -> str:
        tool_name = decision.tool_name
        tool_task = decision.tool_task
        if not tool_name or not tool_task:
            return "Missing tool_name or tool_task"
        if tool_name == "ComputationTool":
            return self.computation_tool.handle_task(tool_task)
        if tool_name == "WebSearchTool":
            return self.web_search_tool.handle_task(tool_task)
        if tool_name == "QuantitiesFinder":
            return self.quantities_finder.handle_task(tool_task)
        return f"Unknown tool: {tool_name}"

    @staticmethod
    def _log_user_mode_action(decision: PlannerDecision) -> None:
        if decision.action_type == "call_tool":
            _planner_logger.info(
                "📋 Planner Agent: Using %s for '%s'",
                decision.tool_name,
                decision.tool_task,
            )
        elif decision.action_type == "draft_plan":
            _planner_logger.info("📋 Planner Agent: Drafting plan")
        elif decision.action_type == "provide_plan":
            _planner_logger.info("📋 Planner Agent: Finalizing plan")
