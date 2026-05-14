"""Agent implementations.

Three agents make up the system:

* :class:`CoachAgent` — orchestrator. Picks one action per turn
  (``call_agent`` / ``ask_user`` / ``write_memory`` / ``compose_response``)
  and self-reviews the plan against medical flags before composing.
* :class:`MedicalAssessmentAgent` — runs deterministic clinical formulas
  *first*, then asks the LLM to interpret the numbers and emit flags /
  recommendations / evidence. The agent overwrites whatever the LLM
  emitted for ``calculations`` with the deterministic values so the math
  is exact by construction.
* :class:`PlannerAgent` — drafts a meal plan, looks up nutrition via the
  WebSearchTool, runs the PuLP solver, and runs :func:`check_plan` on the
  result before returning. Up to two internal revisions resolve any
  deterministic violations (allergies / calorie or macro deviations)
  without involving the Coach.

The public :func:`check_plan` mirrors the Planner's internal critic so the
eval harness can exercise the same code path against hand-built fixtures.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from config import get_settings
from logging_setup import get_logger
from nutrition_formulas import full_assessment
from schemas import (
    Calculations,
    CoachDecision,
    MacroTargets,
    MedicalAssessmentDecision,
    MedicalAssessmentResult,
    PlannerDecision,
)
from tools import QuantitiesFinder, WebSearchTool
from utils import save_to_json, should_debug, update_memory_partition

_coach_logger = get_logger("agents.coach")
_medical_logger = get_logger("agents.medical")
_planner_logger = get_logger("agents.planner")


# Tolerances are module-level so tests / configs can override.
CALORIE_TOLERANCE = 0.03  # +/- 3 %
MACRO_TOLERANCE = 0.05  # +/- 5 %


def _decode_plan_field(value: Any) -> Optional[Dict[str, Any]]:
    """Normalise a Planner plan field to a dict.

    Structured decoding usually returns a dict, but mocked tests and some
    SDK paths pass the same payload as a JSON-encoded string. Both shapes
    flow through here, so the agent only ever sees ``Optional[dict]``.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            _planner_logger.warning(
                "Planner plan field was not valid JSON; using raw string. Preview: %s",
                text[:200],
            )
            return {"raw": text}
        if isinstance(decoded, dict):
            return decoded
        return {"value": decoded}
    return None


def _is_plan_empty(plan: Dict[str, Any]) -> bool:
    """True when ``plan`` contains no renderable content.

    Catches the specific anti-pattern ``{"days": [{}]}`` that constrained
    decoding can emit when the model fills the envelope but skips the
    body. Anything with non-empty entries under ``days`` is accepted —
    clinical quality is the post-LP check's job, not this guard's.
    """
    if not isinstance(plan, dict):
        return True
    days = plan.get("days")
    if not days:
        return True
    if isinstance(days, list):
        if all(not isinstance(d, (dict, list)) or not d for d in days):
            return True
    return False


# ---------------------------------------------------------------------------
# Plan checker (public — used by Planner internally and by the eval harness)
# ---------------------------------------------------------------------------
def _extract_foods_and_totals(plan: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Walk a plan and collect (list-of-foods, summed-totals).

    Supports both the canonical ``{"days": [[FoodItem, ...], ...]}`` shape
    and the flatter ``{"days": [{food}, ...]}`` shape the LLM tends to emit
    when constrained decoding inlines the inner list. ``daily_totals`` (if
    present) overrides the walked sums.
    """
    foods: List[Dict[str, Any]] = []
    totals: Dict[str, float] = {
        "calories": 0.0,
        "protein": 0.0,
        "fat": 0.0,
        "carbohydrates": 0.0,
    }

    def _walk(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                _walk(item)
        elif isinstance(node, dict):
            if "name" in node and any(k in node for k in ("calories", "calories_g", "kcal")):
                foods.append(node)
                totals["calories"] += float(node.get("calories", node.get("kcal", 0)) or 0)
                totals["protein"] += float(node.get("protein_g", node.get("protein", 0)) or 0)
                totals["fat"] += float(node.get("fat_g", node.get("fat", 0)) or 0)
                totals["carbohydrates"] += float(
                    node.get("carbohydrates_g", node.get("carbohydrates", 0)) or 0
                )
            else:
                for v in node.values():
                    _walk(v)

    _walk(plan)

    if isinstance(plan, dict) and "daily_totals" in plan:
        dt = plan["daily_totals"] or {}
        for k in ("calories", "protein", "fat", "carbohydrates"):
            if k in dt:
                totals[k] = float(dt[k])
            # Also accept the ``_g`` variants surfaced by the Planner schema.
            g_key = f"{k}_g" if k != "calories" else None
            if g_key and g_key in dt:
                totals[k] = float(dt[g_key])
    return foods, totals


def check_plan(plan: Dict[str, Any], memory: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Deterministic safety / target check over a finalised plan.

    Returns a list of issue dicts ``{code, description, severity}``. The
    Planner uses the severity to drive its internal revision loop:

    * ``high``   — allergy violations. Hard block; force a revision.
    * ``medium`` — calorie deviation > 3 % or any macro deviation > 5 %.
    * ``low``    — disliked-food advisory.

    Empty list means the plan passes every deterministic check. The
    Coach's self-review covers the LLM-graded checks (medical-flag
    respect, citation presence, cultural fit).
    """
    issues: List[Dict[str, Any]] = []

    user_profile = memory.get("user_profile", {}) or {}
    allergies = {
        str(a).strip().lower()
        for a in user_profile.get("allergies", []) or []
        if a
    }
    dislikes_raw = user_profile.get("food_dislikes", "") or ""
    dislikes = {d.strip().lower() for d in str(dislikes_raw).split(",") if d.strip()}

    flags = memory.get("flags_and_assessments", {}) or {}
    calc = flags.get("calculations", {}) or {}
    target_calories = calc.get("daily_target_calories")
    macro_targets = calc.get("macro_targets") or {}

    foods, totals = _extract_foods_and_totals(plan)

    for food in foods:
        name = str(food.get("name") or "").lower()
        for allergen in allergies:
            if allergen and allergen in name:
                issues.append(
                    {
                        "code": "allergy_violation",
                        "description": f"Food '{name}' matches allergen '{allergen}'.",
                        "severity": "high",
                    }
                )

    for food in foods:
        name = str(food.get("name") or "").lower()
        for d in dislikes:
            if d and d in name:
                issues.append(
                    {
                        "code": "disliked_food",
                        "description": f"Food '{name}' matches user dislike '{d}'.",
                        "severity": "low",
                    }
                )

    if target_calories and totals.get("calories"):
        dev = abs(totals["calories"] - target_calories) / target_calories
        if dev > CALORIE_TOLERANCE:
            issues.append(
                {
                    "code": "calorie_deviation",
                    "description": (
                        f"Plan total {totals['calories']:.0f} kcal vs target "
                        f"{target_calories} kcal ({dev*100:.1f}% deviation)."
                    ),
                    "severity": "medium",
                }
            )

    macro_map = {"protein_g": "protein", "fat_g": "fat", "carbohydrates_g": "carbohydrates"}
    for tgt_key, plan_key in macro_map.items():
        target = macro_targets.get(tgt_key)
        actual = totals.get(plan_key)
        if target and actual:
            dev = abs(actual - target) / target
            if dev > MACRO_TOLERANCE:
                issues.append(
                    {
                        "code": f"{plan_key}_deviation",
                        "description": (
                            f"{plan_key} total {actual:.0f}g vs target {target}g "
                            f"({dev*100:.1f}% deviation)."
                        ),
                        "severity": "medium",
                    }
                )

    return issues


# ---------------------------------------------------------------------------
# Coach
# ---------------------------------------------------------------------------
_COACH_SYSTEM_PROMPT = """\
You are the Coach Agent (central orchestrator) of a nutrition Multi-Agent System.

Primary responsibilities:
- Translate user intent into a concrete workflow of response_steps.
- Enforce system rules (MedicalAssessment must complete before Planner runs).
- Decide and perform exactly one action per turn: call_agent, ask_user,
  write_memory, or compose_response. There is no call_tool — tools are
  invoked exclusively by the worker agents.

Inputs each turn:
- observation (string built from user query + memory + history)
- memory partitions: user_profile, medical_history, flags_and_assessments, plans
- response_steps (list, may be empty on the first turn)

Behaviour rules (mandatory):
1. If response_steps is empty, generate ordered steps (max 6). Each step
   must include id, actor, prerequisites, and status "pending".
   Standard personal-workflow:
     1) Validate required user data (height, weight, age, sex, activity_level,
        allergies, goal). If missing -> ask_user.
     2) Update memory if the user provided new data [action: write_memory].
     3) Call MedicalAssessmentAgent with a task to assess the user.
     4) Call PlannerAgent with the relevant task.
     5) Self-review the resulting plan; revise via PlannerAgent if needed.
     6) compose_response.
2. When calling any agent, set the called step status to "in_progress" and
   include prerequisites satisfied by your observation.
3. Only call PlannerAgent if memory.flags_and_assessments has
   "assessment_status" of "assessment_complete". If missing, call
   MedicalAssessmentAgent first.
4. Self-review (after PlannerAgent returns with memory.plans.current_plan
   populated). The deterministic checks — allergy / calorie / macro
   tolerances — have already been enforced inside the Planner, so do NOT
   re-check the math. Focus on:
     a) Medical-flag respect: for each entry in flags_and_assessments.flags
        (e.g. "diabetes_risk", "high_ldl"), confirm the plan does not
        contain foods that contraindicate the flag.
     b) Recommendation coverage: every item in
        flags_and_assessments.recommendations should be reflected in the
        plan or notes; mention any unaddressed one.
     c) Citation presence: clinical claims in the assessment require sources.
        If memory.plans.current_plan.sources is empty AND the assessment
        had medical flags, that is a revision-worthy gap.
     d) Cultural / preference fit: if user_profile.country is set, prefer
        regional foods. Major mismatches are revision-worthy; small ones
        can be noted in compose_response.
   If any (a)-(c) issue is found, call PlannerAgent with task =
     "Revise the plan to address: " + each issue joined by "; ".
   Cap revisions at 2; on the third attempt, compose_response with the
   best plan available and append the unresolved issues as warnings.
5. When new personal data appears in user input, add steps to: propose
   memory update (write_memory), call MedicalAssessmentAgent if needed,
   re-plan if needed.
6. For any write_memory action, provide the full partition contents in
   params.data (not diffs). Top-level partitions are merged; dotted paths
   like "plans.current_plan" replace the leaf wholesale.
7. HITL escalation: when
   flags_and_assessments.requires_professional_consultation is true,
   compose_response MUST append the marker
   "<<HITL:CLINICIAN_REVIEW_REQUIRED>>" on its own line at the end and
   strongly recommend consulting a clinician before following the plan.

Output JSON shape (enforced by schema):
{
  "observation": "...",
  "thought": "...",
  "response_steps": [ ... ],
  "action": "call_agent | ask_user | write_memory | compose_response",
  "params": { ... }
}

Required params per action:
- call_agent:       {"agent_name": "...", "task": "..."}
- ask_user:         {"prompt": "..."}
- write_memory:     {"partition": "...", "data": {...}}
- compose_response: {"text": "...markdown..."}

Composition rules:
- When composing the response, extract relevant information from memory
  state (calorie target, plan details, dietary restrictions, citations)
  in markdown.
- Always include a "Trace" line summarising which agents/tools contributed.
- For high-risk profiles (requires_professional_consultation == true),
  end with the HITL marker on its own line.
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
                    {
                        "role": "assistant",
                        "content": msg["content"][:200] + "... (full response in memory)",
                    }
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
        new_steps = [s.model_dump() for s in decision.response_steps] or state.get(
            "response_steps", []
        )

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
                "params": {
                    "text": f"Sorry — I hit an internal error while planning. ({message})"
                },
                "_parse_error": True,
            },
            "num_turns": state["num_turns"] + 1,
            "agent_result": None,
        }


# ---------------------------------------------------------------------------
# Medical Assessment
# ---------------------------------------------------------------------------
_MEDICAL_SYSTEM_PROMPT = """\
You are the Medical Assessment Agent. You receive a user's profile +
medical history AND a deterministic computation of BMI / BMR / TDEE /
daily_target_calories / macro_targets that has already been run for you.

Your job is to interpret those numbers and emit clinical flags,
recommendations, and evidence sources — NOT to recompute the math.

Available tools: WebSearchTool (optional, for fetching authoritative
clinical guidelines from WHO / USDA / ADA / EFSA / NICE).

Mandatory behaviour:
1. The system has already computed and given you the calculations block.
   Echo it back in result.calculations exactly — do not alter the numbers.
2. Read the user's flags / conditions / medications / lab results and emit:
   - flags_to_set: short stable codes (e.g. "diabetes_risk", "high_ldl",
     "hypertension_risk"). Use stable, lowercase, snake_case strings.
   - recommendations: 3-6 short bullets the Planner can act on.
   - requires_professional_consultation: True for medically sensitive
     cases (HbA1c > 6.5 %, eGFR concerns, severe hypertension, pregnancy,
     active eating disorder, etc.).
3. Optionally call WebSearchTool ONCE to fetch authoritative guidelines
   when the user has a flagged condition. Use tool_task = "..." with a
   focused question. Capture the URLs in evidence_sources.
4. When complete, set action_type="assessment_complete" and populate
   result with assessment_summary + flags_to_set + recommendations +
   requires_professional_consultation + calculations + evidence_sources +
   trace + data_confidence.
5. If a critical field is missing (age, sex, height, weight,
   activity_level), set action_type="ask_user" and list the missing names
   in ``fields``.

Output JSON shape (enforced by schema):
{
  "medical_reasoning": "...",
  "observation": "...",
  "risk_assessment_priorities": [...],
  "assessment_plan": [...],
  "action_type": "call_tool" | "ask_user" | "assessment_complete",
  "tool_task": "..." | null,        // only when calling WebSearchTool
  "fields": [...],                  // only when ask_user
  "result": { ... }                 // only when assessment_complete
}
"""


# Critical anthropometric fields needed before any computation.
_CRITICAL_FIELDS = ("age", "sex", "height", "weight", "activity_level", "goal")


class MedicalAssessmentAgent:
    # 4 iterations is plenty: one LLM call to enrich the assessment, plus
    # an optional WebSearchTool round-trip + recovery iterations.
    MAX_ITERATIONS = 4

    def __init__(self, llm_instance, web_search_tool: WebSearchTool):
        self.llm = llm_instance
        self.web_search_tool = web_search_tool

    def handle_task(self, task: str, memory: Dict[str, Any]) -> str:
        _medical_logger.info("\n👨🏻‍⚕️ MEDICAL ASSESSMENT AGENT STARTED")
        settings = get_settings()

        user_profile = memory.get("user_profile", {}) or {}

        # 1. Precheck — short-circuit before paying for an LLM call if we
        # don't even have the data to compute on.
        missing = [f for f in _CRITICAL_FIELDS if not user_profile.get(f)]
        if missing:
            msg = (
                f"Missing critical fields: {', '.join(missing)}. "
                "Please provide the following information to continue the assessment."
            )
            _medical_logger.info(
                "👨🏻‍⚕️ MEDICAL ASSESSMENT AGENT: User query needed - %s", msg
            )
            return msg

        # 2. Deterministic clinical math (no LLM, no subprocess).
        deterministic_calcs = full_assessment(
            weight_kg=float(user_profile["weight"]),
            height_cm=float(user_profile["height"]),
            age_years=float(user_profile["age"]),
            sex=str(user_profile["sex"]),
            activity_level=str(user_profile["activity_level"]),
            goal=str(user_profile["goal"]),
        )

        relevant_memory = {
            "user_profile": user_profile,
            "medical_history": memory.get("medical_history", {}),
        }
        memory_str = json.dumps(relevant_memory, indent=2, default=str)
        calc_str = json.dumps(deterministic_calcs, indent=2, default=str)
        tool_results: List[str] = []
        assessment_plan: List[dict] = []

        # 3. LLM enrichment loop. Usually one iteration; up to MAX in case
        # the model wants a WebSearchTool round-trip before finalising.
        for iteration in range(self.MAX_ITERATIONS):
            tool_results_str = (
                "\n".join(f"Tool Result {i+1}: {r}" for i, r in enumerate(tool_results))
                or "None"
            )
            assessment_plan_str = (
                json.dumps(assessment_plan, indent=2, default=str)
                if assessment_plan
                else "None"
            )

            prompt = (
                f"{_MEDICAL_SYSTEM_PROMPT}\n\n--- Task & State ---\n"
                f"Task: {task}\n"
                f"Current Memory: {memory_str}\n"
                f"Deterministic calculations (use these EXACTLY in result.calculations): "
                f"{calc_str}\n"
                f"Current Assessment Plan: {assessment_plan_str}\n"
                f"Previous Tool Results: {tool_results_str}\n"
            )

            if should_debug("agents", "MedicalAssessmentAgent"):
                _medical_logger.debug(
                    "--- Medical Assessment Iteration %d ---", iteration + 1
                )
                if settings.debug_level == "full":
                    _medical_logger.debug("Raw LLM input:\n%s", prompt)

            decision = self.llm.call_typed(prompt, MedicalAssessmentDecision)
            if decision is None:
                _medical_logger.error(
                    "Medical decision parse failed at iteration %d", iteration + 1
                )
                # Fall through to the deterministic-only fallback below.
                break

            if should_debug("agents", "MedicalAssessmentAgent"):
                _medical_logger.debug(
                    "Medical decision:\n%s", decision.model_dump_json(indent=2)
                )

            if decision.assessment_plan:
                assessment_plan = [s.model_dump() for s in decision.assessment_plan]

            if not settings.debug_mode:
                self._log_user_mode_action(decision)

            if decision.action_type == "call_tool":
                if not decision.tool_task:
                    tool_results.append("WebSearchTool: missing tool_task")
                    continue
                tool_results.append(
                    f"WebSearchTool: {self.web_search_tool.handle_task(decision.tool_task)}"
                )

            elif decision.action_type == "ask_user":
                fields = decision.fields or []
                msg = (
                    f"Missing critical fields: {', '.join(fields)}. "
                    "Please provide the following information to continue the assessment."
                )
                _medical_logger.info(
                    "👨🏻‍⚕️ MEDICAL ASSESSMENT AGENT: User query needed - %s", msg
                )
                return msg

            elif decision.action_type == "assessment_complete":
                return self._finalize(
                    task,
                    decision,
                    memory,
                    relevant_memory,
                    tool_results,
                    deterministic_calcs,
                )

            else:
                _medical_logger.error("Unknown action_type: %s", decision.action_type)
                break

        # 4. Deterministic-only fallback. Reached when the LLM never emits
        # assessment_complete; we still ship a usable assessment so the
        # Coach can proceed instead of stalling.
        _medical_logger.warning(
            "👨🏻‍⚕️ MEDICAL ASSESSMENT AGENT: LLM never finalised; using "
            "deterministic-only fallback."
        )
        return self._finalize_deterministic_only(
            task, memory, relevant_memory, tool_results, deterministic_calcs
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _log_user_mode_action(decision: MedicalAssessmentDecision) -> None:
        if decision.action_type == "call_tool":
            _medical_logger.info(
                "👨🏻‍⚕️ Medical Assessment Agent: Using WebSearchTool for '%s'",
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
        deterministic_calcs: Dict[str, Any],
    ) -> str:
        result: Optional[MedicalAssessmentResult] = decision.result
        if result is None:
            _medical_logger.error("assessment_complete decision missing result payload")
            return self._finalize_deterministic_only(
                task, memory, relevant_memory, tool_results, deterministic_calcs
            )

        # Overwrite whatever calculations the LLM emitted with the
        # deterministic values. The LLM may have re-derived (or invented)
        # the numbers; only the formula output is allowed to drive the
        # Planner.
        result.calculations = Calculations(
            BMI=deterministic_calcs["BMI"],
            BMR=deterministic_calcs["BMR"],
            TDEE=deterministic_calcs["TDEE"],
            daily_target_calories=deterministic_calcs["daily_target_calories"],
            macro_targets=MacroTargets(**deterministic_calcs["macro_targets"]),
        )

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
        _medical_logger.info(
            "👨🏻‍⚕️ MEDICAL ASSESSMENT AGENT COMPLETED: %s", result.assessment_summary
        )
        return result.assessment_summary

    def _finalize_deterministic_only(
        self,
        task: str,
        memory: Dict[str, Any],
        relevant_memory: Dict[str, Any],
        tool_results: List[str],
        deterministic_calcs: Dict[str, Any],
    ) -> str:
        """Persist a deterministic-only assessment when the LLM bails.

        Lower ``data_confidence`` so the Coach can decide whether to
        proceed or escalate; flags are empty because we have no LLM
        interpretation to attach.
        """
        summary = (
            f"Deterministic assessment only (LLM enrichment unavailable). "
            f"Daily target: {deterministic_calcs['daily_target_calories']} kcal, "
            f"macros: {deterministic_calcs['macro_targets']}."
        )
        update_memory_partition(
            memory,
            "flags_and_assessments",
            {
                "assessment_summary": summary,
                "flags": [],
                "recommendations": [],
                "requires_professional_consultation": False,
                "calculations": deterministic_calcs,
                "evidence_sources": [],
                "data_confidence": 0.6,
                "trace": "deterministic-only fallback (no LLM enrichment)",
                "assessment_status": "assessment_complete",
                "assessment_timestamp": datetime.now().isoformat(),
            },
        )
        save_to_json(
            {
                "task": task,
                "memory_input": relevant_memory,
                "tool_results": tool_results,
                "deterministic_only": True,
                "calculations": deterministic_calcs,
                "timestamp": datetime.now().isoformat(),
            },
            f"medical_assessment_{datetime.now().isoformat()}.json",
            subdirectory="MedicalAssessment",
        )
        return summary


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------
_PLANNER_SYSTEM_PROMPT = """\
You are the Planner Agent. Create personalised meal plans constrained by
the medical assessment.

Available tools: WebSearchTool, QuantitiesFinder.

Mandatory behaviour:
1. Precondition: do NOT plan unless flags_and_assessments has an
   "assessment_status" of "assessment_complete". If missing, return
   action_type="provide_plan" with final_plan={"error": "..."} explaining
   the blocker and suggesting MedicalAssessmentAgent.
2. Batch tool calls: fetch nutrition facts for ALL foods in one
   WebSearchTool call rather than one call per item.
3. For each food in the draft, look up per-100g nutrition (calories,
   protein, fat, carbohydrates). If WebSearchTool fails for >2 items,
   fall back to internal knowledge.
4. Tolerances: calories +/- 3%, each macro +/- 5% of target. These are
   enforced after the QuantitiesFinder solves; if a deviation is flagged
   you will receive "Revision issues" in the prompt and must adjust the
   draft (swap foods, change anchors) before the next solver call.
5. Exclude allergens and disliked foods. Propose alternatives if necessary.
6. Multi-day requests: emit a 1-2 day plan and instruct the user to rotate.
7. QuantitiesFinder format: tool_task MUST be a JSON STRING containing
   {"foods": [...], "targets": {...}}. Each food needs name, calories,
   protein, fat, carbohydrates (per 100g) and estimated_g. Targets MUST
   use the keys calories, protein, fat, carbohydrates (NOT protein_g
   etc.) — the solver validates this strictly.

Planning Steps (used when none provided):
  1. Draft a realistic plan; assign a realistic estimated_g per food.
  2. Batch-gather nutrition facts via WebSearchTool.
  3. Call QuantitiesFinder with foods + targets to compute precise grams.
  4. Update the draft with the solver's quantities.
  5. Provide the final plan via action_type="provide_plan".

Output JSON shape (enforced by schema):
{
  "observation": "...",
  "thought": "...",
  "planning_steps": [...],
  "action_type": "call_tool" | "draft_plan" | "provide_plan",
  "tool_name": "WebSearchTool" | "QuantitiesFinder" | null,
  "tool_task": "..." | null,
  "drafted_plan": { ... } | null,
  "final_plan": { ... } | null
}

Final-plan shape (when action_type="provide_plan"):
- final_plan.days MUST be a non-empty list of fully populated food
  objects (name, meal_group, grams from the solver, calories, protein_g,
  fat_g, carbohydrates_g).
- final_plan.daily_totals mirrors the solver's ``achieved`` block.
- final_plan.sources lists the citation URIs.
- final_plan.trace summarises which agents/tools contributed.
- An empty envelope ({"days": [{}], ...}) is rejected; always transcribe
  the solver output before returning.
"""


# Bounded internal revision count. After this, the agent returns whatever
# it has plus an ``unresolved_issues`` block so the Coach can decide.
_MAX_INTERNAL_REVISIONS = 2


class PlannerAgent:
    MAX_ITERATIONS = 12

    def __init__(
        self,
        llm_instance,
        web_search_tool: WebSearchTool,
        quantities_finder: QuantitiesFinder,
    ):
        self.llm = llm_instance
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
        revision_count = 0
        last_issues: List[Dict[str, Any]] = []

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
                "\n".join(f"Tool Result {i+1}: {r}" for i, r in enumerate(tool_results))
                or "None"
            )
            planning_steps_str = (
                json.dumps(planning_steps, indent=2, default=str)
                if planning_steps
                else "None"
            )
            revision_block = ""
            if last_issues:
                revision_block = (
                    "\n--- Revision issues (must address before next provide_plan) ---\n"
                    + "\n".join(
                        f"- [{i['severity']}] {i['code']}: {i['description']}"
                        for i in last_issues
                    )
                    + f"\nRevision attempt: {revision_count}/{_MAX_INTERNAL_REVISIONS}\n"
                )

            prompt = (
                f"{_PLANNER_SYSTEM_PROMPT}\n\n--- Task & State ---\n"
                f"Task: {task}\n"
                f"Current Memory: {memory_str}\n"
                f"Current Planning Steps: {planning_steps_str}\n"
                f"Previous Tool Results: {tool_results_str}"
                f"{revision_block}"
            )

            if should_debug("agents", "PlannerAgent"):
                _planner_logger.debug("--- Planner Iteration %d ---", iteration + 1)
                if settings.debug_level == "full":
                    _planner_logger.debug("Raw LLM input:\n%s", prompt)

            decision = self.llm.call_typed(prompt, PlannerDecision)
            if decision is None:
                _planner_logger.error(
                    "Planner decision parse failed at iteration %d", iteration + 1
                )
                return "Planner failed: could not parse LLM decision."

            if should_debug("agents", "PlannerAgent"):
                _planner_logger.debug(
                    "Planner decision:\n%s", decision.model_dump_json(indent=2)
                )

            if decision.planning_steps:
                planning_steps = [s.model_dump() for s in decision.planning_steps]

            if not settings.debug_mode:
                self._log_user_mode_action(decision)

            if decision.action_type == "call_tool":
                tool_results.append(f"{decision.tool_name}: {self._dispatch_tool(decision)}")

            elif decision.action_type == "draft_plan":
                drafted = _decode_plan_field(decision.drafted_plan)
                if drafted is not None:
                    memory.setdefault("plans", {})["drafted_plan"] = drafted
                    tool_results.append("Plan drafted and stored in memory")
                else:
                    tool_results.append("Drafted plan not provided")

            elif decision.action_type == "provide_plan":
                final = _decode_plan_field(decision.final_plan) or memory.get(
                    "plans", {}
                ).get("drafted_plan")

                # Error escape hatch (e.g. precondition not met). The
                # schema hint advertises an optional ``error`` key, so the
                # LLM often emits ``"error": ""`` alongside a valid plan;
                # only treat it as a real error when the value is truthy.
                if isinstance(final, dict) and final.get("error"):
                    _planner_logger.error("📋 PLANNER AGENT ERROR: %s", final["error"])
                    return json.dumps({"error": final["error"]})

                if not final:
                    tool_results.append("Cannot finalize: missing plan")
                    continue

                if _is_plan_empty(final):
                    tool_results.append(
                        "final_plan.days had no food entries. Transcribe the "
                        "QuantitiesFinder quantities into days[] as fully "
                        "populated food objects and emit provide_plan again."
                    )
                    _planner_logger.warning(
                        "📋 Planner Agent: empty final_plan rejected; re-iterating."
                    )
                    continue

                # Post-LP deterministic check.
                issues = check_plan(final, memory)
                blocking = [i for i in issues if i["severity"] in {"medium", "high"}]

                if blocking and revision_count < _MAX_INTERNAL_REVISIONS:
                    revision_count += 1
                    last_issues = issues
                    _planner_logger.info(
                        "📋 Planner Agent: revising (attempt %d/%d) — %d blocking issue(s).",
                        revision_count,
                        _MAX_INTERNAL_REVISIONS,
                        len(blocking),
                    )
                    continue

                # Either no blocking issues, or we've exhausted internal
                # revisions and must return what we have.
                memory.setdefault("plans", {})
                memory["plans"]["current_plan"] = final
                memory["plans"]["plan_timestamp"] = datetime.now().isoformat()
                memory["plans"]["revision_count"] = revision_count
                memory["plans"]["post_lp_issues"] = issues
                memory["plans"].pop("drafted_plan", None)

                unresolved = [i for i in issues if i["severity"] == "high"]
                envelope = {
                    "plan": final,
                    "revisions": revision_count,
                    "unresolved_issues": unresolved,
                }

                save_to_json(
                    {
                        "task": task,
                        "memory_input": relevant_memory,
                        "tool_results": tool_results,
                        "final_response": decision.model_dump(),
                        "post_lp_issues": issues,
                        "revision_count": revision_count,
                        "timestamp": datetime.now().isoformat(),
                    },
                    f"planner_agent_{datetime.now().isoformat()}.json",
                    subdirectory="PlannerAgent",
                )
                _planner_logger.info("\n📋 PLANNER AGENT COMPLETED")
                return json.dumps(envelope)

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


__all__ = [
    "CoachAgent",
    "MedicalAssessmentAgent",
    "PlannerAgent",
    "check_plan",
    "CALORIE_TOLERANCE",
    "MACRO_TOLERANCE",
]
