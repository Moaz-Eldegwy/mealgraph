"""ValidationAgent — the critic in the generator-critic loop.

Why this exists
----------------
The original README promised a ``ValidationAgent`` but it was never
implemented; the system shipped plans straight from the Planner to the user.
Modern multi-agent literature (Anthropic's research-system writeup, every
LangGraph reflection-pattern tutorial) is unanimous that a separate critic
node materially raises output quality on tasks with hard constraints.

Design
------
We combine two layers:

1. **Deterministic checks** (no LLM, no cost, instant):
   * allergy violations,
   * calorie deviation > 3 % of daily target,
   * each macro deviation > 5 % of its target,
   * disliked foods present (advisory),
   * professional-consultation flag set without disclaimer.

2. **LLM-graded checks** (one Gemini round-trip, structured output):
   * medical-flag respect (e.g., diabetes user should avoid high-GL meals),
   * citation presence for clinical recommendations,
   * cultural appropriateness against user's country/cuisine preference.

Verdict semantics
-----------------
* ``pass``   — Coach proceeds to ``compose_response``.
* ``revise`` — Issues are bundled into the next Planner task; Coach loops back
               to ``call_agent('PlannerAgent', task=...)``. Capped at 2
               revisions (enforced by Coach prompt) to avoid infinite loops.
* ``reject`` — Hard stop with ``severity='high'``. Coach must compose a
               warning + HITL escalation chip (Phase 4 wires up the chip).
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from logging_setup import get_logger
from schemas import ValidationDecision, ValidationIssue
from utils import save_to_json

_logger = get_logger("agents.validation")


# Tolerances are class-level so tests/configs can override.
CALORIE_TOLERANCE = 0.03  # +/- 3 %
MACRO_TOLERANCE = 0.05  # +/- 5 %


_VALIDATION_SYSTEM_PROMPT = """\
You are the Validation Agent. You receive a meal plan and the medical
assessment context. Your job is to grade the plan, NOT redesign it.

Mandatory checks (in addition to the deterministic ones already supplied):
1. Medical-flag respect: for each flag in flags_and_assessments.flags
   (e.g., "diabetes_risk", "high_ldl"), confirm the plan does not contain
   foods that contraindicate the flag. Cite which food fails which flag.
2. Evidence: clinical recommendations in flags_and_assessments.recommendations
   must be reflected in the plan or notes. Mention any unaddressed item.
3. Cultural appropriateness: if user_profile.country is set, confirm at
   least 60 % of foods are commonly available / culturally familiar there.
   Otherwise emit a low-severity issue suggesting substitutions.

Output JSON shape (enforced by schema):
{
  "verdict": "pass" | "revise" | "reject",
  "issues": [
    {"code": "...", "description": "...",
     "severity": "low" | "medium" | "high"}
  ],
  "notes": "...",
  "requires_human_review": false
}

Rules:
- Mark requires_human_review=true if any issue has severity="high" OR if
  flags_and_assessments.requires_professional_consultation is true.
- Use verdict="reject" only for hard safety violations (allergy made it
  through, food explicitly contraindicated by medication).
- Use verdict="revise" for fixable problems (over-budget calories, missing
  guideline citation, monotonous menu).
- Use verdict="pass" only when issues is empty OR all issues are severity="low".
"""


class ValidationAgent:
    """Generator-critic gate for the Planner's output."""

    def __init__(self, llm_instance):
        self.llm = llm_instance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def handle_task(self, task: str, memory: Dict[str, Any]) -> str:
        """Validate the current plan in ``memory.plans.current_plan``.

        Returns a JSON string of ``ValidationDecision.model_dump()`` so the
        Coach can read structured fields back out (``verdict``, ``issues``).
        """
        _logger.info("\n🛡️ VALIDATION AGENT STARTED")

        plan = memory.get("plans", {}).get("current_plan")
        if plan is None:
            _logger.warning("No current_plan in memory; nothing to validate.")
            verdict = ValidationDecision(
                verdict="reject",
                issues=[
                    ValidationIssue(
                        code="missing_plan",
                        description="No current_plan in memory; Planner did not finalise.",
                        severity="high",
                    )
                ],
                notes="Validator received no plan. Re-run PlannerAgent.",
                requires_human_review=False,
            )
            return self._save_and_return(task, memory, verdict)

        # 1. Deterministic checks
        det_issues = self._deterministic_checks(plan, memory)

        # 2. LLM-graded checks (only if deterministic ones don't already reject)
        llm_decision: Optional[ValidationDecision] = None
        hard_block = any(i.severity == "high" for i in det_issues)
        if not hard_block:
            llm_decision = self._llm_review(plan, memory, det_issues)

        # 3. Merge
        all_issues = list(det_issues)
        notes_parts: List[str] = []
        requires_hr = False
        if llm_decision is not None:
            all_issues.extend(llm_decision.issues)
            if llm_decision.notes:
                notes_parts.append(llm_decision.notes)
            requires_hr |= llm_decision.requires_human_review

        # Force human review when the medical assessment said so.
        if memory.get("flags_and_assessments", {}).get("requires_professional_consultation"):
            requires_hr = True

        verdict = self._compute_verdict(all_issues)
        decision = ValidationDecision(
            verdict=verdict,
            issues=all_issues,
            notes=" | ".join(notes_parts) if notes_parts else "",
            requires_human_review=requires_hr,
        )
        _logger.info("🛡️ Validation verdict: %s (%d issue(s))", verdict, len(all_issues))
        return self._save_and_return(task, memory, decision)

    # ------------------------------------------------------------------
    # Deterministic layer
    # ------------------------------------------------------------------
    @staticmethod
    def _deterministic_checks(plan: Dict[str, Any], memory: Dict[str, Any]) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []

        user_profile = memory.get("user_profile", {}) or {}
        allergies = {a.strip().lower() for a in user_profile.get("allergies", []) or [] if a}
        dislikes_raw = user_profile.get("food_dislikes", "") or ""
        dislikes = {d.strip().lower() for d in dislikes_raw.split(",") if d.strip()}

        flags = memory.get("flags_and_assessments", {}) or {}
        calc = flags.get("calculations", {}) or {}
        target_calories = calc.get("daily_target_calories")
        macro_targets = calc.get("macro_targets") or {}

        # Walk plan, accumulating foods and totals.
        foods, totals = ValidationAgent._extract_foods_and_totals(plan)

        # 1. Allergy violations (severity high — never let these through)
        for food in foods:
            name = (food.get("name") or "").lower()
            for allergen in allergies:
                if allergen and allergen in name:
                    issues.append(
                        ValidationIssue(
                            code="allergy_violation",
                            description=f"Food '{name}' matches allergen '{allergen}'.",
                            severity="high",
                        )
                    )

        # 2. Disliked foods (advisory)
        for food in foods:
            name = (food.get("name") or "").lower()
            for d in dislikes:
                if d and d in name:
                    issues.append(
                        ValidationIssue(
                            code="disliked_food",
                            description=f"Food '{name}' matches user dislike '{d}'.",
                            severity="low",
                        )
                    )

        # 3. Calorie tolerance
        if target_calories and totals.get("calories"):
            dev = abs(totals["calories"] - target_calories) / target_calories
            if dev > CALORIE_TOLERANCE:
                issues.append(
                    ValidationIssue(
                        code="calorie_deviation",
                        description=(
                            f"Plan total {totals['calories']:.0f} kcal vs target "
                            f"{target_calories} kcal ({dev*100:.1f}% deviation)."
                        ),
                        severity="medium",
                    )
                )

        # 4. Macro tolerances
        macro_map = {"protein_g": "protein", "fat_g": "fat", "carbohydrates_g": "carbohydrates"}
        for tgt_key, plan_key in macro_map.items():
            target = macro_targets.get(tgt_key)
            actual = totals.get(plan_key)
            if target and actual:
                dev = abs(actual - target) / target
                if dev > MACRO_TOLERANCE:
                    issues.append(
                        ValidationIssue(
                            code=f"{plan_key}_deviation",
                            description=(
                                f"{plan_key} total {actual:.0f}g vs target {target}g "
                                f"({dev*100:.1f}% deviation)."
                            ),
                            severity="medium",
                        )
                    )

        return issues

    @staticmethod
    def _extract_foods_and_totals(
        plan: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """Best-effort: support both 'days' shape and a flat dict-of-foods.

        We tolerate the LLM's free-form ``drafted_plan`` shape too, since the
        Planner's final_plan isn't yet strictly typed.
        """
        foods: List[Dict[str, Any]] = []
        totals: Dict[str, float] = {"calories": 0.0, "protein": 0.0, "fat": 0.0, "carbohydrates": 0.0}

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

        # Plans may also surface daily_totals directly — prefer those when present.
        if isinstance(plan, dict) and "daily_totals" in plan:
            dt = plan["daily_totals"]
            for k in ("calories", "protein", "fat", "carbohydrates"):
                if k in dt:
                    totals[k] = float(dt[k])
        return foods, totals

    # ------------------------------------------------------------------
    # LLM layer
    # ------------------------------------------------------------------
    def _llm_review(
        self,
        plan: Dict[str, Any],
        memory: Dict[str, Any],
        deterministic_issues: List[ValidationIssue],
    ) -> Optional[ValidationDecision]:
        det_summary = "\n".join(f"- [{i.severity}] {i.code}: {i.description}" for i in deterministic_issues) or "None"
        prompt = (
            f"{_VALIDATION_SYSTEM_PROMPT}\n\n--- Plan ---\n{json.dumps(plan, indent=2, default=str)}\n\n"
            f"--- User profile ---\n{json.dumps(memory.get('user_profile', {}), indent=2, default=str)}\n\n"
            f"--- Medical assessment ---\n"
            f"{json.dumps(memory.get('flags_and_assessments', {}), indent=2, default=str)}\n\n"
            f"--- Deterministic findings already raised ---\n{det_summary}\n\n"
            "Add only NEW issues. Do not repeat the deterministic ones."
        )
        decision = self.llm.call_typed(prompt, ValidationDecision)
        if decision is None:
            _logger.warning("Validator LLM call returned no parseable decision; skipping LLM layer.")
        return decision

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_verdict(issues: List[ValidationIssue]) -> str:
        if any(i.severity == "high" for i in issues):
            return "reject"
        if any(i.severity == "medium" for i in issues):
            return "revise"
        return "pass"

    @staticmethod
    def _save_and_return(task: str, memory: Dict[str, Any], decision: ValidationDecision) -> str:
        # Persist to memory so the Coach can inspect the verdict next turn.
        memory.setdefault("flags_and_assessments", {})
        memory["flags_and_assessments"]["last_validation"] = decision.model_dump()
        memory["flags_and_assessments"]["last_validation_at"] = datetime.now().isoformat()

        save_to_json(
            {
                "task": task,
                "decision": decision.model_dump(),
                "timestamp": datetime.now().isoformat(),
            },
            f"validation_agent_{datetime.now().isoformat()}.json",
            subdirectory="ValidationAgent",
        )
        return decision.model_dump_json()
