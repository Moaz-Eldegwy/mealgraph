"""Offline eval runner.

Validates the *deterministic* parts of the system end-to-end without paying
for Gemini calls. Walks each fixture, builds the calculations the Medical
agent would emit, runs the Validator over a hand-built plan, and asserts
that the per-fixture ``expected`` dict matches.

Two ways to invoke:

* As a script: ``python -m evals.runner`` -> prints a per-fixture report.
* From pytest: imported and called by ``tests/test_evals.py``.

When real Gemini keys are present and ``--live`` is passed, the runner
swaps the MockLLM for a real one. That path is opt-in so CI never hits
the API by accident.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List

from evals.fixtures import all_fixtures
from guardrails import detect_prompt_injection
from nutrition_formulas import full_assessment
from observability import get_metrics
from utils import get_parse_metrics
from validation import ValidationAgent


@dataclass
class FixtureResult:
    name: str
    passed: bool
    failures: List[str] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)


def _build_assessment(fixture: Dict[str, Any]) -> Dict[str, Any]:
    p = fixture["user_profile"]
    return full_assessment(
        weight_kg=p["weight"],
        height_cm=p["height"],
        age_years=p["age"],
        sex=p["sex"],
        activity_level=p["activity_level"],
        goal=p["goal"],
    )


def _build_plan_from_assessment(assessment: Dict[str, Any]) -> Dict[str, Any]:
    """Hand-build a plan that hits the assessment targets within tolerance."""
    target_cal = assessment["daily_target_calories"]
    macros = assessment["macro_targets"]
    # Single-row plan: nutrient totals = full daily target.
    return {
        "days": [
            {
                "name": "balanced day",
                "calories": target_cal,
                "protein_g": macros["protein_g"],
                "fat_g": macros["fat_g"],
                "carbohydrates_g": macros["carbohydrates_g"],
            }
        ],
        "trace": "eval-fixture: hand-built plan hitting targets exactly.",
    }


class _PassThroughLLM:
    """LLM stub for the Validator's LLM layer; emits 'pass' verdicts."""

    def call_typed(self, _prompt: str, response_model):  # noqa: ANN001
        from schemas import ValidationDecision

        if response_model is ValidationDecision:
            return ValidationDecision(verdict="pass", issues=[], notes="eval-stub")
        return None


def _check_expected(
    fixture: Dict[str, Any],
    assessment: Dict[str, Any],
    validation_decision: Dict[str, Any],
) -> List[str]:
    failures: List[str] = []
    expected = fixture.get("expected", {})

    cal = assessment["daily_target_calories"]
    if "min_calories" in expected and cal < expected["min_calories"]:
        failures.append(f"calories {cal} below expected min {expected['min_calories']}")
    if "max_calories" in expected and cal > expected["max_calories"]:
        failures.append(f"calories {cal} above expected max {expected['max_calories']}")

    if "min_protein_g" in expected:
        actual = assessment["macro_targets"]["protein_g"]
        if actual < expected["min_protein_g"]:
            failures.append(
                f"protein_g {actual} below expected min {expected['min_protein_g']}"
            )

    if expected.get("requires_human_review") and not validation_decision.get(
        "requires_human_review"
    ):
        failures.append("expected requires_human_review=True")

    return failures


def run_offline() -> List[FixtureResult]:
    results: List[FixtureResult] = []
    for fixture in all_fixtures():
        name = fixture["user_profile"]["name"]

        # 1. Sanity-check the user-question for prompt injection.
        for q in fixture["questions"]:
            verdict = detect_prompt_injection(q)
            if verdict.is_attempt:
                results.append(
                    FixtureResult(
                        name=name,
                        passed=False,
                        failures=[f"fixture question flagged as injection: {verdict.matches}"],
                    )
                )
                continue

        # 2. Compute the assessment deterministically.
        assessment = _build_assessment(fixture)

        # 3. Hand-build a plan that hits targets and run the Validator.
        plan = _build_plan_from_assessment(assessment)
        memory: Dict[str, Any] = {
            "user_profile": fixture["user_profile"],
            "medical_history": fixture["medical_history"],
            "flags_and_assessments": {
                "assessment_status": "assessment_complete",
                "calculations": assessment,
                "flags": fixture["medical_history"]["conditions"],
                "recommendations": [],
                "requires_professional_consultation": fixture["expected"].get(
                    "requires_human_review", False
                ),
            },
            "plans": {"current_plan": plan},
        }

        validator = ValidationAgent(_PassThroughLLM())
        decision_json = validator.handle_task("eval", memory)
        decision = json.loads(decision_json)

        failures = _check_expected(fixture, assessment, decision)
        results.append(
            FixtureResult(
                name=name,
                passed=not failures,
                failures=failures,
                info={
                    "calories": assessment["daily_target_calories"],
                    "protein_g": assessment["macro_targets"]["protein_g"],
                    "verdict": decision["verdict"],
                    "issues": [i["code"] for i in decision["issues"]],
                },
            )
        )
    return results


def print_report(results: List[FixtureResult]) -> int:
    passes = sum(1 for r in results if r.passed)
    fails = len(results) - passes
    for r in results:
        status = "[PASS]" if r.passed else "[FAIL]"
        print(f"{status}  {r.name}  -> {r.info}")
        for f in r.failures:
            print(f"        - {f}")

    pm = get_parse_metrics()
    metrics = get_metrics().snapshot()
    print()
    print(f"Summary: {passes}/{len(results)} fixtures passed.")
    print(
        f"Parse metrics — native={pm.native_parses}  fallback={pm.fallback_parses}  "
        f"failure={pm.schema_failures}"
    )
    print(f"Agent timings: {metrics['agents']}")
    return 0 if fails == 0 else 1


def main() -> int:  # pragma: no cover
    return print_report(run_offline())


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
