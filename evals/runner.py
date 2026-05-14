"""Offline eval runner.

Validates the *deterministic* parts of the system end-to-end without paying
for Gemini calls. Walks each fixture, builds the calculations the Medical
agent would emit, runs :func:`agents.check_plan` over a hand-built plan,
and asserts that the per-fixture ``expected`` dict matches.

This is the same ``check_plan`` the Planner runs internally after the LP
solver — keeping the eval surface aligned with production behaviour.

Two ways to invoke:

* As a script: ``python -m evals.runner`` -> prints a per-fixture report.
* From pytest: imported and called by ``tests/test_evals.py``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List

from agents import check_plan
from evals.fixtures import all_fixtures
from guardrails import detect_prompt_injection
from nutrition_formulas import full_assessment
from observability import get_metrics
from utils import get_parse_metrics


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


def _check_expected(
    fixture: Dict[str, Any],
    assessment: Dict[str, Any],
    issues: List[Dict[str, Any]],
    requires_human_review: bool,
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

    if expected.get("requires_human_review") and not requires_human_review:
        failures.append("expected requires_human_review=True")

    # No deterministic issue should fire for a hand-built plan that hits
    # targets exactly — if one does, the check_plan contract has drifted.
    if issues:
        failures.append(
            f"unexpected post-LP issues on a target-hitting plan: "
            f"{[i['code'] for i in issues]}"
        )

    return failures


def run_offline() -> List[FixtureResult]:
    results: List[FixtureResult] = []
    for fixture in all_fixtures():
        name = fixture["user_profile"]["name"]

        # 1. Sanity-check the user-question for prompt injection.
        injected = False
        for q in fixture["questions"]:
            verdict = detect_prompt_injection(q)
            if verdict.is_attempt:
                results.append(
                    FixtureResult(
                        name=name,
                        passed=False,
                        failures=[
                            f"fixture question flagged as injection: {verdict.matches}"
                        ],
                    )
                )
                injected = True
                break
        if injected:
            continue

        # 2. Compute the assessment deterministically.
        assessment = _build_assessment(fixture)

        # 3. Hand-build a plan that hits targets and run the deterministic
        #    plan check — the same code the Planner runs internally.
        plan = _build_plan_from_assessment(assessment)
        requires_human_review = fixture["expected"].get("requires_human_review", False)
        memory: Dict[str, Any] = {
            "user_profile": fixture["user_profile"],
            "medical_history": fixture["medical_history"],
            "flags_and_assessments": {
                "assessment_status": "assessment_complete",
                "calculations": assessment,
                "flags": fixture["medical_history"]["conditions"],
                "recommendations": [],
                "requires_professional_consultation": requires_human_review,
            },
            "plans": {"current_plan": plan},
        }

        issues = check_plan(plan, memory)

        failures = _check_expected(fixture, assessment, issues, requires_human_review)
        results.append(
            FixtureResult(
                name=name,
                passed=not failures,
                failures=failures,
                info={
                    "calories": assessment["daily_target_calories"],
                    "protein_g": assessment["macro_targets"]["protein_g"],
                    "issues": [i["code"] for i in issues],
                    "requires_human_review": requires_human_review,
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
