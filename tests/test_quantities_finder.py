"""Smoke tests for the PuLP-backed QuantitiesFinder.

Pure deterministic tool — no LLM, no network. Should be the fastest test in
the suite and the one we trust most.
"""

from __future__ import annotations

import json

from tools import QuantitiesFinder


def test_basic_two_food_balance() -> None:
    """Two foods, simple targets — solver must find quantities within a few
    percent of the target."""
    qf = QuantitiesFinder()
    payload = {
        "foods": [
            {
                "name": "chicken_breast",
                "calories": 165,
                "protein": 31,
                "fat": 3.6,
                "carbohydrates": 0,
                "estimated_g": 200,
            },
            {
                "name": "rice_cooked",
                "calories": 130,
                "protein": 2.7,
                "fat": 0.3,
                "carbohydrates": 28,
                "estimated_g": 200,
            },
        ],
        "targets": {"calories": 700, "protein": 65, "fat": 8, "carbohydrates": 60},
    }
    result = json.loads(qf.handle_task(json.dumps(payload)))
    assert "quantities" in result and "achieved" in result, f"Bad shape: {result}"

    achieved = result["achieved"]
    # The solver minimises weighted deviation across all 4 nutrients. With only
    # two foods (chicken and rice) it cannot hit every target tightly — it will
    # nail fat/carbs (constrained by rice) and trade off calories/protein.
    # We assert it lands within 20% of every target, which is the realistic
    # feasibility envelope for a 2-food problem.
    for nut, target in [("calories", 700), ("protein", 65), ("fat", 8), ("carbohydrates", 60)]:
        deviation = abs(achieved[nut] - target) / target
        assert deviation < 0.20, f"{nut} achieved={achieved[nut]} target={target} dev={deviation:.2%}"


def test_invalid_payload_returns_error() -> None:
    qf = QuantitiesFinder()
    bad = {"foods": [{"name": "x"}], "targets": {}}  # missing required keys
    result = json.loads(qf.handle_task(json.dumps(bad)))
    assert "error" in result, f"Expected an error key, got {result}"


def test_min_max_bounds_respected() -> None:
    qf = QuantitiesFinder()
    payload = {
        "foods": [
            {
                "name": "egg",
                "calories": 155,
                "protein": 13,
                "fat": 11,
                "carbohydrates": 1.1,
                "estimated_g": 100,
                "min_g": 50,
                "max_g": 120,
            },
            {
                "name": "oats",
                "calories": 389,
                "protein": 17,
                "fat": 7,
                "carbohydrates": 66,
                "estimated_g": 80,
                "min_g": 30,
                "max_g": 150,
            },
        ],
        "targets": {"calories": 500, "protein": 25, "fat": 15, "carbohydrates": 50},
    }
    result = json.loads(qf.handle_task(json.dumps(payload)))
    qty = result["quantities"]
    assert 50 <= qty["egg"] <= 120
    assert 30 <= qty["oats"] <= 150
