"""Fixture users for the eval harness.

Each fixture is a (user_profile, medical_history, questions, expected) tuple.
Three personas exercise the main risk axes:

* ``athlete``  — gain-muscle goal, high TDEE; tests macro_split protein floor.
* ``diabetic`` — high LDL + diabetes flag; tests medical guideline adherence.
* ``vegan_budget`` — vegan + low-budget; tests allergy/dislike handling AND
  the cultural-appropriateness branch of the Validator.

The expected dict is what the system *should* produce; the eval runner uses
it to compute a pass/fail per metric.
"""

from __future__ import annotations

from typing import Any, Dict, List

ATHLETE = {
    "user_profile": {
        "name": "Athlete Test",
        "age": 28,
        "sex": "male",
        "height": 180,
        "weight": 78,
        "activity_level": "very active",
        "goal": "gain muscle",
        "job": "personal trainer",
        "dietary_restrictions": "",
        "food_likes": "chicken, rice, eggs",
        "food_dislikes": "",
        "allergies": [],
        "country": "USA",
        "currency": "USD",
    },
    "medical_history": {
        "conditions": [],
        "medications": [],
        "past_issues": [],
        "lab_results": "all within range",
    },
    "questions": ["Build me a one-day meal plan for muscle gain."],
    "expected": {
        "min_calories": 2900,
        "max_calories": 3600,
        "min_protein_g": int(1.6 * 78),
        "must_be_uncited_safe": True,  # athlete: no medical flag, citation optional
    },
}

DIABETIC = {
    "user_profile": {
        "name": "Diabetic Test",
        "age": 55,
        "sex": "female",
        "height": 162,
        "weight": 82,
        "activity_level": "lightly active",
        "goal": "lose weight",
        "job": "office worker",
        "dietary_restrictions": "low carbohydrate",
        "food_likes": "vegetables, fish",
        "food_dislikes": "white bread",
        "allergies": [],
        "country": "Egypt",
        "currency": "EGP",
    },
    "medical_history": {
        "conditions": ["type 2 diabetes", "hypertension"],
        "medications": ["metformin", "lisinopril"],
        "past_issues": ["high LDL last year"],
        "lab_results": "HbA1c 7.4, LDL 145",
    },
    "questions": ["Plan a one-day diet that respects my diabetes and BP medication."],
    "expected": {
        "max_calories": 2000,  # weight-loss + sedentary woman
        "must_have_assessment_flag": True,
        "must_have_citation": True,  # clinical claims require sources
        "requires_human_review": True,
    },
}

VEGAN_BUDGET = {
    "user_profile": {
        "name": "Vegan Budget Test",
        "age": 32,
        "sex": "female",
        "height": 170,
        "weight": 60,
        "activity_level": "moderately active",
        "goal": "maintain weight",
        "job": "graphic designer",
        "dietary_restrictions": "vegan",
        "food_likes": "lentils, tofu, oats",
        "food_dislikes": "okra",
        "allergies": ["peanut"],
        "country": "India",
        "currency": "INR",
    },
    "medical_history": {
        "conditions": [],
        "medications": [],
        "past_issues": [],
        "lab_results": "borderline low B12",
    },
    "questions": ["Make me a one-day vegan plan on a tight budget."],
    "expected": {
        "must_exclude": ["peanut", "chicken", "milk", "egg"],
        "must_warn_about": "B12",
        "regional_bias": "India",
    },
}


def all_fixtures() -> List[Dict[str, Any]]:
    return [ATHLETE, DIABETIC, VEGAN_BUDGET]


__all__ = ["ATHLETE", "DIABETIC", "VEGAN_BUDGET", "all_fixtures"]
