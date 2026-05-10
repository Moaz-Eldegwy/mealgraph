"""Closed-form nutrition formulas.

Pure-Python deterministic calculations for BMI, BMR (Mifflin-St Jeor), TDEE,
calorie targets, and macro splits. Used by the Phase 4 ``ComputationTool``
to replace the previous LLM-generated ``subprocess.run(['python', ...])``
path — closing the remote-code-execution vector.

Numbers come from standard clinical guidelines (ACSM activity multipliers,
Mifflin-St Jeor 1990 BMR equation). Conservative defaults; the Validator
will still grade the resulting plan.
"""

from __future__ import annotations

from typing import Dict, Literal


Sex = Literal["male", "female"]
ActivityLevel = Literal[
    "sedentary",
    "lightly active",
    "moderately active",
    "very active",
    "extra active",
]
Goal = Literal["lose weight", "maintain weight", "gain muscle", "gain weight"]


_ACTIVITY_MULTIPLIERS: Dict[str, float] = {
    "sedentary": 1.2,
    "lightly active": 1.375,
    "moderately active": 1.55,
    "very active": 1.725,
    "extra active": 1.9,
}


def bmi(weight_kg: float, height_cm: float) -> float:
    """Body Mass Index. Returns kg/m²."""
    if weight_kg <= 0 or height_cm <= 0:
        raise ValueError("weight_kg and height_cm must be positive.")
    height_m = height_cm / 100.0
    return weight_kg / (height_m * height_m)


def bmr_mifflin_st_jeor(weight_kg: float, height_cm: float, age_years: float, sex: str) -> float:
    """Basal Metabolic Rate via Mifflin-St Jeor (1990).

    Male:   10*kg + 6.25*cm - 5*age + 5
    Female: 10*kg + 6.25*cm - 5*age - 161
    """
    sex_n = sex.strip().lower()
    if sex_n not in ("male", "female"):
        raise ValueError(f"Sex must be 'male' or 'female', got {sex!r}.")
    if weight_kg <= 0 or height_cm <= 0 or age_years <= 0:
        raise ValueError("weight, height, and age must be positive.")
    base = 10 * weight_kg + 6.25 * height_cm - 5 * age_years
    return base + (5 if sex_n == "male" else -161)


def tdee(bmr_value: float, activity_level: str) -> float:
    """Total Daily Energy Expenditure = BMR * activity multiplier."""
    if bmr_value <= 0:
        raise ValueError("bmr_value must be positive.")
    key = activity_level.strip().lower()
    if key not in _ACTIVITY_MULTIPLIERS:
        raise ValueError(
            f"Unknown activity_level {activity_level!r}. "
            f"Valid: {list(_ACTIVITY_MULTIPLIERS)}"
        )
    return bmr_value * _ACTIVITY_MULTIPLIERS[key]


def daily_calorie_target(tdee_value: float, goal: str) -> int:
    """Apply a goal-driven adjustment to TDEE.

    * lose weight  -> TDEE − 500 kcal (≈ 0.45 kg/week deficit)
    * maintain     -> TDEE
    * gain muscle  -> TDEE + 300 kcal (lean surplus)
    * gain weight  -> TDEE + 500 kcal
    """
    g = goal.strip().lower()
    if g.startswith("lose"):
        return int(round(tdee_value - 500))
    if g.startswith("gain muscle"):
        return int(round(tdee_value + 300))
    if g.startswith("gain"):
        return int(round(tdee_value + 500))
    if g.startswith("maintain"):
        return int(round(tdee_value))
    raise ValueError(
        "goal must be 'lose weight', 'maintain weight', 'gain muscle' or 'gain weight'."
    )


def macro_split(
    daily_target_calories: int,
    goal: str,
    weight_kg: float | None = None,
) -> Dict[str, int]:
    """Return ``{"protein_g", "fat_g", "carbohydrates_g"}`` for the day.

    Uses goal-aware percentages. When ``weight_kg`` is supplied, protein is
    floor-capped at 1.6 g/kg for "gain muscle" / 1.2 g/kg otherwise — covers
    the common case where percentage-based split underweights protein for
    smaller folks on muscle-gain goals.
    """
    g = goal.strip().lower()
    # (protein%, fat%, carb%) by goal — sum to 1.0
    if g.startswith("lose"):
        pct = (0.30, 0.30, 0.40)
    elif g.startswith("gain muscle"):
        pct = (0.30, 0.25, 0.45)
    elif g.startswith("gain"):
        pct = (0.25, 0.25, 0.50)
    elif g.startswith("maintain"):
        pct = (0.25, 0.30, 0.45)
    else:
        raise ValueError("goal must be lose/maintain/gain muscle/gain weight.")

    protein_kcal = daily_target_calories * pct[0]
    fat_kcal = daily_target_calories * pct[1]
    carb_kcal = daily_target_calories * pct[2]

    protein_g = protein_kcal / 4.0
    fat_g = fat_kcal / 9.0
    carbs_g = carb_kcal / 4.0

    if weight_kg and weight_kg > 0:
        floor = 1.6 * weight_kg if g.startswith("gain muscle") else 1.2 * weight_kg
        if protein_g < floor:
            protein_g = floor

    return {
        "protein_g": int(round(protein_g)),
        "fat_g": int(round(fat_g)),
        "carbohydrates_g": int(round(carbs_g)),
    }


def full_assessment(
    weight_kg: float,
    height_cm: float,
    age_years: float,
    sex: str,
    activity_level: str,
    goal: str,
) -> Dict[str, object]:
    """Convenience wrapper returning every standard derived value at once."""
    b = bmi(weight_kg, height_cm)
    bmr_v = bmr_mifflin_st_jeor(weight_kg, height_cm, age_years, sex)
    tdee_v = tdee(bmr_v, activity_level)
    cals = daily_calorie_target(tdee_v, goal)
    macros = macro_split(cals, goal, weight_kg=weight_kg)
    return {
        "BMI": round(b, 2),
        "BMR": round(bmr_v, 1),
        "TDEE": round(tdee_v, 1),
        "daily_target_calories": cals,
        "macro_targets": macros,
    }


__all__ = [
    "bmi",
    "bmr_mifflin_st_jeor",
    "tdee",
    "daily_calorie_target",
    "macro_split",
    "full_assessment",
]
