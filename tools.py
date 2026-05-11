"""Tools layer.

Phase 1: prints -> namespaced loggers; settings via :func:`config.get_settings`.

Phase 4 (safety):
    * :class:`ComputationTool` no longer shells out to ``subprocess.run(['python', ...])``.
      It now dispatches to closed-form clinical formulas (``nutrition_formulas``)
      for the standard cases (BMI / BMR / TDEE / calorie target / macro split)
      and falls back to an AST-restricted math evaluator for arbitrary numeric
      expressions. **Closes the remote-code-execution vector.**
    * No subprocess, no eval, no exec, no file/network access from the tool.

Phase 7+:
    * :class:`WebSearchTool` is now a thin wrapper around Gemini's built-in
      ``google_search`` grounding. Gemini generates queries, runs them, and
      synthesises a cited answer in one round-trip. DuckDuckGo / DDGS removed.
"""

from __future__ import annotations

import ast
import json
import operator as op
import re
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from pulp import (
    LpMinimize,
    LpProblem,
    LpStatus,
    LpVariable,
    PULP_CBC_CMD,
    lpSum,
    value,
)

from logging_setup import get_logger
from nutrition_formulas import (
    bmi,
    bmr_mifflin_st_jeor,
    daily_calorie_target,
    full_assessment,
    macro_split,
    tdee,
)
from utils import save_to_json

_qf_logger = get_logger("tools.quantities_finder")
_comp_logger = get_logger("tools.computation")
_web_logger = get_logger("tools.web_search")


# ---------------------------------------------------------------------------
# QuantitiesFinder (PuLP LP solver)
# ---------------------------------------------------------------------------
class QuantitiesFinder:
    """Linear-program solver that turns an LLM-drafted plan into precise grams.

    The schema is:

        {
            "foods": [{name, calories, protein, fat, carbohydrates,
                       estimated_g, [min_g, max_g, meal_group, estimate_weight]}, ...],
            "targets": {calories, protein, fat, carbohydrates},
            "meal_constraints": [{group_name, [max_<nut>], [min_<nut>]}, ...]   # optional
        }
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def _round(v: Any) -> float:
        if v is None:
            return 0.0
        return round(float(v), 2)

    @staticmethod
    def _round_structure(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: QuantitiesFinder._round_structure(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [QuantitiesFinder._round_structure(v) for v in obj]
        if isinstance(obj, (int, float)):
            return QuantitiesFinder._round(obj)
        return obj

    def handle_task(self, task: str) -> str:
        _qf_logger.info("\n📊 ENHANCED QUANTITIES FINDER (V3) TOOL STARTED")
        # Priority 1: hit daily totals; Priority 2: stay close to per-item estimates.
        W_NUTRITION = 1.0
        W_ESTIMATE_DEFAULT = 0.1

        try:
            data = json.loads(task)
            foods = data["foods"]
            targets = data["targets"]

            # 1. Validation
            required_nutrients = ["calories", "protein", "fat", "carbohydrates"]
            for food in foods:
                if not all(key in food for key in ["name"] + required_nutrients + ["estimated_g"]):
                    raise ValueError(
                        "Each food must have name, calories, protein, fat, carbohydrates, and estimated_g."
                    )
            if not all(key in targets for key in required_nutrients):
                raise ValueError("Targets must include calories, protein, fat, carbohydrates.")

            prob = LpProblem("Nutrient_Optimization", LpMinimize)

            # 2. Variables
            g = {}
            for food in foods:
                g[food["name"]] = LpVariable(
                    f"g_{food['name']}",
                    lowBound=food.get("min_g", 0),
                    upBound=food.get("max_g"),
                )

            # 3. Nutrition deviations
            totals = {
                nut: lpSum((g[f["name"]] / 100) * f[nut] for f in foods) for nut in required_nutrients
            }
            d_pos = {nut: LpVariable(f"d_pos_{nut}", lowBound=0) for nut in required_nutrients}
            d_neg = {nut: LpVariable(f"d_neg_{nut}", lowBound=0) for nut in required_nutrients}
            for nut in required_nutrients:
                prob += totals[nut] - targets[nut] <= d_pos[nut]
                prob += targets[nut] - totals[nut] <= d_neg[nut]

            # 3.5 Optional meal-level constraints
            for constraint in data.get("meal_constraints", []) or []:
                group_name = constraint.get("group_name")
                if not group_name:
                    continue
                group_foods = [f for f in foods if f.get("meal_group") == group_name]
                if not group_foods:
                    _qf_logger.warning("No foods found for meal_group '%s'", group_name)
                    continue
                for nut in required_nutrients:
                    meal_total = lpSum((g[f["name"]] / 100) * f[nut] for f in group_foods)
                    if (max_val := constraint.get(f"max_{nut}")) is not None:
                        prob += (meal_total <= max_val, f"Meal_{group_name}_max_{nut}")
                        _qf_logger.debug("Constraint: %s max %s <= %s", group_name, nut, max_val)
                    if (min_val := constraint.get(f"min_{nut}")) is not None:
                        prob += (meal_total >= min_val, f"Meal_{group_name}_min_{nut}")
                        _qf_logger.debug("Constraint: %s min %s >= %s", group_name, nut, min_val)

            # 4. Estimate deviations (per-item soft anchor)
            dev_est_pos = {f["name"]: LpVariable(f"dev_est_pos_{f['name']}", lowBound=0) for f in foods}
            dev_est_neg = {f["name"]: LpVariable(f"dev_est_neg_{f['name']}", lowBound=0) for f in foods}
            for food in foods:
                name = food["name"]
                est = food["estimated_g"]
                prob += g[name] - est <= dev_est_pos[name]
                prob += est - g[name] <= dev_est_neg[name]

            # 5. Objective
            nutrition_objective = lpSum(
                (d_pos[nut] + d_neg[nut]) / max(targets[nut], 1) for nut in required_nutrients
            )
            estimate_objective = lpSum(
                f.get("estimate_weight", W_ESTIMATE_DEFAULT)
                * (dev_est_pos[f["name"]] + dev_est_neg[f["name"]])
                / max(f["estimated_g"], 1)
                for f in foods
                if f["estimated_g"] > 0
            )
            prob += (W_NUTRITION * nutrition_objective) + estimate_objective

            # 6. Solve
            prob.solve(PULP_CBC_CMD(msg=0))
            if LpStatus[prob.status] != "Optimal":
                raise ValueError(
                    "No optimal solution found (problem may be infeasible). Check your targets and constraints."
                )

            quantities = {name: value(g[name]) for name in g}
            achieved = {nut: value(totals[nut]) for nut in required_nutrients}
            result = QuantitiesFinder._round_structure({"quantities": quantities, "achieved": achieved})

            _qf_logger.info("Solution Status: %s", LpStatus[prob.status])
            _qf_logger.info("Quantities (g): %s", json.dumps(result["quantities"], indent=2))
            _qf_logger.info(
                "Achieved Nutrition (around): %s",
                json.dumps(result["achieved"], indent=2),
            )
            _qf_logger.info(
                "Target Nutrition: %s",
                json.dumps(QuantitiesFinder._round_structure(targets), indent=2),
            )
            _qf_logger.info("\n📊 QUANTITIES FINDER COMPLETED")
            return json.dumps(result)

        except Exception as e:  # noqa: BLE001
            _qf_logger.error("QuantitiesFinder Error: %s", str(e))
            return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# ComputationTool (Phase 4: deterministic + sandboxed; no subprocess/eval/exec)
# ---------------------------------------------------------------------------
_OP_MAP: Dict[str, Callable[[str], str]]


class ComputationTool:
    """Deterministic numerical helpers for the agent system.

    Accepts either:

    1. A structured JSON task: ``{"op": "<name>", ...args}``. Supported ops:
       ``bmi``, ``bmr``, ``tdee``, ``calorie_target``, ``macro_split``,
       ``full_assessment``, ``eval`` (sandboxed math expression).
    2. A free-form English task (legacy). The tool's regex parser tries to
       extract numeric arguments and dispatch to the right formula. If
       parsing is ambiguous, the tool returns a structured error asking the
       agent to retry with the JSON form — no LLM-generated code, no
       subprocess.
    """

    # Activity / sex / goal lexicon used by the free-form parser.
    _SEX_PATTERN = re.compile(r"\b(male|female)\b", re.I)
    _ACTIVITY_PATTERN = re.compile(
        r"\b(sedentary|lightly active|moderately active|very active|extra active)\b",
        re.I,
    )
    _GOAL_PATTERN = re.compile(
        r"\b(lose weight|maintain weight|gain muscle|gain weight)\b", re.I
    )

    def __init__(self, llm_instance: Optional[Any] = None) -> None:
        # llm_instance retained only for backwards-compat constructor signature;
        # this tool no longer calls the LLM.
        self.llm = llm_instance

    # ------------------------------------------------------------------
    def handle_task(self, task_description: str) -> str:
        _comp_logger.info("\n🤖 COMPUTATION TOOL STARTED (deterministic)")
        try:
            result = self._dispatch(task_description)
        except Exception as e:  # noqa: BLE001
            result = {"error": f"{type(e).__name__}: {e}"}

        result_json = json.dumps(result)
        save_to_json(
            {
                "input": task_description,
                "result": result,
                "timestamp": datetime.now().isoformat(),
            },
            f"computation_tool_{datetime.now().isoformat()}.json",
            subdirectory="ComputationTool",
        )
        _comp_logger.info("🤖 COMPUTATION COMPLETED %s", result_json)
        return result_json

    # ------------------------------------------------------------------
    def _dispatch(self, task: str) -> Dict[str, Any]:
        # 1. Structured JSON dispatch
        try:
            data = json.loads(task)
            if isinstance(data, dict) and "op" in data:
                return self._run_op(data)
        except (json.JSONDecodeError, TypeError):
            pass

        # 2. Free-form English -> formula
        return self._parse_free_form(task)

    def _run_op(self, data: Dict[str, Any]) -> Dict[str, Any]:
        op_name = str(data.get("op", "")).lower()
        if op_name == "bmi":
            return {"BMI": round(bmi(float(data["weight_kg"]), float(data["height_cm"])), 2)}
        if op_name == "bmr":
            return {
                "BMR": round(
                    bmr_mifflin_st_jeor(
                        float(data["weight_kg"]),
                        float(data["height_cm"]),
                        float(data["age_years"]),
                        str(data["sex"]),
                    ),
                    1,
                )
            }
        if op_name == "tdee":
            return {"TDEE": round(tdee(float(data["bmr"]), str(data["activity_level"])), 1)}
        if op_name == "calorie_target":
            return {
                "daily_target_calories": daily_calorie_target(
                    float(data["tdee"]), str(data["goal"])
                )
            }
        if op_name == "macro_split":
            return {
                "macro_targets": macro_split(
                    int(data["daily_target_calories"]),
                    str(data["goal"]),
                    weight_kg=float(data["weight_kg"]) if "weight_kg" in data else None,
                )
            }
        if op_name == "full_assessment":
            return full_assessment(
                weight_kg=float(data["weight_kg"]),
                height_cm=float(data["height_cm"]),
                age_years=float(data["age_years"]),
                sex=str(data["sex"]),
                activity_level=str(data["activity_level"]),
                goal=str(data["goal"]),
            )
        if op_name == "eval":
            return {"result": safe_math_eval(str(data["expression"]))}
        raise ValueError(
            f"Unknown op {op_name!r}. Valid: bmi, bmr, tdee, calorie_target, "
            "macro_split, full_assessment, eval."
        )

    # ------------------------------------------------------------------
    def _parse_free_form(self, task: str) -> Dict[str, Any]:
        """Extract numeric kwargs from a free-form sentence and dispatch."""
        t = task.lower()
        nums = _extract_numbers(task)

        sex_m = self._SEX_PATTERN.search(task)
        activity_m = self._ACTIVITY_PATTERN.search(task)
        goal_m = self._GOAL_PATTERN.search(task)

        sex = sex_m.group(1).lower() if sex_m else None
        activity = activity_m.group(1).lower() if activity_m else None
        goal = goal_m.group(1).lower() if goal_m else None

        # Heuristic intent detection.
        wants_full = any(
            k in t for k in ("bmi", "bmr", "tdee", "calorie", "macro", "assessment")
        ) and {"weight_kg", "height_cm", "age"}.issubset(_label_numbers(task, nums).keys())

        labelled = _label_numbers(task, nums)

        if wants_full and sex and activity and goal:
            return full_assessment(
                weight_kg=labelled["weight_kg"],
                height_cm=labelled["height_cm"],
                age_years=labelled["age"],
                sex=sex,
                activity_level=activity,
                goal=goal,
            )

        if "bmi" in t and "weight_kg" in labelled and "height_cm" in labelled:
            return {"BMI": round(bmi(labelled["weight_kg"], labelled["height_cm"]), 2)}

        # Pure arithmetic fallback ("compute 2700 * 0.30 / 4")
        candidate = _strip_to_expression(task)
        if candidate and re.fullmatch(r"[\d\.\s\+\-\*\/\(\)]+", candidate):
            return {"result": safe_math_eval(candidate)}

        return {
            "error": (
                "ComputationTool could not parse the task. Re-issue using JSON: "
                '{"op": "full_assessment", "weight_kg": ..., "height_cm": ..., '
                '"age_years": ..., "sex": "male"|"female", '
                '"activity_level": "sedentary"|..., "goal": "lose weight"|...}'
            ),
            "received": task,
        }


# ---------------------------------------------------------------------------
# Safe math evaluator (no names, no calls, no attribute access)
# ---------------------------------------------------------------------------
_SAFE_BIN_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
}
_SAFE_UNARY_OPS = {ast.UAdd: op.pos, ast.USub: op.neg}


def safe_math_eval(expression: str) -> float:
    """Evaluate ``expression`` using a strict AST whitelist.

    Allowed: numeric literals, the seven arithmetic binary operators above,
    unary +/-, and parentheses. Anything else (names, calls, attribute access,
    subscripts, comprehensions, comparisons, lambdas, ...) raises ``ValueError``.
    """
    if len(expression) > 200:
        raise ValueError("expression too long")
    tree = ast.parse(expression, mode="eval")

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError(f"Forbidden constant: {node.value!r}")
        if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_BIN_OPS:
            return _SAFE_BIN_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_UNARY_OPS:
            return _SAFE_UNARY_OPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"Forbidden expression node: {type(node).__name__}")

    return float(_eval(tree))


# ---------------------------------------------------------------------------
# Free-form parser helpers
# ---------------------------------------------------------------------------
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _extract_numbers(text: str) -> list[float]:
    return [float(m) for m in _NUMBER_RE.findall(text)]


def _label_numbers(text: str, nums: list[float]) -> Dict[str, float]:
    """Best-effort: associate numbers with role labels by scanning units around them."""
    labelled: Dict[str, float] = {}
    for m in _NUMBER_RE.finditer(text):
        n = float(m.group(0))
        # Look at the next 12 chars after the number for a unit hint.
        tail = text[m.end() : m.end() + 12].lower()
        head = text[max(0, m.start() - 25) : m.start()].lower()
        if "kg" in tail and "weight_kg" not in labelled:
            labelled["weight_kg"] = n
        elif "cm" in tail and "height_cm" not in labelled:
            labelled["height_cm"] = n
        elif (
            "year" in tail
            or "years" in tail
            or "yo" in tail
            or "y/o" in tail
            or "age" in head
        ) and "age" not in labelled:
            labelled["age"] = n
    return labelled


def _strip_to_expression(text: str) -> str | None:
    """Pull out an obvious arithmetic substring like '2700 * 0.30 / 4'."""
    m = re.search(r"[\d\.\(\)\+\-\*\/\s]{4,}", text)
    if not m:
        return None
    candidate = m.group(0).strip()
    return candidate if any(c in candidate for c in "+-*/") else None


# ---------------------------------------------------------------------------
# WebSearchTool (Gemini google_search grounding)
# ---------------------------------------------------------------------------
class WebSearchTool:
    """Single-pass grounded web search.

    Backed by Gemini's built-in ``google_search`` tool: one round-trip in
    which Gemini decides which queries to run, searches Google for them,
    synthesises a cited answer, and returns ``groundingMetadata``. No
    third-party search provider, no separate query-generation or synthesis
    pass — the model owns the whole loop.

    The injected ``llm_instance`` must expose
    :meth:`utils.GeminiLLM.call_grounded`. In tests, the ``MockLLM`` fixture
    provides the same surface.
    """

    _SYSTEM_INSTRUCTION = (
        "You are a nutrition / clinical research assistant. Answer the "
        "question below using up-to-date sources you can find via Google "
        "Search. Prefer authoritative domains (WHO, USDA / FDC, EFSA, NICE, "
        "ADA, NIH, peer-reviewed journals, government health agencies). "
        "Return a concise, factual answer; cite source URLs inline. "
        "If the question asks for nutrition facts, give per-100g values for "
        "calories, protein, fat, and carbohydrates when available."
    )

    def __init__(self, llm_instance: Any) -> None:
        self.llm = llm_instance

    def handle_task(self, research_task: str) -> str:
        _web_logger.info("\n🌐 WEB SEARCH TOOL STARTED")

        question = self._extract_question(research_task)
        prompt = f"{self._SYSTEM_INSTRUCTION}\n\nQuestion: {question}\n\nAnswer:"

        if not hasattr(self.llm, "call_grounded"):
            msg = (
                "WebSearchTool requires a GeminiLLM with call_grounded(); "
                f"got {type(self.llm).__name__}."
            )
            _web_logger.error(msg)
            return msg

        text, citations, queries = self.llm.call_grounded(prompt)

        answer = self._append_sources(text, citations)
        timestamp = datetime.now().isoformat()
        save_to_json(
            {
                "input": research_task,
                "question": question,
                "queries_run": queries,
                "answer": answer,
                "citations": citations,
                "timestamp": timestamp,
            },
            f"web_search_tool_{timestamp}.json",
            subdirectory="WebSearchTool",
        )
        _web_logger.info(
            "🌐 WEB SEARCH TOOL completed (%d citations, queries=%s)",
            len(citations),
            queries,
        )
        return answer

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_question(task: str) -> str:
        """Accept legacy ``{"queries": [...]}`` JSON or a free-form string."""
        try:
            data = json.loads(task)
        except (json.JSONDecodeError, TypeError):
            return task
        if isinstance(data, dict):
            if isinstance(data.get("queries"), list) and data["queries"]:
                return " | ".join(str(q) for q in data["queries"])
            if isinstance(data.get("query"), str):
                return data["query"]
            if isinstance(data.get("question"), str):
                return data["question"]
        return task

    @staticmethod
    def _append_sources(text: str, citations: list[Dict[str, str]]) -> str:
        if not citations:
            return text
        # Skip duplicates while preserving order.
        seen: set[str] = set()
        lines: list[str] = []
        for c in citations:
            uri = c.get("uri", "")
            if not uri or uri in seen:
                continue
            seen.add(uri)
            title = c.get("title") or uri
            lines.append(f"- [{title}]({uri})")
        if not lines:
            return text
        return f"{text}\n\nSources:\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# Note: ``execute_python_code_raw`` (subprocess-based) was REMOVED in Phase 4.
# It executed LLM-generated Python on the host with no sandbox — a remote
# code execution vector. Replaced by deterministic formulas + safe_math_eval.
# ``search_web_raw`` (DDGS) was REMOVED in Phase 7+. Replaced by Gemini's
# built-in ``google_search`` grounding via GeminiLLM.call_grounded().
