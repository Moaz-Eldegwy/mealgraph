"""Tool implementations used by the agents.

Two tools, both safe by construction:

* :class:`QuantitiesFinder` — PuLP linear-program solver. Given a list of
  candidate foods (with per-100g macros and an ``estimated_g`` anchor) and
  the daily targets, it returns gram quantities that minimise the weighted
  deviation from both the daily macro targets and the per-item anchors.
  Optional ``meal_constraints`` express per-meal macro caps / floors.

* :class:`WebSearchTool` — single-pass wrapper around Gemini's built-in
  ``google_search`` grounding. Gemini decides the queries, runs them,
  synthesises a cited answer, and returns ``grounding_metadata`` in one
  round-trip. Citations are appended to the answer string so downstream
  code can persist them with the plan.

No LLM-generated code path exists anywhere in this module: clinical math
runs through :mod:`nutrition_formulas` (called directly by the agents) and
``QuantitiesFinder`` is a pure LP. There is no ``eval``, no subprocess,
and no filesystem or network access beyond the Gemini SDK.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List

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
from utils import save_to_json

_qf_logger = get_logger("tools.quantities_finder")
_web_logger = get_logger("tools.web_search")


# ---------------------------------------------------------------------------
# QuantitiesFinder (PuLP LP solver)
# ---------------------------------------------------------------------------
class QuantitiesFinder:
    """Linear-program solver that turns an LLM-drafted plan into precise grams.

    Input schema (tool_task must be a JSON string):

        {
            "foods": [
                {
                    "name": str,
                    "calories": float,        # per 100g
                    "protein": float,         # per 100g
                    "fat": float,             # per 100g
                    "carbohydrates": float,   # per 100g
                    "estimated_g": float,     # LLM's anchor weight
                    # optional:
                    "min_g": float,
                    "max_g": float,
                    "meal_group": str,
                    "estimate_weight": float,
                },
                ...
            ],
            "targets": {
                "calories": float,
                "protein": float,
                "fat": float,
                "carbohydrates": float,
            },
            "meal_constraints": [   # optional
                {"group_name": str, "max_<nut>": float, "min_<nut>": float},
                ...
            ]
        }

    Default per-food bounds (when ``min_g``/``max_g`` are not supplied)::

        min_g = max(20,  estimated_g * 0.3)
        max_g = min(400, estimated_g * 2.5)

    These stop the LP from suggesting 1 g of butter or 900 g of broccoli
    to chase a macro target. The estimate-anchor weight defaults to 0.3
    (was 0.1 in earlier revisions) so the LP must have a strong reason to
    drift away from the LLM's drafted serving sizes — small deviations are
    penalised harder, which keeps the output realistic.
    """

    def __init__(self) -> None:
        pass

    # Priority 1: hit daily totals; Priority 2: stay close to per-item
    # estimates. The default estimate weight is intentionally non-trivial
    # so the LP cannot wander far from the LLM's draft.
    W_NUTRITION = 1.0
    W_ESTIMATE_DEFAULT = 0.3
    MIN_BOUND_FLOOR = 20.0
    MAX_BOUND_CAP = 400.0
    MIN_BOUND_RATIO = 0.3
    MAX_BOUND_RATIO = 2.5

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

    @classmethod
    def _default_bounds(cls, est_g: float) -> tuple[float, float]:
        """Default ``(min_g, max_g)`` derived from ``estimated_g``."""
        if est_g <= 0:
            return cls.MIN_BOUND_FLOOR, cls.MAX_BOUND_CAP
        min_g = max(cls.MIN_BOUND_FLOOR, est_g * cls.MIN_BOUND_RATIO)
        max_g = min(cls.MAX_BOUND_CAP, est_g * cls.MAX_BOUND_RATIO)
        if min_g > max_g:
            # Degenerate case: bounds collide. Fall back to the est anchor.
            min_g, max_g = max(0.0, est_g - 1), est_g + 1
        return min_g, max_g

    def handle_task(self, task: str) -> str:
        _qf_logger.info("\n📊 QUANTITIES FINDER TOOL STARTED")

        try:
            data = json.loads(task)
            foods = data["foods"]
            targets = data["targets"]

            # 1. Validation
            required_nutrients = ["calories", "protein", "fat", "carbohydrates"]
            for food in foods:
                if not all(
                    key in food for key in ["name"] + required_nutrients + ["estimated_g"]
                ):
                    raise ValueError(
                        "Each food must have name, calories, protein, fat, "
                        "carbohydrates, and estimated_g."
                    )
            if not all(key in targets for key in required_nutrients):
                raise ValueError(
                    "Targets must include calories, protein, fat, carbohydrates."
                )

            prob = LpProblem("Nutrient_Optimization", LpMinimize)

            # 2. Variables (with realistic default bounds)
            g: Dict[str, LpVariable] = {}
            for food in foods:
                est = float(food["estimated_g"])
                default_min, default_max = self._default_bounds(est)
                min_g = float(food.get("min_g", default_min))
                max_g = float(food.get("max_g", default_max))
                g[food["name"]] = LpVariable(
                    f"g_{food['name']}",
                    lowBound=min_g,
                    upBound=max_g,
                )

            # 3. Nutrition deviations.
            # ``g[name] * (per100 / 100)`` keeps the LpVariable on the LEFT of
            # the multiplication so PuLP returns an LpAffineExpression. The
            # earlier ``g[name] / 100 * per100`` form trips Python's operator
            # precedence: ``g / 100`` raises ``LpVariable / int`` which is
            # rejected by PuLP at expression-build time.
            totals = {
                nut: lpSum(g[f["name"]] * (float(f[nut]) / 100.0) for f in foods)
                for nut in required_nutrients
            }
            d_pos = {nut: LpVariable(f"d_pos_{nut}", lowBound=0) for nut in required_nutrients}
            d_neg = {nut: LpVariable(f"d_neg_{nut}", lowBound=0) for nut in required_nutrients}
            for nut in required_nutrients:
                prob += totals[nut] - float(targets[nut]) <= d_pos[nut]
                prob += float(targets[nut]) - totals[nut] <= d_neg[nut]

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
                    meal_total = lpSum(
                        g[f["name"]] * (float(f[nut]) / 100.0) for f in group_foods
                    )
                    if (max_val := constraint.get(f"max_{nut}")) is not None:
                        prob += (meal_total <= max_val, f"Meal_{group_name}_max_{nut}")
                    if (min_val := constraint.get(f"min_{nut}")) is not None:
                        prob += (meal_total >= min_val, f"Meal_{group_name}_min_{nut}")

            # 4. Estimate deviations (per-item soft anchor)
            dev_est_pos = {f["name"]: LpVariable(f"dev_est_pos_{f['name']}", lowBound=0) for f in foods}
            dev_est_neg = {f["name"]: LpVariable(f"dev_est_neg_{f['name']}", lowBound=0) for f in foods}
            for food in foods:
                name = food["name"]
                est = float(food["estimated_g"])
                prob += g[name] - est <= dev_est_pos[name]
                prob += est - g[name] <= dev_est_neg[name]

            # 5. Objective
            nutrition_objective = lpSum(
                (d_pos[nut] + d_neg[nut]) / max(float(targets[nut]), 1.0)
                for nut in required_nutrients
            )
            estimate_objective = lpSum(
                float(f.get("estimate_weight", self.W_ESTIMATE_DEFAULT))
                * (dev_est_pos[f["name"]] + dev_est_neg[f["name"]])
                / max(float(f["estimated_g"]), 1.0)
                for f in foods
                if float(f["estimated_g"]) > 0
            )
            prob += (self.W_NUTRITION * nutrition_objective) + estimate_objective

            # 6. Solve
            prob.solve(PULP_CBC_CMD(msg=0))
            if LpStatus[prob.status] != "Optimal":
                raise ValueError(
                    "No optimal solution found (problem may be infeasible). "
                    "Check your targets and constraints."
                )

            quantities = {name: value(g[name]) for name in g}
            achieved = {nut: value(totals[nut]) for nut in required_nutrients}
            result = QuantitiesFinder._round_structure(
                {"quantities": quantities, "achieved": achieved}
            )

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
    can stub the same surface.
    """

    _SYSTEM_INSTRUCTION = (
        "You are a nutrition / clinical research assistant. Answer the "
        "question below using up-to-date sources you can find via Google "
        "Search. Prefer authoritative domains (WHO, USDA / FDC, EFSA, NICE, "
        "ADA, NIH, MedlinePlus, peer-reviewed journals, government health "
        "agencies). Return a concise, factual answer; cite source URLs "
        "inline. If the question asks for nutrition facts, give per-100g "
        "values for calories, protein, fat, and carbohydrates when available."
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
    def _append_sources(text: str, citations: List[Dict[str, str]]) -> str:
        if not citations:
            return text
        seen: set[str] = set()
        lines: List[str] = []
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


__all__ = ["QuantitiesFinder", "WebSearchTool"]
