"""Tools layer.

Phase 1 cleanup notes:

* Replaced ``print`` with namespaced loggers so user-mode emoji output is
  filterable and the API/UI in Phase 7 can subscribe to it as events.
* Reads ``settings.debug_mode`` via :func:`config.get_settings` instead of the
  legacy module-level globals.

The :class:`ComputationTool` still shells out to ``subprocess.run(['python', ...])``
- **this is a known security issue**, fixed in Phase 4 by either deterministic
formula functions or a ``RestrictedPython`` sandbox.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from datetime import datetime
from time import sleep
from typing import Any

from ddgs import DDGS
from pulp import (
    LpMinimize,
    LpProblem,
    LpStatus,
    LpVariable,
    PULP_CBC_CMD,
    lpSum,
    value,
)

from config import get_settings
from logging_setup import get_logger
from utils import save_to_json, should_debug

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
# ComputationTool (LLM-generated Python; ⚠ replace in Phase 4)
# ---------------------------------------------------------------------------
class ComputationTool:
    def __init__(self, llm_instance):
        self.llm = llm_instance

    def handle_task(self, task_description: str) -> str:
        _comp_logger.info("\n🤖 COMPUTATION TOOL STARTED")
        settings = get_settings()
        instruction = (
            "You are a Python coding assistant. Generate only the Python code required "
            "to perform the given task. Do not forget to print the result. Do not add explanations."
        )
        prompt = f"{instruction}\n\nTask: {task_description}\n\nCode:"

        if should_debug("tools", "ComputationTool") and settings.debug_level == "full":
            _comp_logger.debug("Computation Tool Prompt:\n%s", prompt)
        code_response = self.llm(prompt)[0]
        if should_debug("tools", "ComputationTool"):
            _comp_logger.debug("Computation Tool Response:\n%s", code_response)

        match = re.search(r"```python\n(.*?)\n```", code_response, re.DOTALL)
        if not match:
            match = re.search(r"```\n(.*?)\n```", code_response, re.DOTALL)
        code_to_execute = match.group(1).strip() if match else code_response.strip()

        execution_result = execute_python_code_raw(code_to_execute)

        save_to_json(
            {
                "instruction": instruction,
                "input": task_description,
                "output": code_to_execute,
                "execution_result": execution_result,
                "timestamp": datetime.now().isoformat(),
            },
            f"computation_tool_{datetime.now().isoformat()}.json",
            subdirectory="ComputationTool",
        )
        _comp_logger.info("🤖 COMPUTATION COMPLETED\n%s", execution_result)
        return execution_result


# ---------------------------------------------------------------------------
# WebSearchTool (DuckDuckGo + LLM synthesis)
# ---------------------------------------------------------------------------
class WebSearchTool:
    def __init__(self, llm_instance):
        self.llm = llm_instance

    def handle_task(self, research_task: str) -> str:
        _web_logger.info("\n🌐 WEB SEARCH TOOL STARTED")
        settings = get_settings()

        try:
            task_data = json.loads(research_task)
            if (
                isinstance(task_data, dict)
                and "queries" in task_data
                and isinstance(task_data["queries"], list)
            ):
                _web_logger.info("JSON query list detected. Converting to single text task.")
                research_question = " ".join(task_data["queries"])
            else:
                _web_logger.info("Single question mode (non-query JSON). Generating queries.")
                research_question = research_task
        except (json.JSONDecodeError, TypeError):
            _web_logger.info("Single question mode (plain text). Generating queries.")
            research_question = research_task

        query_instruction = (
            "Formulate concise search queries for DuckDuckGo based on the given question. "
            "Output only the queries, one per line."
        )
        query_prompt = f"{query_instruction}\n\nQuestion: {research_question}\n\nQueries:"

        if should_debug("tools", "WebSearchTool") and settings.debug_level == "full":
            _web_logger.debug("Web Search Query Prompt:\n%s", query_prompt)
        search_queries_text = self.llm(query_prompt)[0]
        if should_debug("tools", "WebSearchTool"):
            _web_logger.debug("Web Search Query Response:\n%s", search_queries_text)

        search_queries = [q.strip() for q in search_queries_text.split("\n") if q.strip()] or [
            research_question
        ]
        if should_debug("tools", "WebSearchTool"):
            _web_logger.debug("Parsed queries: %s", search_queries)

        all_raw_results = []
        for query in search_queries:
            raw_results = search_web_raw(query, num_results=10)
            _web_logger.info("Search results: %s...", raw_results[:200])
            all_raw_results.append(f"Results for '{query}':\n{raw_results}")
            sleep(1)

        raw_search_output = "\n\n".join(all_raw_results)
        synthesis_instruction = (
            f"Synthesize a concise answer to:\n"
            f"Question: {research_question}\n"
            f"Based on:\n---\n{raw_search_output}\n---\n"
        )

        if should_debug("tools", "WebSearchTool") and settings.debug_level == "full":
            _web_logger.debug("Web Search Synthesis Instruction:\n%s", synthesis_instruction)
        synthesized_answer = self.llm(synthesis_instruction)[0]
        if should_debug("tools", "WebSearchTool"):
            _web_logger.debug("Web Search Synthesis Response:\n%s", synthesized_answer)

        timestamp = datetime.now().isoformat()
        save_to_json(
            {
                "instruction": query_instruction,
                "input": research_question,
                "output": search_queries_text,
                "timestamp": timestamp,
            },
            f"web_search_tool_queries_{timestamp}.json",
            subdirectory="WebSearchTool",
        )
        save_to_json(
            {
                "instruction": synthesis_instruction,
                "input": raw_search_output,
                "output": synthesized_answer,
                "timestamp": timestamp,
            },
            f"web_search_tool_synthesis_{timestamp}.json",
            subdirectory="WebSearchTool",
        )
        _web_logger.info("\n🌐 WEB SEARCH TOOL Result:\n%s\n", synthesized_answer)
        return synthesized_answer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def execute_python_code_raw(code_string: str) -> str:
    """⚠ Phase 4 will replace this with a sandbox or deterministic functions."""
    settings = get_settings()
    if should_debug("tools", "ComputationTool") and settings.debug_level == "full":
        _comp_logger.debug("🐍 Executing Code (raw):\n%s", code_string)
    script_path = ""
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_script:
            tmp_script.write(code_string)
            script_path = tmp_script.name
        process = subprocess.run(
            ["python", script_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if process.returncode == 0:
            return f"Output:\n{process.stdout if process.stdout else 'Code executed successfully.'}"
        return f"Error:\n{process.stderr}"
    except Exception as e:  # noqa: BLE001
        return f"Execution Exception: {str(e)}"
    finally:
        if script_path and os.path.exists(script_path):
            os.remove(script_path)


def search_web_raw(query: str, num_results: int = 3) -> str:
    _web_logger.info("🌐 Searching Web (raw) for: %s", query)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=num_results, timelimit="m"))
            if not results:
                return "No search results found."
            return "\n".join(
                f"Title: {r.get('title')}\nURL: {r.get('href')}\nSnippet: {r.get('body')}"
                for r in results
            )
        except Exception as e:  # noqa: BLE001
            if attempt < max_retries - 1:
                sleep(1)
                continue
            return f"Search Exception after {max_retries} attempts: {str(e)}"
    return "Search Exception: exhausted retries"
