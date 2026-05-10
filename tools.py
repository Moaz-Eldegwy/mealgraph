import re
import subprocess
import tempfile
import time
from time import sleep
import os
from datetime import datetime
from utils import save_to_json, should_debug
from ddgs import DDGS
import config
import json
from pulp import *

class QuantitiesFinder:

    def __init__(self):
        pass

    @staticmethod
    def _round(v):
        if v is None:
            return 0.0
        return round(float(v), 2)

    @staticmethod
    def _round_structure(obj):
        if isinstance(obj, dict):
            return {k: QuantitiesFinder._round_structure(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [QuantitiesFinder._round_structure(v) for v in obj]
        if isinstance(obj, (int, float)):
            return QuantitiesFinder._round(obj)
        return obj

    def handle_task(self, task: str) -> str:
        print(f"\n📊 ENHANCED QUANTITIES FINDER (V3) TOOL STARTED")
        # --- Define Weights ---
        W_NUTRITION = 1.0  # Priority 1: Hitting daily totals
        W_ESTIMATE_DEFAULT = 0.1  # Priority 2: Default "soft" estimate penalty

        try:
            data = json.loads(task)
            foods = data["foods"]
            targets = data["targets"]

            # --- 1. VALIDATION ---
            required_nutrients = ["calories", "protein", "fat", "carbohydrates"]
            for food in foods:
                if not all(
                    key in food
                    for key in ["name"] + required_nutrients + ["estimated_g"]
                ):
                    raise ValueError(
                        "Each food must have name, calories, protein, fat, carbohydrates, and estimated_g."
                    )

            if not all(key in targets for key in required_nutrients):
                raise ValueError(
                    "Targets must include calories, protein, fat, carbohydrates."
                )

            prob = LpProblem("Nutrient_Optimization", LpMinimize)

            # --- 2. VARIABLES (Unchanged from V2) ---
            g = {}
            for food in foods:
                food_name = food["name"]
                min_bound = food.get("min_g", 0)
                max_bound = food.get("max_g")
                g[food_name] = LpVariable(
                    f"g_{food_name}", lowBound=min_bound, upBound=max_bound
                )

            # --- 3. NUTRITION DEVIATIONS (Unchanged) ---
            nutrients = required_nutrients
            totals = {}
            for nut in nutrients:
                totals[nut] = lpSum(
                    (g[food["name"]] / 100) * food[nut] for food in foods
                )

            d_pos = {nut: LpVariable(f"d_pos_{nut}", lowBound=0) for nut in nutrients}
            d_neg = {nut: LpVariable(f"d_neg_{nut}", lowBound=0) for nut in nutrients}

            for nut in nutrients:
                prob += totals[nut] - targets[nut] <= d_pos[nut]
                prob += targets[nut] - totals[nut] <= d_neg[nut]

            # --- 3.5 MEAL-LEVEL CONSTRAINTS (Unchanged from V2) ---
            meal_constraints = data.get("meal_constraints", [])
            if meal_constraints:
                print("Applying meal-level constraints...")
                for constraint in meal_constraints:
                    group_name = constraint.get("group_name")
                    if not group_name:
                        continue
                    group_foods = [
                        f for f in foods if f.get("meal_group") == group_name
                    ]
                    if not group_foods:
                        print(f"Warning: No foods found for meal_group '{group_name}'")
                        continue
                    
                    for nut in nutrients:
                        max_val = constraint.get(f"max_{nut}")
                        if max_val is not None:
                            meal_total = lpSum(
                                (g[f["name"]] / 100) * f[nut] for f in group_foods
                            )
                            prob += (
                                meal_total <= max_val,
                                f"Meal_{group_name}_max_{nut}",
                            )
                            print(f"  -> Constraint: {group_name} max {nut} <= {max_val}")

                        min_val = constraint.get(f"min_{nut}")
                        if min_val is not None:
                            meal_total = lpSum(
                                (g[f["name"]] / 100) * f[nut] for f in group_foods
                            )
                            prob += (
                                meal_total >= min_val,
                                f"Meal_{group_name}_min_{nut}",
                            )
                            print(f"  -> Constraint: {group_name} min {nut} >= {min_val}")

            # --- 4. ESTIMATE DEVIATIONS (ENHANCED) ---
            # This section now reads a per-item 'estimate_weight'
            dev_est_pos = {
                food["name"]: LpVariable(f"dev_est_pos_{food['name']}", lowBound=0)
                for food in foods
            }
            dev_est_neg = {
                food["name"]: LpVariable(f"dev_est_neg_{food['name']}", lowBound=0)
                for food in foods
            }

            for food in foods:
                food_name = food["name"]
                estimate = food["estimated_g"]
                prob += g[food_name] - estimate <= dev_est_pos[food_name]
                prob += estimate - g[food_name] <= dev_est_neg[food_name]

            # --- 5. OBJECTIVE FUNCTION (ENHANCED) ---
            # Goal 1: (Unchanged)
            nutrition_objective = lpSum(
                (d_pos[nut] + d_neg[nut]) / max(targets[nut], 1) for nut in nutrients
            )
            
            # Goal 2: (ENHANCED)
            # Now uses the per-item 'estimate_weight' if provided,
            # otherwise, it falls back to the default.
            estimate_objective = lpSum(
                (
                    f.get("estimate_weight", W_ESTIMATE_DEFAULT)
                    * (dev_est_pos[f["name"]] + dev_est_neg[f["name"]])
                )
                / max(f["estimated_g"], 1)
                for f in foods
                if f["estimated_g"] > 0
            )

            # Combined objective
            prob += (W_NUTRITION * nutrition_objective) + estimate_objective

            # --- 6. SOLVE & RETURN (Unchanged) ---
            prob.solve(PULP_CBC_CMD(msg=0))

            if LpStatus[prob.status] != "Optimal":
                raise ValueError(
                    "No optimal solution found (problem may be infeasible). Check your targets and constraints."
                )

            quantities = {name: value(g[name]) for name in g}
            achieved = {nut: value(totals[nut]) for nut in nutrients}

            result = {"quantities": quantities, "achieved": achieved}
            result = QuantitiesFinder._round_structure(result)

            print(f"Solution Status: {LpStatus[prob.status]}")
            print(f"Quantities (g): {json.dumps(result['quantities'], indent=2)}")
            print(
                f"Achieved Nutrition (around): {json.dumps(result['achieved'], indent=2)}"
            )
            print(
                f"Target Nutrition: {json.dumps(QuantitiesFinder._round_structure(targets), indent=2)}"
            )

            print(f"\n📊 QUANTITIES FINDER COMPLETED")
            return json.dumps(result)

        except Exception as e:
            error_result = {"error": str(e)}
            print(f"QuantitiesFinder Error: {str(e)}")
            return json.dumps(error_result)

class ComputationTool:
    def __init__(self, llm_instance):
        self.llm = llm_instance

    def handle_task(self, task_description: str) -> str:
        print(f"\n🤖 COMPUTATION TOOL STARTED")
        instruction = "You are a Python coding assistant. Generate only the Python code required to perform the given task. Do not forget to print the result. Do not add explanations."
        prompt = f"{instruction}\n\nTask: {task_description}\n\nCode:"

        if should_debug('tools', 'ComputationTool') and config.DEBUG_LEVEL == 'full':
            print(f"Computation Tool Prompt:\n{prompt}")
        code_response = self.llm(prompt)[0]
        if should_debug('tools', 'ComputationTool'):
            print(f"Computation Tool Response:\n{code_response}")

        # Try to extract code from markdown blocks first, then use raw response
        code_match = re.search(r"```python\n(.*?)\n```", code_response, re.DOTALL)
        if not code_match:
            code_match = re.search(r"```\n(.*?)\n```", code_response, re.DOTALL)

        if code_match:
            code_to_execute = code_match.group(1).strip()
            # print(f"Extracted code from markdown blocks")
        else:
            code_to_execute = code_response.strip()
            # print(f"Using raw response as code")

        execution_result = execute_python_code_raw(code_to_execute)

        log_data = {
            "instruction": instruction,
            "input": task_description,
            "output": code_to_execute,
            "execution_result": execution_result,
            "timestamp": datetime.now().isoformat()
        }
        save_to_json(log_data, f'computation_tool_{datetime.now().isoformat()}.json', subdirectory='ComputationTool')

        print(f"🤖 COMPUTATION COMPLETED\n{execution_result}")
        return execution_result


class WebSearchTool:
    def __init__(self, llm_instance):
        self.llm = llm_instance

    def handle_task(self, research_task: str) -> str:
        print(f"\n🌐 WEB SEARCH TOOL STARTED")

        try:
            task_data = json.loads(research_task)
            if isinstance(task_data, dict) and 'queries' in task_data and isinstance(task_data['queries'], list):
                print("JSON query list detected. Converting to single text task.")
                research_question = " ".join(task_data['queries'])
            else:
                print("Single question mode detected (non-query JSON). Generating queries.")
                research_question = research_task
        except (json.JSONDecodeError, TypeError):
            print("Single question mode detected (plain text). Generating queries.")
            research_question = research_task

        query_instruction = "Formulate concise search queries for DuckDuckGo based on the given question. Output only the queries, one per line."
        query_prompt = f"{query_instruction}\n\nQuestion: {research_question}\n\nQueries:"

        if should_debug('tools', 'WebSearchTool') and config.DEBUG_LEVEL == 'full':
            print(f"Web Search Query Prompt:\n{query_prompt}")
        search_queries_text = self.llm(query_prompt)[0]
        if should_debug('tools', 'WebSearchTool'):
            print(f"Web Search Query Response:\n{search_queries_text}")

        search_queries = [q.strip() for q in search_queries_text.split('\n') if q.strip()] or [research_question]
        if should_debug('tools', 'WebSearchTool'):
            print(f"Parsed queries: {search_queries}")

        all_raw_results = []
        for i, query in enumerate(search_queries):
            raw_results = search_web_raw(query, num_results=10)
            print(f"Search results:\n{raw_results[:200]}...")
            all_raw_results.append(f"Results for '{query}':\n{raw_results}")
            sleep(1)

        raw_search_output = "\n\n".join(all_raw_results)

        synthesis_instruction = f"""Synthesize a concise answer to:
        Question: {research_question}
        Based on:
        ---
        {raw_search_output}
        ---
        """

        if should_debug('tools', 'WebSearchTool') and config.DEBUG_LEVEL == 'full':
            print(f"Web Search Synthesis Instruction:\n{synthesis_instruction}")
        synthesized_answer = self.llm(synthesis_instruction)[0]
        if should_debug('tools', 'WebSearchTool'):
            print(f"Web Search Synthesis Response:\n{synthesized_answer}")

        timestamp = datetime.now().isoformat()
        save_to_json({
            "instruction": query_instruction,
            "input": research_question,
            "output": search_queries_text,
            "timestamp": timestamp
        }, f'web_search_tool_queries_{timestamp}.json', subdirectory='WebSearchTool')

        save_to_json({
            "instruction": synthesis_instruction,
            "input": raw_search_output,
            "output": synthesized_answer,
            "timestamp": timestamp
        }, f'web_search_tool_synthesis_{timestamp}.json', subdirectory='WebSearchTool')

        print(f"\n🌐 WEB SEARCH TOOL Result:\n{synthesized_answer}\n")
        return synthesized_answer

def execute_python_code_raw(code_string: str) -> str:
    if should_debug('tools', 'ComputationTool') and config.DEBUG_LEVEL == 'full':
            print(f"🐍 Executing Code (raw):\n{code_string}")
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_script:
            tmp_script.write(code_string)
            script_path = tmp_script.name
        process = subprocess.run(["python", script_path], capture_output=True, text=True, timeout=30)
        os.remove(script_path)
        if process.returncode == 0:
            return f"Output:\n{process.stdout if process.stdout else 'Code executed successfully.'}"
        else:
            return f"Error:\n{process.stderr}"
    except Exception as e:
        return f"Execution Exception: {str(e)}"
    finally:
        if os.path.exists(script_path):
            os.remove(script_path)

def search_web_raw(query: str, num_results: int = 3) -> str:
    print(f"🌐 Searching Web (raw) for: {query}")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=num_results, timelimit="m"))
            if not results:
                return "No search results found."
            return "\n".join([f"Title: {r.get('title')}\nURL: {r.get('href')}\nSnippet: {r.get('body')}" for r in results])
        except Exception as e:
            if attempt < max_retries - 1:
                sleep(1)
                continue
            return f"Search Exception after {max_retries} attempts: {str(e)}"



