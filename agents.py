from typing import Dict, Any
from utils import extract_and_parse_json, set_nested, update_memory_partition, save_to_json, should_debug
from tools import ComputationTool, WebSearchTool, QuantitiesFinder
from datetime import datetime
import json
import config

class CoachAgent:
    def __init__(self, llm_instance):
        self.llm = llm_instance

    def handle_task(self, state: Dict[str, Any]) -> Dict[str, Any]:
        memory_str = json.dumps(state["memory"], indent=2)
        response_steps = state.get("response_steps", [])
        response_steps_str = json.dumps(response_steps, indent=2) if response_steps else "None"
        truncated_history = []
        for msg in state["conversation_history"]:
            if msg["role"] == "assistant" and len(msg["content"]) > 200:
                truncated_content = msg["content"][:200] + "... (full response in memory)"
                truncated_history.append({"role": "assistant", "content": truncated_content})
            else:
                truncated_history.append(msg)
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in truncated_history])

        observation = f"""User query: {state['user_question']}
        Memory State: {memory_str}
        Current Response Steps: {response_steps_str}
        Previous Tool Result: {state.get('agent_result', 'None')}
        Conversation history: {history_str}"""

        prompt = f"""
        You are the Coach Agent (central orchestrator) of a nutrition MAS. 

        Current State: {observation}

        Primary responsibilities:
        - Translate user intent to a concrete workflow of response_steps (use the shared response_step schema).
        - Enforce system rules (MedicalAssessment must be completed before Planner.
        - Decide and perform actions: call_agent, call_tool, ask_user, write_memory, compose_response.

        Inputs:
        - observation (string)
        - memory partitions: user_profile, medical_history, flags_and_assessments, plans
        - response_steps (may be None or list)

        Behavior rules (mandatory):
        1. If response_steps is None or empty, generate a response_steps list with explicit ordered steps (max 6 steps). Each step must include id, actor, prerequisites, and status "pending".
          - Typical personal-workflow (if user asks for personalized plan): 
            1) Validate required user data (height, weight, age, sex, activity_level, allergies, goal). If missing -> ask_user.
            2) Update memory (if user provided new data). [action: write_memory]
            3) Call MedicalAssessmentAgent with task to assess user.
            4) Wait for assessment to be completed and stored into memory.
            5) Call PlannerAgent with relevent task.
        2. When calling any agent, set the called step status to "in_progress" and include `prerequisites` satisfied by your observation.
        3. Only call PlannerAgent if memory.flags_and_assessments exists and contains "assessment_status":"assessment_complete". If not, call MedicalAssessmentAgent.
        4. When new user personal data is detected in user input, add steps to:
          - propose memory update (write_memory)
          - call MedicalAssessmentAgent if needed
          - re-plan if needed
        5. For any "write_memory" action, provide the full partition contents in params.data (not diffs). The Coach is responsible to merge and store.

        Action outputs: respond with a JSON object:
        {{
          "observation": "...",
          "thought": "...",
          "response_steps": [ ... ],
          "action": "call_agent | call_tool | ask_user | write_memory | compose_response",
          "params": {{ ... }}
        }}

        Examples:
        - call_agent params: {{"agent_name":"MedicalAssessmentAgent", "task":"task description"}}
        - compose_response params:{{"text":"Complete response in markdown"}}


        Rules: 
        - When composing the response, extract and include relevant information from the memory state (e.g., calorie target, plan details, dietary restrictions) in markdown format for readability.
        - Always include a "trace" field in composed responses summarizing which agents/tools were called for and which sources were used.
        - For high-risk profiles (e.g., requires_professional_consultation: true); in such cases append a bold warning at the end of the diet plan response advising professional consultation before implementation.
        """

        if should_debug('agents', 'CoachAgent'):
            print(f"\n--- Coach Agent Turn {state['num_turns'] + 1} ---")
        if should_debug('agents', 'CoachAgent') and config.DEBUG_LEVEL == 'full':
            print(f"Raw LLM input:\n{prompt}")
        response = self.llm(prompt)[0]
        if should_debug('agents', 'CoachAgent'):
            print(f"Coach Raw Response:\n{response}")

        parsed = extract_and_parse_json(response)

        # Add high-level print for user mode
        if not config.DEBUG_MODE:
            action = parsed.get("action")
            params = parsed.get("params", {})
            print_str = "\n🏋️‍♂️Coach Agent: "
            if action == "call_agent":
                print_str += f"Calling {params.get('agent_name')} with task '{params.get('task')}'"
            elif action == "call_tool":
                print_str += f"Using {params.get('tool_name')} with task '{params.get('task')}'"
            elif action == "ask_user":
                print_str += f"Asking user: {params.get('prompt')}"
            elif action == "write_memory":
                print_str += f"Writing to memory partition '{params.get('partition')}'"
            elif action == "compose_response":
                print_str += "Composing final response"
            print(print_str)

        current_action = {
            "action": parsed.get("action"),
            "params": parsed.get("params", {})
        }

        response_steps = parsed.get("response_steps", state.get("response_steps", []))
        
        log_data = {
            "prompt": prompt,
            "output":response,
            "parsed": parsed,
            "timestamp": datetime.now().isoformat()
        }
        save_to_json(log_data, f'coach_agent_{datetime.now().isoformat()}.json', subdirectory='CoachAgent')
        
        return {
            **state,
            "current_action": current_action,
            "response_steps": response_steps,
            "num_turns": state["num_turns"] + 1,
            "agent_result": None
        }

        
class MedicalAssessmentAgent:
    def __init__(self, llm_instance, computation_tool: ComputationTool, web_search_tool: WebSearchTool):
        self.llm = llm_instance
        self.computation_tool = computation_tool
        self.web_search_tool = web_search_tool

    def handle_task(self, task: str, memory: Dict[str, Any]) -> str:
        print(f"\n👨🏻‍⚕️ MEDICAL ASSESSMENT AGENT STARTED")

        # Build relevant memory context
        relevant_memory = {
            "user_profile": memory.get("user_profile", {}),
            "medical_history": memory.get("medical_history", {}),
        }
        memory_str = json.dumps(relevant_memory, indent=2)
        tool_results = []
        assessment_plan = []
        max_iterations = 15
        iteration = 0

        while iteration < max_iterations:

            tool_results_str = "\n".join([f"Tool Result {i+1}: {result}" for i, result in enumerate(tool_results)])

            prompt = f"""
            You are the Medical Assessment Agent. Your job: produce an evidence-based assessment and the set of clinical flags and calculations needed by the Planner and Validation agents.
            Task: {task}
            Current Memory: {memory_str}
            Current Assessment Plan: {assessment_plan}
            Previous Tool Results: {tool_results_str}
            Available tools: ComputationTool, WebSearchTool
            Mandatory behavior (do not skip):
            1. Critical data check: confirm presence of age, sex, height, weight, activity_level, allergies, medications. If any critical field is missing -> action: ask_user (return which fields).
            2. Use ComputationTool for all numeric calculations (BMI, BMR, TDEE, calorie targets, macro targets). Provide computation inputs with inside the task description.
            3. Use WebSearchTool to fetch authoritative guidelines where relevant (WHO, USDA, clinical guidelines). Always capture the source(s) used with timestamped citations.
            4. Produce a compact assessment_plan (3-6 steps max) that lists each computational/search step, its status, and result.
            - When generating the assessment_plan (if empty or None), follow this exact sequence (assuming critical data is present; if not, prepend a step for ask_user):
              1. Call ComputationTool to calculate BMI, BMR, TDEE, and a single daily_target_calories (integer) based on the user's goal, all in one tool call.
              2. Call ComputationTool to calculate macro_targets (protein_g, fat_g, carbohydrates_g as single integers) optimized for the user's goal given the daily_target_calories.
              3. Call WebSearchTool to find dietary guidelines related to the user based on their profile and medical history to manage conditions.
              4-6. Additional steps if needed (e.g., synthesis, further searches/computations for specific risks).
            5. Return a `assessment_complete` containing:
              - assessment_summary
              - calculations: {{BMI, BMR, TDEE, daily_target_calories, macro_targets}}
                - daily_target_calories: a single integer value (e.g., 2750)
                - macro_targets: {{"protein_g": int, "fat_g": int, "carbohydrates_g": int}} (single integer values for each, no ranges)
              - flags_to_set: [e.g., "high_ldl", "diabetes_risk"]
              - recommendations: clinical dietary constraints or urgent issues (e.g., "refer to PCP for suspected iron deficiency")
              - requires_professional_consultation: True/False (True if the case is medically sensitive)
              - trace: a single paragraph summarizing which agents/tools were called and key steps.
            6. If any calculation or guideline retrieval fails due to tool error:
              - fallback to best-known guideline values only if necessary (mark "data_confidence": 0.xx).
              - set "requires_tool_retry": true in the response.
            Response JSON must contain:
            - medical_reasoning: detailed rationale
            - observation: missing/available info
            - risk_assessment_priorities: ordered list of 1-4 priorities
            - assessment_plan: list of response_step objects (schema above)
            - action: either {{"type":"call_tool","tool_name":"ComputationTool" or "WebSearchTool","tool_task": "<task string>"}} or {{"type":"assessment_complete",...}}
            """

            if should_debug('agents', 'MedicalAssessmentAgent'):
                print(f"\n--- Medical Assessment Agent Iteration {iteration + 1} ---")
            if should_debug('agents', 'MedicalAssessmentAgent') and config.DEBUG_LEVEL == 'full':
                print(f"Raw LLM input:\n{prompt}")
            response = self.llm(prompt)[0]
            if should_debug('agents', 'MedicalAssessmentAgent'):
                print(f"Medical Assessment Raw Response:\n{response}")

            parsed = extract_and_parse_json(response)

            # Add high-level print for user mode
            if not config.DEBUG_MODE:
                action_type = parsed.get("action", {}).get("type")
                if action_type == "call_tool":
                    tool_name = parsed["action"].get("tool_name")
                    tool_task = parsed["action"].get("tool_task")
                    print(f"👨🏻‍⚕️ Medical Assessment Agent: Using {tool_name} for '{tool_task}'")
                elif action_type == "ask_user":
                    fields = parsed["action"].get("fields", [])
                    print(f"👨🏻‍⚕️ Medical Assessment Agent: Asking user for missing fields: {', '.join(fields)}")
                elif action_type == "assessment_complete":
                    print("👨🏻‍⚕️ Medical Assessment Agent: Completing assessment")

            if "assessment_plan" in parsed:
                assessment_plan = parsed["assessment_plan"]

            action = parsed.get("action", {})
            action_type = action.get("type")

            if action_type == "call_tool":
                tool_name = action.get("tool_name")
                tool_task = action.get("tool_task")

                if tool_name == "ComputationTool":
                    if tool_task:
                        result = self.computation_tool.handle_task(tool_task)
                    else:
                        result = "Missing 'tool_task' for ComputationTool"
                elif tool_name == "WebSearchTool":
                    if tool_task:
                        result = self.web_search_tool.handle_task(tool_task)
                    else:
                        result = "Missing 'tool_task' for WebSearchTool"
                else:
                    result = f"Unknown tool: {tool_name}"

                tool_results.append(f"{tool_name}: {result}")

            elif action_type == "ask_user":
                fields = action.get("fields", [])  
                result = f"Missing critical fields: {', '.join(fields)}. Please provide the following information to continue the assessment."
                print(f"👨🏻‍⚕️ MEDICAL ASSESSMENT AGENT: User query needed - {result}")
                return result  
                
            elif action_type == "assessment_complete":
                assessment_summary = action.get("assessment_summary")
                flags_to_set = action.get("flags_to_set", [])
                recommendations = action.get("recommendations", [])
                requires_professional_consultation = action.get("requires_professional_consultation", False)
                calculations = action.get("calculations", {})  # Now a dict as per new prompt
                evidence_sources = action.get("evidence_sources", [])
                trace = action.get("trace", "")

                if action.get("requires_tool_retry", False):
                    result = "Assessment requires tool retry due to failures. Please re-run with fixed tools."
                    print(f"👨🏻‍⚕️ MEDICAL ASSESSMENT AGENT: Tool retry needed - {result}")
                    return result  # Return early without updating memory

                # Update memory using update_memory_partition
                update_memory_partition(memory, "flags_and_assessments", {
                    "assessment_summary": assessment_summary,
                    "flags": flags_to_set,
                    "recommendations": recommendations,
                    "requires_professional_consultation": requires_professional_consultation,
                    "calculations": calculations,
                    "evidence_sources": evidence_sources,
                    "trace": trace,
                    "assessment_timestamp": datetime.now().isoformat()  # Retained timestamp
                })

                # Log the assessment (updated to include new fields)
                log_data = {
                    "task": task,
                    "memory_input": relevant_memory,
                    "tool_results": tool_results,
                    "assessment_summary": assessment_summary,
                    "flags_set": flags_to_set,
                    "recommendations": recommendations,
                    "requires_professional_consultation": requires_professional_consultation,
                    "evidence_sources": evidence_sources,
                    "trace": trace,
                    "timestamp": datetime.now().isoformat()
                }
                save_to_json(log_data, f'medical_assessment_{datetime.now().isoformat()}.json', subdirectory='MedicalAssessment')

                result = assessment_summary
                print(f"👨🏻‍⚕️ MEDICAL ASSESSMENT AGENT COMPLETED: {result}")
                return result

            else:
                print(f"Unknown action type: {parsed}")
                break

            iteration += 1

        # Fallback if max iterations reached
        result = f"Medical assessment stopped after {max_iterations} iterations"
        print(f"👨🏻‍⚕️ MEDICAL ASSESSMENT AGENT Stopped (MAX ITERATIONS)")
        return result

class PlannerAgent:
    def __init__(self, llm_instance, computation_tool: ComputationTool, web_search_tool: WebSearchTool, quantities_finder: QuantitiesFinder):
        self.llm = llm_instance
        self.computation_tool = computation_tool
        self.web_search_tool = web_search_tool
        self.quantities_finder = quantities_finder

    def handle_task(self, task: str, memory: Dict[str, Any]) -> str:
        print(f"\n📋 PLANNER AGENT STARTED")

        relevant_memory = {
            "user_profile": memory.get("user_profile", {}),
            "flags_and_assessments": memory.get("flags_and_assessments", {}),
        }
        memory_str = json.dumps(relevant_memory, indent=2)
        tool_results = []
        planning_steps = []
        max_iterations = 15
        iteration = 0

        while iteration < max_iterations:
            tool_results_str = "\n".join([f"Tool Result {i+1}: {res}" for i, res in enumerate(tool_results)]) if tool_results else "None"
            planning_steps_str = json.dumps(planning_steps, indent=2) if planning_steps else "None"
            plan_status = relevant_memory.get("flags_and_assessments", {}).get("assessment_status", "none")
            prompt = f"""
You are the Planner Agent. Create personalized meal plans constrained by the medical assessment.

Task: {task}
Current Memory: {memory_str}
Current Planning Steps: {planning_steps_str}
Previous Tool Results: {tool_results_str}

Available Tools: WebSearchTool, QuantitiesFinder

Mandatory behavior & rules:
1. Precondition: Do NOT start planing unless user medical assessment exists in memory (flags_and_assessments is not empty). If missing, return action: {{"type":"provide_plan", "final_plan":{{"Can't draft plan as flags_and_assessments is empty, please use MedicalAssessmentAgent"}}}}

2. Batch behavior:
   - Always group related items when using tools. Example: fetch nutrition facts for all foods in one WebSearchTool call instead of multiple calls.

3. For each food in the draft:
   - Use WebSearchTool to fetch nutrition facts for a standard serving size (or 100g cooked) (e.g., "Find nutrition facts (calories, protein, fat, carbohydrates) for the following items,...").
   - If WebSearchTool fails for >2 items, stop retrying and use your internal knowledge.

4. Acceptable tolerances:
   - Calories: within ±3% of daily_target_calories
   - Macronutrients: within ±5% of each macro target

5. Exclude all items listed in allergies and avoid disliked foods unless necessary for balance, in which case propose alternatives.

6. Flexible Planning: If task requests a multi-day plan (e.g., 7 days), fall back to a shorter balanced plan (1–2 unique days) and instruct user to repeat/rotate.

7. QuantitiesFinder Format: When calling 'QuantitiesFinder', the 'tool_task' MUST be a JSON STRING. This string is the serialized version of an object containing "foods" and "targets".
    - "foods": A list of dictionaries. Each dictionary must have:
      - name, calories, protein, fat, carbohydrates (per 100g)
      - estimated_g: Your "best guess" for a realistic quantity (e.g., 150g). The solver will be penalized for deviating from this, so it will try to stay close.
    - "targets": A dictionary containing: calories, protein, fat, carbohydrates.
    - Example: "tool_task": "{{\"foods\": [...], \"targets\": {{...}}}}"

Planning Steps Handling:
- If Current Planning Steps is empty or 'None', you MUST adopt the following fixed 6-step plan as your primary workflow.
[
{{"id": 1, "description": "Analyze requirements, "Draft a realistic diet plan. For each food, assign a realistic 'estimated_g' (e.g., 150g chicken)."", "status": "pending"}},
{{"id": 1, "description": "Analyze drafted plan, determine a list of all ingredients in the darafted plan, and batch-gather their nutritional facts (calories, protein, fat, carbohydrates) using WebSearchTool.", "status": "pending"}},
{{"id": 3, "description": "Call 'QuantitiesFinder' (PuLP solver) with all nutritional data, targets, and bounds to calculate precise quantities.", "status": "pending"}},
{{"id": 4, "description": "Update the drafted plan with the precise quantities returned by the QuantitiesFinder.", "status": "pending"}},
{{"id": 4, "description": "Provide the final plan 'provide_plan'", "status": "pending"}}
]

- If Current Planning Steps is provided... You may remain in a step for multiple iterations if necessary to meet all targets, as outlined in the Iterative Correction Loop rule.

Return JSON:
- observation, thought
- planning_steps (full list of response_step objects)
- action: one of {{
    "type":"call_tool","tool_name":...,"tool_task":...,
    "type":"draft_plan","drafted_plan":{{...}},
    "type":"provide_plan","final_plan":{{...}}
}}

Notes:
- Keep each plan realistic and culturally appropriate (regional foods if provided).
- Trace: at the end of the plan, summarize which agents/tools were called.
- Always include the full updated planning_steps in your response JSON to persist across iterations.
"""

            if should_debug('agents', 'PlannerAgent'):
                print(f"\n--- Planner Agent Iteration {iteration + 1} ---")
            if should_debug('agents', 'PlannerAgent') and config.DEBUG_LEVEL == 'full':
                print(f"Raw LLM input:\n{prompt}")
            response = self.llm(prompt)[0]
            if should_debug('agents', 'PlannerAgent'):
                print(f"Planner Raw Response:\n{response}")

            parsed = extract_and_parse_json(response)

            # Add high-level print for user mode
            if not config.DEBUG_MODE:
                action_type = parsed.get("action", {}).get("type")
                print_str = "📋 Planner Agent: "
                if action_type == "call_tool":
                    tool_name = parsed["action"].get("tool_name")
                    tool_task = parsed["action"].get("tool_task")
                    print_str += f"Using {tool_name} for '{tool_task}'"
                elif action_type == "draft_plan":
                    print_str += "Drafting plan"
                elif action_type == "provide_plan":
                    print_str += "Finalizing plan"
                print(print_str)

            planning_steps = parsed.get("planning_steps", planning_steps)

            action = parsed.get("action", {})
            action_type = action.get("type")

            if action_type == "call_tool":
                tool_name = action.get("tool_name")
                tool_task = action.get("tool_task")
                if tool_name and tool_task:
                    print(f"Calling {tool_name} with task: {tool_task}")
                    if tool_name == "ComputationTool":
                        result = self.computation_tool.handle_task(tool_task)
                    elif tool_name == "WebSearchTool":
                        result = self.web_search_tool.handle_task(tool_task)
                    elif tool_name == "QuantitiesFinder":
                        result = self.quantities_finder.handle_task(tool_task)
                    else:
                        result = f"Unknown tool: {tool_name}"
                    tool_results.append(f"{tool_name}: {result}")
                else:
                    print("Missing tool_name or tool_task")

            elif action_type == "draft_plan":
                drafted_plan = action.get("drafted_plan")
                if drafted_plan:
                    if "plans" not in memory:
                        memory["plans"] = {}
                    memory["plans"]["drafted_plan"] = drafted_plan
                    result = "Plan drafted and stored in memory"
                    tool_results.append(result)
                else:
                    result = "Drafted plan not provided"
                    tool_results.append(result)

            elif action_type == "provide_plan":
                final_plan = action.get("final_plan")
                if "error" in final_plan:
                    print(f"\n📋 PLANNER AGENT ERROR: {final_plan}")
                    return json.dumps(final_plan)
                else:
                    final_plan = final_plan or memory["plans"].get("drafted_plan")
                    if final_plan:
                        memory["plans"]["current_plan"] = final_plan
                        memory["plans"]["plan_timestamp"] = datetime.now().isoformat()
                        if "drafted_plan" in memory["plans"]:
                            del memory["plans"]["drafted_plan"]
                        result = "Planning completed with validated plan"
                        tool_results.append(result)
                        log_data = {
                            "task": task,
                            "memory_input": relevant_memory,
                            "tool_results": tool_results,
                            "final_response": parsed,
                            "timestamp": datetime.now().isoformat()
                        }
                        save_to_json(log_data, f'planner_agent_{datetime.now().isoformat()}.json', subdirectory='PlannerAgent')
                        print(f"\n📋 PLANNER AGENT COMPLETED: {result}")
                        return json.dumps(final_plan) if isinstance(final_plan, dict) else final_plan
                    else:
                        result = "Cannot finalize: missing plan"
                        tool_results.append(result)

            else:
                print(f"Unknown action type: {action_type}")
                break

            iteration += 1
            memory_str = json.dumps({
                "user_profile": memory.get("user_profile", {}),
                "flags_and_assessments": memory.get("flags_and_assessments", {}),
                "plans": memory.get("plans", {})
            }, indent=2)

        result = f"Planning stopped after {max_iterations} iterations with {len(tool_results)} actions"
        print(f"📋 PLANNER AGENT Stopped (MAX ITERATIONS)")
        return result