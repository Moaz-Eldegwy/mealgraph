---
title: Nutrition Multi-Agent System
emoji: 🥗
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
license: mit
short_description: Multi-agent nutrition planner with LangGraph + Gemini
---

# 🥗 Nutrition Multi-Agent System

**Live demo:** <https://huggingface.co/spaces/moazeldegwy/mealgraph>

A clinical-nutrition planner built on **LangGraph** and **Gemini 3.x**
(`gemini-pro-latest` · `gemini-flash-latest` · `gemini-flash-lite-latest`).

A `CoachAgent` orchestrates four specialists — `MedicalAssessmentAgent`,
`PlannerAgent`, `ValidationAgent`, and `KnowledgeAgent` — and routes
between them based on a typed action plan. The Planner uses a **PuLP
linear-program solver** to translate LLM-drafted meals into exact gram
quantities; the Validator runs both deterministic checks (allergy
violations, calorie / macro tolerances) and an LLM-graded clinical pass
(medical-flag respect, citations, cultural fit) before any plan reaches
the user.

```
                ┌───────────────────────────────────────────┐
                │                  Coach                    │
                │  one typed action per turn (LangGraph)    │
                └──────────────┬────────────────────────────┘
                               │ call_agent / call_tool / ask_user
                               │ write_memory / compose_response
       ┌───────────────┬───────┴────────┬───────────────┬───────────────┐
       ▼               ▼                ▼               ▼               ▼
  Medical          Planner          Validation      Knowledge        Tools
  Assessment      (PuLP LP)        (critic loop)   (citations)    Compute /
                                                                  WebSearch /
                                                                  Quantities
```

## Quick start

```bash
pip install -r requirements.txt
python app.py
```

Open the Gradio UI, paste a Gemini API key, fill in the profile sidebar,
and ask for a plan.

The same repository deploys directly as a **Hugging Face Space** — the
YAML front-matter above is the Space manifest, and `app.py` is the
auto-detected entry point.

## Architecture

| Component | Role | Key file |
|---|---|---|
| **CoachAgent** | Orchestrator. Picks one action per turn (`call_agent` / `call_tool` / `ask_user` / `write_memory` / `compose_response`). | `agents.py` |
| **MedicalAssessmentAgent** | BMI / BMR / TDEE / macros, clinical flags, evidence sources. | `agents.py` |
| **PlannerAgent** | Drafts meals; runs `QuantitiesFinder` LP for exact grams. | `agents.py` |
| **ValidationAgent** | Generator-critic gate. Deterministic allergy / calorie / macro checks + LLM-graded medical-flag respect and citation requirement. | `validation.py` |
| **KnowledgeAgent** | Citation-first lookup, biased toward authoritative domains (USDA / WHO / ADA / EFSA / NICE). | `knowledge.py` |
| **QuantitiesFinder** | PuLP linear-program meal-quantity solver. Deterministic. | `tools.py` |
| **ComputationTool** | Closed-form clinical formulas (Mifflin-St Jeor BMR, ACSM activity multipliers). No subprocess, no `eval`. | `tools.py`, `nutrition_formulas.py` |
| **WebSearchTool** | Single-pass Gemini call with built-in `google_search` grounding; returns answer + cited URLs. | `tools.py` |
| **LongTermMemory** | SQLite-backed semantic / procedural / episodic tiers. | `memory.py` |
| **Guardrails** | Prompt-injection sniff, PII redaction, HITL escalation marker. | `guardrails.py` |
| **MCP server** | Exposes the same tools to Claude Desktop, Cursor, and any MCP-aware client. | `mcp_server.py` |
| **Agent cards** | A2A capability descriptors with an in-process registry. | `agent_cards.py` |
| **Observability** | LangSmith passthrough + in-process metrics surface. | `observability.py` |
| **Eval harness** | Three fixture personas; runs offline (no Gemini calls). | `evals/` |

### Models and rate limits

Three Gemini 3.x rolling aliases, mapped per role. The free-tier RPM /
RPD limits below are conservative defaults; override with
`enable_rate_limiting=False` (or pass a paid quota) if you have one.

| Alias | RPM | RPD | Default role |
|---|---:|---:|---|
| `gemini-pro-latest` | 5 | 100 | Coach, Medical, Planner |
| `gemini-flash-latest` | 10 | 250 | Available for overrides |
| `gemini-flash-lite-latest` | 15 | 500 | Tools, Validator, simulator |

### Validator semantics

The Validator returns one of three verdicts. The Coach reacts to each
verdict via its system prompt; the loop is bounded so revisions cannot
run away.

| Verdict | Coach response |
|---|---|
| `pass` | Proceed to `compose_response`. |
| `revise` | Re-call the Planner with the issues bundled into the task. Capped at two revisions. |
| `reject` | Refuse the request and emit a human-in-the-loop escalation marker. |

### Run the offline eval harness

Three persona fixtures (athlete, diabetic, vegan-budget) exercise the
deterministic surface — no Gemini calls needed:

```bash
python -m evals.runner
```

### Run the test suite

84 tests cover schemas, solver behaviour, safety surface, rate-limit
pool, memory tiers, and full Coach ↔ specialist loops via a mock LLM:

```bash
pytest -ra
```

---

## Library usage

The same code runs as a library. Import the `nutritionmas` module,
provide API keys, and call a few setup functions.

### 1. Import

```python
import nutritionmas
```

### 2. API keys

Provide a list of keys; the system rotates through them and respects
each model's RPM / RPD limit. A single key is enough for evaluation.

```python
api_keys = [
    "your_api_key1",
    "your_api_key2",
    # add more as needed
]
```

### 3. (Optional) Model overrides

Override `model_name` or `params` per role. Other configuration is fixed
in the module.

```python
model_overrides = {
    "main":       {"model_name": "gemini-pro-latest", "params": {"temperature": 0.5}},
    "agents_llm": {"model_name": "gemini-flash-latest", "params": {"max_tokens": 6000}},
}
```

Pass `None` (or skip the argument) to use the defaults.

### 4. (Optional) Toggle rate limiting

Rate limiting is on by default. Disable it for local development or when
you have paid-tier quota:

```python
nutritionmas.create_llm_instances(api_keys, model_overrides, enable_rate_limiting=False)
```

### 5. (Optional) Debug mode

User mode prints high-level progress:

```
Coach Agent: Calling MedicalAssessmentAgent with task 'Assess user health'
Medical Assessment Agent: Using ComputationTool for 'Calculate BMI'
Coach Agent: Composing final response
```

Debug mode adds raw LLM I/O:

```python
nutritionmas.debug(level='full', scopes={'agents': ['CoachAgent'], 'tools': ['all']})
```

* `level`: `'full'` (inputs + outputs) or `'output'` (outputs only).
* `scopes`: `{'agents': ['all' | <name>...], 'tools': ['all' | <name>...]}`.

### 6. Persistent logging

Dump every agent / tool I/O to a directory:

```python
nutritionmas.logging("path/to/log/dir")
```

### 7. Initialise

```python
nutritionmas.create_llm_instances(api_keys, model_overrides, enable_rate_limiting=True)
nutritionmas.initialize_tools()
nutritionmas.initialize_agents()
nutritionmas.setup_workflow()
```

### 8. Run

Either interactive mode (collects user data via stdin) or simulation
mode (drives one or more synthetic users through a fixed question list):

```python
nutritionmas.run(simulate=False)
# or
nutritionmas.run(simulate=True, simulated_users=[...])
```

### Full example

```python
import nutritionmas

api_keys = ["your_api_key1", "your_api_key2"]
model_overrides = {
    "main": {"model_name": "gemini-pro-latest", "params": {"temperature": 0.5}},
}

nutritionmas.logging("path/to/log/dir")
nutritionmas.create_llm_instances(api_keys, model_overrides, enable_rate_limiting=True)
nutritionmas.initialize_tools()
nutritionmas.initialize_agents()
nutritionmas.setup_workflow()
nutritionmas.run(simulate=False)
```

## Behaviour notes

* **User mode output** — high-level progress lines, one per agent / tool
  action.
* **Debug mode output** — raw LLM input / output (or output only),
  scoped per agent / tool.
* **API-key pooling** — the manager rotates keys and (when rate limiting
  is on) enforces per-model RPM / RPD. Keys that exhaust their daily
  quota are dropped from the pool until the next UTC day. Wait windows
  are computed *after* the response completes so generation time does
  not count against the budget. If every key is currently saturated,
  the next call sleeps until the earliest slot frees up.
* **Interactive mode** — prompts for profile fields, then accepts free
  questions; type `exit` to quit.
* **Simulation mode** — each entry in `simulated_users` is a dict with
  `user_profile`, `medical_history`, and `questions`; the loop drives
  each user's questions sequentially.
* **Error handling** — provide at least one API key (else `ValueError`).
  Each initialisation function checks that its predecessor has been
  called.
