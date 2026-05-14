---
title: Nutrition Multi-Agent System
emoji: 🥗
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
license: cc-by-nc-4.0
short_description: Multi-agent nutrition planner with LangGraph + Gemini
---

# 🥗 MealGraph — Nutrition Multi-Agent System

**Live demo:** <https://huggingface.co/spaces/moazeldegwy/mealgraph>

A clinical-nutrition planner built on **LangGraph** and **Gemini 3.x**
(`gemini-pro-latest` · `gemini-flash-latest` · `gemini-flash-lite-latest`).

Three agents — **Coach**, **MedicalAssessmentAgent**, and
**PlannerAgent** — sit on top of two safe-by-construction tools (a PuLP
linear-program meal solver and a Gemini-grounded web search). Clinical
math runs through closed-form Python formulas (Mifflin-St Jeor BMR,
ACSM activity multipliers); the LLM interprets the numbers but never
recomputes them. The Planner runs a deterministic plan check after the
LP solver and self-revises on allergy / calorie / macro violations
before returning. The Coach does an LLM-graded self-review (medical
flag respect, citation presence, cultural fit) before composing.

```
                ┌───────────────────────────────────────────┐
                │                  Coach                    │
                │  one typed action per turn (LangGraph)    │
                │  + self-review of the Planner's output    │
                └──────────────┬────────────────────────────┘
                               │ call_agent / ask_user
                               │ write_memory / compose_response
                ┌──────────────┴──────────────┐
                ▼                             ▼
       MedicalAssessment                  Planner
       deterministic math                draft -> LP -> check_plan
       + LLM enrichment                  (≤ 2 internal revisions)
                │                             │
                ▼                             ▼
         nutrition_formulas        QuantitiesFinder (PuLP)
         (BMI / BMR / TDEE)        WebSearchTool (grounded)
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
| **CoachAgent** | Orchestrator. Picks one action per turn (`call_agent` / `ask_user` / `write_memory` / `compose_response`). After the Planner returns, runs an LLM-graded self-review (medical-flag respect, citation presence, cultural fit) and triggers a revision when needed. | [agents.py](agents.py) |
| **MedicalAssessmentAgent** | Deterministic clinical math first (`full_assessment` → BMI / BMR / TDEE / macros), then an LLM step that emits flags / recommendations / evidence. The agent **overwrites** the LLM's `calculations` with the deterministic values so the math is exact by construction. | [agents.py](agents.py) |
| **PlannerAgent** | Drafts meals, batches nutrition lookups via the grounded `WebSearchTool`, runs the `QuantitiesFinder` LP, then runs `check_plan` (allergy / calorie / macro tolerances) inline. Up to two internal revisions resolve any blocking issue before returning. | [agents.py](agents.py) |
| **`check_plan`** | Deterministic post-LP critic. Allergy → severity `high` (hard block); calorie ±3 % / macro ±5 % → severity `medium`; disliked food → severity `low`. Same code path the Planner uses internally and the eval harness asserts against. | [agents.py](agents.py) |
| **QuantitiesFinder** | PuLP linear-program meal-quantity solver. Default per-food bounds `min = max(20, est × 0.3)`, `max = min(400, est × 2.5)` keep the LP from suggesting 1 g of butter or 900 g of broccoli. Estimate-anchor weight is `0.3`. | [tools.py](tools.py) |
| **WebSearchTool** | Single round-trip wrapper around Gemini's built-in `google_search` grounding. Returns answer + citations + queries from `grounding_metadata`. Prompt biases toward USDA / WHO / ADA / EFSA / NICE / FDA / MedlinePlus. | [tools.py](tools.py) |
| **LongTermMemory** | SQLite-backed semantic / procedural / episodic tiers. | [memory.py](memory.py) |
| **Guardrails** | Prompt-injection sniff, PII redaction, HITL escalation marker (`<<HITL:CLINICIAN_REVIEW_REQUIRED>>`). | [guardrails.py](guardrails.py) |
| **MCP server** | Exposes `QuantitiesFinder` and `assess_user` to Claude Desktop, Cursor, and any MCP-aware client. | [mcp_server.py](mcp_server.py) |
| **Agent cards** | A2A capability descriptors (three cards) with an in-process registry. | [agent_cards.py](agent_cards.py) |
| **Observability** | LangSmith passthrough + in-process metrics surface. | [observability.py](observability.py) |
| **Eval harness** | Three fixture personas; runs offline (no Gemini calls) against `check_plan`. | [evals/](evals/) |

### Models and rate limits

Three Gemini 3.x rolling aliases, mapped per role. The free-tier RPM /
RPD limits below are conservative defaults; override with
`enable_rate_limiting=False` (or pass a paid quota) if you have one.

| Alias | RPM | RPD | Default role |
|---|---:|---:|---|
| `gemini-pro-latest` | 5 | 100 | Coach, Medical, Planner |
| `gemini-flash-latest` | 10 | 250 | Available for overrides |
| `gemini-flash-lite-latest` | 15 | 500 | Tools (WebSearch), simulator |

### Safety guarantees

| Guarantee | Where it lives |
|---|---|
| **Allergies never appear in the plan** | Planner's `check_plan` — severity `high` → hard block → internal revision. |
| **Calorie target hit within ±3 %** | Planner's `check_plan` — severity `medium` → revision. |
| **Each macro hit within ±5 %** | Planner's `check_plan` — severity `medium` → revision. |
| **Medical flags respected** | Coach's self-review turn (LLM-graded). |
| **Clinical claims carry citations** | `WebSearchTool` returns `grounding_chunks` natively; Coach checks for them in self-review. |
| **Serious cases escalate** | Medical sets `requires_professional_consultation=True`; Coach appends `<<HITL:CLINICIAN_REVIEW_REQUIRED>>`. |
| **No RCE via LLM-generated code** | No code-from-LLM path exists at all. `nutrition_formulas` is closed-form Python; `QuantitiesFinder` is a pure LP. |
| **Deterministic math** | `full_assessment()` runs server-side; the Medical agent overwrites whatever the LLM emitted for `calculations`. |

### Run the offline eval harness

Three persona fixtures (athlete, diabetic, vegan-budget) exercise the
deterministic surface — no Gemini calls needed:

```bash
python -m evals.runner
```

### Run the test suite

```bash
pytest -ra
```

Coverage: schemas, solver behaviour, safety surface, rate-limit pool,
memory tiers, and full Coach ↔ specialist loops via a mock LLM. The
post-LP allergy revision and deterministic-calculation overwrite are
both unit-tested.

---

## Library usage

The same code runs as a library. Import the `mealgraph` module,
provide API keys, and call a few setup functions.

### 1. Import

```python
import mealgraph
```

### 2. API keys

Provide a list of keys; the system rotates through them and respects
each model's RPM / RPD limit. A single key is enough for evaluation.

```python
api_keys = [
    "your_api_key1",
    "your_api_key2",
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

### 4. Initialise

```python
mealgraph.create_llm_instances(api_keys, model_overrides, enable_rate_limiting=True)
mealgraph.initialize_tools()
mealgraph.initialize_agents()
mealgraph.setup_workflow()
```

### 5. Run

Either interactive mode (collects user data via stdin) or simulation
mode (drives one or more synthetic users through a fixed question list):

```python
mealgraph.run(simulate=False)
# or
mealgraph.run(simulate=True, simulated_users=[...])
```

## Behaviour notes

* **User mode output** — high-level progress lines, one per agent / tool
  action.
* **Debug mode output** — raw LLM input / output (or output only),
  scoped per agent / tool. Enable with `mealgraph.debug(level='full',
  scopes={'agents': ['CoachAgent'], 'tools': ['all']})`.
* **API-key pooling** — the manager rotates keys and (when rate
  limiting is on) enforces per-model RPM / RPD. Keys that exhaust their
  daily quota are dropped from the pool until the next UTC day.
* **Interactive mode** — prompts for profile fields, then accepts free
  questions; type `exit` to quit.
* **Simulation mode** — each entry in `simulated_users` is a dict with
  `user_profile`, `medical_history`, and `questions`; the loop drives
  each user's questions sequentially.
* **Error handling** — provide at least one API key (else
  `ValueError`). Each initialisation function checks that its
  predecessor has been called.
