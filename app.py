"""Gradio demo app for the Nutrition MAS — entry point for the Hugging Face Space.

The app is intentionally thin: settings sidebar -> chat -> trace pane. The
heavy lifting stays in the agent system. The whole point is to *show* the
multi-agent architecture, not to build a product UI.

Run locally::

    pip install -r requirements.txt gradio
    python app.py

On Hugging Face Spaces, this file is the auto-detected entry point. The
README's metadata block (sdk: gradio, app_file: app.py) tells the platform
what to do.
"""

from __future__ import annotations

import json
import logging
import traceback
from io import StringIO
from typing import Any, Dict, List, Tuple

import gradio as gr

import nutritionmas
from agent_cards import build_default_registry
from logging_setup import get_logger
from observability import get_metrics, init_langsmith, span
from state import initialize_empty_memory

_logger = get_logger("app")


# ---------------------------------------------------------------------------
# Per-session state
# ---------------------------------------------------------------------------
class SessionState:
    """Holds per-user MAS state. One instance per Gradio session.

    Kept deepcopy-safe (no threads, no locks) because Gradio's ``gr.State``
    deep-copies the initial value on every session. Concurrency safety is
    provided by Gradio's per-session queue, which serialises handler calls
    for a single browser tab.
    """

    def __init__(self) -> None:
        self.initialised: bool = False
        self.memory: Dict[str, Any] = initialize_empty_memory()
        self.conversation_history: List[Dict[str, str]] = []
        self.previous_actions: List[str] = []
        self.thread_id: str = "session-default"


# ---------------------------------------------------------------------------
# Bootstrapping
# ---------------------------------------------------------------------------
def initialise_system(
    api_keys_text: str,
    coach_model: str,
    workers_model: str,
    tools_model: str,
    rate_limit: bool,
    debug_on: bool,
) -> str:
    """Spin up the MAS once with the supplied keys + per-role model overrides."""
    keys = [k.strip() for k in api_keys_text.splitlines() if k.strip()]
    if not keys:
        return "❌ Please paste at least one Gemini API key (one per line)."

    overrides = {
        "main": {"model_name": coach_model},
        "agents_llm": {"model_name": workers_model},
        "planner_agent": {"model_name": workers_model},
        "validation_agent": {"model_name": tools_model},
        "tools_llm": {"model_name": tools_model},
    }
    try:
        if debug_on:
            nutritionmas.debug(level="output")
        nutritionmas.create_llm_instances(keys, overrides, enable_rate_limiting=rate_limit)
        nutritionmas.initialize_tools()
        nutritionmas.initialize_agents()
        nutritionmas.setup_workflow()
        nutritionmas.initialize_long_term_memory()
        init_langsmith()
        return (
            f"✅ System initialised with {len(keys)} key(s). "
            f"Coach={coach_model}, Workers={workers_model}, Tools/Validator={tools_model}."
        )
    except Exception as e:  # noqa: BLE001
        return f"❌ Initialisation failed: {e}\n\n{traceback.format_exc()}"


# ---------------------------------------------------------------------------
# Per-call log capture
# ---------------------------------------------------------------------------
class _BufferHandler(logging.Handler):
    """Captures every nutrition_mas.* log line into a string buffer for the UI."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.buffer = StringIO()
        self.setFormatter(logging.Formatter("%(name)s — %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        self.buffer.write(self.format(record) + "\n")

    def text(self) -> str:
        return self.buffer.getvalue()


def _attach_buffer() -> _BufferHandler:
    handler = _BufferHandler()
    root = logging.getLogger("nutrition_mas")
    root.addHandler(handler)
    return handler


def _detach_buffer(handler: _BufferHandler) -> None:
    logging.getLogger("nutrition_mas").removeHandler(handler)


# ---------------------------------------------------------------------------
# Profile builder (sidebar form)
# ---------------------------------------------------------------------------
def build_user_profile(
    name: str,
    age: float,
    sex: str,
    height_cm: float,
    weight_kg: float,
    activity: str,
    goal: str,
    allergies: str,
    dislikes: str,
    country: str,
    conditions: str,
    medications: str,
) -> Dict[str, Any]:
    return {
        "user_profile": {
            "name": name or "Anonymous",
            "age": age,
            "sex": sex,
            "height": height_cm,
            "weight": weight_kg,
            "activity_level": activity,
            "goal": goal,
            "food_dislikes": dislikes,
            "allergies": [a.strip() for a in allergies.split(",") if a.strip()],
            "country": country,
            "currency": "USD",
        },
        "medical_history": {
            "conditions": [c.strip() for c in conditions.split(",") if c.strip()],
            "medications": [m.strip() for m in medications.split(",") if m.strip()],
            "past_issues": [],
            "lab_results": "",
        },
    }


# ---------------------------------------------------------------------------
# Chat handler
# ---------------------------------------------------------------------------
def chat(
    user_message: str,
    history: List[Dict[str, str]],
    session: SessionState,
    profile_json: str,
) -> Tuple[List[Dict[str, str]], str, str, SessionState]:
    """Single-turn handler. Returns (chat_history, trace_log, metrics_md, session).

    Uses Gradio 5+ "messages" Chatbot format: ``[{"role": "user"/"assistant",
    "content": "..."}, ...]``.
    """
    if session is None:
        session = SessionState()
    if not history:
        history = []
    history = history + [{"role": "user", "content": user_message}]
    if nutritionmas.APP is None:
        history.append(
            {"role": "assistant", "content": "❌ System not initialised. Use the sidebar Initialize button."}
        )
        return history, "", "", session

    # Update profile if user changed it.
    try:
        profile_data = json.loads(profile_json) if profile_json.strip() else {}
        if profile_data:
            session.memory["user_profile"] = profile_data.get("user_profile", session.memory["user_profile"])
            session.memory["medical_history"] = profile_data.get(
                "medical_history", session.memory["medical_history"]
            )
    except json.JSONDecodeError:
        pass

    handler = _attach_buffer()
    final_response = ""
    error_text = ""
    try:
        state = {
            "memory": session.memory,
            "user_question": user_message,
            "conversation_history": session.conversation_history
            + [{"role": "user", "content": user_message}],
            "current_action": None,
            "agent_result": None,
            "num_turns": 0,
            "max_turns": 12,
            "previous_actions": session.previous_actions,
            "response_steps": [],
        }
        with span("end_to_end_chat", kind="agent"):
            final_state = nutritionmas.APP.invoke(
                state, config={"configurable": {"thread_id": session.thread_id}}
            )
        session.memory = final_state["memory"]
        session.conversation_history = final_state["conversation_history"]
        session.previous_actions = final_state["previous_actions"]
        final_response = final_state.get("agent_result") or "(no response)"
    except Exception as e:  # noqa: BLE001
        error_text = f"\n\n⚠ Error: {e}"
    finally:
        log_text = handler.text()
        _detach_buffer(handler)

    history.append({"role": "assistant", "content": str(final_response) + error_text})
    metrics = get_metrics().snapshot()
    metrics_md = _render_metrics(metrics)
    return history, log_text, metrics_md, session


def _render_metrics(snap: Dict[str, Any]) -> str:
    lines = ["### System metrics", "", "| Component | Calls | Total (s) | Errors |", "|---|---|---|---|"]
    for name, m in snap["agents"].items():
        lines.append(f"| agent · {name} | {m['calls']} | {m['total_seconds']:.2f} | {m['errors']} |")
    for name, m in snap["tools"].items():
        lines.append(f"| tool · {name} | {m['calls']} | {m['total_seconds']:.2f} | {m['errors']} |")
    p = snap["parsing"]
    lines.append("")
    lines.append(
        f"**Parsing**: native={p['native']}  fallback={p['fallback']}  failure={p['failure']}"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def build_demo() -> gr.Blocks:
    registry = build_default_registry()
    cards_md = "## Active agents\n\n" + "\n".join(
        f"- **{c.name}** ({c.role}) — {c.description}" for c in registry.list()
    )

    # ``theme`` moved to launch() in Gradio 6+; we still support 4/5 by passing
    # it here AND at launch() — the latter wins on newer versions.
    with gr.Blocks(title="Nutrition MAS — Multi-Agent Demo") as demo:
        gr.Markdown(
            """
            # 🥗 Nutrition Multi-Agent System
            A LangGraph + Gemini orchestrator that delegates to a Medical
            Assessment specialist, a Planner (with a PuLP linear-program meal
            solver), a Validator critic, and a citation-first Knowledge
            agent. Bring your own Gemini API keys.
            """
        )

        # gr.State deepcopies the initial value on every session, so seed it
        # with None and let the chat handler instantiate SessionState lazily.
        session_state = gr.State(None)

        with gr.Row():
            # ---------------- Sidebar ----------------
            with gr.Column(scale=1):
                gr.Markdown("### 1. Setup")
                # Multi-line key input — type="password" doesn't allow >1 line in
                # Gradio 5+, so we use plain text and rely on browser/HF Space
                # to keep it ephemeral.
                api_keys = gr.Textbox(
                    label="Gemini API key(s) — one per line",
                    placeholder="AIza...\nAIza...",
                    lines=3,
                )
                coach_model = gr.Dropdown(
                    label="Coach model",
                    choices=["gemini-2.5-pro", "gemini-2.5-flash"],
                    value="gemini-2.5-pro",
                )
                workers_model = gr.Dropdown(
                    label="Workers (Medical / Planner) model",
                    choices=["gemini-2.5-pro", "gemini-2.5-flash"],
                    value="gemini-2.5-pro",
                )
                tools_model = gr.Dropdown(
                    label="Validator / Tools model",
                    choices=["gemini-2.5-flash", "gemini-2.5-pro"],
                    value="gemini-2.5-flash",
                )
                rate_limit = gr.Checkbox(label="Rate-limit Gemini calls", value=True)
                debug_on = gr.Checkbox(label="Debug logging", value=False)
                init_btn = gr.Button("Initialize system", variant="primary")
                init_status = gr.Markdown()

                gr.Markdown("### 2. Your profile")
                p_name = gr.Textbox(label="Name", value="Demo User")
                p_age = gr.Number(label="Age", value=30, precision=0)
                p_sex = gr.Radio(label="Sex", choices=["male", "female"], value="male")
                p_height = gr.Number(label="Height (cm)", value=175)
                p_weight = gr.Number(label="Weight (kg)", value=72)
                p_activity = gr.Dropdown(
                    label="Activity",
                    choices=[
                        "sedentary",
                        "lightly active",
                        "moderately active",
                        "very active",
                        "extra active",
                    ],
                    value="moderately active",
                )
                p_goal = gr.Dropdown(
                    label="Goal",
                    choices=["lose weight", "maintain weight", "gain muscle", "gain weight"],
                    value="maintain weight",
                )
                p_allergies = gr.Textbox(label="Allergies (comma-separated)", value="")
                p_dislikes = gr.Textbox(label="Dislikes", value="")
                p_country = gr.Textbox(label="Country", value="USA")
                p_conditions = gr.Textbox(label="Medical conditions", value="")
                p_medications = gr.Textbox(label="Medications", value="")
                profile_json = gr.Textbox(visible=False)

                def _refresh_profile(*args: Any) -> str:
                    return json.dumps(build_user_profile(*args))

                for component in [
                    p_name, p_age, p_sex, p_height, p_weight, p_activity, p_goal,
                    p_allergies, p_dislikes, p_country, p_conditions, p_medications,
                ]:
                    component.change(
                        _refresh_profile,
                        inputs=[
                            p_name, p_age, p_sex, p_height, p_weight, p_activity, p_goal,
                            p_allergies, p_dislikes, p_country, p_conditions, p_medications,
                        ],
                        outputs=profile_json,
                    )

            # ---------------- Main pane ----------------
            with gr.Column(scale=2):
                # Gradio 6 dropped the `type=` parameter; the messages format
                # ([{role, content}]) is now the only one. Older versions still
                # accept `type="messages"` so we keep the same payload shape.
                chatbot = gr.Chatbot(label="Conversation", height=420)
                user_input = gr.Textbox(
                    label="Your question",
                    placeholder="e.g. Build me a one-day meal plan to gain muscle.",
                    lines=2,
                )
                send_btn = gr.Button("Send", variant="primary")
                with gr.Accordion("🔍 Agent activity (live trace)", open=False):
                    trace_log = gr.Textbox(label="Log", lines=12, interactive=False)
                with gr.Accordion("📈 System metrics", open=False):
                    metrics_md = gr.Markdown()
                with gr.Accordion("🤖 Agent registry (A2A cards)", open=False):
                    gr.Markdown(cards_md)

        init_btn.click(
            initialise_system,
            inputs=[api_keys, coach_model, workers_model, tools_model, rate_limit, debug_on],
            outputs=init_status,
        )
        send_btn.click(
            chat,
            inputs=[user_input, chatbot, session_state, profile_json],
            outputs=[chatbot, trace_log, metrics_md, session_state],
        ).then(lambda: "", None, user_input)
        user_input.submit(
            chat,
            inputs=[user_input, chatbot, session_state, profile_json],
            outputs=[chatbot, trace_log, metrics_md, session_state],
        ).then(lambda: "", None, user_input)

        gr.Markdown(
            """
            ---
            **About**: This demo runs a 5-agent system (Coach, Medical
            Assessment, Planner, Validation, Knowledge). The Validator
            applies *deterministic* checks (allergy violations, calorie /
            macro tolerances, HITL escalation) plus an LLM-graded layer for
            medical-flag respect and citation presence. See the GitHub repo
            for the full architecture writeup.
            """
        )
    return demo


def main() -> None:
    demo = build_demo()
    try:
        demo.queue().launch(theme=gr.themes.Soft())
    except TypeError:
        # Gradio 4.x doesn't accept theme at launch().
        demo.queue().launch()


if __name__ == "__main__":  # pragma: no cover
    main()
