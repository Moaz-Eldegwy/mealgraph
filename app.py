"""Gradio demo app for the MealGraph — entry point for the Hugging Face Space.

The app is intentionally thin: settings sidebar -> chat -> trace pane. The
heavy lifting stays in the agent system. The whole point is to *show* the
multi-agent architecture, not to build a product UI.

Run locally::

    pip install -r requirements.txt gradio
    python app.py

On Hugging Face Spaces, this file is the auto-detected entry point. The
README's metadata block (sdk: gradio, app_file: app.py) tells the platform
what to do.

Two UX affordances on top of the bare-bones chat:

* **Live progress panel** — a textbox that accumulates one line per agent
  / tool start + finish, so the user can see what's happening without
  opening the verbose trace accordion.
* **Stop button** — signals a cooperative interrupt between agent /
  tool steps. We can't kill a running Gemini HTTP request mid-flight,
  but every ``handle_task`` checks the stop flag at entry, so the run
  unwinds at the next boundary (typically within a few seconds) and the
  chatbot receives a "stopped by user" message instead of an answer.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import traceback
from io import StringIO
from typing import Any, Dict, Iterator, List, Optional, Tuple

import gradio as gr

import mealgraph
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
# Progress + cooperative-stop machinery
# ---------------------------------------------------------------------------
class StopRequested(Exception):
    """Raised inside an agent / tool wrapper when the user pressed Stop."""


class ProgressBus:
    """One per chat turn. Carries progress lines + a cooperative stop flag.

    Producers (the agent / tool wrappers) push human-readable lines onto
    ``queue`` and call :meth:`check_stop` at the start of each step. The
    consumer (the chat generator) drains ``queue`` and yields each line
    to the UI. The sentinel value ``None`` signals "no more events".
    """

    def __init__(self) -> None:
        self.queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self.stop_event = threading.Event()

    def emit(self, line: str) -> None:
        self.queue.put(line)

    def check_stop(self) -> None:
        if self.stop_event.is_set():
            raise StopRequested("user requested stop")

    def end(self) -> None:
        self.queue.put(None)


# Module-global handle on the currently-running run. The Stop button uses
# it to flip the stop flag on the active bus. Guarded by a lock so a
# concurrent Stop press and chat-start can't race.
_CURRENT_BUS: Optional[ProgressBus] = None
_CURRENT_BUS_LOCK = threading.Lock()


def _set_current_bus(bus: Optional[ProgressBus]) -> None:
    global _CURRENT_BUS
    with _CURRENT_BUS_LOCK:
        _CURRENT_BUS = bus


def _peek_current_bus() -> Optional[ProgressBus]:
    with _CURRENT_BUS_LOCK:
        return _CURRENT_BUS


def request_stop() -> str:
    """Stop-button handler. Signals the current run; safe no-op if none."""
    bus = _peek_current_bus()
    if bus is None:
        return "ℹ️ No run is in progress."
    bus.stop_event.set()
    return "🛑 Stop requested — unwinding at next agent/tool boundary…"


# Display labels for the three agents. Anything not in this map falls
# through to the raw class name, which is fine for debugging.
_AGENT_LABELS = {
    "CoachAgent": "🏋️ Coach",
    "MedicalAssessmentAgent": "👨‍⚕️ Medical Assessment",
    "PlannerAgent": "📋 Planner",
}
_TOOL_LABELS = {
    "WebSearchTool": "🌐 WebSearchTool",
    "QuantitiesFinder": "📊 QuantitiesFinder",
}


def _install_progress_hooks() -> None:
    """Idempotent: wrap every agent/tool's ``handle_task`` once.

    Each wrapper:
    1. Reads the current bus from the module global (``None`` outside a
       chat turn, in which case the wrapper is transparent).
    2. Calls ``bus.check_stop()`` to honour a pending stop request.
    3. Emits a start line, runs the original, emits a finish line.
    4. For the Coach specifically, decodes the chosen action so the
       progress panel can show "Coach → MedicalAssessmentAgent" rather
       than the opaque "Coach: done".
    """
    if mealgraph.AGENTS is None or mealgraph.TOOLS is None:
        return  # system not initialised yet

    for name, agent in mealgraph.AGENTS.items():
        _wrap_agent(name, agent)
    for name, tool in mealgraph.TOOLS.items():
        _wrap_tool(name, tool)


def _wrap_agent(name: str, agent: Any) -> None:
    if getattr(agent, "_progress_wrapped", False):
        return
    label = _AGENT_LABELS.get(name, name)
    orig = agent.handle_task

    def traced(*args: Any, **kwargs: Any) -> Any:
        bus = _peek_current_bus()
        if bus is not None:
            bus.check_stop()
            bus.emit(f"⏳ {label}: starting…")
        try:
            result = orig(*args, **kwargs)
            if bus is not None:
                bus.emit(_summarise_agent_result(name, label, result))
            return result
        except StopRequested:
            if bus is not None:
                bus.emit(f"🛑 {label}: interrupted")
            raise

    agent.handle_task = traced  # type: ignore[method-assign]
    agent._progress_wrapped = True  # type: ignore[attr-defined]


def _wrap_tool(name: str, tool: Any) -> None:
    if getattr(tool, "_progress_wrapped", False):
        return
    label = _TOOL_LABELS.get(name, name)
    orig = tool.handle_task

    def traced(task: str) -> str:
        bus = _peek_current_bus()
        if bus is not None:
            bus.check_stop()
            bus.emit(f"  └─ {label}: running…")
        try:
            result = orig(task)
            if bus is not None:
                bus.emit(f"  └─ {label}: done")
            return result
        except StopRequested:
            if bus is not None:
                bus.emit(f"  └─ {label}: interrupted")
            raise

    tool.handle_task = traced  # type: ignore[method-assign]
    tool._progress_wrapped = True  # type: ignore[attr-defined]


def _summarise_agent_result(name: str, label: str, result: Any) -> str:
    """One-line summary of an agent step for the progress panel.

    The Coach's ``handle_task`` returns the full updated state dict,
    which includes ``current_action`` — surfacing it makes the trace
    much easier to follow ("Coach → Planner" beats "Coach: done").
    Workers return free-form text, so we fall back to "done".
    """
    if name == "CoachAgent" and isinstance(result, dict):
        action = (result.get("current_action") or {}).get("action")
        params = (result.get("current_action") or {}).get("params") or {}
        if action == "call_agent":
            return f"✓ Coach → {params.get('agent_name', '?')}: {params.get('task', '')[:80]}"
        if action == "ask_user":
            return f"✓ Coach: asking user — {params.get('prompt', '')[:80]}"
        if action == "compose_response":
            return "✓ Coach: composing final answer"
        if action == "write_memory":
            return f"✓ Coach: write memory ({params.get('partition', '?')})"
        return f"✓ {label}: {action or 'done'}"
    return f"✓ {label}: done"


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
        "tools_llm": {"model_name": tools_model},
    }
    try:
        if debug_on:
            mealgraph.debug(level="output")
        mealgraph.create_llm_instances(keys, overrides, enable_rate_limiting=rate_limit)
        mealgraph.initialize_tools()
        mealgraph.initialize_agents()
        mealgraph.setup_workflow()
        mealgraph.initialize_long_term_memory()
        init_langsmith()
        # Install progress + stop hooks last so every agent / tool we
        # just wired up gets wrapped exactly once.
        _install_progress_hooks()
        return (
            f"✅ System initialised with {len(keys)} key(s). "
            f"Coach={coach_model}, Workers={workers_model}, Tools (WebSearch)={tools_model}."
        )
    except Exception as e:  # noqa: BLE001
        return f"❌ Initialisation failed: {e}\n\n{traceback.format_exc()}"


# ---------------------------------------------------------------------------
# Per-call log capture
# ---------------------------------------------------------------------------
class _BufferHandler(logging.Handler):
    """Captures every mealgraph.* log line into a string buffer for the UI."""

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
    root = logging.getLogger("mealgraph")
    root.addHandler(handler)
    return handler


def _detach_buffer(handler: _BufferHandler) -> None:
    logging.getLogger("mealgraph").removeHandler(handler)


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
    lab_results: str,
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
            "lab_results": lab_results.strip(),
        },
    }


# ---------------------------------------------------------------------------
# Chat handler (streaming generator)
# ---------------------------------------------------------------------------
ChatYield = Tuple[List[Dict[str, str]], str, str, SessionState, str]


def chat(
    user_message: str,
    history: List[Dict[str, str]],
    session: SessionState,
    profile_json: str,
) -> Iterator[ChatYield]:
    """Streaming handler. Yields ``(chat, trace, metrics, session, progress)``.

    Runs the LangGraph workflow on a background thread and pulls progress
    lines off the ProgressBus as they arrive. Each yield updates the UI;
    the final yield carries the composed answer.
    """
    if session is None:
        session = SessionState()
    if not history:
        history = []
    history = history + [{"role": "user", "content": user_message}]

    if mealgraph.APP is None:
        history.append(
            {"role": "assistant", "content": "❌ System not initialised. Use the sidebar Initialize button."}
        )
        yield history, "", "", session, "(system not initialised)"
        return

    # Update profile if user changed it.
    try:
        profile_data = json.loads(profile_json) if profile_json.strip() else {}
        if profile_data:
            session.memory["user_profile"] = profile_data.get(
                "user_profile", session.memory["user_profile"]
            )
            session.memory["medical_history"] = profile_data.get(
                "medical_history", session.memory["medical_history"]
            )
    except json.JSONDecodeError:
        pass

    bus = ProgressBus()
    _set_current_bus(bus)
    log_handler = _attach_buffer()

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

    result_holder: Dict[str, Any] = {}

    def runner() -> None:
        try:
            with span("end_to_end_chat", kind="agent"):
                final_state = mealgraph.APP.invoke(
                    state, config={"configurable": {"thread_id": session.thread_id}}
                )
            result_holder["state"] = final_state
        except StopRequested:
            result_holder["stopped"] = True
        except Exception as e:  # noqa: BLE001
            result_holder["error"] = e
        finally:
            bus.end()

    worker = threading.Thread(target=runner, daemon=True)
    worker.start()

    progress: List[str] = ["⏳ Starting workflow…"]
    # Initial yield so the UI clears the previous progress and shows the
    # first "Starting workflow" line immediately.
    yield history, log_handler.text(), "", session, "\n".join(progress)

    while True:
        line = bus.queue.get()
        if line is None:
            break
        progress.append(line)
        yield history, log_handler.text(), "", session, "\n".join(progress)

    worker.join(timeout=5)
    _detach_buffer(log_handler)
    _set_current_bus(None)

    # Final yield: append the assistant message to the chat and refresh
    # metrics. Branches over stopped / error / success.
    if result_holder.get("stopped"):
        progress.append("🛑 Run stopped by user.")
        history.append({"role": "assistant", "content": "🛑 Stopped by user before a plan was finalised."})
    elif "error" in result_holder:
        err = result_holder["error"]
        progress.append(f"⚠ Error: {err}")
        history.append({"role": "assistant", "content": f"⚠ Error: {err}"})
    else:
        final_state = result_holder["state"]
        session.memory = final_state["memory"]
        session.conversation_history = final_state["conversation_history"]
        session.previous_actions = final_state["previous_actions"]
        final_response = final_state.get("agent_result") or "(no response)"
        history.append({"role": "assistant", "content": str(final_response)})
        progress.append("✅ Done.")

    metrics_md = _render_metrics(get_metrics().snapshot())
    yield history, log_handler.text(), metrics_md, session, "\n".join(progress)


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
    with gr.Blocks(title="MealGraph — Multi-Agent Demo") as demo:
        gr.Markdown(
            """
            # 🥗 MealGraph — Nutrition Multi-Agent System
            A LangGraph + Gemini orchestrator: a **Coach** delegates to a
            **Medical Assessment** specialist (deterministic clinical
            formulas + LLM interpretation) and a **Planner** that combines
            grounded web search with a PuLP linear-program meal solver
            and runs its own post-solve safety check (allergy / calorie /
            macro tolerances). Bring your own Gemini API keys.
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
                _GEMINI_MODELS = [
                    "gemini-pro-latest",
                    "gemini-flash-latest",
                    "gemini-flash-lite-latest",
                ]
                coach_model = gr.Dropdown(
                    label="Coach model",
                    choices=_GEMINI_MODELS,
                    value="gemini-pro-latest",
                )
                workers_model = gr.Dropdown(
                    label="Workers (Medical / Planner) model",
                    choices=_GEMINI_MODELS,
                    value="gemini-pro-latest",
                )
                tools_model = gr.Dropdown(
                    label="Tools (WebSearch) model",
                    choices=_GEMINI_MODELS,
                    value="gemini-flash-lite-latest",
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
                p_lab_results = gr.Textbox(
                    label="Lab results",
                    placeholder=(
                        "e.g. Fasting glucose 110 mg/dL, HbA1c 6.1%, LDL 145 mg/dL, "
                        "TSH 2.3 µIU/mL — paste a recent panel or summary."
                    ),
                    value="",
                    lines=4,
                )
                profile_json = gr.Textbox(visible=False)

                def _refresh_profile(*args: Any) -> str:
                    return json.dumps(build_user_profile(*args))

                _profile_inputs = [
                    p_name, p_age, p_sex, p_height, p_weight, p_activity, p_goal,
                    p_allergies, p_dislikes, p_country, p_conditions, p_medications,
                    p_lab_results,
                ]
                for component in _profile_inputs:
                    component.change(
                        _refresh_profile,
                        inputs=_profile_inputs,
                        outputs=profile_json,
                    )

            # ---------------- Main pane ----------------
            with gr.Column(scale=2):
                # Gradio 5+ defaults to the "tuples" format when ``type`` is
                # omitted, which clashes with our messages-shaped payload
                # ([{role, content}, ...]) and triggers a postprocess error
                # on the first turn. Pin ``type="messages"`` explicitly; the
                # arg is accepted by 4.40+ and is the only valid choice in
                # Gradio 6.
                try:
                    chatbot = gr.Chatbot(
                        label="Conversation", height=420, type="messages"
                    )
                except TypeError:
                    # Pre-4.40 Gradio doesn't know the ``type`` kwarg; fall
                    # back to the default and let the tuples postprocessor
                    # handle the legacy shape.
                    chatbot = gr.Chatbot(label="Conversation", height=420)
                user_input = gr.Textbox(
                    label="Your question",
                    placeholder="e.g. Build me a one-day meal plan to gain muscle.",
                    lines=2,
                )
                with gr.Row():
                    send_btn = gr.Button("Send", variant="primary", scale=3)
                    stop_btn = gr.Button("⏹ Stop", variant="stop", scale=1)
                # Plain-text progress feed: one short line per agent / tool
                # step. Kept separate from the verbose trace below so the
                # casual viewer sees a clean timeline.
                progress_panel = gr.Textbox(
                    label="Live progress",
                    lines=10,
                    interactive=False,
                    placeholder="Each agent / tool step will appear here as the workflow runs.",
                )
                stop_status = gr.Markdown(visible=False)
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

        # The chat handler is a generator now (streams progress events),
        # so Gradio will hold the connection open and re-render on each
        # yield. We capture the click and submit event handles so the
        # Stop button can ``cancels=`` them — Gradio will tear down the
        # generator on the server side as soon as Stop is clicked, while
        # our own bus.stop_event makes the background workflow unwind
        # cleanly at the next agent / tool boundary.
        chat_inputs = [user_input, chatbot, session_state, profile_json]
        chat_outputs = [chatbot, trace_log, metrics_md, session_state, progress_panel]
        send_event = send_btn.click(
            chat, inputs=chat_inputs, outputs=chat_outputs
        ).then(lambda: "", None, user_input)
        submit_event = user_input.submit(
            chat, inputs=chat_inputs, outputs=chat_outputs
        ).then(lambda: "", None, user_input)

        # Stop button: flip the bus flag + cancel the Gradio-side
        # generator. ``cancels=`` is best-effort across Gradio versions;
        # the bus flag is the actual stop signal.
        stop_btn.click(
            request_stop,
            inputs=None,
            outputs=stop_status,
        )
        # Wire ``cancels`` in a try / except — older Gradio releases
        # don't accept it on a separate ``.click()`` chain.
        try:
            stop_btn.click(  # type: ignore[call-arg]
                lambda: gr.update(visible=True),
                inputs=None,
                outputs=stop_status,
                cancels=[send_event, submit_event],
            )
        except TypeError:
            pass

        gr.Markdown(
            """
            ---
            **About**: This demo runs a 3-agent system (Coach, Medical
            Assessment, Planner) on top of two safe-by-construction tools
            (PuLP `QuantitiesFinder`, Gemini-grounded `WebSearchTool`). The
            Planner runs its own *deterministic* check (allergy / calorie /
            macro tolerances) after the LP solver and self-revises up to
            twice. The Coach then does an LLM-graded self-review (medical
            flag respect, citation presence, cultural fit) before composing.
            See the GitHub repo for the full architecture writeup.
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
