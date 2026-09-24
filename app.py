"""
Web UI for the research agents: streams each agent's progress, then shows the next steps, the comparison
and the evidence behind it. Run with: streamlit run app.py
"""

import asyncio
import json
import re
from pathlib import PureWindowsPath

from dotenv import load_dotenv

# Load environment variables before importing the app modules
load_dotenv()

import streamlit as st  # noqa: E402

from agentic_research.config import NODE_SETTINGS, get_node_settings  # noqa: E402
from agentic_research.llm import supports_temperature  # noqa: E402
from agentic_research.runner import run_research  # noqa: E402
from agentic_research.tools.memory import load_history  # noqa: E402

SOURCE_LINE = re.compile(r"^\[source: (.+?)(?: \| status: ([\w-]+))?\]$", re.M)
STATUS_COLORS = {"current": "green", "historical": "orange", "superseded": "orange", "retracted": "red"}

# Creativity presets for the brainstormer's temperature (0 to 1); the other agents always stay deterministic
CREATIVITY_LEVELS = {"Focused": 0.0, "Balanced": 0.4, "Exploratory": 0.8}

st.set_page_config(page_title="Agentic Research", page_icon=":material/science:", layout="wide")


# ---------- Formatting helpers ----------


def parse_json(text: str) -> dict | None:
    """Parses an agent's JSON answer into {section: [items]}, or None when it isn't a non-empty JSON object."""
    # Models sometimes wrap JSON in ```json fences
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not data:
        return None
    return {key: items if isinstance(items, list) else [items] for key, items in data.items()}


def format_item(item) -> str:
    text = item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
    text = text.replace("$", "\\$")  # avoid Streamlit rendering "$" as LaTeX
    for status, color in STATUS_COLORS.items():
        text = text.replace(f"[{status}]", f":{color}-badge[{status}]")
    return text


def file_name(source: str) -> str:
    # PureWindowsPath understands both / and \ separators, so file names work for sources from any OS
    return PureWindowsPath(source).name


def short_model(model: str) -> str:
    return model.split(":", 1)[-1]


def shorten(text: str, limit: int = 55) -> str:
    # Cuts at a word boundary, so labels never end mid-word
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(",.;:") + "…"


# ---------- Result renderers (shared by live runs, replays and the history sidebar) ----------


def render_raw(box, text: str):
    # Fallback for answers that aren't the expected JSON (or are empty, like "" or "{}")
    if text.strip() in ("", "{}", "[]"):
        box.caption("Nothing reported.")
    else:
        box.markdown(format_item(text))


def render_next_steps(box, ideas_text: str):
    data = parse_json(ideas_text)
    if data is None:
        render_raw(box, ideas_text)
        return
    solutions, ideas = data.get("solutions", []), data.get("ideas", [])
    if solutions:
        box.success(f"**Start here:** {format_item(solutions[0])}", icon=":material/flag:")
        if len(solutions) > 1:
            box.markdown("\n".join(f"{i}. {format_item(s)}" for i, s in enumerate(solutions[1:], start=2)))
    if ideas:
        box.markdown("**More ideas**")
        box.markdown("\n".join(f"- {format_item(idea)}" for idea in ideas))


def render_comparison(box, comparison_text: str):
    data = parse_json(comparison_text)
    if data is None:
        render_raw(box, comparison_text)
        return
    tabs = box.tabs([f"{key.capitalize()} ({len(items)})" for key, items in data.items()])
    for tab, items in zip(tabs, data.values(), strict=True):
        tab.markdown("\n".join(f"- {format_item(item)}" for item in items) or "_Nothing reported._")


def render_code_analysis(box, analysis_text: str):
    data = parse_json(analysis_text)
    if data is None:
        render_raw(box, analysis_text)
        return
    for key, items in data.items():
        box.markdown(f"**{key.replace('_', ' ').capitalize()}**")
        box.markdown("\n".join(f"- {format_item(item)}" for item in items) or "_Nothing reported._")


def render_sources(box, text: str):
    sources = SOURCE_LINE.findall(text)
    if not sources:
        box.caption("No excerpts retrieved.")
        return
    for source, status in sources:
        badge = f":{STATUS_COLORS.get(status, 'gray')}-badge[{status}] " if status else ""
        box.markdown(f"- {badge}`{file_name(source)}`")
    box.caption("Excerpts")
    box.text(text)


# ---------- Live run layout ----------


def create_boxes() -> dict:
    progress = st.status("Researching…", expanded=True)

    next_steps = st.container(border=True)
    next_steps.subheader("Next steps", anchor=False)
    next_steps_body = next_steps.empty()
    next_steps_body.caption("Waiting for the comparison…")

    comparison = st.container(border=True)
    comparison.subheader("Comparison", anchor=False)
    comparison_body = comparison.empty()
    comparison_body.caption("Waiting for the code analysis and sources…")

    evidence = st.expander("Evidence: code analysis, project docs and papers", icon=":material/menu_book:")
    code_tab, internal_tab, papers_tab = evidence.tabs(["Code analysis", "Project docs", "Papers"])

    return {
        "progress": progress,
        "next_steps": next_steps_body,
        "comparison": comparison_body,
        "code": code_tab,
        "internal": internal_tab,
        "papers": papers_tab,
    }


def describe_tool_call(call: dict) -> str:
    args = ", ".join(f"{key}={value!r}" for key, value in call["args"].items())
    return f"`{call['name']}({args[:90]}{'…' if len(args) > 90 else ''})`"


def draw(boxes: dict, node: str, output: dict):
    # Draws one node update; used both while streaming and when replaying the last run
    progress = boxes["progress"]
    if node == "code_reader":
        message = output["messages"][0]
        for call in message.tool_calls:
            progress.markdown(f":material/search: Reading the codebase: {describe_tool_call(call)}")
        if not message.tool_calls:
            progress.markdown(f":material/code: Code analysis ready ({output['tool_steps'] - 1} tool calls)")
            render_code_analysis(boxes["code"], output["code_analysis"])
    elif node == "internal_librarian":
        count = len(SOURCE_LINE.findall(output["internal_knowledge"]))
        progress.markdown(f":material/folder_open: Retrieved {count} excerpts from the project docs")
        render_sources(boxes["internal"], output["internal_knowledge"])
    elif node == "external_librarian":
        count = len(SOURCE_LINE.findall(output["external_knowledge"]))
        progress.markdown(f":material/article: Retrieved {count} excerpts from the papers")
        render_sources(boxes["papers"], output["external_knowledge"])
    elif node == "differ":
        progress.markdown(":material/compare_arrows: Comparison ready")
        render_comparison(boxes["comparison"].container(), output["comparison"])
        boxes["next_steps"].caption("Brainstorming…")
    elif node == "brainstormer":
        render_next_steps(boxes["next_steps"].container(), output["ideas"])
        progress.update(label="Research complete", state="complete", expanded=False)


def draw_usage(usage: dict):
    parts = [
        f"{model}: {u['total_tokens']:,} tokens ({u['input_tokens']:,} in · {u['output_tokens']:,} out)"
        for model, u in usage.items()
    ]
    if parts:
        st.caption(":material/toll: " + "  ·  ".join(parts))


async def run_graph(question: str, boxes: dict, temperature: float | None) -> tuple[list, dict]:
    events = []

    def on_update(node: str, output: dict):
        events.append((node, output))
        draw(boxes, node, output)

    _, usage = await run_research(question, on_update, brainstormer_temperature=temperature)
    return events, usage


# ---------- Inputs ----------


@st.cache_data
def cached_supports_temperature(node: str, model: str) -> bool | None:
    # Looked up once per node and model, not on every Streamlit rerun
    return supports_temperature(node)


def creativity_input() -> float | None:
    """Compact creativity selector, disabled with an explanation when the brainstormer's model has no temperature."""
    model = get_node_settings("brainstormer")["model"]
    supported = cached_supports_temperature("brainstormer", model)
    level = st.segmented_control(
        "Creativity",
        options=list(CREATIVITY_LEVELS),
        default="Focused",
        required=True,
        disabled=supported is False,
        help="How varied the brainstormed ideas are (brainstormer temperature: "
        + ", ".join(f"{name} {value}" for name, value in CREATIVITY_LEVELS.items())
        + "). The code analysis and the comparison always stay deterministic.",
    )
    if supported is False:
        st.caption(
            f":material/info: The brainstormer model (`{model}`) does not accept a temperature with its current "
            "settings, so creativity can't be adjusted (see `agentic_research/config.py`)."
        )
        return None
    if supported is None:
        st.caption(f":material/info: Couldn't verify whether `{model}` accepts a temperature.")
    return CREATIVITY_LEVELS[level]


def draw_empty_state():
    steps = [
        (":material/code:", "Reads your code", "A code reader explores the codebase with read-only tools."),
        (":material/library_books:", "Retrieves evidence", "Your project docs and your selected papers."),
        (":material/lightbulb:", "Compares and brainstorms", "Differences with the papers, then ranked next steps."),
    ]
    for column, (icon, title, text) in zip(st.columns(len(steps)), steps, strict=True):
        with column.container(border=True):
            st.markdown(f"{icon} **{title}**")
            st.caption(text)


# ---------- Sidebar ----------


def draw_sidebar():
    with st.sidebar:
        st.markdown("### :material/history: Research history")
        history = load_history()
        if not history:
            st.caption("Finished runs appear here.")
        for record in history:
            with st.expander(shorten(record["question"])):
                details = [record["date"].replace("T", " ")]
                if record.get("brainstormer_temperature") is not None:
                    details.append(f"creativity {record['brainstormer_temperature']}")
                st.caption(" · ".join(details))
                st.markdown(f"**{format_item(record['question'])}**")
                next_steps_tab, comparison_tab = st.tabs(["Next steps", "Comparison"])
                render_next_steps(next_steps_tab, record["ideas"])
                render_comparison(comparison_tab, record["comparison"])

        st.divider()
        # Two trailing spaces make a Markdown line break
        lines = [
            f"{node.replace('_', ' ').capitalize()}: {short_model(settings['model'])}"
            for node, settings in NODE_SETTINGS.items()
            if node != "default"
        ]
        st.caption("  \n".join(["**Models**", *lines]))


# ---------- Page ----------

draw_sidebar()

st.title("Agentic Research", anchor=False)
st.caption("Compares your codebase and internal docs with your selected papers to brainstorm research directions.")

with st.form("research"):
    question = st.text_area(
        "Research question",
        placeholder="e.g. Our recall is stuck. What do the selected papers do differently? What should we try first?",
        height=110,
        label_visibility="collapsed",
    )
    creativity_column, run_column = st.columns([4, 1], vertical_alignment="bottom")
    with creativity_column:
        temperature = creativity_input()
    with run_column:
        run = st.form_submit_button("Run research", type="primary", icon=":material/play_arrow:", width="stretch")

if run and not question.strip():
    st.warning("Type a research question first.", icon=":material/edit:")
elif run:
    boxes = create_boxes()
    try:
        events, usage = asyncio.run(run_graph(question, boxes, temperature))
    except Exception as error:
        boxes["progress"].update(label="Research failed", state="error", expanded=True)
        st.error(f"{type(error).__name__}: {error}", icon=":material/error:")
    else:
        st.session_state["last_run"] = {"events": events, "usage": usage}
        draw_usage(usage)
elif "last_run" in st.session_state:
    # Streamlit reruns the script on every interaction, so the last run is redrawn from session state
    boxes = create_boxes()
    for node, output in st.session_state["last_run"]["events"]:
        draw(boxes, node, output)
    draw_usage(st.session_state["last_run"]["usage"])
else:
    draw_empty_state()
