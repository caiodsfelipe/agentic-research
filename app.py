import asyncio
import json
import re
from pathlib import PureWindowsPath

from dotenv import load_dotenv

# Load environment variables before importing the app modules
load_dotenv()

import streamlit as st  # noqa: E402

from agentic_research.runner import run_research  # noqa: E402
from agentic_research.tools.memory import load_history  # noqa: E402

# Streamlit UI for the research graph: streams each node's output as the graph runs.
# Run with: streamlit run app.py

SOURCE_LINE = re.compile(r"^\[source: (.+?)(?: \| status: ([\w-]+))?\]$", re.M)
STATUS_COLORS = {"current": "green", "historical": "orange", "superseded": "orange", "retracted": "red"}

st.set_page_config(page_title="Agentic Research", layout="wide")


def parse_json(text: str):
    # Models sometimes wrap JSON in ```json fences
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def format_item(item) -> str:
    text = item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
    text = text.replace("$", "\\$")  # avoid Streamlit rendering "$" as LaTeX
    for status, color in STATUS_COLORS.items():
        text = text.replace(f"[{status}]", f":{color}-badge[{status}]")
    return text


def render_result(box, text: str):
    # Renders a JSON answer ({"key": [items]}) as titled bullet lists, falling back to raw text
    data = parse_json(text)
    if not isinstance(data, dict):
        box.markdown(format_item(text))
        return
    for key, items in data.items():
        box.markdown(f"**{key.replace('_', ' ').capitalize()}**")
        for item in items if isinstance(items, list) else [items]:
            box.markdown(f"- {format_item(item)}")


def render_sources(box, text: str):
    for source, status in SOURCE_LINE.findall(text):
        badge = f":{STATUS_COLORS.get(status, 'gray')}-badge[{status}] " if status else ""
        # PureWindowsPath understands both / and \ separators, so file names work for sources from any OS
        box.markdown(f"- {badge}`{PureWindowsPath(source).name}`")
    with box.expander("Retrieved excerpts"):
        st.text(text)


def create_boxes() -> dict:
    code = st.status("Code reader", expanded=False)
    internal_column, external_column = st.columns(2)
    internal = internal_column.container(border=True)
    internal.markdown("#### Internal knowledge")
    external = external_column.container(border=True)
    external.markdown("#### External knowledge")
    comparison = st.container(border=True)
    comparison.markdown("#### Comparison")
    ideas = st.container(border=True)
    ideas.markdown("#### Ideas")
    return {"code": code, "internal": internal, "external": external, "comparison": comparison, "ideas": ideas}


def draw(boxes: dict, node: str, output: dict):
    # Draws one node update into its box; used both while streaming and when replaying a stored run
    if node == "code_reader":
        message = output["messages"][0]
        for call in message.tool_calls:
            boxes["code"].markdown(f"`{call['name']}` {json.dumps(call['args'], ensure_ascii=False)}")
        if not message.tool_calls:
            boxes["code"].update(label=f"Code reader: done ({output['tool_steps'] - 1} tool calls)", state="complete")
            render_result(boxes["code"], output["code_analysis"])
    elif node == "execute_tools":
        for message in output["messages"]:
            boxes["code"].caption(message.text[:300])
    elif node == "internal_librarian":
        render_sources(boxes["internal"], output["internal_knowledge"])
    elif node == "external_librarian":
        render_sources(boxes["external"], output["external_knowledge"])
    elif node == "differ":
        render_result(boxes["comparison"], output["comparison"])
    elif node == "brainstormer":
        render_result(boxes["ideas"], output["ideas"])


def draw_usage(usage: dict):
    st.markdown("#### Token usage")
    columns = st.columns(max(len(usage), 1))
    # strict=False: with no usage there is still one (empty) column
    for column, (model, model_usage) in zip(columns, usage.items(), strict=False):
        column.metric(model, f"{model_usage['total_tokens']:,} tokens")
        column.caption(f"input {model_usage['input_tokens']:,} · output {model_usage['output_tokens']:,}")


async def run_graph(question: str, boxes: dict) -> tuple[list, dict]:
    events = []

    def on_update(node: str, output: dict):
        events.append((node, output))
        draw(boxes, node, output)

    _, usage = await run_research(question, on_update)
    return events, usage


def draw_history_sidebar():
    # Browsable record of past runs; it is never fed back into the agents
    st.sidebar.markdown("### Research history")
    history = load_history()
    if not history:
        st.sidebar.caption("No saved runs yet.")
    for record in history:
        with st.sidebar.expander(f"{record['date'].replace('T', ' ')} · {record['question'][:60]}"):
            st.markdown(f"**Question:** {format_item(record['question'])}")
            render_result(st, record["comparison"])
            render_result(st, record["ideas"])


draw_history_sidebar()

st.title("Agentic Research")
st.caption("Compares the codebase and internal docs with the state of the art, then proposes next steps.")

question = st.text_area("Research question", height=120)
run = st.button("Run", type="primary", disabled=not question.strip())

if run:
    boxes = create_boxes()
    boxes["code"].update(label="Code reader: running…", state="running", expanded=True)
    events, usage = asyncio.run(run_graph(question, boxes))
    st.session_state["last_run"] = {"events": events, "usage": usage}
    draw_usage(usage)
elif "last_run" in st.session_state:
    # Streamlit reruns the script on every interaction, so the last run is redrawn from session state
    boxes = create_boxes()
    for node, output in st.session_state["last_run"]["events"]:
        draw(boxes, node, output)
    draw_usage(st.session_state["last_run"]["usage"])
