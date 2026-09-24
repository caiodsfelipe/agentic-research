import functools
import os

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from agentic_research.nodes.brainstormer import brainstormer
from agentic_research.nodes.code_reader import code_reader
from agentic_research.nodes.differ import differ
from agentic_research.nodes.external_librarian import external_librarian
from agentic_research.nodes.internal_librarian import internal_librarian
from agentic_research.state import AppState
from agentic_research.tools.project_map import build_project_map

LIBRARIANS = ["internal_librarian", "external_librarian"]


# Runs the requested tools, or fans out to the librarians once the code reader has answered
def route_after_code_reader(state: AppState) -> str | list[str]:
    if state["messages"][-1].tool_calls:
        return "execute_tools"
    return LIBRARIANS


def build_graph(mcp_tools: list) -> CompiledStateGraph:
    graph = StateGraph(AppState)

    codebase_path = os.getenv("CODEBASE_PATH")
    if not codebase_path:
        raise ValueError("CODEBASE_PATH is not defined in .env")

    bound_reader = functools.partial(
        code_reader,
        mcp_tools=mcp_tools,
        codebase_path=codebase_path,
        project_map=build_project_map(codebase_path),
    )

    graph.add_node("code_reader", bound_reader)
    graph.add_node("execute_tools", ToolNode(mcp_tools))
    graph.add_node("internal_librarian", internal_librarian)
    graph.add_node("external_librarian", external_librarian)
    graph.add_node("differ", differ)
    graph.add_node("brainstormer", brainstormer)

    graph.add_conditional_edges(
        "code_reader",
        route_after_code_reader,
        ["execute_tools", *LIBRARIANS],
    )

    graph.add_edge(START, "code_reader")
    graph.add_edge("execute_tools", "code_reader")

    # differ waits for all librarians to finish
    graph.add_edge(LIBRARIANS, "differ")
    graph.add_edge("differ", "brainstormer")
    graph.add_edge("brainstormer", END)

    return graph.compile()
