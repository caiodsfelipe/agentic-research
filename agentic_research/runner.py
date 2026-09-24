from collections.abc import Callable

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import HumanMessage

from agentic_research.graph import build_graph
from agentic_research.tools.mcp_loader import code_search_tools
from agentic_research.tools.memory import save_research


async def run_research(
    question: str, on_update: Callable[[str, dict], None] | None = None, save_to_memory: bool = True
) -> tuple[dict, dict]:
    """
    Runs the research graph for a question, passing each node's update to on_update as it happens.
    With save_to_memory, the finished run is appended to the research history (a record, never fed back to the agents).
    Returns the final state and the token usage (input/output) per model.
    """
    usage = UsageMetadataCallbackHandler()
    state = {"messages": [HumanMessage(content=question)], "user_input": question}

    async with code_search_tools() as mcp_tools:
        graph = build_graph(mcp_tools)
        stream = graph.astream(state, config={"callbacks": [usage]}, stream_mode=["updates", "values"])
        async for mode, chunk in stream:
            if mode == "values":
                state = chunk
            elif on_update:
                for node, output in chunk.items():
                    on_update(node, output)

    if save_to_memory and state.get("ideas"):
        save_research(state, question)

    return state, usage.usage_metadata
