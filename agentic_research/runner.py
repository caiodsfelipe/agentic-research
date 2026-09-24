from collections.abc import Callable

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import HumanMessage

from agentic_research.graph import build_graph
from agentic_research.llm import validate_temperature
from agentic_research.tools.mcp_loader import code_search_tools
from agentic_research.tools.memory import save_research


async def run_research(
    question: str,
    on_update: Callable[[str, dict], None] | None = None,
    save_to_memory: bool = True,
    brainstormer_temperature: float | None = None,
) -> tuple[dict, dict]:
    """
    Runs the research graph for a question, passing each node's update to on_update as it happens.
    With save_to_memory, the finished run is appended to the research history (a record, never fed back to the agents).
    brainstormer_temperature sets how creative the brainstormer's ideas are (0 to 1; None uses the default, 0).
    Returns the final state and the token usage (input/output) per model.
    """
    # Checked before the run, so an unsupported temperature fails immediately instead of after all the other agents
    if brainstormer_temperature is not None:
        validate_temperature("brainstormer", brainstormer_temperature)

    usage = UsageMetadataCallbackHandler()
    state = {"messages": [HumanMessage(content=question)], "user_input": question}

    async with code_search_tools() as mcp_tools:
        graph = build_graph(mcp_tools)
        config = {"callbacks": [usage], "configurable": {"brainstormer_temperature": brainstormer_temperature}}
        stream = graph.astream(state, config=config, stream_mode=["updates", "values"])
        async for mode, chunk in stream:
            if mode == "values":
                state = chunk
            elif on_update:
                for node, output in chunk.items():
                    on_update(node, output)

    if save_to_memory and state.get("ideas"):
        save_research(state, question, brainstormer_temperature)

    return state, usage.usage_metadata
