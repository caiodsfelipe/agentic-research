from langchain_core.messages import SystemMessage

from agentic_research.config import MAX_TOOL_STEPS
from agentic_research.llm import get_llm
from agentic_research.prompts import CODE_READER_PROMPT
from agentic_research.state import AppState
from agentic_research.token_budget import fit_messages


# Agent that explores the codebase (through Serena's read-only tools) to understand the project itself.
# The internal docs describe the project, but the code itself may have nuances worth exploring.
async def code_reader(state: AppState, mcp_tools: list, codebase_path: str, project_map: str) -> dict:
    tool_steps = state.get("tool_steps", 0)

    # On the last allowed step, tools stay visible but can't be called, forcing a final answer
    tool_choice = "none" if tool_steps >= MAX_TOOL_STEPS else "auto"
    # One tool call per step, so each result fits the input budget
    llm = get_llm("code_reader").bind_tools(mcp_tools, tool_choice=tool_choice, parallel_tool_calls=False)

    system_prompt = CODE_READER_PROMPT.replace("{codebase_path}", codebase_path).replace("{project_map}", project_map)

    messages = [SystemMessage(content=system_prompt)] + state.get("messages", [])

    # Use the LLM to process the messages and generate a response
    response = await llm.ainvoke(fit_messages(messages, "code_reader"))

    return {
        "messages": [response],
        # .text, not .content: some providers return content as a list of blocks
        "code_analysis": response.text,
        "tool_steps": tool_steps + 1,
    }
