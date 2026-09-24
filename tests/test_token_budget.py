from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately

from agentic_research.config import MAX_TOOL_RESULT_TOKENS, NODE_SETTINGS
from agentic_research.token_budget import fit_messages, truncate_text


def tool_call(call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": "read_file", "args": {}, "id": call_id}])


def test_truncate_text_keeps_short_text():
    assert truncate_text("short", max_tokens=10) == "short"


def test_truncate_text_cuts_long_text():
    result = truncate_text("x" * 1000, max_tokens=10)
    assert result.startswith("x" * 40) and result.endswith("[truncated]")


def test_fit_messages_respects_budget_and_keeps_question():
    messages = [SystemMessage("system"), HumanMessage("question")]
    for i in range(20):
        messages += [tool_call(str(i)), ToolMessage("result " * 200, tool_call_id=str(i))]

    fitted = fit_messages(messages, "code_reader")

    assert fitted[:2] == messages[:2]
    assert count_tokens_approximately(fitted) <= NODE_SETTINGS["code_reader"]["max_input"]


def test_fit_messages_never_starts_history_with_orphan_tool_result():
    messages = [SystemMessage("system"), HumanMessage("question")]
    for i in range(20):
        messages += [tool_call(str(i)), ToolMessage("result " * 200, tool_call_id=str(i))]

    fitted = fit_messages(messages, "code_reader")

    assert isinstance(fitted[2], AIMessage)


def test_fit_messages_truncates_mcp_content_blocks():
    # MCP tools return a list of content blocks instead of a string
    huge_result = ToolMessage([{"type": "text", "text": "y" * 100_000}], tool_call_id="1")
    messages = [SystemMessage("system"), HumanMessage("question"), tool_call("1"), huge_result]

    fitted = fit_messages(messages, "code_reader")

    assert len(fitted[-1].text) <= MAX_TOOL_RESULT_TOKENS * 4 + len("\n...[truncated]")
