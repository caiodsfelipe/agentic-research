from langchain_core.messages import AIMessage

from agentic_research.graph import route_after_code_reader


def test_routes_to_tools_when_code_reader_calls_a_tool():
    message = AIMessage(content="", tool_calls=[{"name": "read_file", "args": {}, "id": "1"}])
    assert route_after_code_reader({"messages": [message]}) == "execute_tools"


def test_fans_out_to_both_librarians_when_code_reader_answers():
    message = AIMessage(content='{"structure": []}')
    assert route_after_code_reader({"messages": [message]}) == ["internal_librarian", "external_librarian"]
