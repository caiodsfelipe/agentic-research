from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


# Shared state of the research graph; each node writes only its own field
class AppState(TypedDict, total=False):
    user_input: str  # The research question
    messages: Annotated[list, add_messages]  # The code reader's tool-calling transcript
    tool_steps: int  # Number of code_reader calls so far
    code_analysis: str  # Output of the code reader
    internal_knowledge: str  # Output of the internal librarian
    external_knowledge: str  # Output of the external librarian
    comparison: str  # Output of the differ
    ideas: str  # Output of the brainstormer
