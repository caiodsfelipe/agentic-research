from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately

from agentic_research.config import MAX_TOOL_RESULT_TOKENS, get_node_settings


def truncate_text(text: str, max_tokens: int) -> str:
    max_chars = max_tokens * 4  # ~4 characters per token
    return text if len(text) <= max_chars else text[:max_chars] + "\n...[truncated]"


def fit_messages(messages: list[BaseMessage], node: str) -> list[BaseMessage]:
    """
    Trims messages to the node's input budget.
    The system prompt and the first human message (the user's question) are always kept;
    the remaining history is trimmed from the oldest side, starting on an AI message
    so tool results are never separated from the tool call that produced them.
    """
    max_input = get_node_settings(node)["max_input"]
    messages = [_truncate_tool_result(m) if isinstance(m, ToolMessage) else m for m in messages]

    pinned = []
    rest = list(messages)
    if rest and isinstance(rest[0], SystemMessage):
        pinned.append(rest.pop(0))
    if rest and isinstance(rest[0], HumanMessage):
        pinned.append(rest.pop(0))

    remaining = max_input - count_tokens_approximately(pinned)
    if remaining <= 0:
        # Pinned messages alone exceed the budget: cut the text of the last pinned message.
        return trim_messages(
            pinned,
            max_tokens=max_input,
            token_counter=count_tokens_approximately,
            strategy="first",
            allow_partial=True,
        )

    trimmed = (
        trim_messages(
            rest,
            max_tokens=remaining,
            token_counter=count_tokens_approximately,
            strategy="last",
            start_on="ai",
            allow_partial=False,
        )
        if rest
        else []
    )

    return pinned + trimmed


def _truncate_tool_result(message: ToolMessage) -> ToolMessage:
    max_chars = MAX_TOOL_RESULT_TOKENS * 4  # ~4 characters per token
    # MCP tools return a list of content blocks, so the plain text is used instead of the raw content
    text = message.text
    if len(text) > max_chars:
        return message.model_copy(update={"content": text[:max_chars] + "\n...[truncated]"})
    return message
