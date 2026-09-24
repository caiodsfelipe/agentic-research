import asyncio

from agentic_research.state import AppState
from agentic_research.tools.rag import search_internal


# Agent that acts as a librarian, looking into the project's own docs for information relevant to the question.
async def internal_librarian(state: AppState) -> dict:
    # The Chroma search is synchronous, so it runs in a thread to keep the parallel librarians concurrent
    relevant_info = await asyncio.to_thread(search_internal, state["user_input"])

    return {"internal_knowledge": relevant_info}
