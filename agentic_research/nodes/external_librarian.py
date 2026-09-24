import asyncio

from agentic_research.state import AppState
from agentic_research.tools.rag import search_external


# Agent that acts as a librarian, looking into the selected papers for information relevant to the question.
async def external_librarian(state: AppState) -> dict:
    # The Chroma search is synchronous, so it runs in a thread to keep the parallel librarians concurrent
    relevant_info = await asyncio.to_thread(search_external, state["user_input"])

    return {"external_knowledge": relevant_info}
