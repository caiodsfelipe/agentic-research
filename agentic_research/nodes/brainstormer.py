from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from agentic_research.llm import get_llm
from agentic_research.prompts import BRAINSTORMER_PROMPT
from agentic_research.state import AppState
from agentic_research.token_budget import fit_messages


class Ideas(BaseModel):
    ideas: list[str]
    solutions: list[str]  # ordered by expected impact: the first one is what to do right now


# Gets the comparison from the differ and brainstorms ideas and next steps for the project.
# The goal is to find implementation holes, methodology flaws, areas for improvement, among others.
async def brainstormer(state: AppState) -> dict:
    # Create a system message with the brainstorming prompt
    system_message = SystemMessage(content=BRAINSTORMER_PROMPT)

    # Create a human message with the research question and the comparison result from the differ node
    human_message = HumanMessage(
        content=(f"Research Question:\n{state['user_input']}\n\nComparison Result:\n{state.get('comparison', '')}")
    )

    # Get the LLM instance, constrained to the Ideas schema so the output is always valid JSON
    llm = get_llm("brainstormer").with_structured_output(Ideas)

    # Use the LLM to brainstorm ideas based on the comparison result
    brainstorming_result = await llm.ainvoke(fit_messages([system_message, human_message], "brainstormer"))

    return {"ideas": brainstorming_result.model_dump_json(indent=2)}
