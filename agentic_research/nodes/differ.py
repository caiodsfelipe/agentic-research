from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from agentic_research.config import MAX_SECTION_TOKENS
from agentic_research.llm import get_llm
from agentic_research.prompts import COMPARE_KNOWLEDGE_PROMPT
from agentic_research.state import AppState
from agentic_research.token_budget import fit_messages, truncate_text


class Comparison(BaseModel):
    differences: list[str]
    similarities: list[str]
    implications: list[str]


# Compares the internal knowledge (code analysis and internal docs) with the external knowledge (the literature).
# The goal is to understand how the project's implementation differs from the state of the art.
async def differ(state: AppState) -> dict:
    # Create a system message with the comparison prompt
    system_message = SystemMessage(content=COMPARE_KNOWLEDGE_PROMPT)

    # Each section gets its own token share, so trimming never drops a whole section
    sections = {
        "code_analysis": state.get("code_analysis", ""),
        "internal_knowledge": state.get("internal_knowledge", ""),
        "external_knowledge": state.get("external_knowledge", ""),
    }
    knowledge = "\n\n".join(
        f"<{name}>\n{truncate_text(text, MAX_SECTION_TOKENS)}\n</{name}>" for name, text in sections.items()
    )

    # Create a human message with the research question, the internal and the external knowledge
    human_message = HumanMessage(
        content=f"<research_question>\n{state['user_input']}\n</research_question>\n\n{knowledge}"
    )

    # Get the LLM instance, constrained to the Comparison schema so the output is always valid JSON
    llm = get_llm("differ").with_structured_output(Comparison)

    # Use the LLM to compare the knowledge and get the differences
    comparison_result = await llm.ainvoke(fit_messages([system_message, human_message], "differ"))

    return {"comparison": comparison_result.model_dump_json(indent=2)}
