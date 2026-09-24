"""
Evaluators for the research graph. Each one scores a single run and returns LangSmith feedback.

Three families, from cheapest to most subjective:
- Deterministic (code): retrieval_recall checks that the expected source documents were retrieved.
- LLM-as-judge with a standard prompt (openevals): groundedness (RAGAS calls it "faithfulness").
- LLM-as-judge with custom rubrics: conclusion_coverage checks that the reference answer's key
  conclusions are reached (extra correct detail is not penalized), and status_discipline checks that
  past, failed or superseded approaches are not presented as current or re-proposed without justification.
"""

from langchain.chat_models import init_chat_model
from openevals.llm import create_async_llm_as_judge
from openevals.prompts import RAG_GROUNDEDNESS_PROMPT

# openevals' CORRECTNESS_PROMPT penalizes any detail absent from the reference, which rewards short answers.
# A research answer can correctly add context, so this rubric scores the reference's conclusions instead.
CONCLUSION_COVERAGE_PROMPT = """You are an expert reviewer comparing a research answer with an expert's
reference answer.

<question>
{inputs}
</question>

<answer>
{outputs}
</answer>

<reference_answer>
{reference_outputs}
</reference_answer>

1. List the key conclusions of the reference answer: its direct answer to the question, its recommendations,
   and anything it explicitly advises against. Ignore supporting details such as exact numbers.
2. For each key conclusion, decide whether the answer reaches it (fully, partially, or not at all).
3. Check whether the answer contradicts any key conclusion, for example by recommending something the
   reference advises against.

Score from 0 to 1: the share of key conclusions the answer reaches, with a large penalty for each contradiction.
Do NOT penalize additional details, context or recommendations that do not contradict the reference;
they may be correct information the reference simply omits."""

STATUS_DISCIPLINE_PROMPT = """You are an expert reviewer of research recommendations.

The project has approaches that were already tried and abandoned, or replaced (historical):
<historical_approaches>
{historical_approaches}
</historical_approaches>

<question>
{inputs}
</question>

<answer>
{outputs}
</answer>

Score the answer as correct only if it does NOT present any historical approach as part of the
current system, and does NOT recommend any of them as a new idea without explicitly acknowledging
that it was already tried and explaining what would be different this time.
Mentioning a historical approach as a past attempt is fine."""


def retrieval_recall(outputs: dict, reference_outputs: dict) -> dict:
    # Fraction of the expected source documents that appear in the retrieved context
    expected = reference_outputs["relevant_sources"]
    retrieved = outputs["retrieved"]
    found = [source for source in expected if source in retrieved]
    return {
        "key": "retrieval_recall",
        "score": len(found) / len(expected),
        "comment": f"Missing: {[s for s in expected if s not in found]}"
        if len(found) < len(expected)
        else "All expected sources retrieved",
    }


def build_llm_evaluators(judge_model: str) -> list:
    """Builds the LLM-as-judge evaluators. The judge should be at least as strong as the models being judged."""
    judge = init_chat_model(judge_model, temperature=0)

    groundedness_judge = create_async_llm_as_judge(
        prompt=RAG_GROUNDEDNESS_PROMPT, judge=judge, feedback_key="groundedness", continuous=True
    )
    coverage_judge = create_async_llm_as_judge(
        prompt=CONCLUSION_COVERAGE_PROMPT, judge=judge, feedback_key="conclusion_coverage", continuous=True
    )
    status_judge = create_async_llm_as_judge(
        prompt=STATUS_DISCIPLINE_PROMPT, judge=judge, feedback_key="status_discipline"
    )

    # LangSmith passes (inputs, outputs, reference_outputs); these wrappers map them to each judge's prompt variables
    async def groundedness(outputs: dict) -> dict:
        return await groundedness_judge(outputs=outputs["answer"], context=outputs["context"])

    async def conclusion_coverage(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
        return await coverage_judge(
            inputs=inputs["question"],
            outputs=outputs["answer"],
            reference_outputs=reference_outputs["reference_answer"],
        )

    async def status_discipline(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
        return await status_judge(
            inputs=inputs["question"],
            outputs=outputs["answer"],
            historical_approaches="\n".join(f"- {a}" for a in reference_outputs["historical_approaches"]),
        )

    return [groundedness, conclusion_coverage, status_discipline]
