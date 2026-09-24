"""
Command-line entry point.

    python main.py "Our recall is stuck. What does the literature do differently?"
    python main.py        # asks for the question interactively
"""

import argparse
import asyncio

from dotenv import load_dotenv

# Load environment variables before importing the app modules
load_dotenv()

from agentic_research.runner import run_research  # noqa: E402


async def main():
    parser = argparse.ArgumentParser(description="Run the research agents on a question.")
    parser.add_argument("question", nargs="?", help="The research question (asked interactively if omitted).")
    question = parser.parse_args().question or input("Research question: ")

    # Execute the graph, printing each node as it finishes
    final_state, usage = await run_research(question, on_update=lambda node, _: print(f"[{node}] done"))

    # Print the results
    print("Comparison:", final_state.get("comparison"))
    print("Ideas:", final_state.get("ideas"))

    # Print the token usage per model
    for model_name, model_usage in usage.items():
        print(
            f"Token usage [{model_name}]: input={model_usage['input_tokens']} "
            f"output={model_usage['output_tokens']} total={model_usage['total_tokens']}"
        )


if __name__ == "__main__":
    asyncio.run(main())
