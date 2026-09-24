"""
Prints the agent graph as a Mermaid diagram, or writes it as a PNG with --png.

    python -m scripts.draw_graph         # Mermaid text (paste into a README; GitHub renders it)
    python -m scripts.draw_graph --png   # agent_architecture.png (rendered by the mermaid.ink web service)
"""

import argparse
import os

from dotenv import load_dotenv

load_dotenv()

from agentic_research.graph import build_graph  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw the agent graph.")
    parser.add_argument("--png", action="store_true", help="Write agent_architecture.png instead of printing Mermaid.")
    args = parser.parse_args()

    # The graph's structure depends on neither the tools nor the analyzed project, so no Serena server
    # or configured codebase is needed
    os.environ.setdefault("CODEBASE_PATH", ".")
    graph = build_graph(mcp_tools=[]).get_graph()

    if args.png:
        with open("agent_architecture.png", "wb") as file:
            file.write(graph.draw_mermaid_png())
    else:
        print(graph.draw_mermaid())
