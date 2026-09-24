"""
Runs the evaluation suite as a LangSmith experiment.

    python -m evals.run_eval                      # all examples
    python -m evals.run_eval --limit 1            # quick, cheap smoke test
    python -m evals.run_eval --split dev          # iterate on the dev split; keep test for the final check
    python -m evals.run_eval --rescore <name>     # re-score an existing experiment with the current evaluators

The system is pointed at the fictional fixtures (codebase, internal docs, papers) instead of your
real project, so results are reproducible and the suite can be shared. To evaluate on your own data,
pass --fixtures and --dataset with the same folder layout and JSON format.
"""

import argparse
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

EVALS_DIR = Path(__file__).resolve().parent


def point_to_fixtures(fixtures: Path, vector_store_dir: Path, domain: str):
    # Overriding these before the app modules are imported redirects the whole system to the fixtures
    os.environ["CODEBASE_PATH"] = str(fixtures / "codebase")
    os.environ["VECTOR_STORE_DIR"] = str(vector_store_dir)
    os.environ["RESEARCH_DOMAIN"] = domain


def ingest_fixtures(fixtures: Path, vector_store_dir: Path):
    from agentic_research.ingest import ingest_documents_to_vector_store

    # Idempotent: re-running replaces each file's chunks instead of duplicating them
    ingest_documents_to_vector_store(str(fixtures / "internal_docs"), str(vector_store_dir / "internal"))
    ingest_documents_to_vector_store(str(fixtures / "external_papers"), str(vector_store_dir / "external"))


def sync_dataset(client, dataset_name: str, examples: list[dict]):
    # Updates examples in place (matched by question) instead of recreating the dataset: LangSmith versions
    # every change, and past experiments stay linked to the version they ran on
    if not client.has_dataset(dataset_name=dataset_name):
        client.create_dataset(
            dataset_name, description="Research questions with reference answers for the agentic research graph."
        )
    remote = {example.inputs["question"]: example for example in client.list_examples(dataset_name=dataset_name)}
    for example in examples:
        existing = remote.get(example["inputs"]["question"])
        split = example.get("split")
        if existing:
            # Only changed examples are updated, so unchanged runs don't create new dataset versions
            split_changed = split and split not in (existing.metadata or {}).get("dataset_split", [])
            if existing.outputs != example["outputs"] or split_changed:
                client.update_example(existing.id, outputs=example["outputs"], split=split)
        else:
            client.create_examples(dataset_name=dataset_name, examples=[example])


async def target(inputs: dict) -> dict:
    # The system under test: one full graph run per question
    from agentic_research.runner import run_research

    # Not saved to the research history: evaluation runs are not real research
    state, _ = await run_research(inputs["question"], save_to_memory=False)
    retrieved = f"{state.get('internal_knowledge', '')}\n\n{state.get('external_knowledge', '')}"
    return {
        "answer": f"Comparison:\n{state.get('comparison', '')}\n\nIdeas:\n{state.get('ideas', '')}",
        "context": f"Code analysis:\n{state.get('code_analysis', '')}\n\n{retrieved}",
        "retrieved": retrieved,
    }


async def main():
    parser = argparse.ArgumentParser(description="Run the evaluation suite on LangSmith.")
    parser.add_argument("--dataset-name", default="agentic-research-eval", help="LangSmith dataset name.")
    parser.add_argument("--dataset", type=Path, default=EVALS_DIR / "dataset.json", help="Local dataset JSON.")
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=EVALS_DIR / "fixtures",
        help="Folder with codebase/, internal_docs/ and external_papers/.",
    )
    parser.add_argument(
        "--judge-model",
        default="openai:gpt-5.4-mini",
        help="Model used by the LLM-as-judge evaluators (provider:model).",
    )
    parser.add_argument(
        "--domain",
        default="AI applied to medical imaging",
        help="Research domain of the fixtures (sets RESEARCH_DOMAIN).",
    )
    parser.add_argument("--limit", type=int, help="Evaluate only the first N examples.")
    parser.add_argument(
        "--split", choices=["dev", "test"], help="Evaluate only one split (tune on dev, confirm on test)."
    )
    parser.add_argument(
        "--rescore", help="Name of an existing experiment to re-score with the current evaluators (no new graph runs)."
    )
    args = parser.parse_args()

    # Chroma needs an ASCII-only path, so the eval stores live in the home folder by default
    vector_store_dir = Path(os.getenv("EVAL_VECTOR_STORE_DIR", Path.home() / ".agentic-research" / "eval_vector_store"))

    from langsmith import Client, aevaluate, aevaluate_existing

    from agentic_research.config import NODE_SETTINGS
    from evals.evaluators import build_llm_evaluators, retrieval_recall

    examples = json.loads(args.dataset.read_text(encoding="utf-8"))
    split_of = {example["inputs"]["question"]: example.get("split", "all") for example in examples}
    evaluators = [retrieval_recall, *build_llm_evaluators(args.judge_model)]
    client = Client()

    if args.rescore:
        # Only the evaluators run again, on the stored outputs: isolates a judge change from a system change
        results = await aevaluate_existing(args.rescore, evaluators=evaluators, max_concurrency=2)
    else:
        point_to_fixtures(args.fixtures, vector_store_dir, args.domain)
        ingest_fixtures(args.fixtures, vector_store_dir)
        sync_dataset(client, args.dataset_name, examples)

        data = client.list_examples(
            dataset_name=args.dataset_name, splits=[args.split] if args.split else None, limit=args.limit
        )
        results = await aevaluate(
            target,
            data=data,
            evaluators=evaluators,
            experiment_prefix="research-graph",
            # Stored with the experiment, so runs with different models or splits can be compared in LangSmith
            metadata={
                "node_settings": NODE_SETTINGS,
                "judge_model": args.judge_model,
                "domain": args.domain,
                "split": args.split or "all",
            },
            max_concurrency=1,  # one run at a time: keeps cost predictable and avoids parallel Serena servers
        )

    # Average score per metric and split, printed locally; the full breakdown is in the LangSmith experiment
    scores: dict[tuple[str, str], list[float]] = {}
    async for result in results:
        split = split_of.get(result["example"].inputs["question"], "all")
        for evaluation in result["evaluation_results"]["results"]:
            if evaluation.score is not None:
                scores.setdefault((evaluation.key, split), []).append(float(evaluation.score))
    print(f"\nExperiment: {results.experiment_name}")
    for (key, split), values in sorted(scores.items()):
        print(f"  {key} [{split}]: {sum(values) / len(values):.2f}  (n={len(values)})")


if __name__ == "__main__":
    asyncio.run(main())
