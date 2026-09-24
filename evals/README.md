# Evaluation

How do you measure a system whose output is a research recommendation? There is no single right
answer, but there are things a good answer must do: use the right sources, stay faithful to them,
reach the expert's conclusion, and never present a failed past attempt as the current system.
This suite turns each of those into a score, and runs it as a [LangSmith](https://smith.langchain.com)
experiment so every change to the system can be compared against the previous ones.

## Concepts

- **Dataset** (`dataset.json`): questions (`inputs`) with what an expert expects (`outputs`): a reference
  answer, the source documents that should be retrieved, and the approaches that were already tried.
  It is uploaded to LangSmith once and versioned there.
- **Target**: the system under test. Here, one full graph run per question (`run_eval.py:target`).
- **Evaluators** (`evaluators.py`): functions that score one run. From cheapest to most subjective:

  | Metric | Type | Question it answers |
  |---|---|---|
  | `retrieval_recall` | code | Were the expected source documents retrieved? |
  | `groundedness` | LLM-as-judge ([openevals](https://github.com/langchain-ai/openevals)) | Are the answer's claims supported by the retrieved context? (RAGAS calls this "faithfulness") |
  | `conclusion_coverage` | LLM-as-judge, custom rubric | Does the answer reach the reference's key conclusions without contradicting them? Extra correct detail is not penalized (openevals' standard correctness prompt does penalize it, which rewards short answers). |
  | `status_discipline` | LLM-as-judge, custom rubric | Are historical/superseded approaches kept out of the "current" story and not re-proposed blindly? |

- **Experiment**: one pass of the target over the dataset, with all scores. The model settings are
  stored as metadata, so in LangSmith you can compare experiments side by side and see exactly which
  question got better or worse after a change.

Retrieval scores tell you whether a bad answer is a *retrieval* problem or a *reasoning* problem:
low `retrieval_recall` with low `correctness` means fix the retrieval first.

## Fixtures

`fixtures/` is a small **fictional** research project (a polyp detector), so the suite does not depend
on any private data:

- `codebase/`: the code the code reader explores.
- `internal_docs/`: project docs with a `status` in their front matter (`current`, `historical`, `superseded`).
- `external_papers/`: invented papers (the "selected papers" of the fictional project).

The facts are planted so that each question has a known best answer (for example, the dataset is split
by frame, and a paper shows that frame-level splits inflate F1).

## Running

Requires the eval extra (`pip install -e ".[evals]"`, included in `.[dev]`) and `LANGSMITH_API_KEY` and
`OPENAI_API_KEY` in `.env`.

```bash
python -m evals.run_eval --limit 1            # smoke test on one question
python -m evals.run_eval --split dev          # iterate on the dev split
python -m evals.run_eval                      # full experiment (dev and test)
python -m evals.run_eval --rescore <name>     # re-score an existing experiment after changing an evaluator
```

Each full run costs one graph run per question plus three judge calls per question; the default judge
is `openai:gpt-5.4-mini` (change it with `--judge-model`). The fixture vector stores are built under
`~/.agentic-research/eval_vector_store` (override with `EVAL_VECTOR_STORE_DIR`; Chroma requires an
ASCII-only path).

## Using your own data

Keep the same layout and format:

```bash
python -m evals.run_eval --fixtures path/to/my_fixtures --dataset path/to/my_dataset.json --dataset-name my-eval
```

Tips for a good dataset:

- Start with 10-30 real questions; write the reference answers yourself, as the domain expert.
- Include "trap" questions, where the tempting answer is a past failed attempt.
- Spot-check the judges: read a few scored runs in LangSmith and confirm you agree with the scores.
  A judge you disagree with is a bug in the rubric.
- Change one thing at a time (a prompt, a model, a retrieval setting) and compare experiments.
- Avoid overfitting to the dataset: tune on the `dev` split only, and look at `test` once, at the end.
  Fixes should be general rules, never facts about specific questions. If a change helps `dev` but not
  `test`, it fit the questions rather than improving the system.
- When you change an evaluator, re-score an existing experiment (`--rescore`) instead of re-running the
  system, so the judge change is measured on its own.
- The dataset syncs automatically on every run (updated in place, matched by question), and LangSmith
  keeps each version, so past experiments remain comparable.
