from evals.evaluators import retrieval_recall


def test_retrieval_recall_counts_expected_sources():
    outputs = {"retrieved": "[source: docs/CURRENT_PIPELINE.md | status: current]\ntext\n[source: papers/a.md]\ntext"}
    reference = {"relevant_sources": ["CURRENT_PIPELINE.md", "a.md", "missing.md"]}

    result = retrieval_recall(outputs, reference)

    assert result["score"] == 2 / 3
    assert "missing.md" in result["comment"]
