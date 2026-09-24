from agentic_research.ingest import ingest_documents_to_vector_store
from agentic_research.tools.rag import search_external, search_internal


def write_doc(folder, name, status, text):
    (folder / name).write_text(f"---\nstatus: {status}\n---\n{text}", encoding="utf-8")


def test_internal_search_labels_status_and_drops_retracted_docs(tmp_path, vector_store_dir, fake_embeddings):
    docs = tmp_path / "docs"
    docs.mkdir()
    write_doc(docs, "production.md", "current", "The production detector uses focal loss.")
    write_doc(docs, "old_experiment.md", "historical", "We tried a transformer backbone and it overfit.")
    write_doc(docs, "old_policy.md", "superseded", "The first release used a threshold of 0.5.")
    write_doc(docs, "wrong.md", "retracted", "This analysis was wrong.")
    (docs / "no_status.md").write_text("A doc without front matter.", encoding="utf-8")
    ingest_documents_to_vector_store(str(docs), str(vector_store_dir / "internal"), embeddings=fake_embeddings)

    result = search_internal("How does the detector work?")

    past = [result.index(label) for label in ("status: historical]", "status: superseded]") if label in result]
    assert "production.md | status: current]" in result
    assert past, "past attempts should be retrieved too"
    assert "wrong.md" not in result
    # A doc without a declared status counts as current
    assert "no_status.md | status: current]" in result
    # What is in production comes first
    assert result.index("status: current]") < min(past)


def test_external_search_labels_sources(tmp_path, vector_store_dir, fake_embeddings):
    papers = tmp_path / "papers"
    papers.mkdir()
    (papers / "paper.md").write_text("A paper about detection.", encoding="utf-8")
    ingest_documents_to_vector_store(str(papers), str(vector_store_dir / "external"), embeddings=fake_embeddings)

    result = search_external("detection")

    assert result.startswith("[source: ") and "paper.md]" in result
