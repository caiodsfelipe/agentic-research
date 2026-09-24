from pathlib import Path

from langchain_chroma import Chroma

from agentic_research.ingest import ingest_documents_to_vector_store, load_documents


def test_load_documents_reads_text_files_and_skips_others(tmp_path):
    (tmp_path / "notes.md").write_text("markdown", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "paper.txt").write_text("text", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"not a document")

    documents = load_documents(str(tmp_path))

    assert sorted(doc.page_content for doc in documents) == ["markdown", "text"]


def test_ingesting_twice_does_not_duplicate_chunks(tmp_path, fake_embeddings):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("first document " * 200, encoding="utf-8")
    (docs / "b.md").write_text("second document", encoding="utf-8")
    store_path = str(tmp_path / "store")

    first = ingest_documents_to_vector_store(str(docs), store_path, embeddings=fake_embeddings)
    ingest_documents_to_vector_store(str(docs), store_path, embeddings=fake_embeddings)

    stored = Chroma(persist_directory=store_path, embedding_function=fake_embeddings).get()["ids"]
    assert len(stored) == first


def test_reingesting_removes_chunks_of_deleted_files(tmp_path, fake_embeddings):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "keep.md").write_text("kept document", encoding="utf-8")
    (docs / "remove.md").write_text("document that will be deleted", encoding="utf-8")
    other = tmp_path / "other"
    other.mkdir()
    (other / "unrelated.md").write_text("from another directory", encoding="utf-8")
    store_path = str(tmp_path / "store")
    ingest_documents_to_vector_store(str(docs), store_path, embeddings=fake_embeddings)
    ingest_documents_to_vector_store(str(other), store_path, embeddings=fake_embeddings)

    (docs / "remove.md").unlink()
    ingest_documents_to_vector_store(str(docs), store_path, embeddings=fake_embeddings)

    store = Chroma(persist_directory=store_path, embedding_function=fake_embeddings)
    sources = {Path(metadata["source"]).name for metadata in store.get(include=["metadatas"])["metadatas"]}
    assert sources == {"keep.md", "unrelated.md"}
