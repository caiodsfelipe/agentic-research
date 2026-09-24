"""
Ingestion of documents (PDF, Markdown, text) into a Chroma vector store.
Used by scripts/ingest.py (CLI) and by the evaluation suite to build its fixture stores.
"""

import hashlib
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

TEXT_EXTENSIONS = {".md", ".txt"}
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def load_documents(document_directory: str) -> list[Document]:
    # One document per file: PDFs are read with pypdf, text files as plain text.
    # Sources are absolute paths, so internal docs' status can be read from any working directory.
    documents = []
    for path in sorted(Path(document_directory).resolve().rglob("*")):
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
        elif suffix in TEXT_EXTENSIONS:
            text = path.read_text(encoding="utf-8", errors="ignore")
        else:
            continue
        documents.append(Document(page_content=text, metadata={"source": str(path)}))
    return documents


def ingest_documents_to_vector_store(document_directory: str, vector_store_path: str, embeddings=None) -> int:
    """
    Loads, splits and embeds every document of a directory into the vector store, returning the chunk count.
    The store is kept in sync with the directory: re-ingesting replaces each file's chunks instead of
    duplicating them, and removes the chunks of files that were deleted from the directory.
    """
    # Chroma can't load its index from paths with non-ASCII characters (e.g. "Área")
    if not str(Path(vector_store_path).resolve()).isascii():
        raise ValueError(f"Vector store path must be ASCII-only: {Path(vector_store_path).resolve()}")

    documents = load_documents(document_directory)
    chunks = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP).split_documents(
        documents
    )

    vector_store = Chroma(persist_directory=vector_store_path, embedding_function=embeddings or OpenAIEmbeddings())

    # Every chunk previously ingested from this directory is removed: the chunks of changed files are
    # re-added below, and those of deleted files are gone for good. Other directories' chunks are kept.
    directory = Path(document_directory).resolve()
    stored_sources = {metadata["source"] for metadata in vector_store.get(include=["metadatas"])["metadatas"]}
    for source in stored_sources:
        if Path(source).is_relative_to(directory):
            vector_store.delete(where={"source": source})

    # Deterministic ids (source + chunk position) make repeated runs idempotent
    ids = []
    chunk_index: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata["source"]
        chunk_index[source] = chunk_index.get(source, -1) + 1
        ids.append(hashlib.sha1(f"{source}:{chunk_index[source]}".encode()).hexdigest())

    if chunks:
        vector_store.add_documents(chunks, ids=ids)

    print(f"Vector database built successfully! {len(documents)} files, {len(chunks)} chunks.")
    return len(chunks)
