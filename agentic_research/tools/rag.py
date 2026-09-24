"""
Retrieval from the two knowledge bases in VECTOR_STORE_DIR:
- internal/: the project's own docs, balanced between what is in production and past attempts.
- external/: the state of the art literature.

Every excerpt is prefixed with "[source: <path>]" (plus "| status: <status>" for internal docs),
so the agents and the UI can tell documents apart.
"""

import os
import re
from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Chunks retrieved per search: external papers, and internal docs per status group
EXTERNAL_RESULTS = 3
INTERNAL_RESULTS_PER_GROUP = 2
# Internal status groups, retrieved separately so past attempts can't crowd out what is in production.
# Docs with any other status ("retracted", or "unknown" when the file no longer exists) are never retrieved.
INTERNAL_STATUS_GROUPS = (("current",), ("historical", "superseded"))


@lru_cache(maxsize=2)
def _get_vector_store(name: str) -> Chroma:
    # Built once per store and reused across calls. Read at call time (not import time),
    # so it works regardless of when load_dotenv() runs and can be redirected (e.g. by the eval suite).
    vector_store_dir = os.getenv("VECTOR_STORE_DIR")
    if not vector_store_dir:
        raise ValueError("VECTOR_STORE_DIR is not defined in .env")
    persist_directory = Path(vector_store_dir) / name

    # Chroma silently creates an empty store for a missing directory, so fail loudly instead
    if not persist_directory.is_dir():
        raise FileNotFoundError(f"Vector store not found: {persist_directory}. Check VECTOR_STORE_DIR in .env")

    return Chroma(persist_directory=str(persist_directory), embedding_function=OpenAIEmbeddings())


def _document_status(source: str) -> str:
    # Internal docs may declare a status (current, historical, superseded, retracted) in their YAML front matter;
    # a doc without one describes the project as it is, so it counts as "current".
    # Read on every search (not cached), so a changed status applies without restarting the app.
    try:
        with open(source, encoding="utf-8", errors="ignore") as file:
            front_matter = re.match(r"^---\s*\n(.*?)\n---", file.read(2000), re.S)
    except OSError:
        return "unknown"  # the file was moved or deleted since it was ingested
    status = front_matter and re.search(r"^status:\s*[\"']?([\w-]+)", front_matter.group(1), re.M)
    return status.group(1) if status else "current"


def _internal_sources_by_status() -> dict[str, list[str]]:
    # Groups every internal source file by its status, so searches can be filtered with Chroma's metadata filter
    metadatas = _get_vector_store("internal").get(include=["metadatas"])["metadatas"]
    sources_by_status: dict[str, list[str]] = {}
    for source in {metadata["source"] for metadata in metadatas}:
        sources_by_status.setdefault(_document_status(source), []).append(source)
    return sources_by_status


def search_external(query: str) -> str:
    # MMR picks relevant but diverse chunks, avoiding near-duplicates
    results = _get_vector_store("external").max_marginal_relevance_search(query, k=EXTERNAL_RESULTS, fetch_k=10)
    return "\n\n".join(f"[source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}" for doc in results)


def search_internal(query: str) -> str:
    vector_store = _get_vector_store("internal")
    sources = _internal_sources_by_status()
    sections = []
    for statuses in INTERNAL_STATUS_GROUPS:
        allowed = [source for status in statuses for source in sources.get(status, [])]
        if not allowed:
            continue
        results = vector_store.max_marginal_relevance_search(
            query, k=INTERNAL_RESULTS_PER_GROUP, fetch_k=10, filter={"source": {"$in": allowed}}
        )
        for doc in results:
            source = doc.metadata["source"]
            sections.append(f"[source: {source} | status: {_document_status(source)}]\n{doc.page_content}")
    return "\n\n".join(sections)
