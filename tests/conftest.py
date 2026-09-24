import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from agentic_research.tools import rag


@pytest.fixture
def fake_embeddings():
    # Offline, deterministic embeddings: no API calls in tests
    return DeterministicFakeEmbedding(size=32)


@pytest.fixture
def vector_store_dir(tmp_path, monkeypatch, fake_embeddings):
    # Points retrieval at an empty temporary VECTOR_STORE_DIR that uses the fake embeddings
    monkeypatch.setenv("VECTOR_STORE_DIR", str(tmp_path))
    monkeypatch.setattr(rag, "OpenAIEmbeddings", lambda: fake_embeddings)
    rag._get_vector_store.cache_clear()
    yield tmp_path
    rag._get_vector_store.cache_clear()
