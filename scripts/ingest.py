"""
Ingests a folder of documents (PDF, Markdown, text) into a Chroma vector store.

    python scripts/ingest.py --document_directory path/to/docs --vector_store_path path/to/vector_store/internal
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

# Allows running this file directly (python scripts/ingest.py) as well as a module (python -m scripts.ingest)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
load_dotenv()

from agentic_research.ingest import ingest_documents_to_vector_store  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into a vector store.")
    parser.add_argument("--document_directory", required=True, help="Path to the directory containing documents.")
    parser.add_argument(
        "--vector_store_path", required=True, help="Path to the directory where the vector store will be saved."
    )
    args = parser.parse_args()

    ingest_documents_to_vector_store(args.document_directory, args.vector_store_path)
