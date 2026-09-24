from agentic_research.tools.project_map import build_project_map
from agentic_research.tools.rag import _document_status


def test_document_status_reads_front_matter(tmp_path):
    doc = tmp_path / "doc.md"
    doc.write_text("---\ntitle: X\nstatus: historical\n---\nBody", encoding="utf-8")
    assert _document_status(str(doc)) == "historical"


def test_document_without_status_counts_as_current(tmp_path):
    doc = tmp_path / "doc.md"
    doc.write_text("No front matter", encoding="utf-8")
    assert _document_status(str(doc)) == "current"


def test_document_status_missing_file(tmp_path):
    assert _document_status(str(tmp_path / "missing.md")) == "unknown"


def test_project_map_skips_artifact_folders(tmp_path):
    (tmp_path / "detector" / "src").mkdir(parents=True)
    (tmp_path / "detector" / "data").mkdir()
    (tmp_path / "detector" / "train.py").write_text("")
    (tmp_path / "detector" / "image.png").write_text("")
    (tmp_path / "venv").mkdir()

    project_map = build_project_map(str(tmp_path))

    assert project_map == "detector/: src/, train.py"
