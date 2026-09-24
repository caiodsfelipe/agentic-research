import pytest

from agentic_research.tools.memory import load_history, save_research


@pytest.fixture(autouse=True)
def history_file(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_HISTORY_PATH", str(tmp_path / "history.jsonl"))


def test_empty_history():
    assert load_history() == []


def test_saved_runs_are_listed_newest_first():
    save_research({"comparison": "c1", "ideas": "i1"}, "First question?")
    save_research({"comparison": "c2", "ideas": "i2"}, "Second question?")

    history = load_history()

    assert [record["question"] for record in history] == ["Second question?", "First question?"]
    assert history[0]["ideas"] == "i2" and "date" in history[0]


def test_malformed_lines_are_skipped(tmp_path):
    save_research({"comparison": "c", "ideas": "i"}, "Valid question?")
    with (tmp_path / "history.jsonl").open("a", encoding="utf-8") as file:
        file.write('{"date": "2026-01-01", "question": "interrupt')

    assert [record["question"] for record in load_history()] == ["Valid question?"]
