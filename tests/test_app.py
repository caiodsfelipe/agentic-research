from pathlib import Path

from streamlit.testing.v1 import AppTest

from agentic_research.prompts import BRAINSTORMER_PROMPT, CODE_READER_PROMPT, COMPARE_KNOWLEDGE_PROMPT

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_prompts_have_the_domain_filled_in():
    for prompt in (COMPARE_KNOWLEDGE_PROMPT, BRAINSTORMER_PROMPT, CODE_READER_PROMPT):
        assert "{domain}" not in prompt


def test_app_loads_and_lists_saved_history(tmp_path, monkeypatch):
    history = tmp_path / "history.jsonl"
    history.write_text('{"date": "2026-01-01T10:00", "question": "Q?", "comparison": "{}", "ideas": "{}"}\n')
    monkeypatch.setenv("RESEARCH_HISTORY_PATH", str(history))

    app = AppTest.from_file(str(APP)).run()

    assert not app.exception
    assert [expander.label for expander in app.sidebar.expander] == ["2026-01-01 10:00 · Q?"]
