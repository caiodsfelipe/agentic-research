import json
from pathlib import Path

import pytest
import streamlit as st
from langchain_core.messages import AIMessage
from streamlit.testing.v1 import AppTest

from agentic_research import llm, runner
from agentic_research.prompts import BRAINSTORMER_PROMPT, CODE_READER_PROMPT, COMPARE_KNOWLEDGE_PROMPT

APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture(autouse=True)
def empty_history(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_HISTORY_PATH", str(tmp_path / "history.jsonl"))
    st.cache_data.clear()
    yield tmp_path / "history.jsonl"
    st.cache_data.clear()


def test_prompts_have_the_domain_filled_in():
    for prompt in (COMPARE_KNOWLEDGE_PROMPT, BRAINSTORMER_PROMPT, CODE_READER_PROMPT):
        assert "{domain}" not in prompt


def test_app_loads_and_lists_saved_history(empty_history):
    question = "Our detector misses many flat polyps compared with the papers. How can we improve recall?"
    record = {"date": "2026-01-01T10:00", "question": question, "comparison": "{}", "ideas": "{}"}
    empty_history.write_text(json.dumps(record) + "\n")

    app = AppTest.from_file(str(APP)).run()

    assert not app.exception
    # Long questions are cut at a word boundary; the date moves inside the entry
    assert [expander.label for expander in app.sidebar.expander] == [
        "Our detector misses many flat polyps compared with the…"
    ]
    assert app.sidebar.caption[0].value == "2026-01-01 10:00"


def test_creativity_is_disabled_when_the_model_does_not_accept_a_temperature(monkeypatch):
    monkeypatch.setattr(llm, "supports_temperature", lambda node: False)

    app = AppTest.from_file(str(APP)).run()

    assert app.segmented_control[0].disabled
    assert any("does not accept a temperature" in caption.value for caption in app.caption)


def test_submitting_without_a_question_asks_for_one():
    app = AppTest.from_file(str(APP)).run()
    app.button[0].click().run()

    assert app.warning[0].value == "Type a research question first."


def test_a_run_streams_results_and_shows_the_next_steps(monkeypatch):
    received = {}

    async def fake_run_research(question, on_update, brainstormer_temperature=None):
        received.update(question=question, temperature=brainstormer_temperature)
        on_update("code_reader", {"messages": [AIMessage(content="{}")], "code_analysis": "{}", "tool_steps": 1})
        on_update("internal_librarian", {"internal_knowledge": "[source: /docs/a.md | status: current]\ntext"})
        on_update("external_librarian", {"external_knowledge": "[source: /papers/p.pdf]\ntext"})
        comparison = {"differences": ["[current] uses X"], "similarities": [], "implications": ["try Y"]}
        on_update("differ", {"comparison": json.dumps(comparison)})
        on_update("brainstormer", {"ideas": json.dumps({"ideas": ["idea"], "solutions": ["Do Y first", "Then Z"]})})
        return {}, {"gpt-test": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}}

    monkeypatch.setattr(runner, "run_research", fake_run_research)
    monkeypatch.setattr(llm, "supports_temperature", lambda node: True)

    app = AppTest.from_file(str(APP)).run()
    app.text_area[0].input("How do we improve recall?")
    app.segmented_control[0].set_value("Exploratory")
    app.button[0].click().run()

    assert not app.exception
    assert received == {"question": "How do we improve recall?", "temperature": 0.8}
    assert app.success[0].value == "**Start here:** Do Y first"
    comparison_tabs = [tab.label for tab in app.tabs if "(" in tab.label]
    assert comparison_tabs == ["Differences (1)", "Similarities (0)", "Implications (1)"]
    assert app.status[0].label == "Research complete"
