import asyncio

import pytest

from agentic_research import config, llm, runner
from agentic_research.llm import get_llm, supports_temperature, validate_temperature


@pytest.fixture(autouse=True)
def dummy_api_key(monkeypatch):
    # Creating a model client needs a key, but reading its profile makes no API call
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def use_model(monkeypatch, model, options=None):
    monkeypatch.setitem(
        config.NODE_SETTINGS,
        "brainstormer",
        {"model": model, "max_input": 1500, "max_output": 600, "options": options or {}},
    )


def test_configured_brainstormer_accepts_temperature():
    assert supports_temperature("brainstormer") is True


def test_models_whose_profile_rejects_temperature_are_detected(monkeypatch):
    use_model(monkeypatch, "openai:o3")
    assert supports_temperature("brainstormer") is False


def test_temperature_dropped_by_the_integration_is_detected(monkeypatch):
    # gpt-5 models only keep a temperature with reasoning_effort="none"
    use_model(monkeypatch, "openai:gpt-5.4-mini")
    assert supports_temperature("brainstormer") is False
    use_model(monkeypatch, "openai:gpt-5.4-mini", {"reasoning_effort": "none"})
    assert supports_temperature("brainstormer") is True


def test_out_of_range_temperature_is_rejected():
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        validate_temperature("brainstormer", 1.5)


def test_explicit_temperature_on_unsupported_model_is_rejected(monkeypatch):
    monkeypatch.setattr(llm, "supports_temperature", lambda node: False)

    with pytest.raises(ValueError, match="does not accept a temperature"):
        get_llm("brainstormer", temperature=0.7)


def test_default_temperature_is_omitted_for_unsupported_models(monkeypatch):
    monkeypatch.setattr(llm, "supports_temperature", lambda node: False)
    assert get_llm("brainstormer").temperature is None

    monkeypatch.setattr(llm, "supports_temperature", lambda node: True)
    assert get_llm("brainstormer").temperature == 0
    assert get_llm("brainstormer", temperature=0.7).temperature == 0.7


def test_run_fails_before_starting_when_temperature_is_unsupported(monkeypatch):
    monkeypatch.setattr(llm, "supports_temperature", lambda node: False)

    def must_not_start():
        raise AssertionError("the graph should not start")

    monkeypatch.setattr(runner, "code_search_tools", must_not_start)

    with pytest.raises(ValueError, match="does not accept a temperature"):
        asyncio.run(runner.run_research("question", brainstormer_temperature=0.5))
