from agentic_research.tools.mcp_loader import DEFAULT_SERENA_COMMAND, serena_server_config


def test_default_command_runs_pinned_serena_with_uvx(monkeypatch):
    monkeypatch.delenv("SERENA_COMMAND", raising=False)

    config = serena_server_config("/path/to/project")

    assert DEFAULT_SERENA_COMMAND.startswith("uvx ")
    assert config["command"] == "uvx"
    assert config["args"][-3:] == ["start-mcp-server", "--project", "/path/to/project"]


def test_command_can_be_overridden(monkeypatch):
    monkeypatch.setenv("SERENA_COMMAND", "serena")

    config = serena_server_config("/path/to/project")

    assert config["command"] == "serena"
    assert config["args"] == ["start-mcp-server", "--project", "/path/to/project"]


def test_api_keys_are_not_passed_to_serena(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("UV_CACHE_DIR", "/tmp/uv-cache")

    env = serena_server_config("/path/to/project")["env"]

    assert "OPENAI_API_KEY" not in env
    assert env["UV_CACHE_DIR"] == "/tmp/uv-cache"
    assert "PATH" in env
