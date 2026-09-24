import os
import shlex
from contextlib import asynccontextmanager

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp.client.stdio import get_default_environment

# How Serena is launched. By default uvx downloads and runs a pinned version on demand, so the only
# prerequisite is uv. Override with SERENA_COMMAND in .env (e.g. "serena" for an existing install).
DEFAULT_SERENA_COMMAND = "uvx --from serena-agent==1.7.0 serena"

# find_file is excluded: it scans every file (ignoring Serena's ignored_paths), which is very slow on large codebases
ALLOWED_TOOLS = {"list_dir", "get_symbols_overview", "read_file", "search_for_pattern"}


def serena_server_config(codebase_path: str) -> dict:
    # On Windows, non-POSIX splitting keeps backslashes in paths intact (quotes are then stripped by hand)
    windows = os.name == "nt"
    parts = shlex.split(os.getenv("SERENA_COMMAND", DEFAULT_SERENA_COMMAND), posix=not windows)
    command, *args = [part.strip('"') for part in parts] if windows else parts
    return {
        "transport": "stdio",
        "command": command,
        "args": [*args, "start-mcp-server", "--project", codebase_path],
        "env": {
            # Only the variables a subprocess needs (PATH, home and temp folders), so API keys are not passed on
            **get_default_environment(),
            # uv settings (cache, index), used when Serena is launched with uvx
            **{name: value for name, value in os.environ.items() if name.startswith("UV_")},
            # UTF-8 output, so Serena's logs don't fail on non-ASCII file names in the Windows console
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
        },
    }


# Keeps one Serena session open for the whole run; without it, every tool call starts a new Serena server
@asynccontextmanager
async def code_search_tools():
    codebase_path = os.getenv("CODEBASE_PATH")
    if not codebase_path:
        raise ValueError("CODEBASE_PATH is not defined in .env")

    mcp_client = MultiServerMCPClient({"serena": serena_server_config(codebase_path)})

    async with mcp_client.session("serena") as session:
        tools = await load_mcp_tools(session)

        # Keep only read-only tools: every tool schema is sent on each LLM call, so fewer tools means fewer input tokens
        yield [tool for tool in tools if tool.name in ALLOWED_TOOLS]
