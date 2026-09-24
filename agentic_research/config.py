import os

# Research domain the agents act as experts in (used in every prompt's role).
RESEARCH_DOMAIN = os.getenv("RESEARCH_DOMAIN", "AI and machine learning research")

# Central place for every model choice and token limit in the project.
# model: "provider:model" as accepted by LangChain's init_chat_model (e.g. "openai:gpt-5.4-mini",
#   "google_genai:gemini-3.6-flash", "anthropic:claude-sonnet-5"); the stronger model only where reasoning matters most.
# max_input: tokens sent to the model (system prompt + messages), trimmed before each call.
# max_output: tokens the model is allowed to generate.
# options: extra provider-specific arguments for init_chat_model. OpenAI gpt-5 models accept a temperature only
#   with reasoning_effort="none" (LangChain silently drops it otherwise), hence the option below.
OPENAI_NO_REASONING = {"reasoning_effort": "none"}
NODE_SETTINGS = {
    "code_reader": {
        "model": "openai:gpt-5.4-nano",
        "max_input": 3500,
        "max_output": 500,
        "options": OPENAI_NO_REASONING,
    },
    "differ": {"model": "openai:gpt-5.4-mini", "max_input": 4000, "max_output": 600, "options": OPENAI_NO_REASONING},
    "brainstormer": {
        "model": "openai:gpt-5.4-mini",
        "max_input": 1500,
        "max_output": 600,
        "options": OPENAI_NO_REASONING,
    },
    "default": {"model": "openai:gpt-5.4-nano", "max_input": 1500, "max_output": 300, "options": OPENAI_NO_REASONING},
}

# Maximum number of code_reader <-> tools round trips before it must answer.
MAX_TOOL_STEPS = 4

# Maximum tokens kept from a single tool result (e.g. a large file read).
MAX_TOOL_RESULT_TOKENS = 800

# Maximum tokens kept from each knowledge section (code analysis, internal, external) sent to the differ.
MAX_SECTION_TOKENS = 1100


def get_node_settings(node: str) -> dict:
    return NODE_SETTINGS.get(node, NODE_SETTINGS["default"])
