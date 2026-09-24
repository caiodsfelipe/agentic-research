from langchain.chat_models import init_chat_model

from agentic_research.config import get_node_settings

# Temperatures accepted from users (CLI --temperature, UI creativity slider). 1 is the highest value every major
# provider accepts; above it, answers tend to lose coherence rather than become more creative.
MIN_TEMPERATURE, MAX_TEMPERATURE = 0.0, 1.0

# Value used to check whether a model keeps a temperature it is given
_PROBE_TEMPERATURE = 0.5


def _create_model(node: str, **kwargs):
    settings = get_node_settings(node)
    return init_chat_model(settings["model"], **settings.get("options", {}), **kwargs)


def supports_temperature(node: str) -> bool | None:
    """
    Whether the node's model, with its configured options, accepts a temperature:
    - False when LangChain's model profile says the model doesn't (e.g. reasoning models such as o3), or when
      the integration drops the value (e.g. OpenAI gpt-5 models unless reasoning_effort is "none");
    - True when a probe temperature is kept;
    - None when it can't be determined (e.g. the model can't be created without an API key).
    No API call is made: only the local client is created.
    """
    try:
        profile = getattr(_create_model(node), "profile", None) or {}
        if profile.get("temperature") is False:
            return False
        probe = _create_model(node, temperature=_PROBE_TEMPERATURE)
    except Exception:
        return None
    if not hasattr(probe, "temperature"):
        return None
    return probe.temperature == _PROBE_TEMPERATURE


def validate_temperature(node: str, temperature: float) -> None:
    """Raises a ValueError explaining why a requested temperature can't be used for a node's model."""
    if not MIN_TEMPERATURE <= temperature <= MAX_TEMPERATURE:
        raise ValueError(f"Temperature must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}, got {temperature}.")
    if supports_temperature(node) is False:
        model = get_node_settings(node)["model"]
        raise ValueError(
            f"The {node} model ({model}) does not accept a temperature with its current settings, so it can't be "
            "adjusted. Some models don't support it at all (e.g. reasoning models such as o3); OpenAI gpt-5 models "
            'only accept it with reasoning_effort="none" in the node\'s options. '
            f"Run without a temperature, or change the {node} settings in agentic_research/config.py."
        )


def get_llm(node: str = "default", temperature: float | None = None):
    """
    Returns an instance of the LLM, with the model, options and output limit taken from the node's settings.
    Any provider supported by init_chat_model works, as long as its integration package is installed.

    temperature: None uses 0 (deterministic) when the model accepts a temperature, and none when it doesn't;
    an explicit value is validated and raises a ValueError if the model doesn't accept it.
    """
    if temperature is not None:
        validate_temperature(node, temperature)
    elif supports_temperature(node) is not False:
        temperature = 0

    kwargs = {"max_tokens": get_node_settings(node)["max_output"]}
    if temperature is not None:
        kwargs["temperature"] = temperature
    return _create_model(node, **kwargs)
