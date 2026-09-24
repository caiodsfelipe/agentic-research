from langchain.chat_models import init_chat_model

from agentic_research.config import get_node_settings


def get_llm(node: str = "default", temperature: float = 0):
    """
    Returns an instance of the LLM, with the model and output limit taken from the node's settings.
    Any provider supported by init_chat_model works, as long as its integration package is installed.
    """
    settings = get_node_settings(node)
    return init_chat_model(settings["model"], temperature=temperature, max_tokens=settings["max_output"])
