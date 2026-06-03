"""
lambdagent.extractors — Extract lambdagent-compatible configs from external frameworks.

Phase 7c (I06-I09): Reverse-engineer LangChain/CrewAI/AutoGen runtime objects
into normalized YAML configs that can be fed through lambdagent's lint, type
checker, cost predictor, and parallel safety verifier.

Usage:
    from lambdagent.extractors import extract_config
    config = extract_config(langchain_agent_executor)  # auto-detect framework
"""

from .base import FrameworkExtractor, ExtractionError
from .langchain_extractor import LangChainExtractor
from .crewai_extractor import CrewAIExtractor
from .autogen_extractor import AutoGenExtractor

_EXTRACTORS = [
    LangChainExtractor(),
    CrewAIExtractor(),
    AutoGenExtractor(),
]


def extract_config(framework_object, framework: str = "auto") -> dict:
    """
    Auto-detect framework and extract normalized lambdagent config.

    Args:
        framework_object: Runtime object (AgentExecutor, Crew, GroupChatManager, etc.)
        framework: Force specific framework ("langchain", "crewai", "autogen")
                   or "auto" for auto-detection.

    Returns:
        dict: Normalized YAML config compatible with lambdagent from_config.

    Raises:
        ExtractionError: If framework not detected or extraction fails.
    """
    if framework != "auto":
        for ext in _EXTRACTORS:
            if ext.framework_name == framework:
                return ext.extract(framework_object)
        raise ExtractionError(f"Unknown framework: {framework}")

    for ext in _EXTRACTORS:
        if ext.detect(framework_object):
            return ext.extract(framework_object)

    raise ExtractionError(
        f"Cannot extract config from {type(framework_object).__name__}. "
        f"Supported: LangChain AgentExecutor, CrewAI Crew, AutoGen GroupChatManager."
    )


__all__ = [
    "extract_config",
    "FrameworkExtractor", "ExtractionError",
    "LangChainExtractor", "CrewAIExtractor", "AutoGenExtractor",
]
