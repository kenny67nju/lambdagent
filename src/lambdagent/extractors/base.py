"""
extractors.base — Abstract extractor interface (I06).

Every extractor converts a framework-specific runtime object into a
normalized dict that conforms to lambdagent YAML schema.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict


class ExtractionError(Exception):
    """Raised when extraction from a framework object fails."""

    pass


class FrameworkExtractor(ABC):
    """
    Extract lambdagent-compatible config from a framework runtime object.

    Each subclass handles one framework (LangChain, CrewAI, AutoGen).
    The extracted config dict must conform to lambdagent YAML schema:
      - type: simple | react | chain | router | parallel
      - model: { provider, name, temperature, maxTokens }
      - systemPrompt: str
      - react: { maxSteps }
      - mcp: { localTools: [...] }
      - memory: { enabled, strategy, size }
    """

    @property
    @abstractmethod
    def framework_name(self) -> str:
        """Short name: 'langchain', 'crewai', 'autogen'."""
        ...

    @abstractmethod
    def detect(self, obj: Any) -> bool:
        """Return True if obj is an instance of this framework's agent."""
        ...

    @abstractmethod
    def extract(self, obj: Any) -> Dict[str, Any]:
        """Extract normalized config dict from framework object."""
        ...

    def _safe_getattr(self, obj: Any, *attrs, default=None):
        """Safely traverse nested attributes."""
        current = obj
        for attr in attrs:
            current = getattr(current, attr, None)
            if current is None:
                return default
        return current
