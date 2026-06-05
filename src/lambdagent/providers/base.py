"""
lambdagent.providers.base — Unified LLM Provider interface.

All providers implement the same contract:
    messages: list[dict] -> response: str

This decouples conversation management (ConversationLam) from
the transport layer (HTTP API / CLI subprocess / local inference).

L01: Added ChatMessage, ChatResponse dataclasses and unified interface
methods (achat, provider_name, default_model) for multi-provider support.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ============================================================
# L01: Unified message and response types
# ============================================================


@dataclass
class ChatMessage:
    """Unified chat message for the new provider interface."""

    role: str  # "system" | "user" | "assistant" | "tool"
    content: str


@dataclass
class ChatResponse:
    """Structured response from an LLM provider."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    finish_reason: str = ""


# ============================================================
# Legacy types (backward compatibility)
# ============================================================


@dataclass
class Message:
    """A single message in the conversation."""

    role: str  # "system", "user", "assistant"
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""

    model: str = ""
    temperature: float = 0.3
    max_tokens: int = 4096
    timeout: int = 600
    context_window: int = 200000  # max tokens the model can handle
    extra: Dict[str, Any] = field(default_factory=dict)


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    All providers accept a list of messages and return a text response.
    This is the lowest-level abstraction — no history management, no
    tool parsing, no retry logic. Just messages in, text out.

    L01: Unified interface adds typed chat/achat with ChatMessage/ChatResponse,
    plus provider_name and default_model properties. Legacy chat(list[dict]) -> str
    remains for backward compatibility.
    """

    def __init__(self, config: ProviderConfig):
        self.config = config

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Send messages to the LLM and return the response text.

        Args:
            messages: List of {"role": "system/user/assistant", "content": "..."}

        Returns:
            The assistant's response text.

        Raises:
            ProviderError: On API/connection failures.
        """
        ...

    # ── L01: Unified interface methods ──

    def chat_typed(
        self,
        messages: List[ChatMessage],
        model: str = "",
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> ChatResponse:
        """
        Typed chat completion using ChatMessage/ChatResponse.

        Default implementation delegates to legacy chat() for backward compat.
        Providers can override for richer response metadata.
        """
        dict_msgs = [{"role": m.role, "content": m.content} for m in messages]
        # Temporarily override config if params differ
        orig_temp = self.config.temperature
        orig_max = self.config.max_tokens
        orig_model = self.config.model
        try:
            if temperature != 0.0:
                self.config.temperature = temperature
            if max_tokens != 4096:
                self.config.max_tokens = max_tokens
            if model:
                self.config.model = model
            text = self.chat(dict_msgs)
        finally:
            self.config.temperature = orig_temp
            self.config.max_tokens = orig_max
            self.config.model = orig_model
        return ChatResponse(text=text, model=model or self.config.model)

    async def achat(
        self,
        messages: List[ChatMessage],
        model: str = "",
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> ChatResponse:
        """
        Async chat completion. Default wraps sync chat_typed in executor.
        """
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.chat_typed(messages, model, temperature, max_tokens)
        )

    @property
    def provider_name(self) -> str:
        """Provider identifier (e.g. 'anthropic', 'openai', 'ollama')."""
        return self.__class__.__name__.replace("Provider", "").lower()

    @property
    def default_model(self) -> str:
        """Default model for this provider."""
        return self.config.model

    @property
    def model_name(self) -> str:
        return self.config.model

    @property
    def context_window(self) -> int:
        return self.config.context_window


class ProviderError(Exception):
    """Base exception for provider errors."""

    def __init__(self, message: str, provider: str = "", retryable: bool = False):
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable
