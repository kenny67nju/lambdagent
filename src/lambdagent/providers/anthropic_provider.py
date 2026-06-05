"""
lambdagent.providers.anthropic_provider — Anthropic API provider.

Uses the official anthropic SDK. Requires ANTHROPIC_API_KEY.

L02: Added chat_typed() for ChatMessage/ChatResponse support,
overrides provider_name and default_model properties.
"""
from __future__ import annotations

import os
from typing import Dict, List

from .base import LLMProvider, ProviderConfig, ProviderError, ChatMessage, ChatResponse


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API (messages endpoint)."""

    def __init__(self, config: ProviderConfig, api_key: str = ""):
        super().__init__(config)
        self._client = None
        self._api_key = api_key  # L02: Allow explicit api_key param

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ProviderError("anthropic package not installed: pip install anthropic", "anthropic")
            api_key = self._api_key or os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                raise ProviderError("ANTHROPIC_API_KEY not set", "anthropic")
            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def chat(self, messages: List[Dict[str, str]]) -> str:
        client = self._get_client()

        # Anthropic API: system is separate from messages
        system_prompt = ""
        chat_messages = []
        for m in messages:
            if m["role"] == "system":
                system_prompt = m["content"]
            else:
                chat_messages.append(m)

        # Ensure messages alternate user/assistant
        if not chat_messages:
            chat_messages = [{"role": "user", "content": ""}]

        try:
            response = client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=system_prompt,
                messages=chat_messages,
            )
            return response.content[0].text.strip()
        except Exception as e:
            raise ProviderError(f"Anthropic API error: {e}", "anthropic", retryable="overloaded" in str(e).lower())

    # ── L02: Unified interface overrides ──

    def chat_typed(self, messages: List[ChatMessage],
                   model: str = "", temperature: float = 0.0,
                   max_tokens: int = 4096) -> ChatResponse:
        """Typed chat with full response metadata."""
        client = self._get_client()

        system_msg = ""
        api_msgs = []
        for m in messages:
            if m.role == "system":
                system_msg = m.content
            else:
                api_msgs.append({"role": m.role, "content": m.content})
        if not api_msgs:
            api_msgs = [{"role": "user", "content": ""}]

        try:
            response = client.messages.create(
                model=model or self.default_model,
                system=system_msg,
                messages=api_msgs,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return ChatResponse(
                text=response.content[0].text,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                model=response.model,
                finish_reason=response.stop_reason or "",
            )
        except Exception as e:
            raise ProviderError(f"Anthropic API error: {e}", "anthropic",
                                retryable="overloaded" in str(e).lower())

    async def achat(self, messages: List[ChatMessage],
                    model: str = "", temperature: float = 0.0,
                    max_tokens: int = 4096) -> ChatResponse:
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.chat_typed(messages, model, temperature, max_tokens)
        )

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def default_model(self) -> str:
        return self.config.model or "claude-sonnet-4-20250514"
