"""
lambdagent.providers — Unified LLM provider implementations.

All providers implement LLMProvider.chat(messages) -> str.

L02: Added ChatMessage, ChatResponse exports and create_provider() factory
for unified multi-provider instantiation.
"""
from .base import LLMProvider, ProviderConfig, ProviderError, Message, ChatMessage, ChatResponse
from .claude_code import ClaudeLam  # backward compat
from .claude_code_provider import ClaudeCodeProvider
from .anthropic_provider import AnthropicProvider
from .openai_compat_provider import OpenAICompatProvider


def create_provider(provider_name: str, **kwargs) -> LLMProvider:
    """
    Factory: create provider by name.

    Args:
        provider_name: One of 'anthropic', 'openai', 'dashscope', 'deepseek',
                       'moonshot', 'ollama', 'zhipu', 'claude-code'.
        **kwargs: Provider-specific arguments (api_key, base_url, model, etc.)
                  Also accepts ProviderConfig fields: temperature, max_tokens, timeout.

    Returns:
        LLMProvider instance.
    """
    # Extract ProviderConfig fields from kwargs
    config_kwargs = {}
    for key in ("temperature", "max_tokens", "timeout", "context_window"):
        if key in kwargs:
            config_kwargs[key] = kwargs.pop(key)

    if provider_name == "anthropic":
        config = ProviderConfig(
            model=kwargs.pop("model", "claude-sonnet-4-20250514"),
            **config_kwargs,
        )
        return AnthropicProvider(config, api_key=kwargs.pop("api_key", ""))

    if provider_name == "claude-code":
        config = ProviderConfig(
            model=kwargs.pop("model", "sonnet"),
            **config_kwargs,
        )
        return ClaudeCodeProvider(config)

    if provider_name in ("openai", "dashscope", "deepseek", "moonshot", "ollama", "zhipu"):
        preset = OpenAICompatProvider.PRESETS.get(provider_name, ("", "gpt-4o", ""))
        config = ProviderConfig(
            model=kwargs.pop("model", preset[1]),
            **config_kwargs,
        )
        return OpenAICompatProvider(
            config,
            provider_name=provider_name,
            api_key=kwargs.pop("api_key", ""),
            base_url=kwargs.pop("base_url", ""),
        )

    raise ValueError(
        f"Unknown provider: {provider_name}. "
        f"Supported: anthropic, openai, dashscope, deepseek, moonshot, ollama, zhipu, claude-code"
    )


__all__ = [
    "LLMProvider", "ProviderConfig", "ProviderError", "Message",
    "ChatMessage", "ChatResponse",
    "ClaudeLam",  # backward compat for PersonalAssistant
    "ClaudeCodeProvider",
    "AnthropicProvider",
    "OpenAICompatProvider",
    "create_provider",
]
