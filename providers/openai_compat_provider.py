"""
lambdagent.providers.openai_compat_provider — OpenAI-compatible API provider.

Covers: OpenAI, Ollama, DashScope, DeepSeek, Moonshot, Zhipu, and any
OpenAI-compatible endpoint.

L02: Added chat_typed() for ChatMessage/ChatResponse support,
overrides provider_name and default_model properties. Added PRESETS
class attribute for easy provider lookup.
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Dict, List

from .base import LLMProvider, ProviderConfig, ProviderError, ChatMessage, ChatResponse


# Default base URLs and env keys per provider
_PROVIDER_DEFAULTS = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "ollama": ("http://127.0.0.1:11434/v1", ""),
    "dashscope": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"),
    "deepseek": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
    "moonshot": ("https://api.moonshot.cn/v1", "MOONSHOT_API_KEY"),
    "zhipu": ("https://open.bigmodel.cn/api/paas/v4", "ZHIPU_API_KEY"),
}


class OpenAICompatProvider(LLMProvider):
    """OpenAI-compatible chat completion API."""

    # L02: Provider configs: name -> (base_url, default_model, api_key_env)
    PRESETS = {
        "openai": ("https://api.openai.com/v1", "gpt-4o", "OPENAI_API_KEY"),
        "dashscope": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-max", "DASHSCOPE_API_KEY"),
        "deepseek": ("https://api.deepseek.com/v1", "deepseek-chat", "DEEPSEEK_API_KEY"),
        "moonshot": ("https://api.moonshot.cn/v1", "moonshot-v1-128k", "MOONSHOT_API_KEY"),
        "ollama": ("http://127.0.0.1:11434/v1", "qwen2.5:7b", ""),
        "zhipu": ("https://open.bigmodel.cn/api/paas/v4", "glm-4", "ZHIPU_API_KEY"),
    }

    def __init__(self, config: ProviderConfig, provider_name: str = "openai",
                 api_key: str = "", base_url: str = "", model: str = ""):
        super().__init__(config)
        self._provider_name = provider_name

        # Resolve base URL and API key
        defaults = _PROVIDER_DEFAULTS.get(provider_name, ("", ""))
        self.base_url = base_url or config.extra.get("base_url", defaults[0])
        env_key = defaults[1]
        self.api_key = api_key or config.extra.get("api_key", "")
        if not self.api_key and env_key:
            self.api_key = os.environ.get(env_key, "")
        if provider_name == "ollama":
            self.api_key = self.api_key or "ollama"

        # L02: Override model from explicit param if provided
        if model:
            self.config.model = model

    def chat(self, messages: List[Dict[str, str]]) -> str:
        url = f"{self.base_url.rstrip('/')}/chat/completions"

        body = json.dumps({
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }, ensure_ascii=False).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(url, data=body, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                data = json.loads(resp.read())
                return data["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            error_body = e.read().decode()[:500]
            raise ProviderError(f"{self._provider_name} API error {e.code}: {error_body}",
                                self._provider_name, retryable=e.code >= 500)
        except urllib.error.URLError as e:
            raise ProviderError(f"{self._provider_name} connection error: {e}",
                                self._provider_name, retryable=True)
        except Exception as e:
            raise ProviderError(f"{self._provider_name} error: {e}", self._provider_name)

    # ── L02: Unified interface overrides ──

    def chat_typed(self, messages: List[ChatMessage],
                   model: str = "", temperature: float = 0.0,
                   max_tokens: int = 4096) -> ChatResponse:
        """Typed chat with response metadata via urllib (no openai SDK required)."""
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        api_msgs = [{"role": m.role, "content": m.content} for m in messages]

        body = json.dumps({
            "model": model or self.default_model,
            "messages": api_msgs,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }, ensure_ascii=False).encode("utf-8")

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(url, data=body, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                data = json.loads(resp.read())
                choice = data["choices"][0]
                usage = data.get("usage", {})
                return ChatResponse(
                    text=(choice.get("message", {}).get("content") or "").strip(),
                    input_tokens=usage.get("prompt_tokens", 0),
                    output_tokens=usage.get("completion_tokens", 0),
                    model=data.get("model", ""),
                    finish_reason=choice.get("finish_reason", ""),
                )
        except urllib.error.HTTPError as e:
            error_body = e.read().decode()[:500]
            raise ProviderError(f"{self._provider_name} API error {e.code}: {error_body}",
                                self._provider_name, retryable=e.code >= 500)
        except urllib.error.URLError as e:
            raise ProviderError(f"{self._provider_name} connection error: {e}",
                                self._provider_name, retryable=True)
        except Exception as e:
            raise ProviderError(f"{self._provider_name} error: {e}", self._provider_name)

    async def achat(self, messages: List[ChatMessage],
                    model: str = "", temperature: float = 0.0,
                    max_tokens: int = 4096) -> ChatResponse:
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.chat_typed(messages, model, temperature, max_tokens)
        )

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def default_model(self) -> str:
        return self.config.model
