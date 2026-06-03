"""agentruntime.llm_adapter — Multi-provider LLM unified interface (sync + async streaming)"""
from __future__ import annotations
import asyncio
import os
import json
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

@dataclass
class LLMResponse:
    text: str = ""
    model: str = ""
    usage: TokenUsage = None
    finish_reason: str = ""

    def __post_init__(self):
        if self.usage is None:
            self.usage = TokenUsage()


class LLMAdapter:
    """
    Unified LLM call interface.
    Lambda: LLMAdapter = lambda (model, system, user). LLM_{model}(system, user)

    Routes by model name prefix:
        claude-* / anthropic/*  -> Anthropic
        gpt-* / openai/*       -> OpenAI
        qwen* / dashscope/*    -> DashScope
        other                  -> custom endpoint
    """

    def __init__(self, config=None, fallback_models: List[str] = None):
        self.config = config
        self.fallback_models = fallback_models or []
        self._anthropic_client = None
        self._openai_client = None

    def call(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        stop_sequences: List[str] = None,
    ) -> LLMResponse:
        """Unified LLM call. Routes to appropriate provider."""
        model_lower = model.lower()

        if model_lower.startswith("claude") or model_lower.startswith("anthropic/"):
            actual_model = model.replace("anthropic/", "")
            return self._call_anthropic(actual_model, system, user, temperature, max_tokens, stop_sequences)
        elif model_lower.startswith("gpt") or model_lower.startswith("openai/"):
            actual_model = model.replace("openai/", "")
            return self._call_openai(actual_model, system, user, temperature, max_tokens, stop_sequences)
        elif model_lower.startswith("qwen") or model_lower.startswith("dashscope/"):
            actual_model = model.replace("dashscope/", "")
            return self._call_dashscope(actual_model, system, user, temperature, max_tokens)
        else:
            # Try anthropic as default
            return self._call_anthropic(model, system, user, temperature, max_tokens, stop_sequences)

    def call_with_fallback(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        stop_sequences: List[str] = None,
    ) -> LLMResponse:
        """Call with automatic fallback to alternative models on failure.

        Tries primary model first. On 429/500/timeout/connection errors,
        falls through to fallback_models in order.
        """
        models_to_try = [model] + self.fallback_models
        last_error = None

        for i, m in enumerate(models_to_try):
            try:
                return self.call(m, system, user, temperature, max_tokens, stop_sequences)
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                # Only fallback on transient/rate-limit errors
                is_retryable = any(s in error_str for s in [
                    "429", "rate", "overloaded", "500", "502", "503",
                    "timeout", "connection", "urlopen",
                ])
                if not is_retryable or i == len(models_to_try) - 1:
                    raise
                import sys
                print(f"[Fallback] {m} failed: {e}. Trying {models_to_try[i+1]}...",
                      file=sys.stderr)

        raise last_error

    def _call_anthropic(self, model, system, user, temperature, max_tokens, stop_sequences):
        try:
            import anthropic
        except ImportError:
            raise RuntimeError("anthropic package not installed. Run: pip install anthropic")

        if self._anthropic_client is None:
            self._anthropic_client = anthropic.Anthropic()

        kwargs = dict(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        if stop_sequences:
            kwargs["stop_sequences"] = stop_sequences

        response = self._anthropic_client.messages.create(**kwargs)
        text = response.content[0].text.strip()
        usage = TokenUsage(
            input_tokens=getattr(response.usage, 'input_tokens', 0),
            output_tokens=getattr(response.usage, 'output_tokens', 0),
            total_tokens=getattr(response.usage, 'input_tokens', 0) + getattr(response.usage, 'output_tokens', 0),
        )
        return LLMResponse(text=text, model=model, usage=usage, finish_reason=response.stop_reason or "")

    def _call_openai(self, model, system, user, temperature, max_tokens, stop_sequences):
        try:
            import openai
        except ImportError:
            raise RuntimeError("openai package not installed. Run: pip install openai")

        if self._openai_client is None:
            self._openai_client = openai.OpenAI()

        kwargs = dict(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        if stop_sequences:
            kwargs["stop"] = stop_sequences

        response = self._openai_client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        usage = TokenUsage(
            input_tokens=getattr(response.usage, 'prompt_tokens', 0),
            output_tokens=getattr(response.usage, 'completion_tokens', 0),
            total_tokens=getattr(response.usage, 'total_tokens', 0),
        )
        return LLMResponse(text=choice.message.content.strip(), model=model, usage=usage,
                          finish_reason=choice.finish_reason or "")

    def _call_dashscope(self, model, system, user, temperature, max_tokens):
        """DashScope via HTTP (no external dependency)."""
        import urllib.request
        api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY not set")

        url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        body = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }).encode("utf-8")

        req = urllib.request.Request(url, data=body, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        })

        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        choice = data["choices"][0]
        u = data.get("usage", {})
        usage = TokenUsage(
            input_tokens=u.get("prompt_tokens", 0),
            output_tokens=u.get("completion_tokens", 0),
            total_tokens=u.get("total_tokens", 0),
        )
        return LLMResponse(text=choice["message"]["content"].strip(), model=model,
                          usage=usage, finish_reason=choice.get("finish_reason", ""))

    # ════════════════════════════════════════════════════════════
    # Async call — runs sync call in thread pool
    # ════════════════════════════════════════════════════════════

    async def acall(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        stop_sequences: List[str] = None,
    ) -> LLMResponse:
        """Async version of call(). Runs provider SDK in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.call, model, system, user, temperature, max_tokens, stop_sequences
        )

    # ════════════════════════════════════════════════════════════
    # Streaming — async generator yielding tokens
    # ════════════════════════════════════════════════════════════

    async def stream(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        stop_sequences: List[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream LLM response token by token.

        Yields individual text chunks as they arrive.
        Falls back to acall() for providers without streaming support.
        """
        model_lower = model.lower()

        if model_lower.startswith("claude") or model_lower.startswith("anthropic/"):
            actual_model = model.replace("anthropic/", "")
            async for token in self._stream_anthropic(
                actual_model, system, user, temperature, max_tokens, stop_sequences
            ):
                yield token
        elif model_lower.startswith("gpt") or model_lower.startswith("openai/"):
            actual_model = model.replace("openai/", "")
            async for token in self._stream_openai(
                actual_model, system, user, temperature, max_tokens, stop_sequences
            ):
                yield token
        else:
            # DashScope and others: fall back to non-streaming
            response = await self.acall(model, system, user, temperature, max_tokens, stop_sequences)
            yield response.text

    async def _stream_anthropic(self, model, system, user, temperature, max_tokens, stop_sequences):
        """Stream from Anthropic Claude API."""
        try:
            import anthropic
        except ImportError:
            raise RuntimeError("anthropic package not installed. Run: pip install anthropic")

        if self._anthropic_client is None:
            self._anthropic_client = anthropic.Anthropic()

        kwargs = dict(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        if stop_sequences:
            kwargs["stop_sequences"] = stop_sequences

        # Anthropic streaming runs in thread pool (sync SDK)
        loop = asyncio.get_event_loop()

        def _do_stream():
            chunks = []
            with self._anthropic_client.messages.stream(**kwargs) as stream:
                for text in stream.text_stream:
                    chunks.append(text)
            return chunks

        chunks = await loop.run_in_executor(None, _do_stream)
        for chunk in chunks:
            yield chunk

    async def _stream_openai(self, model, system, user, temperature, max_tokens, stop_sequences):
        """Stream from OpenAI API."""
        try:
            import openai
        except ImportError:
            raise RuntimeError("openai package not installed. Run: pip install openai")

        if self._openai_client is None:
            self._openai_client = openai.OpenAI()

        kwargs = dict(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            stream=True,
        )
        if stop_sequences:
            kwargs["stop"] = stop_sequences

        loop = asyncio.get_event_loop()

        def _do_stream():
            chunks = []
            response = self._openai_client.chat.completions.create(**kwargs)
            for chunk in response:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    chunks.append(delta.content)
            return chunks

        chunks = await loop.run_in_executor(None, _do_stream)
        for chunk in chunks:
            yield chunk
