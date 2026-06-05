"""
lambdagent.conversation — ConversationLam: conversation-aware Lambda abstraction.

Wraps any LLMProvider with conversation history management.
Each apply() appends the new input as a user message, calls the provider
with the full (or windowed) history, and records the response.

This is what eliminates hallucination: the LLM sees its complete
conversation history, not a lossy compressed state string.

L03: Added ChatMessage-based apply_typed() path for richer response metadata.
ConversationLam now supports both dict-based (legacy) and ChatMessage-based
(unified) conversation flows.

Usage:
    provider = ClaudeCodeProvider(config)
    lam = ConversationLam("agent", provider, system_prompt="...")

    r1 = lam.apply("read the README")     # creates conversation
    r2 = lam.apply("[tool result] ...")     # continues with full memory
    r3 = lam.apply("[tool result] ...")     # still remembers r1, r2
"""
from __future__ import annotations

import time
from typing import Any, Callable, List, Optional

from lambdagent.core import Term, Context
from lambdagent.providers.base import LLMProvider, ProviderError, ChatMessage, ChatResponse


class ConversationLam(Term):
    """
    Lambda abstraction with conversation persistence.

    Lambda semantics preserved:
        ConversationLam(provider, prompt) = lambda x. provider(history + x)
        apply() = beta-reduction with memory

    The key difference from stateless Lam:
        Lam:             each apply() is independent
        ConversationLam: each apply() builds on all previous calls

    L03: Now supports both legacy dict-based and ChatMessage-based flows.
    """

    def __init__(
        self,
        name: str,
        provider: LLMProvider,
        system_prompt: str,
        max_history_tokens: int = 80000,
        keep_recent_turns: int = 20,
        output_parser: Callable[[str], Any] | None = None,
        # L03: New unified interface params
        model: str = "",
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ):
        super().__init__(name)
        self.provider = provider
        self.system_prompt = system_prompt
        self.max_history_tokens = max_history_tokens
        self.keep_recent_turns = keep_recent_turns
        self.output_parser = output_parser or (lambda x: x)
        # L03: Store model/temperature/max_tokens for chat_typed calls
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

        # Conversation history (system message always first)
        self.messages: List[dict] = [
            {"role": "system", "content": system_prompt}
        ]

        # L03: ChatMessage-based history (parallel to dict-based for typed path)
        self._typed_messages: List[ChatMessage] = []

    # Expose model name for react_step logging
    @property
    def model(self) -> str:
        return self._model or self.provider.model_name

    # Expose _session_id for react_step session detection
    @property
    def _session_id(self):
        return getattr(self.provider, '_session_id', None)

    def apply(self, input: Any, ctx: Context | None = None) -> Any:
        """Beta-reduction with conversation memory."""
        ctx = ctx or Context()
        t0 = time.time()

        # Append user message
        self.messages.append({"role": "user", "content": str(input)})

        # Manage context window before sending
        managed = self._manage_context()

        # Call provider
        tokens_used = 0
        try:
            response = self.provider.chat(managed)
        except ProviderError as e:
            response = f"[{e.provider.upper()}_ERROR] {e}"

        # Record assistant response
        self.messages.append({"role": "assistant", "content": response})

        duration = (time.time() - t0) * 1000
        result = self.output_parser(response)
        ctx.log(self._name, self._trace_id, input, result, duration, self.model, tokens_used)
        return result

    # ── L03: ChatMessage-based typed path ──

    def apply_typed(self, input: Any, ctx: Context | None = None) -> ChatResponse:
        """
        Apply with ChatMessage/ChatResponse for richer metadata.

        Uses provider.chat_typed() instead of provider.chat().
        Returns ChatResponse with token counts, model info, etc.
        """
        ctx = ctx or Context()
        t0 = time.time()

        # Build messages
        messages = [ChatMessage(role="system", content=self.system_prompt)]
        messages.extend(self._typed_messages)
        messages.append(ChatMessage(role="user", content=str(input)))

        # Compact if over budget
        messages = self._compact_typed(messages)

        # Call provider via typed interface
        try:
            response = self.provider.chat_typed(
                messages=messages,
                model=self._model,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        except ProviderError as e:
            response = ChatResponse(
                text=f"[{e.provider.upper()}_ERROR] {e}",
                model=self.model,
            )

        duration = (time.time() - t0) * 1000

        # Update typed history
        self._typed_messages.append(ChatMessage(role="user", content=str(input)))
        self._typed_messages.append(ChatMessage(role="assistant", content=response.text))

        # Also sync to dict-based history for compatibility
        self.messages.append({"role": "user", "content": str(input)})
        self.messages.append({"role": "assistant", "content": response.text})

        # Log to context
        ctx.log(self._name, self._trace_id, str(input)[:200],
                response.text[:200], duration, self.model,
                response.input_tokens + response.output_tokens)

        return response

    def _compact_typed(self, messages: List[ChatMessage]) -> List[ChatMessage]:
        """Compact ChatMessage history if token estimate exceeds budget."""
        total_chars = sum(len(m.content) for m in messages)
        estimated_tokens = total_chars // 4

        if estimated_tokens <= self.max_history_tokens:
            return messages

        # Keep system + last N messages that fit
        system = messages[:1]
        history = messages[1:]

        for keep in range(len(history), 0, -1):
            kept = system + history[-keep:]
            est = sum(len(m.content) for m in kept) // 4
            if est <= self.max_history_tokens:
                return kept

        return system  # Worst case: only system prompt

    @property
    def history(self) -> List[ChatMessage]:
        """Return typed message history."""
        return list(self._typed_messages)

    # ── Legacy dict-based context management ──

    def _manage_context(self) -> List[dict]:
        """
        Ensure messages fit within max_history_tokens.

        Strategy:
          - Always keep system message
          - Always keep recent N turns (user+assistant pairs)
          - Summarize older messages into a single "history summary" message
        """
        total_tokens = self._estimate_tokens(self.messages)

        if total_tokens <= self.max_history_tokens:
            return list(self.messages)

        # Keep system + recent turns
        system = self.messages[0]
        # Each turn = 1 user + 1 assistant = 2 messages
        keep_count = self.keep_recent_turns * 2
        recent = self.messages[-keep_count:] if len(self.messages) > keep_count else self.messages[1:]
        old = self.messages[1:-keep_count] if len(self.messages) > keep_count + 1 else []

        if not old:
            return list(self.messages)

        # Compress old messages into summary
        summary_lines = []
        for m in old:
            role = m["role"]
            content = m["content"][:150]
            summary_lines.append(f"[{role}] {content}...")

        summary = "[对话历史摘要]\n" + "\n".join(summary_lines)

        return [system, {"role": "user", "content": summary}] + list(recent)

    def _estimate_tokens(self, messages: List[dict]) -> int:
        """Rough token estimate: 4 chars ≈ 1 token."""
        return sum(len(m.get("content", "")) for m in messages) // 4

    def reset(self):
        """Clear conversation history, start fresh."""
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self._typed_messages = []
        if hasattr(self.provider, 'reset_session'):
            self.provider.reset_session()

    def __rshift__(self, other):
        from lambdagent.primitives import Compose
        return Compose(self, other)
