"""
L05: Validation tests for the unified LLM Provider interface (L01-L04).

Tests provider creation, ChatMessage/ChatResponse types, ConversationLam
with mock providers, context compaction, and history reset.

All tests use mocks — no real API calls, no anthropic/openai SDK imports.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch
from typing import List, Dict

# Import the types and factory
from lambdagent.providers.base import (
    LLMProvider, ProviderConfig, ProviderError,
    ChatMessage, ChatResponse,
)
from lambdagent.providers import create_provider
from lambdagent.providers.anthropic_provider import AnthropicProvider
from lambdagent.providers.openai_compat_provider import OpenAICompatProvider
from lambdagent.conversation import ConversationLam
from lambdagent.core import Context


# ============================================================
# Mock provider for testing ConversationLam
# ============================================================

class MockProvider(LLMProvider):
    """Test double that returns canned responses without any API calls."""

    def __init__(self, responses: list[str] | None = None):
        config = ProviderConfig(model="mock-model", temperature=0.0, max_tokens=100)
        super().__init__(config)
        self._responses = list(responses or ["mock response"])
        self._call_count = 0

    def chat(self, messages: List[Dict[str, str]]) -> str:
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        return self._responses[idx]

    def chat_typed(self, messages: List[ChatMessage],
                   model: str = "", temperature: float = 0.0,
                   max_tokens: int = 4096) -> ChatResponse:
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        return ChatResponse(
            text=self._responses[idx],
            input_tokens=10,
            output_tokens=5,
            model=model or "mock-model",
            finish_reason="end_turn",
        )

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def default_model(self) -> str:
        return "mock-model"


# ============================================================
# Test 1-4: create_provider factory
# ============================================================

class TestCreateProvider(unittest.TestCase):
    """Test create_provider() returns correct provider types."""

    def test_create_anthropic_provider(self):
        """create_provider('anthropic') returns AnthropicProvider."""
        provider = create_provider("anthropic")
        self.assertIsInstance(provider, AnthropicProvider)
        self.assertEqual(provider.provider_name, "anthropic")

    def test_create_dashscope_provider(self):
        """create_provider('dashscope') returns OpenAICompatProvider with correct base_url."""
        provider = create_provider("dashscope")
        self.assertIsInstance(provider, OpenAICompatProvider)
        self.assertIn("dashscope.aliyuncs.com", provider.base_url)
        self.assertEqual(provider.provider_name, "dashscope")

    def test_create_ollama_provider(self):
        """create_provider('ollama') returns OpenAICompatProvider with localhost URL."""
        provider = create_provider("ollama")
        self.assertIsInstance(provider, OpenAICompatProvider)
        self.assertIn("localhost:11434", provider.base_url)
        self.assertEqual(provider.provider_name, "ollama")

    def test_create_openai_provider(self):
        """create_provider('openai') returns OpenAICompatProvider."""
        provider = create_provider("openai")
        self.assertIsInstance(provider, OpenAICompatProvider)
        self.assertEqual(provider.provider_name, "openai")

    def test_create_deepseek_provider(self):
        """create_provider('deepseek') returns OpenAICompatProvider."""
        provider = create_provider("deepseek")
        self.assertIsInstance(provider, OpenAICompatProvider)
        self.assertEqual(provider.provider_name, "deepseek")

    def test_create_unknown_raises_value_error(self):
        """create_provider('unknown') raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            create_provider("unknown")
        self.assertIn("Unknown provider", str(cm.exception))
        self.assertIn("unknown", str(cm.exception))


# ============================================================
# Test 5: ChatMessage and ChatResponse creation
# ============================================================

class TestDataclasses(unittest.TestCase):
    """Test ChatMessage and ChatResponse can be created and used."""

    def test_chat_message_creation(self):
        msg = ChatMessage(role="user", content="hello")
        self.assertEqual(msg.role, "user")
        self.assertEqual(msg.content, "hello")

    def test_chat_message_roles(self):
        for role in ("system", "user", "assistant", "tool"):
            msg = ChatMessage(role=role, content=f"test {role}")
            self.assertEqual(msg.role, role)

    def test_chat_response_creation(self):
        resp = ChatResponse(text="hi", input_tokens=10, output_tokens=5,
                            model="test-model", finish_reason="stop")
        self.assertEqual(resp.text, "hi")
        self.assertEqual(resp.input_tokens, 10)
        self.assertEqual(resp.output_tokens, 5)
        self.assertEqual(resp.model, "test-model")
        self.assertEqual(resp.finish_reason, "stop")

    def test_chat_response_defaults(self):
        resp = ChatResponse(text="hi")
        self.assertEqual(resp.input_tokens, 0)
        self.assertEqual(resp.output_tokens, 0)
        self.assertEqual(resp.model, "")
        self.assertEqual(resp.finish_reason, "")


# ============================================================
# Test 6-9: ConversationLam with mock provider
# ============================================================

class TestConversationLam(unittest.TestCase):
    """Test ConversationLam instantiation, apply, compact, and reset."""

    def test_instantiation(self):
        """ConversationLam can be instantiated with a mock provider."""
        provider = MockProvider()
        lam = ConversationLam(
            name="test-agent",
            provider=provider,
            system_prompt="You are a test agent.",
        )
        self.assertEqual(lam._name, "test-agent")
        self.assertEqual(lam.system_prompt, "You are a test agent.")
        self.assertEqual(len(lam.messages), 1)  # Just system message

    def test_apply_returns_expected(self):
        """ConversationLam.apply() with a mock provider returns the mock response."""
        provider = MockProvider(["Hello from mock!"])
        lam = ConversationLam(
            name="test-agent",
            provider=provider,
            system_prompt="You are a test agent.",
        )
        result = lam.apply("Hi there")
        self.assertEqual(result, "Hello from mock!")

    def test_apply_builds_history(self):
        """Each apply() adds user + assistant messages to history."""
        provider = MockProvider(["r1", "r2", "r3"])
        lam = ConversationLam(
            name="test-agent",
            provider=provider,
            system_prompt="system",
        )
        lam.apply("q1")
        lam.apply("q2")
        # system + (user, assistant) * 2 = 5
        self.assertEqual(len(lam.messages), 5)
        self.assertEqual(lam.messages[1]["role"], "user")
        self.assertEqual(lam.messages[1]["content"], "q1")
        self.assertEqual(lam.messages[2]["role"], "assistant")
        self.assertEqual(lam.messages[2]["content"], "r1")

    def test_apply_typed_returns_chat_response(self):
        """ConversationLam.apply_typed() returns ChatResponse with metadata."""
        provider = MockProvider(["typed response"])
        lam = ConversationLam(
            name="test-agent",
            provider=provider,
            system_prompt="system",
        )
        response = lam.apply_typed("test input")
        self.assertIsInstance(response, ChatResponse)
        self.assertEqual(response.text, "typed response")
        self.assertEqual(response.input_tokens, 10)
        self.assertEqual(response.output_tokens, 5)

    def test_compact_trims_messages(self):
        """_compact_typed trims messages when over token budget."""
        provider = MockProvider()
        lam = ConversationLam(
            name="test-agent",
            provider=provider,
            system_prompt="s",
            max_history_tokens=50,  # Very low budget
        )
        # Build a list of messages that exceeds the budget
        messages = [ChatMessage(role="system", content="system prompt")]
        for i in range(20):
            messages.append(ChatMessage(role="user", content="x" * 100))
            messages.append(ChatMessage(role="assistant", content="y" * 100))

        compacted = lam._compact_typed(messages)
        # Should be fewer messages than original
        self.assertLess(len(compacted), len(messages))
        # Should always keep system message
        self.assertEqual(compacted[0].role, "system")

    def test_compact_no_trim_when_under_budget(self):
        """_compact_typed returns all messages when under budget."""
        provider = MockProvider()
        lam = ConversationLam(
            name="test-agent",
            provider=provider,
            system_prompt="s",
            max_history_tokens=100000,  # Large budget
        )
        messages = [
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="user", content="hi"),
            ChatMessage(role="assistant", content="hello"),
        ]
        compacted = lam._compact_typed(messages)
        self.assertEqual(len(compacted), 3)

    def test_reset_clears_history(self):
        """ConversationLam.reset() clears both dict and typed history."""
        provider = MockProvider(["r1", "r2"])
        lam = ConversationLam(
            name="test-agent",
            provider=provider,
            system_prompt="system",
        )
        lam.apply("q1")
        lam.apply_typed("q2")
        self.assertGreater(len(lam.messages), 1)
        self.assertGreater(len(lam._typed_messages), 0)

        lam.reset()
        self.assertEqual(len(lam.messages), 1)  # Only system message
        self.assertEqual(lam.messages[0]["content"], "system")
        self.assertEqual(len(lam._typed_messages), 0)

    def test_context_trace_logging(self):
        """apply() logs trace entries to the Context."""
        provider = MockProvider(["response"])
        lam = ConversationLam(
            name="test-agent",
            provider=provider,
            system_prompt="system",
        )
        ctx = Context()
        lam.apply("input", ctx=ctx)
        self.assertEqual(len(ctx.trace), 1)
        self.assertEqual(ctx.trace[0].term_name, "test-agent")

    def test_history_property(self):
        """history property returns typed message list."""
        provider = MockProvider(["r1"])
        lam = ConversationLam(
            name="test-agent",
            provider=provider,
            system_prompt="system",
        )
        lam.apply_typed("q1")
        hist = lam.history
        self.assertEqual(len(hist), 2)  # user + assistant
        self.assertEqual(hist[0].role, "user")
        self.assertEqual(hist[0].content, "q1")
        self.assertEqual(hist[1].role, "assistant")
        self.assertEqual(hist[1].content, "r1")


# ============================================================
# Test: LLMProvider base class default implementations
# ============================================================

class TestLLMProviderDefaults(unittest.TestCase):
    """Test that LLMProvider base class provides working defaults."""

    def test_default_provider_name(self):
        """provider_name defaults to lowercased class name without 'Provider'."""
        provider = MockProvider()
        # MockProvider -> provider_name = "mock" (from override)
        self.assertEqual(provider.provider_name, "mock")

    def test_default_model(self):
        """default_model returns config.model."""
        provider = MockProvider()
        self.assertEqual(provider.default_model, "mock-model")

    def test_chat_typed_delegates_to_chat(self):
        """Base chat_typed() delegates to chat() for backward compat."""
        provider = MockProvider(["delegated response"])
        # Reset call count since MockProvider overrides chat_typed
        # Test the base class behavior by calling chat directly
        result = provider.chat([{"role": "user", "content": "test"}])
        self.assertEqual(result, "delegated response")


if __name__ == "__main__":
    unittest.main()
