"""
lambdagent.token_budget — Token budget tracking and enforcement

Tracks token consumption across agent execution and prevents
budget overrun by raising BudgetExhaustedError.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from .core import LambdagentError


class BudgetExhaustedError(LambdagentError):
    """Raised when token budget is exhausted."""

    def __init__(self, budget: int, used: int):
        self.budget = budget
        self.used = used
        super().__init__(f"Token budget exhausted: {used}/{budget} tokens used")


@dataclass
class TokenBudget:
    """Track and enforce token consumption limits."""

    max_tokens: int = 100000
    used: int = 0
    hard_limit: bool = True
    _by_model: Dict[str, int] = field(default_factory=dict)
    _by_step: list = field(default_factory=list)

    def estimate_cost(self, text: str) -> int:
        """Rough token count estimate (4 chars ≈ 1 token)."""
        return len(text) // 4

    def can_afford(self, estimated_tokens: int) -> bool:
        """Check if budget can afford estimated_tokens more."""
        return self.used + estimated_tokens < self.max_tokens

    def can_afford_prompt(self, prompt: str) -> bool:
        """Check if budget can afford sending this prompt."""
        return self.can_afford(self.estimate_cost(prompt))

    def record(
        self, input_tokens: int, output_tokens: int, model: str = "", step: int = -1
    ):
        """Record token consumption."""
        total = input_tokens + output_tokens
        self.used += total
        if model:
            self._by_model[model] = self._by_model.get(model, 0) + total
        if step >= 0:
            self._by_step.append(
                {
                    "step": step,
                    "input": input_tokens,
                    "output": output_tokens,
                    "model": model,
                }
            )

    def check(self):
        """Raise BudgetExhaustedError if budget exceeded."""
        if self.used >= self.max_tokens:
            raise BudgetExhaustedError(self.max_tokens, self.used)

    def enforce_before_call(self, estimated_tokens: int = 0):
        """S17: Pre-call enforcement. Raises BudgetExhaustedError BEFORE an LLM call
        if hard_limit is True and the budget cannot afford the estimated tokens."""
        if self.hard_limit:
            if self.used >= self.max_tokens:
                raise BudgetExhaustedError(self.max_tokens, self.used)
            if estimated_tokens > 0 and not self.can_afford(estimated_tokens):
                raise BudgetExhaustedError(self.max_tokens, self.used)

    @property
    def remaining(self) -> int:
        return max(0, self.max_tokens - self.used)

    @property
    def utilization(self) -> float:
        """Budget utilization as fraction (0.0 to 1.0+)."""
        return self.used / self.max_tokens if self.max_tokens > 0 else 0

    def summary(self) -> str:
        lines = [
            f"Token Budget: {self.used:,}/{self.max_tokens:,} ({self.utilization:.1%})",
            f"  Remaining: {self.remaining:,}",
        ]
        if self._by_model:
            lines.append("  By model:")
            for model, tokens in sorted(self._by_model.items()):
                lines.append(f"    {model}: {tokens:,}")
        return "\n".join(lines)
