"""
lambdagent.context_manager — Context window management

Prevents unbounded state growth in ReAct loops by compacting
old steps into summaries while preserving recent context.
"""

from __future__ import annotations
from typing import List, Optional, Tuple


class ContextManager:
    """Manages context window size for ReAct loops."""

    def __init__(
        self,
        max_tokens: int = 100000,
        compact_threshold: float = 0.8,
        keep_recent: int = 3,
    ):
        self.max_tokens = max_tokens
        self.compact_threshold = compact_threshold
        self.keep_recent = keep_recent

    def estimate_tokens(self, text: str) -> int:
        """Rough token count estimate (4 chars ≈ 1 token)."""
        return len(text) // 4

    def should_compact(self, state: str) -> bool:
        """Check if state exceeds compact threshold."""
        return self.estimate_tokens(state) > self.max_tokens * self.compact_threshold

    def compact(self, state: str) -> str:
        """Compact state by summarizing old steps, keeping recent ones intact.

        This is a local compaction (no LLM call). For LLM-based summarization,
        use compact_with_llm().
        """
        parts = state.split("\n\n[Step ")
        if len(parts) <= 1:
            return state

        user_input = parts[0]
        steps = [f"[Step {p}" for p in parts[1:]]

        if len(steps) <= self.keep_recent:
            return state

        old_steps = steps[: -self.keep_recent]
        recent_steps = steps[-self.keep_recent :]

        # Summarize old steps (extract action + key observation)
        summary_lines = ["[Previous Steps Summary]"]
        for s in old_steps:
            lines = s.strip().split("\n")
            header = lines[0] if lines else ""
            action = ""
            obs_preview = ""
            for line in lines:
                if line.strip().startswith("Action:"):
                    action = line.strip()
                elif line.strip().startswith("Observation:"):
                    obs_preview = line.strip()[:150]
            summary_lines.append(f"  {header}: {action} → {obs_preview}")

        summary = "\n".join(summary_lines)
        return f"{user_input}\n\n{summary}\n\n" + "\n\n".join(recent_steps)

    async def compact_with_llm(self, state: str, llm_fn) -> str:
        """Compact state using LLM for better summarization.

        Args:
            state: current state string
            llm_fn: async callable(prompt) -> summary string
        """
        parts = state.split("\n\n[Step ")
        if len(parts) <= 1:
            return state

        user_input = parts[0]
        steps = [f"[Step {p}" for p in parts[1:]]

        if len(steps) <= self.keep_recent:
            return state

        old_steps = steps[: -self.keep_recent]
        recent_steps = steps[-self.keep_recent :]

        old_text = "\n\n".join(old_steps)
        summary = await llm_fn(
            f"Summarize these agent execution steps concisely, "
            f"preserving key findings, decisions, and results:\n\n{old_text}"
        )

        return (
            f"{user_input}\n\n"
            f"[Summary of steps 1-{len(old_steps)}]\n{summary}\n\n"
            + "\n\n".join(recent_steps)
        )
