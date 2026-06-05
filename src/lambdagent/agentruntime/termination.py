"""agentruntime.termination — Y combinator base case detection"""

from __future__ import annotations
from typing import List, Optional


class TerminationOracle:
    """
    Determines when the Y combinator reaches its base case.
    Lambda: should_stop = lambda (thought, obs, step).
        IF (action = terminate) THEN TRUE
        ELSE IF (implicit_signal(thought)) THEN TRUE
        ELSE FALSE
    """

    DEFAULT_SIGNALS = [
        "final answer:",
        "task complete",
        "task is done",
        "i have completed",
        "here is the result:",
        "in conclusion,",
    ]

    def __init__(self, signals: List[str] = None, implicit_detection: bool = True):
        self.signals = signals or self.DEFAULT_SIGNALS
        self.implicit_detection = implicit_detection

    def should_stop(self, thought: str, observation: Optional[str], step: int) -> bool:
        """
        Three-layer termination detection:
        Layer 1: Explicit terminate (handled by caller before this)
        Layer 2: Implicit signals in thought text
        Layer 3: Max steps (handled by caller's for loop)
        """
        if not self.implicit_detection:
            return False

        thought_lower = thought.lower()

        # Check implicit signals
        for signal in self.signals:
            if signal in thought_lower:
                return True

        return False
