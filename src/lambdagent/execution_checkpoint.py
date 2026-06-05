"""
lambdagent.execution_checkpoint — Resumable execution state

Extends Checkpoint to capture execution position, enabling
resume from the exact step where execution was interrupted.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .core import Context


@dataclass
class StackFrame:
    """One frame in the execution stack."""

    term_type: str  # 'Loop', 'Compose', 'GroupChat', 'ReAct'
    term_name: str  # Name of the term
    step_index: int  # Current position (loop step, compose stage index, etc.)
    local_state: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "term_type": self.term_type,
            "term_name": self.term_name,
            "step_index": self.step_index,
            "local_state": self.local_state,
        }

    @classmethod
    def from_dict(cls, data: dict) -> StackFrame:
        return cls(**data)


@dataclass
class ExecutionCheckpoint:
    """Checkpoint that captures execution position for resumption."""

    context: Optional[Dict[str, Any]] = None  # Serialized Context
    execution_stack: List[StackFrame] = field(default_factory=list)
    loop_state: Dict[str, Any] = field(default_factory=dict)  # step_count, last_result
    groupchat_state: Dict[str, Any] = field(default_factory=dict)  # round, conversation
    last_input: str = ""
    last_result: str = ""
    description: str = ""
    timestamp: float = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()

    @property
    def resume_step(self) -> int:
        """The step index to resume from."""
        if self.execution_stack:
            return self.execution_stack[-1].step_index
        return self.loop_state.get("step", 0)

    @property
    def term_type(self) -> str:
        """The type of term being executed."""
        if self.execution_stack:
            return self.execution_stack[-1].term_type
        return "unknown"

    def push_frame(
        self, term_type: str, term_name: str, step_index: int, **local_state
    ):
        """Push a new stack frame."""
        self.execution_stack.append(
            StackFrame(
                term_type=term_type,
                term_name=term_name,
                step_index=step_index,
                local_state=local_state,
            )
        )

    def pop_frame(self) -> Optional[StackFrame]:
        """Pop the top stack frame."""
        if self.execution_stack:
            return self.execution_stack.pop()
        return None

    def update_step(self, step_index: int, result: Any = None):
        """Update the current frame's step index."""
        if self.execution_stack:
            self.execution_stack[-1].step_index = step_index
        self.loop_state["step"] = step_index
        if result is not None:
            self.last_result = str(result)[:10000]  # Cap size
            self.loop_state["last_result"] = str(result)[:10000]

    def to_dict(self) -> dict:
        return {
            "version": "2.0.0",
            "timestamp": self.timestamp,
            "description": self.description,
            "last_input": self.last_input,
            "last_result": self.last_result,
            "context": self.context,
            "execution_stack": [f.to_dict() for f in self.execution_stack],
            "loop_state": self.loop_state,
            "groupchat_state": self.groupchat_state,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ExecutionCheckpoint:
        cp = cls(
            context=data.get("context"),
            last_input=data.get("last_input", ""),
            last_result=data.get("last_result", ""),
            description=data.get("description", ""),
            timestamp=data.get("timestamp", 0),
            loop_state=data.get("loop_state", {}),
            groupchat_state=data.get("groupchat_state", {}),
            metadata=data.get("metadata", {}),
        )
        for frame_data in data.get("execution_stack", []):
            cp.execution_stack.append(StackFrame.from_dict(frame_data))
        return cp

    def save(self, path: str) -> str:
        """Save checkpoint to JSON file."""
        os.makedirs(
            os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True
        )
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False, default=str)
        return path

    @classmethod
    def load(cls, path: str) -> ExecutionCheckpoint:
        """Load checkpoint from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"ExecutionCheckpoint: {self.description}",
            f"  Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.timestamp))}",
            f"  Resume step: {self.resume_step}",
            f"  Term type: {self.term_type}",
            f"  Stack depth: {len(self.execution_stack)}",
        ]
        if self.loop_state:
            lines.append(f"  Loop state: step={self.loop_state.get('step', '?')}")
        if self.groupchat_state:
            lines.append(f"  GroupChat: round={self.groupchat_state.get('round', '?')}")
        return "\n".join(lines)


class ExecutionCheckpointManager:
    """Manages auto-saving of execution checkpoints."""

    def __init__(
        self,
        directory: str = ".lambdagent/checkpoints",
        save_every_n_steps: int = 5,
        max_checkpoints: int = 10,
    ):
        self.directory = directory
        self.save_every_n_steps = save_every_n_steps
        self.max_checkpoints = max_checkpoints
        self._current: Optional[ExecutionCheckpoint] = None

    def begin(
        self, term_type: str, term_name: str, input_text: str
    ) -> ExecutionCheckpoint:
        """Begin tracking a new execution."""
        self._current = ExecutionCheckpoint(
            last_input=input_text,
            description=f"{term_type}:{term_name}",
        )
        self._current.push_frame(term_type, term_name, 0)
        return self._current

    def step(self, step_index: int, result: Any = None) -> bool:
        """Record a step. Returns True if checkpoint was saved."""
        if self._current is None:
            return False
        self._current.update_step(step_index, result)
        if step_index > 0 and step_index % self.save_every_n_steps == 0:
            self._auto_save()
            return True
        return False

    def finish(self):
        """Mark execution as complete."""
        self._current = None

    @property
    def current(self) -> Optional[ExecutionCheckpoint]:
        return self._current

    def latest(self) -> Optional[ExecutionCheckpoint]:
        """Load the most recent checkpoint."""
        files = self._list_files()
        if not files:
            return None
        return ExecutionCheckpoint.load(str(files[-1]))

    def _auto_save(self):
        """Save current checkpoint to disk."""
        if self._current is None:
            return
        os.makedirs(self.directory, exist_ok=True)
        idx = len(self._list_files()) + 1
        path = os.path.join(self.directory, f"exec_cp_{idx:04d}.json")
        self._current.save(path)
        self._cleanup()

    def _list_files(self) -> list:
        """List checkpoint files sorted by name."""
        if not os.path.exists(self.directory):
            return []
        files = sorted(
            f
            for f in os.listdir(self.directory)
            if f.startswith("exec_cp_") and f.endswith(".json")
        )
        return [os.path.join(self.directory, f) for f in files]

    def _cleanup(self):
        """Remove old checkpoints beyond max_checkpoints."""
        files = self._list_files()
        while len(files) > self.max_checkpoints:
            os.remove(files.pop(0))
