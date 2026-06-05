"""agentruntime.trace_store — Beta-reduction trace persistence"""
from __future__ import annotations
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class TraceRecord:
    """One beta-reduction record."""
    step: int = 0
    term_name: str = ""
    term_type: str = ""
    duration_ms: float = 0.0
    timestamp: float = 0.0
    input: str = ""
    output: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    thought: Optional[str] = None
    action: Optional[str] = None
    action_input: Optional[Dict] = None
    observation: Optional[str] = None
    terminated: bool = False
    model: str = ""
    temperature: float = 0.0
    error: Optional[str] = None


@dataclass
class TraceStats:
    total_steps: int = 0
    total_time_ms: float = 0.0
    total_tokens: int = 0
    tool_calls: int = 0
    llm_calls: int = 0
    errors: int = 0
    terminated_by: str = ""  # "terminate" | "implicit" | "max_steps"


class TraceStore:
    """
    Beta-reduction trace storage.
    Lambda: TraceStore = List[beta-reduction]
    """

    def __init__(self):
        self._records: List[TraceRecord] = []

    def append(self, record: TraceRecord) -> None:
        if record.timestamp == 0:
            record.timestamp = time.time()
        self._records.append(record)

    def get_all(self) -> List[TraceRecord]:
        return list(self._records)

    def get_step(self, n: int) -> Optional[TraceRecord]:
        if 0 <= n < len(self._records):
            return self._records[n]
        return None

    def to_json(self) -> str:
        return json.dumps([asdict(r) for r in self._records], indent=2, default=str)

    def to_timeline(self) -> str:
        """Format as human-readable timeline."""
        lines = []
        for r in self._records:
            inp_s = str(r.input)[:50]
            out_s = str(r.output)[:50]
            tag = ""
            if r.terminated:
                tag = " (base case)"
            elif r.error:
                tag = f" (ERROR: {r.error})"
            lines.append(
                f"B[{r.step}]  {r.term_name:15s} ({r.duration_ms:.1f}ms){tag}  "
                f"{inp_s} -> {out_s}"
            )
        lines.append("-" * 60)
        stats = self.stats()
        lines.append(
            f"Total: {stats.total_steps} beta-reductions, "
            f"{stats.total_time_ms:.1f}ms, "
            f"~{stats.total_tokens} tokens"
        )
        return "\n".join(lines)

    def stats(self) -> TraceStats:
        s = TraceStats()
        s.total_steps = len(self._records)
        s.total_time_ms = sum(r.duration_ms for r in self._records)
        s.total_tokens = sum(r.input_tokens + r.output_tokens for r in self._records)
        s.tool_calls = sum(1 for r in self._records if r.term_type in ("Tool", "MCP"))
        s.llm_calls = sum(1 for r in self._records if r.term_type == "Lam")
        s.errors = sum(1 for r in self._records if r.error)
        # Determine termination cause
        if self._records:
            last = self._records[-1]
            if last.terminated:
                if last.action == "terminate":
                    s.terminated_by = "terminate"
                else:
                    s.terminated_by = "implicit"
            else:
                s.terminated_by = "max_steps"
        return s
