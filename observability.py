"""
lambdagent.observability — Structured observability for agent execution

Provides OpenTelemetry-compatible tracing for β-reductions.
Works without otel dependency (graceful degradation to no-op).
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class SpanRecord:
    """A single tracing span."""
    name: str
    start_time: float
    end_time: float = 0
    attributes: Dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    parent_id: Optional[str] = None
    span_id: str = ""

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
            "status": self.status,
        }


class AgentTracer:
    """Lightweight tracer for agent execution.

    Collects spans locally. Can export to OpenTelemetry if available.
    """

    def __init__(self, service_name: str = "lambdagent"):
        self.service_name = service_name
        self.spans: list[SpanRecord] = []
        self._otel_tracer = None
        self._span_counter = 0
        self._try_init_otel()

    def _try_init_otel(self):
        """Try to initialize OpenTelemetry tracer. No-op if not installed."""
        try:
            from opentelemetry import trace
            self._otel_tracer = trace.get_tracer(self.service_name)
        except ImportError:
            pass

    def _next_id(self) -> str:
        self._span_counter += 1
        return f"span_{self._span_counter:06d}"

    @contextmanager
    def span(self, name: str, parent_id: Optional[str] = None, **attributes):
        """Create a tracing span."""
        span_id = self._next_id()
        record = SpanRecord(
            name=name,
            start_time=time.time(),
            span_id=span_id,
            parent_id=parent_id,
            attributes=attributes,
        )

        # OTel span if available
        otel_span = None
        otel_ctx = None
        if self._otel_tracer:
            try:
                otel_span = self._otel_tracer.start_span(name, attributes=attributes)
                otel_ctx = otel_span
            except Exception:
                pass

        try:
            yield record
            record.status = "ok"
        except Exception as e:
            record.status = f"error: {type(e).__name__}"
            record.attributes["error"] = str(e)
            raise
        finally:
            record.end_time = time.time()
            self.spans.append(record)
            if otel_span:
                try:
                    for k, v in record.attributes.items():
                        otel_span.set_attribute(str(k), str(v) if not isinstance(v, (int, float, bool)) else v)
                    otel_span.end()
                except Exception:
                    pass

    def record_reduction(self, term_name: str, term_type: str, input_val: Any,
                        output_val: Any, duration_ms: float, model: str = "",
                        tokens: int = 0, parent_id: str = None):
        """Record a β-reduction as a span."""
        self.spans.append(SpanRecord(
            name=f"reduce.{term_type}.{term_name}",
            start_time=time.time() - duration_ms / 1000,
            end_time=time.time(),
            span_id=self._next_id(),
            parent_id=parent_id,
            attributes={
                "term.name": term_name,
                "term.type": term_type,
                "input.length": len(str(input_val)),
                "output.length": len(str(output_val)),
                "model": model,
                "tokens.used": tokens,
                "duration_ms": duration_ms,
            },
        ))

    def export_json(self) -> list:
        """Export all spans as JSON-serializable list."""
        return [s.to_dict() for s in self.spans]

    def summary(self) -> str:
        """Human-readable summary of traced execution."""
        if not self.spans:
            return "No spans recorded."
        total_ms = sum(s.duration_ms for s in self.spans)
        lines = [f"Trace: {len(self.spans)} spans, {total_ms:.0f}ms total"]
        for s in self.spans:
            status = "\u2713" if s.status == "ok" else "\u2717"
            lines.append(f"  {status} {s.name} ({s.duration_ms:.0f}ms)")
        return "\n".join(lines)
