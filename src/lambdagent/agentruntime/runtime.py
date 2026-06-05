"""
agentruntime.runtime — Top-level Runtime class.

Phase 6.5: Updated to support dual engine switching via `engine_mode` param.
  - "recursive" (default): Python call stack (Executor.reduce)
  - "cek": Agent CEK Machine (step-by-step, pause/resume)
  - "adaptive": Auto-select based on term complexity
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from lambdagent.core import Term, Context

from .config import RuntimeConfig
from .executor import Executor
from .trace_store import TraceRecord, TraceStats
from .engine import EngineMode, EngineResult, UnifiedTraceRecord
from .recursive_engine import RecursiveEngine
from .cek_engine import CEKEngine


@dataclass
class RuntimeResult:
    """Result of a runtime execution."""

    result: str
    trace: List[TraceRecord] = field(default_factory=list)
    stats: Optional[TraceStats] = None
    context: Optional[Context] = None
    # Phase 6.5: unified engine output
    engine_result: Optional[EngineResult] = None


def _create_engine(
    engine_mode: str, config: RuntimeConfig | None = None, **engine_opts
):
    """Factory: create engine by mode string."""
    if engine_mode == "cek":
        return CEKEngine(
            cost_budget=engine_opts.get("cost_budget", float("inf")),
            max_steps=engine_opts.get("max_steps", 10000),
            handler=engine_opts.get("handler"),
        )
    elif engine_mode == "adaptive":
        from .adaptive_engine import AdaptiveEngine

        return AdaptiveEngine(config=config, **engine_opts)
    else:
        # Default: recursive
        return RecursiveEngine(config=config)


class Runtime:
    """
    lambdagent runtime.

    Lambda:
        Runtime = (Engine, Gamma, trace)
        Runtime.execute(config, input) =
            let term = compile(config) in
            let ctx  = Context.new() in
            Engine.execute(term, input, ctx)

    Phase 6.5: Supports engine switching via engine_mode param.
    """

    def __init__(
        self, config: RuntimeConfig, engine_mode: str = "recursive", **engine_opts
    ):
        self.config = config
        self.engine_mode = engine_mode
        self.engine = _create_engine(engine_mode, config, **engine_opts)
        # Keep executor for backward compatibility
        self.executor = Executor(config)

    def run(self, term: Term, input_val: str, **opts) -> RuntimeResult:
        """Execute a compiled Term with input."""
        ctx = Context()

        # Use the selected engine
        engine_result = self.engine.execute(term, input_val, ctx, **opts)

        # Extract legacy trace format for backward compatibility
        trace_records = (
            self.executor.trace.get_all() if hasattr(self.executor, "trace") else []
        )
        # If engine populated ctx.trace, build legacy records from that
        if not trace_records and ctx.trace:
            trace_records = [
                TraceRecord(
                    step=i,
                    term_name=e.term_name,
                    term_type="",
                    duration_ms=e.duration_ms,
                    input=str(e.input)[:200],
                    output=str(e.output)[:200],
                    model=e.model,
                    input_tokens=e.tokens_used,
                    output_tokens=0,
                )
                for i, e in enumerate(ctx.trace)
            ]

        stats = TraceStats(
            total_steps=engine_result.steps,
            total_tokens=engine_result.cost.tokens,
            total_time_ms=sum(r.duration_ms for r in engine_result.trace),
            llm_calls=sum(
                1 for r in engine_result.trace if r.action in ("llm_call", "llm")
            ),
            tool_calls=sum(
                1 for r in engine_result.trace if r.action in ("tool_call", "tool")
            ),
        )

        return RuntimeResult(
            result=str(engine_result.value),
            trace=trace_records,
            stats=stats,
            context=ctx,
            engine_result=engine_result,
        )

    @staticmethod
    def execute(config_path: str, input_val: str, **overrides) -> RuntimeResult:
        """
        One-stop execution: compile + init runtime + beta-reduce.

        Supports engine_mode override:
            Runtime.execute("config.yml", "input", engine_mode="cek")
        """
        from lambdagent.fromconfig import from_config

        # Extract engine config from overrides
        engine_mode = overrides.pop("engine_mode", None)
        engine_opts = {}
        for key in ("cost_budget", "max_steps", "handler"):
            if key in overrides:
                engine_opts[key] = overrides.pop(key)

        # 1. Build runtime config
        config = RuntimeConfig.from_yaml(config_path, **overrides)

        # 2. Check YAML for runtime.engine if not overridden
        if engine_mode is None:
            import yaml

            with open(config_path) as f:
                raw = yaml.safe_load(f)
            runtime_cfg = (raw or {}).get("runtime", {})
            engine_mode = runtime_cfg.get("engine", "recursive")
            if "costBudget" in runtime_cfg and "cost_budget" not in engine_opts:
                engine_opts["cost_budget"] = runtime_cfg["costBudget"]
            if "maxSteps" in runtime_cfg and "max_steps" not in engine_opts:
                engine_opts["max_steps"] = runtime_cfg["maxSteps"]

        # 3. Compile Lambda term
        term = from_config(config_path, **overrides)

        # 4. Init runtime with selected engine
        runtime = Runtime(config, engine_mode=engine_mode or "recursive", **engine_opts)

        # 5. Execute
        return runtime.run(term, input_val)
