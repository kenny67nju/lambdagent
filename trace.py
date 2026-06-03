"""
lambdagent.trace — Enhanced β-reduction trace system

Features:
  1. Colorized terminal output with progress bar
  2. parent_step nesting (tree structure)
  3. Anomaly detection (5 algorithms)
  4. Flamegraph HTML export
  5. Replay + diff comparison

Every feature maps to the patent:
  "基于β-规约追踪的大语言模型智能体调试方法及系统"
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple


# ════════════════════════════════════════════════════════════
# 1. Enhanced TraceEntry with parent_step nesting
# ════════════════════════════════════════════════════════════

@dataclass
class TraceEntry:
    """One β-reduction step with nesting support."""
    step: int = 0
    term_type: str = ""      # Lam | Tool | Compose | Loop | Route | Guard | Memory
    name: str = ""
    input: str = ""
    output: str = ""
    elapsed_ms: float = 0.0
    model: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    parent_step: Optional[int] = None   # nesting: which step spawned this
    depth: int = 0                       # nesting depth (0 = top-level)
    terminated: bool = False             # is this the base case?
    error: Optional[str] = None
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


# ════════════════════════════════════════════════════════════
# 2. EnhancedTraceStore
# ════════════════════════════════════════════════════════════

class TraceStore:
    """
    Enhanced β-reduction trace storage.

    Lambda: TraceStore = List[β-reduction] with tree structure
    """

    def __init__(self):
        self._entries: List[TraceEntry] = []
        self._step_counter: int = 0
        self._depth_stack: List[int] = []  # stack of parent steps

    @property
    def entries(self) -> List[TraceEntry]:
        return list(self._entries)

    @property
    def step_count(self) -> int:
        return len(self._entries)

    def push_scope(self, parent_step: int):
        """Enter a nested scope (Compose, Loop, etc.)."""
        self._depth_stack.append(parent_step)

    def pop_scope(self):
        """Leave a nested scope."""
        if self._depth_stack:
            self._depth_stack.pop()

    def record(self, term_type: str, name: str, inp: Any, out: Any,
               elapsed_ms: float, model: str = "", tokens_in: int = 0,
               tokens_out: int = 0, terminated: bool = False,
               error: Optional[str] = None) -> TraceEntry:
        """Record one β-reduction step."""
        entry = TraceEntry(
            step=self._step_counter,
            term_type=term_type,
            name=name,
            input=_truncate(inp, 500),
            output=_truncate(out, 500),
            elapsed_ms=elapsed_ms,
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            parent_step=self._depth_stack[-1] if self._depth_stack else None,
            depth=len(self._depth_stack),
            terminated=terminated,
            error=error,
        )
        self._entries.append(entry)
        self._step_counter += 1
        return entry

    # ── Serialization ──

    def to_json(self, indent: int = 2) -> str:
        return json.dumps([asdict(e) for e in self._entries], indent=indent, default=str)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> "TraceStore":
        store = cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for d in data:
            entry = TraceEntry(**{k: v for k, v in d.items() if k in TraceEntry.__dataclass_fields__})
            store._entries.append(entry)
        store._step_counter = len(store._entries)
        return store

    # ── Stats ──

    def stats(self) -> Dict[str, Any]:
        total_ms = sum(e.elapsed_ms for e in self._entries)
        total_tokens = sum(e.tokens_in + e.tokens_out for e in self._entries)
        llm_calls = sum(1 for e in self._entries if e.term_type == "Lam")
        tool_calls = sum(1 for e in self._entries if e.term_type in ("Tool", "MCP"))
        errors = sum(1 for e in self._entries if e.error)
        terminated_by = "unknown"
        if self._entries:
            last = self._entries[-1]
            if last.terminated:
                terminated_by = "base_case"
            elif last.error:
                terminated_by = "error"
            else:
                terminated_by = "max_steps"
        return {
            "total_steps": len(self._entries),
            "total_ms": total_ms,
            "total_tokens": total_tokens,
            "llm_calls": llm_calls,
            "tool_calls": tool_calls,
            "errors": errors,
            "terminated_by": terminated_by,
            "avg_step_ms": total_ms / max(1, len(self._entries)),
        }


# ════════════════════════════════════════════════════════════
# 3. Colorized Terminal Output
# ════════════════════════════════════════════════════════════

# ANSI color codes
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RED = "\033[31m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_BLUE = "\033[34m"
_MAGENTA = "\033[35m"
_CYAN = "\033[36m"
_WHITE = "\033[37m"
_BG_RED = "\033[41m"

_TYPE_COLORS = {
    "Lam": _BLUE,
    "Tool": _GREEN,
    "Compose": _CYAN,
    "Loop": _MAGENTA,
    "Route": _YELLOW,
    "Guard": _RED,
    "Memory": _WHITE,
    "MCP": _GREEN,
    "Pair": _CYAN,
}


def _supports_color() -> bool:
    """Check if terminal supports color."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def colorize_timeline(store: TraceStore, show_io: bool = True, max_io_len: int = 60) -> str:
    """
    Generate colorized timeline view.

    β[0] ████████████ Lam:think      2.8s  "帮我写排序" → "需要搜索"
    β[1]   ██████     Tool:search    1.2s  "排序算法" → "快排..."
    β[2] ████████████ Lam:think      3.1s  "选择快排" → "开始写代码"
    """
    entries = store.entries
    if not entries:
        return "(empty trace)"

    use_color = _supports_color()
    max_ms = max(e.elapsed_ms for e in entries) if entries else 1
    lines = []

    # Header
    if use_color:
        lines.append(f"{_BOLD}β-reduction trace ({len(entries)} steps){_RESET}")
        lines.append(f"{_DIM}{'─' * 70}{_RESET}")
    else:
        lines.append(f"β-reduction trace ({len(entries)} steps)")
        lines.append("─" * 70)

    for e in entries:
        indent = "  " * e.depth
        bar_len = max(1, int(20 * e.elapsed_ms / max(max_ms, 1)))
        bar = "█" * bar_len

        type_color = _TYPE_COLORS.get(e.term_type, "") if use_color else ""
        reset = _RESET if use_color else ""
        bold = _BOLD if use_color else ""
        dim = _DIM if use_color else ""
        red = _RED if use_color else ""
        green = _GREEN if use_color else ""

        # Step number
        step_str = f"β[{e.step}]"

        # Status indicator
        if e.terminated:
            status = f"{green}■{reset}" if use_color else "■"
        elif e.error:
            status = f"{red}✗{reset}" if use_color else "✗"
        else:
            status = " "

        # Time formatting
        if e.elapsed_ms >= 1000:
            time_str = f"{e.elapsed_ms/1000:.1f}s"
        else:
            time_str = f"{e.elapsed_ms:.0f}ms"

        # Main line
        line = (f"  {step_str:6s} {indent}{status} "
                f"{type_color}{bar} {e.term_type}:{e.name}{reset} "
                f"{dim}({time_str}){reset}")

        if e.error:
            line += f" {red}ERROR: {e.error[:40]}{reset}"

        lines.append(line)

        # I/O detail
        if show_io:
            inp_s = str(e.input)[:max_io_len]
            out_s = str(e.output)[:max_io_len]
            io_line = f"         {indent}  {dim}{inp_s} → {out_s}{reset}"
            lines.append(io_line)

    # Footer: stats
    s = store.stats()
    lines.append(f"{_DIM if use_color else ''}{'─' * 70}{_RESET if use_color else ''}")
    lines.append(
        f"  Total: {s['total_steps']} β-reductions, "
        f"{s['total_ms']/1000:.1f}s, "
        f"~{s['total_tokens']} tokens, "
        f"terminated by: {s['terminated_by']}"
    )

    # Progress bar
    if use_color and entries:
        total_ms = s["total_ms"]
        cumulative = 0
        progress = []
        for e in entries:
            pct = e.elapsed_ms / max(total_ms, 1)
            color = _TYPE_COLORS.get(e.term_type, _WHITE)
            seg_len = max(1, int(60 * pct))
            progress.append(f"{color}{'▓' * seg_len}")
            cumulative += e.elapsed_ms
        lines.append(f"  {_DIM}[{''.join(progress)}{_RESET}{_DIM}]{_RESET}")

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════
# 4. Anomaly Detection (5 algorithms)
# ════════════════════════════════════════════════════════════

@dataclass
class Anomaly:
    """Detected anomaly in trace."""
    type: str           # LATENCY_SPIKE | LOOP_DIVERGENCE | GUARD_STREAK | TOOL_REPEAT | EXCESSIVE_STEPS
    severity: str       # ERROR | WARN
    step: int           # which step triggered it
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


def detect_anomalies(store: TraceStore,
                     latency_factor: float = 3.0,
                     similarity_threshold: float = 0.90,
                     max_guard_fails: int = 3,
                     max_tool_repeats: int = 5,
                     step_warn: int = 50,
                     step_error: int = 100) -> List[Anomaly]:
    """
    Run all 5 anomaly detection algorithms on a trace.
    Returns list of detected anomalies.
    """
    entries = store.entries
    if not entries:
        return []

    anomalies = []

    # ── Algorithm 1: Latency Spike ──
    elapsed_values = [e.elapsed_ms for e in entries if e.elapsed_ms > 0]
    if len(elapsed_values) >= 3:
        mean_ms = sum(elapsed_values) / len(elapsed_values)
        for e in entries:
            if e.elapsed_ms > mean_ms * latency_factor and e.elapsed_ms > 100:
                anomalies.append(Anomaly(
                    type="LATENCY_SPIKE",
                    severity="WARN",
                    step=e.step,
                    message=f"β[{e.step}] {e.term_type}:{e.name} took {e.elapsed_ms:.0f}ms "
                            f"(mean: {mean_ms:.0f}ms, {e.elapsed_ms/mean_ms:.1f}x)",
                    details={"elapsed_ms": e.elapsed_ms, "mean_ms": mean_ms,
                             "factor": e.elapsed_ms / mean_ms},
                ))

    # ── Algorithm 2: Loop Divergence ──
    loop_entries = [e for e in entries if e.term_type == "Loop"]
    if len(loop_entries) >= 3:
        for i in range(2, len(loop_entries)):
            sim = _text_similarity(loop_entries[i].output, loop_entries[i-1].output)
            if sim > similarity_threshold:
                anomalies.append(Anomaly(
                    type="LOOP_DIVERGENCE",
                    severity="WARN",
                    step=loop_entries[i].step,
                    message=f"Loop output similarity {sim:.1%} at β[{loop_entries[i].step}] — "
                            f"output not changing, possible stall",
                    details={"similarity": sim, "consecutive": i},
                ))
                break  # report once

    # ── Algorithm 3: Guard Streak Failure ──
    guard_entries = [e for e in entries if e.term_type == "Guard"]
    consecutive_fails = 0
    for e in guard_entries:
        if e.error or "stuck" in str(e.output).lower():
            consecutive_fails += 1
            if consecutive_fails >= max_guard_fails:
                anomalies.append(Anomaly(
                    type="GUARD_STREAK",
                    severity="ERROR",
                    step=e.step,
                    message=f"Guard failed {consecutive_fails} consecutive times at β[{e.step}]",
                    details={"consecutive_fails": consecutive_fails},
                ))
                break
        else:
            consecutive_fails = 0

    # ── Algorithm 4: Tool Repetition ──
    recent_tools: List[str] = []
    for e in entries:
        if e.term_type in ("Tool", "MCP"):
            recent_tools.append(e.name)
            if len(recent_tools) >= max_tool_repeats:
                last_n = recent_tools[-max_tool_repeats:]
                if all(t == last_n[0] for t in last_n):
                    anomalies.append(Anomaly(
                        type="TOOL_REPEAT",
                        severity="WARN",
                        step=e.step,
                        message=f"Tool '{e.name}' called {max_tool_repeats} times "
                                f"consecutively at β[{e.step}]",
                        details={"tool": e.name, "repeat_count": max_tool_repeats},
                    ))
                    break

    # ── Algorithm 5: Excessive Steps ──
    total = len(entries)
    if total > step_error:
        anomalies.append(Anomaly(
            type="EXCESSIVE_STEPS",
            severity="ERROR",
            step=total - 1,
            message=f"Trace has {total} steps (> {step_error} threshold)",
            details={"total_steps": total, "threshold": step_error},
        ))
    elif total > step_warn:
        anomalies.append(Anomaly(
            type="EXCESSIVE_STEPS",
            severity="WARN",
            step=total - 1,
            message=f"Trace has {total} steps (> {step_warn} threshold)",
            details={"total_steps": total, "threshold": step_warn},
        ))

    return anomalies


def format_anomalies(anomalies: List[Anomaly]) -> str:
    """Format anomalies for terminal output."""
    if not anomalies:
        return "  No anomalies detected ✓"

    use_color = _supports_color()
    lines = []
    for a in anomalies:
        if use_color:
            color = _RED if a.severity == "ERROR" else _YELLOW
            icon = "✗" if a.severity == "ERROR" else "⚠"
            lines.append(f"  {color}{icon} [{a.severity}] {a.type}{_RESET}: {a.message}")
        else:
            icon = "x" if a.severity == "ERROR" else "!"
            lines.append(f"  [{icon}] [{a.severity}] {a.type}: {a.message}")
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════
# 5. Flamegraph (HTML export)
# ════════════════════════════════════════════════════════════

def generate_flamegraph_html(store: TraceStore, title: str = "lambdagent β-reduction flamegraph") -> str:
    """
    Generate a self-contained HTML flamegraph from trace data.
    Each bar = one β-reduction, width = elapsed time, depth = nesting.
    """
    entries = store.entries
    if not entries:
        return "<html><body>Empty trace</body></html>"

    total_ms = sum(e.elapsed_ms for e in entries)
    stats = store.stats()

    # Build rows by depth
    max_depth = max(e.depth for e in entries) if entries else 0

    # Generate SVG bars
    bars_svg = []
    bar_height = 24
    padding = 2
    svg_width = 900
    y_offset = 40  # space for title

    # Group entries by depth level for positioning
    for e in entries:
        if total_ms == 0:
            width_pct = 100
        else:
            width_pct = max(0.5, (e.elapsed_ms / total_ms) * 100)

        # x position: based on cumulative time of entries before this one at same depth
        preceding_ms = sum(
            prev.elapsed_ms for prev in entries[:e.step]
            if prev.depth == e.depth
        )
        x_pct = (preceding_ms / max(total_ms, 1)) * 100

        y = y_offset + (max_depth - e.depth) * (bar_height + padding)

        # Color by type
        colors = {
            "Lam": "#4A90D9", "Tool": "#50B050", "Compose": "#40B0B0",
            "Loop": "#B060B0", "Route": "#D0A030", "Guard": "#D05050",
            "Memory": "#808080", "MCP": "#50B050", "Pair": "#40B0B0",
        }
        color = colors.get(e.term_type, "#999999")

        time_str = f"{e.elapsed_ms:.0f}ms" if e.elapsed_ms < 1000 else f"{e.elapsed_ms/1000:.1f}s"
        label = f"{e.term_type}:{e.name} ({time_str})"

        bars_svg.append(f'''
        <g class="bar" data-step="{e.step}" data-type="{e.term_type}"
           data-name="{e.name}" data-ms="{e.elapsed_ms:.1f}"
           data-input="{_html_escape(_truncate(e.input, 100))}"
           data-output="{_html_escape(_truncate(e.output, 100))}">
          <rect x="{x_pct}%" y="{y}" width="{width_pct}%" height="{bar_height}"
                fill="{color}" rx="3" stroke="#fff" stroke-width="1"
                opacity="0.9"/>
          <text x="{x_pct + width_pct/2}%" y="{y + bar_height/2 + 4}"
                text-anchor="middle" font-size="11" fill="white"
                style="pointer-events:none">{_html_escape(label[:40])}</text>
        </g>''')

    svg_height = y_offset + (max_depth + 1) * (bar_height + padding) + 20

    html = f'''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{_html_escape(title)}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
         margin: 20px; background: #1a1a2e; color: #eee; }}
  h1 {{ font-size: 18px; color: #4A90D9; }}
  .stats {{ font-size: 13px; color: #888; margin-bottom: 10px; }}
  .bar rect:hover {{ opacity: 1; stroke: #FFD700; stroke-width: 2; cursor: pointer; }}
  #tooltip {{ position: fixed; background: #2a2a4a; border: 1px solid #4A90D9;
              border-radius: 6px; padding: 10px; font-size: 12px; display: none;
              max-width: 400px; z-index: 100; box-shadow: 0 4px 12px rgba(0,0,0,0.5); }}
  #tooltip .label {{ color: #4A90D9; font-weight: bold; }}
  #tooltip .io {{ color: #aaa; font-size: 11px; word-break: break-all; }}
  .legend {{ display: flex; gap: 12px; margin: 10px 0; font-size: 12px; }}
  .legend span {{ display: flex; align-items: center; gap: 4px; }}
  .legend .swatch {{ width: 14px; height: 14px; border-radius: 3px; }}
</style>
</head>
<body>
<h1>🔥 {_html_escape(title)}</h1>
<div class="stats">
  {stats['total_steps']} β-reductions · {stats['total_ms']/1000:.1f}s ·
  ~{stats['total_tokens']} tokens · {stats['llm_calls']} LLM calls ·
  {stats['tool_calls']} tool calls · terminated by: {stats['terminated_by']}
</div>
<div class="legend">
  <span><div class="swatch" style="background:#4A90D9"></div> Lam (LLM)</span>
  <span><div class="swatch" style="background:#50B050"></div> Tool</span>
  <span><div class="swatch" style="background:#B060B0"></div> Loop</span>
  <span><div class="swatch" style="background:#D0A030"></div> Route</span>
  <span><div class="swatch" style="background:#D05050"></div> Guard</span>
  <span><div class="swatch" style="background:#40B0B0"></div> Compose</span>
</div>
<svg width="100%" height="{svg_height}" xmlns="http://www.w3.org/2000/svg">
  <text x="10" y="25" font-size="14" fill="#888">
    depth ↑ | time →  (total: {total_ms/1000:.1f}s)
  </text>
  {''.join(bars_svg)}
</svg>
<div id="tooltip"></div>
<script>
document.querySelectorAll('.bar').forEach(bar => {{
  bar.addEventListener('mouseenter', e => {{
    const d = bar.dataset;
    const tt = document.getElementById('tooltip');
    tt.innerHTML = `<div class="label">β[${{d.step}}] ${{d.type}}:${{d.name}}</div>
      <div>${{d.ms}}ms</div>
      <div class="io">Input: ${{d.input}}</div>
      <div class="io">Output: ${{d.output}}</div>`;
    tt.style.display = 'block';
    tt.style.left = (e.clientX + 15) + 'px';
    tt.style.top = (e.clientY + 15) + 'px';
  }});
  bar.addEventListener('mouseleave', () => {{
    document.getElementById('tooltip').style.display = 'none';
  }});
}});
</script>
</body>
</html>'''
    return html


def save_flamegraph(store: TraceStore, path: str, title: str = "lambdagent β-reduction flamegraph"):
    """Save flamegraph as HTML file."""
    html = generate_flamegraph_html(store, title)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


# ════════════════════════════════════════════════════════════
# 6. Replay + Diff
# ════════════════════════════════════════════════════════════

def replay(store: TraceStore, speed: float = 1.0, show_io: bool = True):
    """
    Replay a trace step by step with timing.
    speed=1.0 replays at original speed, speed=2.0 at 2x, etc.
    """
    entries = store.entries
    use_color = _supports_color()

    print(f"\n{'─' * 60}")
    print(f"  Replaying {len(entries)} β-reductions (speed: {speed}x)")
    print(f"{'─' * 60}\n")

    for e in entries:
        indent = "  " * e.depth
        type_color = _TYPE_COLORS.get(e.term_type, "") if use_color else ""
        reset = _RESET if use_color else ""
        dim = _DIM if use_color else ""

        time_str = f"{e.elapsed_ms:.0f}ms" if e.elapsed_ms < 1000 else f"{e.elapsed_ms/1000:.1f}s"

        # Print step
        status = "■" if e.terminated else ("✗" if e.error else "▶")
        print(f"  β[{e.step}] {indent}{status} "
              f"{type_color}{e.term_type}:{e.name}{reset} "
              f"{dim}({time_str}){reset}")

        if show_io:
            inp_s = str(e.input)[:60]
            out_s = str(e.output)[:60]
            print(f"         {indent}  {dim}{inp_s} → {out_s}{reset}")

        # Wait proportional to original time
        wait_ms = e.elapsed_ms / speed / 1000
        if wait_ms > 0.01:  # don't sleep for trivial durations
            time.sleep(min(wait_ms, 3.0))  # cap at 3s per step

    print(f"\n{'─' * 60}")
    print(f"  Replay complete.")
    print(f"{'─' * 60}")


def diff_traces(trace_a: TraceStore, trace_b: TraceStore,
                label_a: str = "A", label_b: str = "B") -> str:
    """
    Compare two traces side by side.
    Aligns by (term_type, name) and reports differences.
    """
    entries_a = trace_a.entries
    entries_b = trace_b.entries
    use_color = _supports_color()

    lines = []
    lines.append(f"Trace diff: {label_a} ({len(entries_a)} steps) vs {label_b} ({len(entries_b)} steps)")
    lines.append("═" * 70)

    max_len = max(len(entries_a), len(entries_b))
    same_count = 0
    changed_count = 0
    added_count = 0
    removed_count = 0

    for i in range(max_len):
        ea = entries_a[i] if i < len(entries_a) else None
        eb = entries_b[i] if i < len(entries_b) else None

        if ea and eb:
            if ea.term_type == eb.term_type and ea.name == eb.name:
                if ea.output == eb.output:
                    # Same
                    same_count += 1
                    if use_color:
                        lines.append(f"  {_DIM}β[{i}] {ea.term_type}:{ea.name} — SAME{_RESET}")
                    else:
                        lines.append(f"  β[{i}] {ea.term_type}:{ea.name} — SAME")
                else:
                    # Changed output
                    changed_count += 1
                    if use_color:
                        lines.append(f"  {_YELLOW}β[{i}] {ea.term_type}:{ea.name} — CHANGED{_RESET}")
                        lines.append(f"    {_RED}  {label_a}: {str(ea.output)[:50]}{_RESET}")
                        lines.append(f"    {_GREEN}  {label_b}: {str(eb.output)[:50]}{_RESET}")
                    else:
                        lines.append(f"  β[{i}] {ea.term_type}:{ea.name} — CHANGED")
                        lines.append(f"    - {label_a}: {str(ea.output)[:50]}")
                        lines.append(f"    + {label_b}: {str(eb.output)[:50]}")
            else:
                # Different structure
                changed_count += 1
                lines.append(f"  β[{i}] STRUCTURE CHANGED: "
                             f"{ea.term_type}:{ea.name} → {eb.term_type}:{eb.name}")
        elif ea:
            removed_count += 1
            if use_color:
                lines.append(f"  {_RED}β[{i}] {ea.term_type}:{ea.name} — REMOVED (only in {label_a}){_RESET}")
            else:
                lines.append(f"  β[{i}] {ea.term_type}:{ea.name} — REMOVED (only in {label_a})")
        elif eb:
            added_count += 1
            if use_color:
                lines.append(f"  {_GREEN}β[{i}] {eb.term_type}:{eb.name} — ADDED (only in {label_b}){_RESET}")
            else:
                lines.append(f"  β[{i}] {eb.term_type}:{eb.name} — ADDED (only in {label_b})")

    lines.append("═" * 70)
    lines.append(f"  Same: {same_count}, Changed: {changed_count}, "
                 f"Added: {added_count}, Removed: {removed_count}")

    # Timing comparison
    ms_a = sum(e.elapsed_ms for e in entries_a)
    ms_b = sum(e.elapsed_ms for e in entries_b)
    if ms_a > 0:
        speedup = ms_a / max(ms_b, 1)
        lines.append(f"  Time: {label_a}={ms_a/1000:.1f}s, {label_b}={ms_b/1000:.1f}s "
                     f"({speedup:.2f}x {'faster' if speedup > 1 else 'slower'})")

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════
# Utility functions
# ════════════════════════════════════════════════════════════

def _truncate(value: Any, max_len: int = 200) -> str:
    s = str(value)
    if len(s) > max_len:
        return s[:max_len - 3] + "..."
    return s


def _text_similarity(a: str, b: str) -> float:
    """Simple Jaccard similarity on word sets."""
    if not a or not b:
        return 0.0
    words_a = set(str(a).lower().split())
    words_b = set(str(b).lower().split())
    if not words_a and not words_b:
        return 1.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / max(len(union), 1)


def _html_escape(s: str) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
