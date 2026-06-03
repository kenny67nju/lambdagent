"""
lambdagent.builtin_tools.terminal_ui — Streaming terminal UI

Renders ReAct execution with:
  - Token-by-token LLM streaming
  - Tool call display (collapsed/expanded)
  - Spinner during execution
  - Colored β-reduction trace
"""
from __future__ import annotations

import sys
import time
from typing import Any


def _has_rich() -> bool:
    try:
        from rich.console import Console
        return True
    except ImportError:
        return False


class TerminalUI:
    """Streaming terminal UI for agent execution."""

    def __init__(self, verbose: bool = False, color: bool = True):
        self.verbose = verbose
        self.color = color
        self._use_rich = color and _has_rich()
        self._console = None
        self._step_count = 0
        self._total_tokens = 0
        self._start_time = 0

        if self._use_rich:
            from rich.console import Console
            self._console = Console()

    def start(self, agent_name: str, agent_type: str, config_path: str = ""):
        """Display agent start info."""
        self._start_time = time.time()
        if self._use_rich:
            self._console.print(f"[bold cyan]lambdagent[/] [dim]v2.0[/]")
            self._console.print(f"  Agent: [bold]{agent_name}[/] ({agent_type})")
            if config_path:
                self._console.print(f"  Config: [dim]{config_path}[/]")
            self._console.print()
        else:
            print(f"lambdagent v2.0")
            print(f"  Agent: {agent_name} ({agent_type})")
            if config_path:
                print(f"  Config: {config_path}")
            print()

    def token(self, text: str, term_name: str = ""):
        """Stream a single token (incremental output)."""
        sys.stdout.write(text)
        sys.stdout.flush()

    def newline(self):
        print()

    def tool_start(self, tool_name: str, tool_input: str):
        """Display tool call start."""
        self._step_count += 1
        input_preview = tool_input[:80].replace("\n", " ")
        if self._use_rich:
            self._console.print(
                f"  [dim]β[{self._step_count}][/] [yellow]{tool_name}[/] "
                f"[dim]{input_preview}...[/]",
                end=""
            )
        else:
            print(f"  β[{self._step_count}] {tool_name}: {input_preview}...", end="")
        sys.stdout.flush()

    def tool_end(self, tool_name: str, result: str, duration_ms: float):
        """Display tool call result."""
        result_preview = result[:60].replace("\n", " ")
        if self._use_rich:
            self._console.print(f" [green]({duration_ms:.0f}ms)[/] {result_preview}")
        else:
            print(f" ({duration_ms:.0f}ms) {result_preview}")

    def tool_error(self, tool_name: str, error: str):
        """Display tool call error."""
        if self._use_rich:
            self._console.print(f" [red]ERROR: {error[:80]}[/]")
        else:
            print(f" ERROR: {error[:80]}")

    def thinking(self, step: int):
        """Display thinking indicator."""
        if self._use_rich:
            self._console.print(f"  [dim]β[{step}] thinking...[/]", end="\r")
        else:
            print(f"  β[{step}] thinking...", end="\r")
        sys.stdout.flush()

    def step_complete(self, step: int, term_name: str, duration_ms: float, tokens: int = 0):
        """Display step completion."""
        self._total_tokens += tokens
        if self.verbose:
            if self._use_rich:
                self._console.print(
                    f"  [dim]β[{step}][/] [blue]{term_name}[/] "
                    f"[dim]{duration_ms:.0f}ms, {tokens} tokens[/]"
                )
            else:
                print(f"  β[{step}] {term_name} ({duration_ms:.0f}ms, {tokens} tokens)")

    def result(self, output: str, trace_count: int):
        """Display final result."""
        elapsed = time.time() - self._start_time
        if self._use_rich:
            self._console.print()
            self._console.print(output)
            self._console.print()
            self._console.print(
                f"[dim]({trace_count} β-reductions, {elapsed:.1f}s, "
                f"~{self._total_tokens} tokens)[/]"
            )
        else:
            print()
            print(output)
            print()
            print(f"({trace_count} β-reductions, {elapsed:.1f}s, ~{self._total_tokens} tokens)")

    def error(self, message: str):
        """Display error."""
        if self._use_rich:
            self._console.print(f"[bold red]ERROR:[/] {message}")
        else:
            print(f"ERROR: {message}", file=sys.stderr)

    def trace(self, entries: list):
        """Display full β-reduction trace."""
        if self._use_rich:
            self._console.print(f"\n[bold]β-reduction trace:[/]")
            colors = {"Lam": "blue", "Tool": "yellow", "Compose": "cyan",
                      "Loop": "magenta", "Route": "green"}
            for i, e in enumerate(entries):
                term = getattr(e, "term_name", str(e.get("term", "?")))
                ms = getattr(e, "duration_ms", e.get("duration_ms", 0))
                ttype = getattr(e, "term_type", "") if hasattr(e, "term_type") else ""
                color = colors.get(ttype, "white")
                inp = str(getattr(e, "input", e.get("input", "")))[:50]
                out = str(getattr(e, "output", e.get("output", "")))[:50]
                self._console.print(
                    f"  β[{i}] [{color}]{term}[/] [dim]({ms:.0f}ms)[/]: "
                    f"{inp} → {out}"
                )
        else:
            print("\nβ-reduction trace:")
            for i, e in enumerate(entries):
                term = getattr(e, "term_name", str(e.get("term", "?")))
                ms = getattr(e, "duration_ms", e.get("duration_ms", 0))
                inp = str(getattr(e, "input", e.get("input", "")))[:50]
                out = str(getattr(e, "output", e.get("output", "")))[:50]
                print(f"  β[{i}] {term} ({ms:.0f}ms): {inp} → {out}")
