"""
quick_start.py — Basic lambdagent usage

Demonstrates the core Lambda calculus constructs:
  Lam (abstraction), Compose (>>), Loop (Y combinator), Tool (oracle).
"""

from lambdagent import Lam, Compose, Loop, Tool, Memory, Guard, Context

# ── 1. Lambda Abstraction: a single agent ──
writer = Lam("writer", "You are a concise technical writer.")

# ── 2. Function Composition: pipeline ──
reviewer = Lam("reviewer", "Review the text for clarity. Output improved version.")
pipeline = writer >> reviewer  # lambda x. reviewer(writer(x))

# ── 3. Y Combinator: iterative agent (ReAct pattern) ──
researcher = Loop(
    body=Lam("think", "Analyze the question. If you have enough info, say DONE."),
    condition=lambda result: "DONE" in str(result),
    max_iterations=5,
)

# ── 4. Memory: persistent context ──
agent_with_memory = Memory(researcher, strategy="local", size=20)

# ── 5. Guard: output validation (dependent type) ──
validated = Guard(
    writer,
    validator=lambda output: len(str(output)) > 50,
    retry=2,
)

# ── 6. Execute = beta-reduction ──
if __name__ == "__main__":
    ctx = Context()
    print("Pipeline structure:", pipeline)
    print()

    # To actually run, you need an LLM backend configured.
    # result = pipeline("Explain quicksort in 3 sentences", ctx)
    # print(result)

    print("All constructs imported successfully.")
    print("To execute agents, configure your LLM provider via environment variables:")
    print("  export DASHSCOPE_API_KEY=your-key-here")
    print("  # or")
    print("  export ANTHROPIC_API_KEY=your-key-here")
