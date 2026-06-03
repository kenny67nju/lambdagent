"""
nl2agent_demo.py — One-sentence agent builder demo

Shows how to use nl2agent.py to build an agent from a natural language description.

Prerequisites:
  export DASHSCOPE_API_KEY=your-key-here

Usage:
  python examples/nl2agent_demo.py
"""

import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nl2agent import one_sentence_to_agent, nl_to_yaml


def demo_generate_only():
    """Generate a YAML config from a natural language description (no execution)."""
    description = (
        "Build a code review pipeline: "
        "first check for security vulnerabilities, "
        "then check code style, "
        "then generate a summary report"
    )
    print("=== Generate YAML from description ===")
    print(f"Description: {description}")
    print()

    yaml_str = nl_to_yaml(description)
    print("Generated YAML:")
    print(yaml_str)


def demo_full_pipeline():
    """Generate config AND execute the agent."""
    description = "Build a research assistant that can search the web, up to 10 steps, with memory"
    task = "What are the latest developments in quantum computing?"

    print("=== Full pipeline: NL -> YAML -> Lambda -> Execute ===")
    result = one_sentence_to_agent(description, task)
    print(f"\nFinal result: {result}")


if __name__ == "__main__":
    if "--full" in sys.argv:
        demo_full_pipeline()
    else:
        demo_generate_only()
