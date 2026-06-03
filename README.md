# lambdagent

**Lambda Calculus Agent DSL** — Every agent is a function. Every composition is function composition. Every loop is a Y combinator.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: BUSL-1.1](https://img.shields.io/badge/License-BUSL--1.1-orange.svg)](./LICENSE)

## Overview

`lambdagent` is a Python DSL that models AI agents as Lambda calculus terms. Instead of ad-hoc agent frameworks, it provides **11 core + 5 multi-agent + 4 skill + sandbox + protocol constructs** with rigorous mathematical foundations — each one maps directly to a concept in Lambda calculus or pi-calculus.

**Core insight**: An LLM-Dataset Pair (M, D) is equivalent to a λ-term. Training defines the function; inference is β-reduction.

**Stats**: ~11,300 lines of Python | 81 exported symbols | 46 source files | 4 patents filed

```
YAML Config ──→ from_config() ──→ Lambda Term ──→ Runtime ──→ Result
                 (compiler)        (Term tree)    (β-reduction)
```

## Installation

```bash
pip install lambdagent

# Or from source:
git clone https://github.com/your-org/lambdagent.git
cd lambdagent && pip install -e .
```

**Dependencies**: `pyyaml` (required), `anthropic` / `openai` (optional, for LLM providers)

## Quick Start

### Python DSL

```python
from lambdagent import Lam, Compose, Tool, Loop, Route, Guard, Memory

# Simple agent (λ abstraction)
agent = Lam("summarizer", "Summarize concisely.", model="claude-sonnet-4-20250514")
result = agent("A long article about quantum computing...")

# Chain / Pipeline (function composition: f >> g >> h)
pipeline = (
    Lam("extract", "Extract key facts from the text.") >>
    Lam("analyze", "Analyze these facts for patterns.") >>
    Lam("report", "Write a structured report.")
)
result = pipeline("Raw research data...")

# Router (generalized Church boolean / CASE)
router = Route(
    classifier=Lam("cls", "Classify as: code, math, or general. Output one word."),
    routes={
        "code": Lam("coder", "You are a coding expert."),
        "math": Lam("math", "You are a math expert."),
    },
    default=Lam("general", "You are a helpful assistant."),
)

# Parallel execution (Church pair)
par = Lam("optimist", "Analyze optimistically.") | Lam("pessimist", "Analyze pessimistically.")
# par("topic") → ("optimistic view...", "pessimistic view...")

# Guard (dependent type: {x:T | P(x)})
safe = Guard(
    Lam("writer", "Write a 200-word essay."),
    validator=lambda x: len(x.split()) >= 150,
    retry=2,
)

# Memory (environment extension: Γ' = Γ ∪ store)
stateful = Memory(
    Lam("assistant", "You are a helpful assistant."),
    store={"user_preference": "concise answers"},
)
```

### YAML Configuration

```yaml
agentId: research-agent
name: ResearchAgent
type: react
systemPrompt: |
  You are a research assistant. Use search tools to find information,
  analyze results, and produce structured reports.
model:
  provider: anthropic
  name: claude-sonnet-4-20250514
  temperature: 0.3
react:
  maxSteps: 15
  observationEnabled: true
mcp:
  onlineTool:
    my-server: [search, calculator]
  localTools: [terminate]
memory:
  enabled: true
  strategy: local
  size: 20
```

```python
from lambdagent.fromconfig import from_config
agent = from_config("config.yml")
result = agent("Research the latest trends in AI agents")
```

### Multi-Agent Group Chat

```python
from lambdagent import Lam, GroupChat

researcher = Lam("researcher", "You research topics thoroughly.")
critic = Lam("critic", "You challenge weak arguments.")

chat = GroupChat([researcher, critic], max_rounds=6, scheduler="round_robin")
result = chat("Should we invest in quantum computing?")
```

### Skill System

```python
from lambdagent import skill, SkillRegistry, SkillAgent, Lam

@skill("summarize", "Summarize text concisely", tags=["writing"])
def summarize(x):
    return f"Summary: {x[:100]}..."

registry = SkillRegistry()
classifier = Lam("cls", "Select the best skill for the task.")
agent = SkillAgent(classifier, registry)
result = agent("Please summarize this article...")
```

### MCP Tools

```python
from lambdagent import mcp_tools

tools = mcp_tools("http://localhost:3000/mcp")
search_tool = tools[0]
result = search_tool({"query": "AI agents"})
```

### RAG

```python
from lambdagent import create_rag

rag = create_rag(["Python is a programming language.", "Lambda calculus is..."])
result = rag("What is Lambda calculus?")
```

### Checkpoint

```python
from lambdagent import Context, save_context, load_context

ctx = Context()
agent("task", ctx)
save_context(ctx, "checkpoint.json")
# Later: ctx = load_context("checkpoint.json")
```

### One-Sentence Agent Builder

```python
# Describe what you need in natural language → auto-generate YAML → compile → run
python nl2agent.py "Build a research assistant that can search and analyze" \
    -t "Research the latest Agent DSL frameworks"
```

## The Constructs

### 11 Core Constructs (Lambda Calculus)

| # | Lambda Calculus | lambdagent DSL | Description |
|---|----------------|----------------|-------------|
| 1 | λx.body | `Lam(name, prompt)` | Lambda abstraction — create an agent |
| 2 | (f x) | `agent(input)` | Function application — β-reduction |
| 3 | λx.g(f(x)) | `f >> g` | Function composition — pipeline |
| 4 | IF c t e | `If(cond, then_, else_)` | Church conditional |
| 5 | Y combinator | `Loop(body, cond, N)` | Bounded recursion (ReAct loop) |
| 6 | PAIR | `Pair(f, g)` | Church pair |
| 7 | FST / SND | `Fst()` / `Snd()` | Projections |
| 8 | Oracle | `Tool(name, fn)` | External function (MCP, CLI) |
| 9 | CASE | `Route(cls, routes)` | Generalized Church boolean |
| 10 | {x:T \| P(x)} | `Guard(agent, P, retry)` | Dependent type (output validation) |
| 11 | Γ' = Γ ∪ s | `Memory(agent, store)` | Environment extension |

### 5 Multi-Agent Constructs (pi-calculus)

| # | Process Calculus | lambdagent DSL | Description |
|---|-----------------|----------------|-------------|
| 12 | c!(v) / c?(x) | `Channel` + `Send` + `Receive` | Inter-agent communication |
| 13 | Γ_shared | `SharedMemory` | Thread-safe shared state |
| 14 | Y_n(Loop+Route) | `GroupChat` | Multi-agent group discussion |
| 15 | Dynamic CASE | `Handoff` | Runtime dynamic delegation |
| 16 | Concurrent β | `AsyncPar` | Thread-pool true parallelism |

### Skill System

| Construct | Description |
|-----------|-------------|
| `Skill(name, term, ...)` | Named, reusable Lambda term with metadata + type signature |
| `SkillPack(name)` | Collection of related skills |
| `SkillRegistry()` | Global singleton registry (search, discover, build_route) |
| `SkillAgent(classifier)` | Auto-discovers and executes best skill from registry |
| `@skill(name, desc, tags)` | Decorator: wrap function/Term as Skill + auto-register |

### Protocol & Storage

| Module | Description |
|--------|-------------|
| `MCPServer` / `MCPTool` | MCP protocol client (HTTP + stdio transport) |
| `A2AServer` / `A2AClient` | Google A2A protocol (publish/discover/call agents) |
| `RAGTool` / `AgenticRAG` | Retrieval-augmented generation (TF-IDF or ChromaDB) |
| `Checkpoint` / `CheckpointManager` | Serialize/restore execution state to JSON |

### Sandbox (Process Isolation)

| Construct | Description |
|-----------|-------------|
| `SandboxedTool(name, fn, policy)` | Tool running in isolated subprocess with resource limits |
| `SandboxPolicy` | Security policy with presets: `.strict()`, `.default()`, `.permissive()` |
| `SecureExecutor` | Auto-wraps all Tools in a term tree with sandbox |
| `ResourceLimiter` | Applies CPU/memory/fd limits via POSIX resource module |
| `@sandboxed(timeout, memory_mb)` | One-line decorator for sandboxed tool creation |

## Architecture

```
lambdagent/                  # ~11,300 lines, 46 .py files, 81 exported symbols
├── __init__.py              # Public API — 81 symbols
├── core.py                  # Term, Context, TraceEntry (base abstractions)
├── primitives.py            # Lam, Compose, If, Loop, Pair, Fst, Snd, Tool
├── extensions.py            # Par, Route, Memory, Guard
├── dataset.py               # Dataset → Lam converter
├── multiagent.py            # Channel, Send, Receive, SharedMemory,
│                            #   GroupChat, Handoff, AsyncPar
├── skills.py                # Skill, SkillSignature, SkillPack,
│                            #   SkillRegistry, SkillAgent, @skill
├── mcp_client.py            # MCPServer, MCPTool, MCPTransport (HTTP+stdio),
│                            #   mcp_tools(), mcp_tool()
├── checkpoint.py            # Checkpoint, CheckpointManager,
│                            #   save_context(), load_context()
├── a2a.py                   # AgentCard, A2AServer, A2AClient, A2ATask,
│                            #   skill_to_agent_card(), registry_to_agent_card()
├── rag.py                   # RAGTool, AgenticRAG, SimpleVectorStore,
│                            #   ChromaStore, Document, SearchResult, create_rag()
├── sandbox.py               # SandboxedTool, SandboxPolicy, SecureExecutor,
│                            #   ResourceLimiter, @sandboxed, SandboxViolation
├── from_config.py           # YAML → Lambda compiler (v1)
├── fromconfig/              # YAML → Lambda Term compiler (v2)
│   ├── compiler.py          #   from_config(), build_agent() — 5 agent types
│   ├── schema.py            #   YAML schema validation
│   ├── lint.py              #   Static analysis (L001-L016)
│   ├── lambda_expr.py       #   Export pure Lambda notation
│   └── errors.py            #   CompileError, SchemaError, SemanticError
├── agentruntime/            # Runtime: Term × Input → Result
│   ├── executor.py          #   β-reduction engine (pattern-match on Term type)
│   ├── react_engine.py      #   ReAct 7-phase loop engine
│   ├── action_parser.py     #   Extract actions from LLM output (JSON/XML/keyword)
│   ├── llm_adapter.py       #   Multi-provider LLM (Anthropic/OpenAI/DashScope)
│   ├── mcp_client.py        #   MCP JSON-RPC 2.0 HTTP client
│   ├── memory_backend.py    #   Local/SQLite/Redis memory backends
│   ├── trace_store.py       #   β-reduction trace recording
│   ├── termination.py       #   Y combinator base case detection
│   ├── config.py            #   RuntimeConfig dataclasses
│   └── runtime.py           #   Top-level Runtime class
└── cli/                     # Command-line interface
    ├── main.py              #   compile, run, repl, lint, lambda, serve
    └── shell_tool.py        #   Shell tool integration
```

## CLI Usage

```bash
# Compile (view Lambda structure, don't execute)
lambdagent compile config.yml

# Run (compile + execute)
lambdagent run config.yml "Your input here"

# Interactive REPL
lambdagent repl config.yml

# Static analysis
lambdagent lint config.yml

# Export Lambda expression
lambdagent lambda config.yml
```

## Agent Types

| Type | Lambda Semantics | Use Case |
|------|-----------------|----------|
| `simple` | `λx. LLM(x)` | Single-turn Q&A |
| `react` | `Y_n(λself.λs. think >> route >> observe)` | Multi-step reasoning with tools |
| `chain` | `λx. h(g(f(x)))` | Sequential pipeline |
| `router` | `CASE (classify x) [(k₁,a₁), ...]` | Intent-based routing |
| `parallel` | `PAIR(f(x), g(x)) >> merge` | Multi-perspective analysis |

## Theoretical Foundation

This project is grounded in the equivalence between LLM-Dataset Pairs and Lambda calculus:

- **Church Numerals**: (M_n, D_n) pairs that apply f to x exactly n times ✓
- **Booleans**: TRUE/FALSE as first/second selector ✓
- **S and K Combinators**: Proven Turing-complete via SKI calculus ✓
- **Arithmetic & Logic**: ADD, MUL, AND, OR, NOT all verified at 94-100% accuracy ✓

See the `experiments/` directory in the [MDPair](https://github.com/your-org/MDPair) repo for verification code.

## Multi-Provider LLM Support

```python
# Anthropic (default)
Lam("agent", "prompt", model="claude-sonnet-4-20250514")

# OpenAI
Lam("agent", "prompt", model="gpt-4o")

# DashScope (Qwen)
Lam("agent", "prompt", model="dashscope/qwen3-max")
```

Provider is auto-detected from model name. API keys are read from environment variables:
- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`
- `DASHSCOPE_API_KEY`

## License

Business Source License 1.1 (BUSL-1.1). Free for non-production use and for production use up to 10 users; converts to Apache 2.0 on 2031-04-05. See [LICENSE](./LICENSE).

## Citation

```bibtex
@software{lambdagent2026,
  title={lambdagent: Lambda Calculus Agent DSL},
  author={Qin Liu},
  year={2026},
  url={https://github.com/your-org/lambdagent}
}
```
