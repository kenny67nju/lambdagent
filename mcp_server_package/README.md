# lambdagent-mcp-server

MCP Server for **lambdagent** static analysis. Lint agent configs, predict costs, type-check pipelines, and verify parallel safety — all from your AI IDE.

## Quick Setup

### Claude Code
```json
// .claude/settings.json
{
  "mcpServers": {
    "lambdagent": {
      "command": "python3",
      "args": ["-m", "lambdagent.mcp_server"]
    }
  }
}
```

### Cursor
```json
// .cursor/mcp.json
{
  "mcpServers": {
    "lambdagent": {
      "command": "python3",
      "args": ["-m", "lambdagent.mcp_server"]
    }
  }
}
```

## Tools

| Tool | Description |
|------|-------------|
| `lint_agent_config` | 26-rule structural lint for LangChain/CrewAI/AutoGen/Dify configs |
| `estimate_agent_cost` | Worst-case cost prediction (tokens, latency, USD, success probability) |
| `check_agent_types` | T-Compose type checking (output(f) <: input(g)) |
| `check_parallel_safety` | Store independence verification (Paper II Proposition 30) |
| `monitor_agent_cost` | Runtime cost anomaly detection (actual vs predicted) |

## Example

```
User: Check my agent config for issues
Claude: [calls lint_agent_config with config_path="agent-config.yml"]

Result: 2 errors found
  L004a: No terminate tool in ReAct loop (maxSteps=200)
  T-COMPOSE: Stage 2 output Json(object) incompatible with Stage 3 input Str
```

## Based On

- **Paper I**: lambdagent — A Formally-Grounded DSL for LLM Agent Composition
- **Paper II**: Operational Semantics for LLM Agent Programs
- **Paper III**: A Type and Effect System for LLM Agent Composition
