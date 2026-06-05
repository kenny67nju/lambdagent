#!/usr/bin/env python3
"""
lambdagent.mcp_server — MCP Server exposing lambdagent static analysis tools.

Phase 7a (I01): Exposes 5 MCP tools via JSON-RPC 2.0 over stdio:
  1. lint_agent_config     — 26-rule structural lint (Paper I)
  2. estimate_agent_cost   — Graded type cost prediction (Paper III)
  3. check_agent_types     — T-Compose type checking (Paper III)
  4. check_parallel_safety — Store independence verification (Paper II Prop. 30)
  5. monitor_agent_cost    — Runtime cost anomaly detection (I03)

Usage:
  # Start as MCP server (stdio transport)
  python -m lambdagent.mcp_server

  # Configure in Claude Code / Cursor:
  # .claude/settings.json:
  #   { "mcpServers": { "lambdagent": { "command": "python3", "args": ["-m", "lambdagent.mcp_server"] } } }

Protocol: MCP (Model Context Protocol) — JSON-RPC 2.0 over stdio
  https://modelcontextprotocol.io/specification/2025-11-25
"""

from __future__ import annotations

import json
import sys
import os
import traceback
from typing import Any, Dict, List, Optional


# ════════════════════════════════════════════════════════════
# JSON-RPC 2.0 over stdio
# ════════════════════════════════════════════════════════════


def _read_message() -> Optional[Dict]:
    """Read a JSON-RPC message from stdin (Content-Length framing)."""
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        line = line.decode("utf-8").rstrip("\r\n")
        if line == "":
            break
        if ":" in line:
            key, val = line.split(":", 1)
            headers[key.strip().lower()] = val.strip()

    length = int(headers.get("content-length", 0))
    if length == 0:
        return None
    body = sys.stdin.buffer.read(length)
    return json.loads(body.decode("utf-8"))


def _write_message(msg: Dict):
    """Write a JSON-RPC message to stdout (Content-Length framing)."""
    body = json.dumps(msg, ensure_ascii=False).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n"
    sys.stdout.buffer.write(header.encode("utf-8"))
    sys.stdout.buffer.write(body)
    sys.stdout.buffer.flush()


def _response(id: Any, result: Any) -> Dict:
    return {"jsonrpc": "2.0", "id": id, "result": result}


def _error(id: Any, code: int, message: str, data: Any = None) -> Dict:
    err = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": id, "error": err}


# ════════════════════════════════════════════════════════════
# MCP Tool Definitions
# ════════════════════════════════════════════════════════════

TOOLS = [
    {
        "name": "lint_agent_config",
        "description": (
            "Lint an agent YAML/JSON config for structural defects. "
            "Works with LangChain, CrewAI, AutoGen, Dify, and generic configs. "
            "Detects: missing terminate conditions, type mismatches, dead routes, "
            "empty loops, and 20+ other defect patterns grounded in Lambda calculus "
            "formal semantics."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "config_path": {
                    "type": "string",
                    "description": "Path to YAML/JSON agent config file",
                },
                "config_content": {
                    "type": "string",
                    "description": "YAML config content as string (alternative to config_path)",
                },
                "level": {
                    "type": "string",
                    "enum": ["error", "warn", "info"],
                    "description": "Minimum severity level to report (default: warn)",
                },
            },
        },
    },
    {
        "name": "estimate_agent_cost",
        "description": (
            "Estimate worst-case cost of an agent pipeline BEFORE execution. "
            "Returns token upper bound, latency estimate, dollar cost, and "
            "end-to-end success probability based on Paper III graded types."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "config_path": {
                    "type": "string",
                    "description": "Path to YAML/JSON agent config file",
                },
                "config_content": {
                    "type": "string",
                    "description": "YAML config content as string (alternative to config_path)",
                },
            },
        },
    },
    {
        "name": "check_agent_types",
        "description": (
            "Type-check an agent pipeline. Verifies that each stage's output "
            "type is compatible with the next stage's input type "
            "(Paper III T-Compose rule: output(f) <: input(g))."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "config_path": {
                    "type": "string",
                    "description": "Path to YAML/JSON agent config file",
                },
                "config_content": {
                    "type": "string",
                    "description": "YAML config content as string",
                },
            },
        },
    },
    {
        "name": "check_parallel_safety",
        "description": (
            "Check if parallel agents have store-independence (no shared "
            "mutable state). Prevents data corruption from race conditions. "
            "Based on Paper II Proposition 30 (Pair Confluence Theorem)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "config_path": {
                    "type": "string",
                    "description": "Path to YAML/JSON agent config file",
                },
                "config_content": {
                    "type": "string",
                    "description": "YAML config content as string",
                },
            },
        },
    },
    {
        "name": "monitor_agent_cost",
        "description": (
            "Compare actual execution cost against predicted upper bound. "
            "Flags anomalies when actual cost exceeds predicted by >2x "
            "(detects issues like Claude Code #34629 prompt-cache regression)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "config_path": {
                    "type": "string",
                    "description": "Path to YAML/JSON agent config file",
                },
                "actual_tokens": {
                    "type": "integer",
                    "description": "Actual tokens consumed in execution",
                },
                "actual_cost_usd": {
                    "type": "number",
                    "description": "Actual cost in USD",
                },
                "threshold": {
                    "type": "number",
                    "description": "Anomaly threshold multiplier (default: 2.0)",
                },
            },
            "required": ["config_path", "actual_tokens"],
        },
    },
]


# ════════════════════════════════════════════════════════════
# Tool Implementations
# ════════════════════════════════════════════════════════════


def _load_config(args: Dict) -> Dict:
    """Load config from path or content string."""
    import yaml

    if "config_content" in args and args["config_content"]:
        return yaml.safe_load(args["config_content"])
    elif "config_path" in args and args["config_path"]:
        path = args["config_path"]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            return yaml.safe_load(f)
    raise ValueError("Either config_path or config_content is required")


def _compile_term(config_or_path):
    """Compile config to lambda term."""
    from lambdagent.fromconfig import from_config
    import tempfile, yaml

    if isinstance(config_or_path, dict):
        # Write to temp file for from_config
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump(config_or_path, f, allow_unicode=True)
            tmp_path = f.name
        try:
            return from_config(tmp_path)
        finally:
            os.unlink(tmp_path)
    else:
        return from_config(config_or_path)


def handle_lint_agent_config(args: Dict) -> Dict:
    """I01: Lint agent config."""
    from lambdagent.fromconfig.lint import lint_config, detect_framework

    cfg = _load_config(args)
    framework = detect_framework(cfg)
    results = lint_config(cfg)

    min_level = args.get("level", "warn").upper()
    level_order = {"ERROR": 0, "WARN": 1, "INFO": 2}
    min_order = level_order.get(min_level, 1)

    filtered = [r for r in results if level_order.get(r.level, 2) <= min_order]

    return {
        "framework": framework,
        "total": len(results),
        "filtered": len(filtered),
        "errors": [
            {
                "rule": r.rule,
                "level": r.level,
                "message": r.message,
                "lambda_meaning": getattr(r, "lambda_meaning", ""),
                "field": getattr(r, "field", ""),
            }
            for r in filtered
        ],
        "summary": (
            f"{sum(1 for r in results if r.level == 'ERROR')} errors, "
            f"{sum(1 for r in results if r.level == 'WARN')} warnings, "
            f"{sum(1 for r in results if r.level == 'INFO')} info"
        ),
    }


def handle_estimate_agent_cost(args: Dict) -> Dict:
    """I01: Estimate agent cost via graded types."""
    from lambdagent.cost_grade import estimate_cost, format_cost_estimate

    cfg = _load_config(args)
    term = _compile_term(cfg)
    grade = estimate_cost(term)

    result = {
        "tokens_upper_bound": grade.tokens,
        "latency_upper_bound_sec": round(grade.latency, 2),
        "cost_upper_bound_usd": round(grade.money, 4),
        "success_probability": round(grade.probability, 4),
    }

    # Add recommendations if success probability is low
    if grade.probability < 0.05:
        result["warning"] = (
            f"End-to-end success probability is only {grade.probability:.1%}. "
            f"Consider reducing parallel agents, lowering maxSteps, or "
            f"increasing Guard retries."
        )

    try:
        result["formatted"] = format_cost_estimate(grade)
    except Exception:
        pass

    return result


def handle_check_agent_types(args: Dict) -> Dict:
    """I01: Type-check agent pipeline."""
    from lambdagent.lam_types import check_compose_types

    cfg = _load_config(args)
    term = _compile_term(cfg)

    # Extract agent_type for compose checking
    agent_type = getattr(term, "agent_type", None)
    errors = []

    # Walk compose chain and check types
    from lambdagent.primitives import Compose

    if isinstance(term, Compose):
        stages = term.stages
        for i in range(len(stages) - 1):
            f_out = getattr(stages[i], "output_type", None)
            g_in = getattr(stages[i + 1], "input_type", None)
            if f_out and g_in:
                from lambdagent.lam_types import is_subtype

                if not is_subtype(f_out, g_in):
                    errors.append(
                        {
                            "stage": i,
                            "composition": f"{stages[i]._name} >> {stages[i + 1]._name}",
                            "output_type": str(f_out),
                            "input_type": str(g_in),
                            "error": f"{f_out} is not subtype of {g_in}",
                        }
                    )

    return {
        "type_safe": len(errors) == 0,
        "errors": errors,
        "agent_type": str(agent_type) if agent_type else "unknown",
    }


def handle_check_parallel_safety(args: Dict) -> Dict:
    """I01: Check store independence for parallel agents."""
    from lambdagent.primitives import Pair
    from lambdagent.extensions import Par

    cfg = _load_config(args)
    term = _compile_term(cfg)

    conflicts = []

    # Find parallel constructs and check store independence
    def _check(t):
        if isinstance(t, (Pair, Par)):
            agents = getattr(t, "agents", None) or [
                getattr(t, "first", None),
                getattr(t, "second", None),
            ]
            agents = [a for a in agents if a is not None]
            try:
                from lambdagent.store_analysis import check_store_independence

                check_store_independence(agents)
            except Exception as e:
                conflicts.append(
                    {
                        "agents": [getattr(a, "_name", "?") for a in agents],
                        "error": str(e),
                    }
                )

        # Recurse into sub-terms
        for attr in (
            "stages",
            "agents",
            "body",
            "agent",
            "first",
            "second",
            "then_",
            "else_",
        ):
            child = getattr(t, attr, None)
            if child is None:
                continue
            if isinstance(child, list):
                for c in child:
                    if hasattr(c, "apply"):
                        _check(c)
            elif hasattr(child, "apply"):
                _check(child)

    _check(term)

    return {
        "safe": len(conflicts) == 0,
        "conflicts": conflicts,
    }


def handle_monitor_agent_cost(args: Dict) -> Dict:
    """I03: Runtime cost anomaly detection."""
    from lambdagent.cost_grade import estimate_cost, validate_cost

    cfg = _load_config(args)
    term = _compile_term(cfg)
    predicted = estimate_cost(term)

    actual_tokens = args.get("actual_tokens", 0)
    actual_cost = args.get("actual_cost_usd", 0.0)
    threshold = args.get("threshold", 2.0)

    validation = validate_cost(predicted, actual_tokens, actual_cost, threshold)

    return {
        "predicted_tokens": predicted.tokens,
        "predicted_cost_usd": round(predicted.money, 4),
        "actual_tokens": actual_tokens,
        "actual_cost_usd": actual_cost,
        "valid": validation["valid"],
        "deviation_tokens": round(validation["deviation_tokens"], 2),
        "deviation_money": round(validation["deviation_money"], 2),
        "alert": validation["alert"],
    }


# ════════════════════════════════════════════════════════════
# MCP Protocol Handlers
# ════════════════════════════════════════════════════════════

_TOOL_HANDLERS = {
    "lint_agent_config": handle_lint_agent_config,
    "estimate_agent_cost": handle_estimate_agent_cost,
    "check_agent_types": handle_check_agent_types,
    "check_parallel_safety": handle_check_parallel_safety,
    "monitor_agent_cost": handle_monitor_agent_cost,
}


def handle_initialize(params: Dict) -> Dict:
    """MCP initialize handshake."""
    return {
        "protocolVersion": "2024-11-05",
        "capabilities": {
            "tools": {"listChanged": False},
        },
        "serverInfo": {
            "name": "lambdagent-analyzer",
            "version": "0.1.0",
        },
    }


def handle_tools_list(params: Dict) -> Dict:
    """MCP tools/list — return available tools."""
    return {"tools": TOOLS}


def handle_tools_call(params: Dict) -> Dict:
    """MCP tools/call — execute a tool."""
    tool_name = params.get("name", "")
    arguments = params.get("arguments", {})

    handler = _TOOL_HANDLERS.get(tool_name)
    if not handler:
        return {
            "content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}],
            "isError": True,
        }

    try:
        result = handler(arguments)
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(result, indent=2, ensure_ascii=False),
                }
            ],
            "isError": False,
        }
    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"Error: {type(e).__name__}: {e}"}],
            "isError": True,
        }


def handle_request(msg: Dict) -> Optional[Dict]:
    """Route JSON-RPC request to handler."""
    method = msg.get("method", "")
    params = msg.get("params", {})
    msg_id = msg.get("id")

    if method == "initialize":
        return _response(msg_id, handle_initialize(params))
    elif method == "notifications/initialized":
        return None  # No response for notifications
    elif method == "tools/list":
        return _response(msg_id, handle_tools_list(params))
    elif method == "tools/call":
        return _response(msg_id, handle_tools_call(params))
    elif method == "ping":
        return _response(msg_id, {})
    else:
        return _error(msg_id, -32601, f"Method not found: {method}")


# ════════════════════════════════════════════════════════════
# Main Loop
# ════════════════════════════════════════════════════════════


def main():
    """MCP Server main loop — reads JSON-RPC from stdin, writes to stdout."""
    while True:
        try:
            msg = _read_message()
            if msg is None:
                break  # EOF

            response = handle_request(msg)
            if response is not None:
                _write_message(response)
        except KeyboardInterrupt:
            break
        except Exception as e:
            # Write error response for any unhandled exception
            msg_id = msg.get("id") if msg else None
            _write_message(_error(msg_id, -32603, str(e)))


if __name__ == "__main__":
    main()
