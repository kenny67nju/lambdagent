"""fromconfig.lambda_expr — Export Lambda calculus notation"""

from __future__ import annotations
from typing import Any, Dict
import yaml


def to_lambda_expr(path_or_cfg) -> str:
    """Export agent config as pure Lambda calculus expression."""
    if isinstance(path_or_cfg, str):
        with open(path_or_cfg, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = path_or_cfg

    return _cfg_to_lambda(cfg)


def _cfg_to_lambda(cfg: Dict[str, Any]) -> str:
    name = cfg.get("name", cfg.get("agentId", "agent"))
    agent_type = cfg.get("type", "simple")
    model_cfg = cfg.get("model", {})
    model_str = model_cfg.get("name", "LLM")
    prompt_short = (cfg.get("systemPrompt", "") or "")[:40]
    if prompt_short:
        prompt_short = prompt_short.replace('"', "'") + "..."

    if agent_type == "simple":
        return f'lambda x. {name}_{{{model_str}}}(x)  -- prompt="{prompt_short}"'

    elif agent_type == "react":
        react_cfg = cfg.get("react", {})
        max_s = react_cfg.get("maxSteps", 10)
        mcp_cfg = cfg.get("mcp", {})
        tools = []
        for tl in mcp_cfg.get("onlineTool", {}).values():
            tools.extend(tl)
        tools.extend(mcp_cfg.get("localTools", []))
        tool_cases = ", ".join(
            f"({t}, lambda x.x)" if t == "terminate" else f"({t}, Tool_MCP)"
            for t in tools
        )
        expr = (
            f"Y_{max_s}(lambda self. lambda state.\n"
            f"  let t = think(state) in\n"
            f"  CASE t [{tool_cases}] >>\n"
            f"  IF is_terminate THEN t ELSE self(state + obs)\n"
            f")"
        )
        memory_cfg = cfg.get("memory", {})
        if memory_cfg.get("enabled"):
            strategy = memory_cfg.get("strategy", "local")
            expr = f"Memory(\n  {expr},\n  {strategy}(size={memory_cfg.get('size')}, ttl={memory_cfg.get('ttl')})\n)"
        return f"{name} = {expr}"

    elif agent_type == "chain":
        chain_cfg = cfg.get("chain", {})
        steps = chain_cfg.get("steps", [])
        step_names = [s.get("name", f"step{i}") for i, s in enumerate(steps)]
        composition = " >> ".join(step_names)
        defs = "\n".join(
            f'  {s.get("name", f"step{i}")} = lambda x. LLM("{(s.get("prompt", ""))[:30]}...")(x)'
            for i, s in enumerate(steps)
        )
        return f"{name} = lambda x. ({composition})(x)\nwhere:\n{defs}"

    elif agent_type == "router":
        router_cfg = cfg.get("router", {})
        categories = list(router_cfg.get("routes", {}).keys())
        cases = ", ".join(f"({c}, {c}_agent)" for c in categories)
        has_default = "default" in router_cfg
        if has_default:
            cases += ", (_, default_agent)"
        return f"{name} = lambda x. CASE (classifier x) [{cases}]"

    elif agent_type == "parallel":
        par_cfg = cfg.get("parallel", {})
        agents = par_cfg.get("agents", [])
        agent_names = [a.get("name", f"agent{i}") for i, a in enumerate(agents)]
        pair_str = " , ".join(f"{n}(x)" for n in agent_names)
        merge = par_cfg.get("merge", "tuple")
        if merge == "custom":
            return f"{name} = lambda x. merge(PAIR({pair_str}))"
        return f"{name} = lambda x. PAIR({pair_str})"

    return f"{name} = lambda x. LLM(x)"


def describe_config(path_or_cfg) -> str:
    """Pretty-print the Lambda structure of an agent config."""
    if isinstance(path_or_cfg, str):
        with open(path_or_cfg, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = path_or_cfg

    name = cfg.get("name", cfg.get("agentId", "agent"))
    agent_type = cfg.get("type", "simple")
    model_cfg = cfg.get("model", {})
    react_cfg = cfg.get("react", {})
    memory_cfg = cfg.get("memory", {})
    mcp_cfg = cfg.get("mcp", {})
    prompt = cfg.get("systemPrompt", "")

    lines = [f"# Lambda structure: {name}", "#"]

    tool_names = []
    for server, tl in mcp_cfg.get("onlineTool", {}).items():
        tool_names.extend(tl)
    tool_names.extend(mcp_cfg.get("localTools", []))

    if agent_type == "react":
        max_s = react_cfg.get("maxSteps", 10)
        lines.append(f"# {name} =")
        if memory_cfg.get("enabled"):
            lines.append(f"#   Memory(")
        lines.append(f"#     Y_{max_s}(lambda self. lambda state.")
        lines.append(f"#       let t = think(state) in")
        if tool_names:
            cases = ", ".join(
                f"({t}, lambda x.x)" if t == "terminate" else f"({t}, Tool_MCP)"
                for t in tool_names
            )
            lines.append(f"#       CASE t [{cases}]")
        lines.append(f"#       >> IF is_terminate THEN t ELSE self(state + obs)")
        lines.append(f"#     )")
        if memory_cfg.get("enabled"):
            s = memory_cfg.get("strategy", "local")
            lines.append(
                f"#     , {s}(size={memory_cfg.get('size')}, ttl={memory_cfg.get('ttl')})"
            )
            lines.append(f"#   )")
        lines.append(f"#")
        lines.append(
            f"# Constructs: Lam, Loop, Route, Tool(x{len(tool_names)})"
            + (", Memory" if memory_cfg.get("enabled") else "")
        )
        lines.append(f"# beta-reduction bound: max {max_s} steps")
        if "terminate" in tool_names:
            lines.append(f"# Base case: terminate = lambda x.x")
    elif agent_type == "chain":
        steps = cfg.get("chain", {}).get("steps", [])
        step_names = [s.get("name", f"step{i}") for i, s in enumerate(steps)]
        lines.append(f"# {name} = {' >> '.join(step_names)}")
        lines.append(
            f"# = lambda x. {step_names[-1]}(...{step_names[0]}(x)...)"
        ) if step_names else None
        lines.append(f"# Constructs: Compose({len(steps)} stages)")
    elif agent_type == "router":
        routes = list(cfg.get("router", {}).get("routes", {}).keys())
        lines.append(f"# {name} = CASE (classifier x) {routes}")
        lines.append(f"# Constructs: Route({len(routes)} branches)")
    elif agent_type == "parallel":
        agents = cfg.get("parallel", {}).get("agents", [])
        names = [a.get("name", f"a{i}") for i, a in enumerate(agents)]
        lines.append(f"# {name} = Par({', '.join(names)})")
        lines.append(f"# Constructs: Par({len(agents)} agents)")
    else:
        lines.append(
            f'# {name} = Lam("{prompt[:50]}...", model={model_cfg.get("name", "default")})'
        )
        lines.append(f"# = lambda x. LLM(x)")

    return "\n".join(lines)
