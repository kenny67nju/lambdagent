"""fromconfig.schema — YAML schema validation"""
from __future__ import annotations
from typing import Any, Dict, List, Tuple


def validate_schema(cfg: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    """
    Validate YAML config against schema.
    Returns list of (level, rule_id, message) tuples.
    """
    errors = []

    # Required: type
    if "type" not in cfg:
        errors.append(("ERROR", "S001", "Missing required field 'type'"))
    else:
        valid_types = ("simple", "react", "chain", "router", "parallel")
        if cfg["type"] not in valid_types:
            errors.append(("ERROR", "S002", f"Invalid type '{cfg['type']}'. Must be one of {valid_types}"))

    # Required: systemPrompt (except router/parallel which may not need top-level)
    agent_type = cfg.get("type", "simple")
    if agent_type in ("simple", "react") and not cfg.get("systemPrompt"):
        errors.append(("ERROR", "S003", "Missing required field 'systemPrompt'"))

    # Type-specific validation
    if agent_type == "react":
        react_cfg = cfg.get("react", {})
        ms = react_cfg.get("maxSteps", 10)
        if isinstance(ms, int) and ms <= 0:
            errors.append(("ERROR", "S004", "react.maxSteps must be > 0"))
        if isinstance(ms, int) and ms > 1000:
            errors.append(("ERROR", "S010", "react.maxSteps exceeds maximum of 1000"))

    elif agent_type == "chain":
        chain_cfg = cfg.get("chain", {})
        steps = chain_cfg.get("steps", [])
        if not steps:
            errors.append(("ERROR", "S005", "chain.steps must not be empty"))

    elif agent_type == "router":
        router_cfg = cfg.get("router", {})
        if not router_cfg.get("routes"):
            errors.append(("ERROR", "S006", "router.routes must not be empty"))
        if not router_cfg.get("classifier"):
            errors.append(("ERROR", "S007", "router.classifier is required"))

    elif agent_type == "parallel":
        par_cfg = cfg.get("parallel", {})
        agents = par_cfg.get("agents", [])
        if len(agents) < 2:
            errors.append(("ERROR", "S008", "parallel.agents must have at least 2 agents"))

    # Model validation
    model_cfg = cfg.get("model", {})
    temp = model_cfg.get("temperature", 0.0)
    if isinstance(temp, (int, float)) and (temp < 0 or temp > 2.0):
        errors.append(("WARN", "S009", f"temperature={temp} outside [0, 2.0]"))

    # Runtime engine validation (Phase 6.5)
    runtime_cfg = cfg.get("runtime", {})
    if runtime_cfg:
        engine = runtime_cfg.get("engine", "recursive")
        valid_engines = ("recursive", "cek", "adaptive")
        if engine not in valid_engines:
            errors.append(("ERROR", "S011",
                           f"Invalid runtime.engine '{engine}'. "
                           f"Must be one of {valid_engines}"))

        cost_budget = runtime_cfg.get("costBudget")
        if cost_budget is not None:
            if not isinstance(cost_budget, (int, float)) or cost_budget <= 0:
                errors.append(("ERROR", "S012",
                               "runtime.costBudget must be a positive number"))

        max_steps = runtime_cfg.get("maxSteps")
        if max_steps is not None:
            if not isinstance(max_steps, int) or max_steps <= 0:
                errors.append(("ERROR", "S013",
                               "runtime.maxSteps must be a positive integer"))
            if isinstance(max_steps, int) and max_steps > 100000:
                errors.append(("WARN", "S014",
                               f"runtime.maxSteps={max_steps} is unusually high"))

    return errors
