"""
fromconfig.lint — Static analysis (lint) for YAML agent configs v3

Framework-aware: detects alternative termination mechanisms (CrewAI internal,
AutoGen is_termination_msg, LangChain AgentFinish, bounded iteration).

26 rules: L001-L026
  ERROR (6):  真缺陷 — 部署前必须修复
  WARN (11):  潜在问题 — 建议修复
  INFO (9):   信息提示 — 帮助理解配置
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
import yaml


@dataclass
class LintResult:
    level: str          # "ERROR" | "WARN" | "INFO"
    rule: str           # e.g. "L001", "L004a"
    message: str
    lambda_meaning: str
    framework: str = "" # detected framework (crewai, langchain, autogen, ...)


# ════════════════════════════════════════════════════════════
# Framework detection
# ════════════════════════════════════════════════════════════

def detect_framework(cfg: dict) -> str:
    """Detect the source framework of a config."""
    keys = set(str(k).lower() for k in cfg.keys())

    # CrewAI: role + goal + backstory
    if "role" in keys and ("goal" in keys or "backstory" in keys):
        return "crewai"

    # AutoGen: llm_config / is_termination_msg / assistant_agent
    if "llm_config" in keys or "is_termination_msg" in keys or "assistant_agent" in keys:
        return "autogen"

    # LangChain: agent_type / chain_type / agent_executor
    if "agent_type" in keys or "chain_type" in keys or "llm_chain" in keys:
        return "langchain"

    # Dify: model_config + prompt_template
    if "model_config" in keys or "prompt_template" in keys:
        return "dify"

    # lambdagent native: agentId
    if "agentid" in keys or "agentId" in cfg:
        return "lambdagent"

    # Multi-agent: agents list
    if "agents" in keys and isinstance(cfg.get("agents"), list):
        return "multi-agent"

    return "generic"


def _has_alternative_termination(cfg: dict) -> Tuple[bool, str]:
    """
    Check if the config has a termination mechanism OTHER than 'terminate' tool.
    Returns (has_mechanism, description).
    """
    reasons = []

    # Check: max_iter / max_iterations / max_steps (bounded Y combinator)
    for field in ("max_iter", "max_iterations", "max_steps", "maxSteps"):
        val = cfg.get(field) or cfg.get("react", {}).get(field)
        if val and isinstance(val, (int, float)) and val > 0:
            reasons.append(f"bounded iteration: {field}={val}")

    # Check: max_turns / max_rounds (multi-agent termination)
    for field in ("max_turns", "max_rounds", "max_round"):
        val = cfg.get(field)
        if val and isinstance(val, (int, float)) and val > 0:
            reasons.append(f"bounded rounds: {field}={val}")

    # Check: is_termination_msg (AutoGen style)
    if cfg.get("is_termination_msg"):
        reasons.append("is_termination_msg (AutoGen string-match termination)")

    # Check: max_consecutive_auto_reply (AutoGen)
    val = cfg.get("max_consecutive_auto_reply")
    if val and isinstance(val, (int, float)) and val > 0:
        reasons.append(f"max_consecutive_auto_reply={val}")

    # Check: stop_condition
    if cfg.get("stop_condition"):
        reasons.append("explicit stop_condition")

    # Check: allow_delegation (CrewAI delegation chain)
    if cfg.get("allow_delegation") is False:
        reasons.append("allow_delegation=false (agent must produce final result)")

    return (len(reasons) > 0, "; ".join(reasons))


# ════════════════════════════════════════════════════════════
# Main lint function
# ════════════════════════════════════════════════════════════

def lint_config(path_or_cfg) -> List[LintResult]:
    """
    Run all lint rules on a config.
    Accepts either a file path (str) or a parsed dict.
    """
    if isinstance(path_or_cfg, str):
        with open(path_or_cfg, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = path_or_cfg

    results = []
    framework = detect_framework(cfg)
    agent_type = cfg.get("type", "simple")
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
    react_cfg = cfg.get("react", {}) if isinstance(cfg.get("react"), dict) else {}
    memory_cfg = cfg.get("memory", {}) if isinstance(cfg.get("memory"), dict) else {}
    mcp_cfg = cfg.get("mcp", {}) if isinstance(cfg.get("mcp"), dict) else {}
    guard_cfg = cfg.get("guard", {}) if isinstance(cfg.get("guard"), dict) else {}
    router_cfg = cfg.get("router", {}) if isinstance(cfg.get("router"), dict) else {}
    chain_cfg = cfg.get("chain", {}) if isinstance(cfg.get("chain"), dict) else {}
    rag_cfg = cfg.get("rag", {}) if isinstance(cfg.get("rag"), dict) else {}

    # ── L001: systemPrompt empty ──
    prompt = cfg.get("systemPrompt", "") or ""
    if not prompt.strip() and agent_type in ("simple", "react"):
        # CrewAI uses role+goal+backstory instead of systemPrompt
        crewai_prompt = f"{cfg.get('role', '')} {cfg.get('goal', '')} {cfg.get('backstory', '')}".strip()
        if not crewai_prompt:
            results.append(LintResult("ERROR", "L001", "Agent has no systemPrompt",
                                      "lambda x. _|_ -- function body undefined",
                                      framework))

    # ── L002: model not configured ──
    model_name = model_cfg.get("name", model_cfg.get("model", ""))
    llm_field = cfg.get("llm", cfg.get("llm_config", {}))
    if not model_cfg and not llm_field and agent_type in ("simple", "react"):
        results.append(LintResult("ERROR", "L002", "No model configured",
                                  "Cannot perform beta-reduction without LLM",
                                  framework))

    # ── L003: maxSteps = 0 ──
    max_steps = react_cfg.get("maxSteps", cfg.get("max_iter", cfg.get("max_iterations", 10)))
    if agent_type == "react" and max_steps == 0:
        results.append(LintResult("ERROR", "L003", "react.maxSteps = 0, agent won't execute",
                                  "Y_0(g) = _|_", framework))

    # ── L004: react without terminate (FRAMEWORK-AWARE) ──
    if agent_type == "react":
        local_tools = mcp_cfg.get("localTools", [])
        if not isinstance(local_tools, list):
            local_tools = []
        has_terminate = "terminate" in local_tools

        if not has_terminate:
            has_alt, alt_desc = _has_alternative_termination(cfg)

            if framework in ("crewai",):
                # CrewAI has built-in termination in Python code
                results.append(LintResult(
                    "INFO", "L004c",
                    f"CrewAI agent: 'terminate' not in YAML (expected: CrewAI handles termination in code)",
                    "Y combinator base case is external to YAML — framework runtime provides it",
                    framework))

            elif framework in ("autogen",) or cfg.get("is_termination_msg"):
                # AutoGen uses is_termination_msg
                results.append(LintResult(
                    "INFO", "L004d",
                    f"AutoGen agent: uses is_termination_msg for termination",
                    "Base case via LLM output string matching (functionally equivalent to lambda x.x)",
                    framework))

            elif framework in ("langchain",):
                # LangChain uses AgentFinish return type
                results.append(LintResult(
                    "INFO", "L004c",
                    f"LangChain agent: uses AgentFinish return type for termination",
                    "Base case via return type discrimination (functionally equivalent to lambda x.x)",
                    framework))

            elif has_alt:
                # Has some alternative termination mechanism
                results.append(LintResult(
                    "WARN", "L004b",
                    f"No 'terminate' tool, but has alternative termination: {alt_desc}",
                    "Y combinator has bounded fallback but no explicit base case — "
                    "agent will force-stop at bound rather than terminate gracefully",
                    framework))

            else:
                # No terminate, no alternative, no known framework mechanism
                results.append(LintResult(
                    "ERROR", "L004a",
                    "type=react, no 'terminate' tool, no alternative termination mechanism detected",
                    "Y combinator has no base case (lambda x.x) AND no bounded fallback -> "
                    "potential infinite loop",
                    framework))

    # ── L005: router with empty routes ──
    if agent_type == "router" and not router_cfg.get("routes"):
        results.append(LintResult("ERROR", "L005", "router has no routes",
                                  "CASE with no branches", framework))

    # ── L006: chain with empty steps ──
    if agent_type == "chain" and not chain_cfg.get("steps"):
        results.append(LintResult("ERROR", "L006", "chain has no steps",
                                  "Empty composition chain", framework))

    # ── L007: high temperature ──
    temp = model_cfg.get("temperature", 0.0)
    if not isinstance(temp, (int, float)):
        temp = 0.0
    if temp > 1.5:
        results.append(LintResult("WARN", "L007", f"temperature={temp} > 1.5, output may be unstable",
                                  "High entropy in stochastic parameter", framework))

    # ── L008: memory ttl=0 ──
    if memory_cfg.get("enabled") and memory_cfg.get("ttl", 3600) == 0:
        results.append(LintResult("WARN", "L008", "memory.ttl=0, bindings never expire",
                                  "Gamma' bindings never reclaimed", framework))

    # ── L009: memory size=0 ──
    if memory_cfg.get("enabled") and memory_cfg.get("size", 20) == 0:
        results.append(LintResult("WARN", "L009", "memory.size=0, memory has no effect",
                                  "Gamma' = Gamma union empty = Gamma", framework))

    # ── L010: large maxSteps ──
    if agent_type == "react" and isinstance(max_steps, (int, float)) and max_steps > 50:
        results.append(LintResult("WARN", "L010",
                                  f"react.maxSteps={max_steps} > 50, long running + high cost",
                                  f"Y_{{{int(max_steps)}}}", framework))

    # ── L011: react with no tools except terminate ──
    if agent_type == "react":
        online = mcp_cfg.get("onlineTool", {})
        all_online = []
        if isinstance(online, dict):
            for tl in online.values():
                if isinstance(tl, list):
                    all_online.extend(tl)
        local_non_term = [t for t in mcp_cfg.get("localTools", [])
                          if isinstance(t, str) and t != "terminate"]
        # Also check top-level 'tools' field (CrewAI/generic)
        top_tools = cfg.get("tools", [])
        if not isinstance(top_tools, list):
            top_tools = []
        if not all_online and not local_non_term and not top_tools:
            if framework not in ("crewai", "langchain", "autogen"):
                results.append(LintResult("WARN", "L011",
                                          "ReAct agent has no tools besides terminate",
                                          "Pure reasoning loop, agent can only think", framework))

    # ── L012: guard retry > 5 ──
    retry = guard_cfg.get("retry", 0)
    if isinstance(retry, (int, float)) and retry > 5:
        results.append(LintResult("WARN", "L012", f"guard.retry={retry} > 5, excessive retries",
                                  "Too many retries on dependent type check", framework))

    # ── L013: router without default ──
    if agent_type == "router" and router_cfg and "default" not in router_cfg:
        results.append(LintResult("WARN", "L013", "router has no default route",
                                  "CASE not exhaustive, unclassified input will error", framework))

    # ── L014: temperature = 0 ──
    if isinstance(temp, (int, float)) and temp == 0:
        results.append(LintResult("INFO", "L014", "temperature=0, deterministic mode",
                                  "Deterministic Lambda", framework))

    # ── L015: memory enabled ──
    if memory_cfg.get("enabled"):
        strategy = memory_cfg.get("strategy", "local")
        results.append(LintResult("INFO", "L015",
                                  f"Memory enabled: strategy={strategy}, "
                                  f"size={memory_cfg.get('size')}, ttl={memory_cfg.get('ttl')}",
                                  f"Gamma' = Gamma union store({strategy})", framework))

    # ── L016: rag enabled ──
    if rag_cfg.get("enabled"):
        results.append(LintResult("INFO", "L016", "RAG enabled, agent can access knowledge base",
                                  "External knowledge Oracle", framework))

    # ════════════════════════════════════════════════════════
    # NEW RULES L017-L026
    # ════════════════════════════════════════════════════════

    # ── L017: max_iter uses framework default (not explicitly set) ──
    if agent_type == "react":
        explicitly_set = any(
            cfg.get(f) is not None or react_cfg.get(f) is not None
            for f in ("max_iter", "max_iterations", "max_steps", "maxSteps")
        )
        if not explicitly_set:
            results.append(LintResult(
                "WARN", "L017",
                "No explicit max_iter/maxSteps — relying on framework default (usually 10-25)",
                "Y_n where n is implicitly determined by framework, not by configuration",
                framework))

    # ── L018: max_iter > 100 (practically unbounded) ──
    all_max = [cfg.get(f) for f in ("max_iter", "max_iterations", "max_steps")]
    all_max.append(react_cfg.get("maxSteps"))
    for val in all_max:
        if isinstance(val, (int, float)) and val > 100:
            results.append(LintResult(
                "WARN", "L018",
                f"max iterations = {int(val)} > 100, practically unbounded",
                f"Y_{{{int(val)}}} ≈ Y (unbounded) — high cost risk", framework))
            break

    # ── L019: is_termination_msg detected ──
    if cfg.get("is_termination_msg"):
        results.append(LintResult(
            "INFO", "L019",
            f"is_termination_msg detected (AutoGen-style string-match termination)",
            "Base case: IF 'TERMINATE' in output THEN state (equivalent to lambda x.x)",
            framework))

    # ── L020: allow_delegation=true but no delegation targets visible ──
    if cfg.get("allow_delegation") is True:
        # Check if there are other agents defined in the same config
        has_peers = bool(cfg.get("agents")) or bool(cfg.get("crew"))
        if not has_peers:
            results.append(LintResult(
                "WARN", "L020",
                "allow_delegation=true but no peer agents visible in this config",
                "Delegation target undefined — delegation call may fail at runtime",
                framework))

    # ── L021: multi-agent with no termination ──
    if agent_type in ("groupchat", "multi-agent") or cfg.get("agents"):
        has_any_term = False
        for f in ("max_turns", "max_rounds", "max_round", "max_iter", "max_iterations"):
            if cfg.get(f):
                has_any_term = True
        if cfg.get("is_termination_msg"):
            has_any_term = True
        if not has_any_term:
            results.append(LintResult(
                "ERROR", "L021",
                "Multi-agent config with no termination condition "
                "(no max_turns, max_rounds, or is_termination_msg)",
                "GroupChat Y combinator has no base case AND no bound -> infinite discussion",
                framework))

    # ── L022: has terminate but no maxSteps fallback ──
    if agent_type == "react":
        local_tools = mcp_cfg.get("localTools", [])
        if isinstance(local_tools, list) and "terminate" in local_tools:
            if not react_cfg.get("maxSteps") and not cfg.get("max_iter"):
                results.append(LintResult(
                    "WARN", "L022",
                    "Has 'terminate' tool but no maxSteps fallback — "
                    "if LLM never calls terminate, loop runs indefinitely",
                    "Y(g) with base case but no bounded fallback Y_n",
                    framework))

    # ── L023: router references undefined target ──
    if agent_type == "router" and router_cfg.get("routes"):
        routes = router_cfg["routes"]
        if isinstance(routes, dict):
            for label, target in routes.items():
                if isinstance(target, str) and not target.strip():
                    results.append(LintResult(
                        "ERROR", "L023",
                        f"Router route '{label}' has empty target",
                        f"CASE branch {label} -> _|_ (undefined)", framework))

    # ── L024: duplicate agents in compose chain ──
    if agent_type == "chain" and chain_cfg.get("steps"):
        steps = chain_cfg["steps"]
        if isinstance(steps, list):
            names = [s.get("name", "") if isinstance(s, dict) else str(s) for s in steps]
            seen = set()
            for n in names:
                if n and n in seen:
                    results.append(LintResult(
                        "WARN", "L024",
                        f"Duplicate agent '{n}' in chain — possible configuration error",
                        f"f >> g >> f — redundant composition", framework))
                    break
                seen.add(n)

    # ── L025: guard exists but no retry strategy ──
    if guard_cfg and not guard_cfg.get("retry"):
        results.append(LintResult(
            "WARN", "L025",
            "Guard validation configured but no retry strategy (retry=0 or missing)",
            "guard e P: if P(result) = false -> stuck with no recovery",
            framework))

    # ── L026: remote dependencies detected ──
    has_mcp_remote = bool(mcp_cfg.get("onlineTool"))
    has_a2a = bool(cfg.get("a2a"))
    if has_mcp_remote or has_a2a:
        deps = []
        if has_mcp_remote:
            servers = list(mcp_cfg.get("onlineTool", {}).keys())
            deps.append(f"MCP: {', '.join(servers)}")
        if has_a2a:
            deps.append("A2A remote agents")
        results.append(LintResult(
            "INFO", "L026",
            f"Remote dependencies detected: {'; '.join(deps)}",
            "Oracle functions depend on external services — ensure availability",
            framework))

    # ════════ Security Lint Rules (S001-S003) ════════

    # S001: Unbounded maxSteps
    if agent_type == "react":
        react_cfg = cfg.get("react", {})
        ms = react_cfg.get("maxSteps", 10)
        if isinstance(ms, int) and ms > 1000:
            results.append(LintResult("ERROR", "S001",
                f"react.maxSteps={ms} exceeds safety limit of 1000. Risk: runaway agent execution.",
                "Y combinator iterations must be bounded for safe evaluation",
                framework))
        elif isinstance(ms, int) and ms > 100:
            results.append(LintResult("WARN", "S001",
                f"react.maxSteps={ms} is very high. Consider lowering for cost control.",
                "Bounded Y combinator prevents resource exhaustion",
                framework))

    # S002: Tool-using agent without Guard
    if agent_type == "react":
        mcp_cfg = cfg.get("mcp", {})
        has_tools = bool(mcp_cfg.get("onlineTool")) or bool(mcp_cfg.get("localTools"))
        guard_cfg = cfg.get("guard", {})
        has_guard = bool(guard_cfg)
        if has_tools and not has_guard:
            results.append(LintResult("WARN", "S002",
                "Agent has tools but no guard config. Add guard.dangerousCommandBlock for safety.",
                "Unguarded tool access = untyped β-reduction (no dependent type constraint)",
                framework))

    # S003: Overly permissive SandboxPolicy
    sandbox_cfg = cfg.get("sandbox", {})
    if sandbox_cfg:
        if sandbox_cfg.get("network", False) and sandbox_cfg.get("filesystem", "full") == "full":
            results.append(LintResult("WARN", "S003",
                "SandboxPolicy allows both network and full filesystem access. Consider restricting.",
                "Principle of least privilege for sandboxed evaluation",
                framework))

    # ── Framework detection info ──
    results.append(LintResult(
        "INFO", "L000",
        f"Detected framework: {framework}",
        f"Framework-specific lint rules applied for {framework}",
        framework))

    return results


def format_lint(results: List[LintResult], config_name: str = "config") -> str:
    """Format lint results for terminal output."""
    lines = [f"lambdagent lint: {config_name}", "=" * 60]
    icons = {"ERROR": "x", "WARN": "!", "INFO": "i"}
    for r in results:
        icon = icons.get(r.level, "?")
        fw = f" [{r.framework}]" if r.framework else ""
        lines.append(f"  [{icon}] [{r.level:5s}] [{r.rule:5s}]{fw} {r.message}")
        lines.append(f"    Lambda: {r.lambda_meaning}")
    errors = sum(1 for r in results if r.level == "ERROR")
    warns = sum(1 for r in results if r.level == "WARN")
    infos = sum(1 for r in results if r.level == "INFO")
    lines.append("-" * 60)
    lines.append(f"  {errors} error(s), {warns} warning(s), {infos} info(s)")
    return "\n".join(lines)
