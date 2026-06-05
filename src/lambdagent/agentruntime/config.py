"""agentruntime.config — Runtime configuration dataclasses"""
from __future__ import annotations
import os
import yaml
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LLMConfig:
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.0
    max_tokens: int = 1024
    api_key: str = ""
    base_url: str = ""

@dataclass
class MCPNodeConfig:
    url: str = ""
    endpoint: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: int = 30
    retry: int = 0

@dataclass
class MemoryConfig:
    enabled: bool = False
    strategy: str = "local"
    size: int = 20
    ttl: int = 3600
    redis_url: str = ""
    db_path: str = ""

@dataclass
class ReActConfig:
    max_steps: int = 10
    tool_timeout: int = 30
    think_timeout: int = 120
    observation_enabled: bool = True
    verbose: bool = False

@dataclass
class TerminationConfig:
    signals: List[str] = field(default_factory=lambda: [
        "final answer:", "task complete", "task is done",
        "i have completed", "here is the result:", "in conclusion,",
    ])
    implicit_detection: bool = True

@dataclass
class TraceConfig:
    enabled: bool = True
    file: str = ""
    format: str = "text"  # "text" | "json"

@dataclass
class TimeoutConfig:
    """Configurable timeouts for all I/O operations."""
    llm_call: int = 120       # LLM API call timeout (seconds)
    tool_call: int = 30       # Tool execution timeout (seconds)
    mcp_call: int = 30        # MCP server call timeout (seconds)
    shell: int = 30           # Shell command timeout (seconds)

@dataclass
class RuntimeConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    mcp: Dict[str, MCPNodeConfig] = field(default_factory=dict)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    react: ReActConfig = field(default_factory=ReActConfig)
    termination: TerminationConfig = field(default_factory=TerminationConfig)
    trace: TraceConfig = field(default_factory=TraceConfig)
    timeout: TimeoutConfig = field(default_factory=TimeoutConfig)

    @staticmethod
    def from_yaml(path: str, **overrides) -> "RuntimeConfig":
        """Build RuntimeConfig from YAML + CLI overrides + env vars."""
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        config = RuntimeConfig()

        # LLM
        model_cfg = cfg.get("model", {})
        config.llm = LLMConfig(
            provider=model_cfg.get("provider", "anthropic"),
            model=overrides.get("model", model_cfg.get("name", "claude-sonnet-4-20250514")),
            temperature=overrides.get("temperature", model_cfg.get("temperature", 0.0)),
            max_tokens=model_cfg.get("maxTokens", 1024),
            api_key=os.environ.get("ANTHROPIC_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
            base_url=model_cfg.get("baseUrl", ""),
        )

        # MCP nodes
        app_nodes = cfg.get("app", {}).get("mcp", {}).get("custom", {}).get("nodes", {})
        for name, node_cfg in app_nodes.items():
            config.mcp[name] = MCPNodeConfig(
                url=node_cfg.get("url", ""),
                endpoint=node_cfg.get("endpoint", ""),
                headers=node_cfg.get("headers", {}),
                timeout=node_cfg.get("timeout", 30),
            )

        # Memory
        mem_cfg = cfg.get("memory", {})
        config.memory = MemoryConfig(
            enabled=mem_cfg.get("enabled", False),
            strategy=mem_cfg.get("strategy", "local"),
            size=mem_cfg.get("size", 20),
            ttl=mem_cfg.get("ttl", 3600),
        )

        # ReAct
        react_cfg = cfg.get("react", {})
        config.react = ReActConfig(
            max_steps=overrides.get("max_steps", react_cfg.get("maxSteps", 10)),
            tool_timeout=react_cfg.get("toolTimeout", 30),
            think_timeout=react_cfg.get("thinkTimeout", 120),
            observation_enabled=react_cfg.get("observationEnabled", True),
            verbose=react_cfg.get("verbose", False),
        )

        # Timeouts
        timeout_cfg = cfg.get("timeout", {})
        config.timeout = TimeoutConfig(
            llm_call=timeout_cfg.get("llmCall", 120),
            tool_call=timeout_cfg.get("toolCall", 30),
            mcp_call=timeout_cfg.get("mcpCall", 30),
            shell=timeout_cfg.get("shell", 30),
        )

        return config
