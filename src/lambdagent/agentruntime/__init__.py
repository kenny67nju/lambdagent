"""
lambdagent.agentruntime — Agent Runtime: from Lambda term to real execution

The runtime is the machine that performs beta-reduction.
Compiler (from_config) builds the Lambda term.
Runtime (this module) executes it.
"""
from .runtime import Runtime, RuntimeResult
from .executor import Executor
from .react_engine import ReActEngine, StepResult
# Phase 6.5: Dual engine abstraction
from .engine import Engine, EngineMode, EngineResult, UnifiedTraceRecord
from .recursive_engine import RecursiveEngine
from .cek_engine import CEKEngine
from .adaptive_engine import AdaptiveEngine
from .action_parser import ActionParser, Action, ParseError
from .termination import TerminationOracle
from .llm_adapter import LLMAdapter, LLMResponse, TokenUsage
from .mcp_client import MCPClient, ToolSchema
from .memory_backend import MemoryBackend, LocalMemory, SQLiteMemory
from .trace_store import TraceStore, TraceRecord, TraceStats
from .config import RuntimeConfig, LLMConfig, MemoryConfig, ReActConfig, MCPNodeConfig

__all__ = [
    "Runtime", "RuntimeResult",
    "Executor",
    "ReActEngine", "StepResult",
    "ActionParser", "Action", "ParseError",
    "TerminationOracle",
    "LLMAdapter", "LLMResponse", "TokenUsage",
    "MCPClient", "ToolSchema",
    "MemoryBackend", "LocalMemory", "SQLiteMemory",
    "TraceStore", "TraceRecord", "TraceStats",
    "RuntimeConfig", "LLMConfig", "MemoryConfig", "ReActConfig", "MCPNodeConfig",
    # Phase 6.5: Dual engine
    "Engine", "EngineMode", "EngineResult", "UnifiedTraceRecord",
    "RecursiveEngine", "CEKEngine", "AdaptiveEngine",
]
