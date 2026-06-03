"""
lambdagent — 基于 Lambda 演算的 Agent DSL

每个 Agent 是一个 Lambda 项。组合 Agent 就是组合函数。

11 个核心构造，严格对应 Lambda 演算：

     Lambda 演算                DSL 构造
     ──────────                ────────
 1.  λ 抽象 λx.body           Lam(name, prompt)
 2.  函数应用 (f x)           agent(input)
 3.  函数组合 λx.g(f(x))     f >> g
 4.  Church 条件 IF c t e     If(cond, then_, else_)
 5.  Y 组合子                 Loop(body, condition)
 6.  Church 对 PAIR           Pair(f, g)
 7.  投影 FST / SND           Fst() / Snd()
 8.  原语 / Oracle            Tool(name, fn)
 9.  广义 Church 布尔 CASE    Route(classifier, routes)
 10. 依赖类型 {x:T|P(x)}     Guard(agent, validator)
 11. 环境扩展 Γ' = Γ ∪ s     Memory(agent, store)

多智能体扩展（π-演算 + 并发）：

     进程演算                   DSL 构造
     ────────                  ────────
 12. π-calculus 通道 c!(v)/c?  Channel + Send + Receive
 13. 共享环境 Γ_shared         SharedMemory
 14. 多Agent群组对话            GroupChat (= Loop + Route 组合)
 15. 动态路由 (动态 CASE)       Handoff
 16. 并发 β-规约               AsyncPar

辅助设施（非构造）：
 -  Context      求值环境 Γ + β-规约追踪（元层级）
 -  Dataset      Lam 的便利构造器（D.to_lam() = Lam(prompt=encode(D))）
 -  from_config  YAML → Lambda 编译器
"""

from .core import Term, Context, TraceEntry
from .core import LambdagentError, UnboundVariable, RouteError, ValidationError
from .trace import (
    TraceStore, TraceEntry as EnhancedTraceEntry, Anomaly,
    colorize_timeline, detect_anomalies, format_anomalies,
    generate_flamegraph_html, save_flamegraph,
    replay, diff_traces,
)
from .primitives import Lam, Compose, If, Loop, Pair, Fst, Snd, Tool
from .extensions import Par, Route, Memory, Guard
from .sandbox import (
    SandboxedTool, SandboxPolicy, SecureExecutor,
    SandboxViolation, TimeoutViolation, MemoryViolation, OutputViolation,
    ResourceLimiter, sandboxed,
)
from .dataset import Dataset
from .multiagent import (
    Channel, Send, Receive,
    SharedMemory,
    GroupChat,
    Handoff,
    AsyncPar,
    ChannelClosed, HandoffError, GroupChatError,
)
from .skills import (
    Skill, SkillSignature, SkillPack,
    SkillRegistry, SkillAgent, skill,
)
from .mcp_client import (
    MCPServer, MCPTool, MCPToolInfo,
    MCPError, MCPConnectionError, MCPToolError,
    mcp_tools, mcp_tool,
)
from .checkpoint import (
    Checkpoint, CheckpointManager,
    save_context, load_context,
    CheckpointError,
)
from .a2a import (
    AgentCard, A2AServer, A2AClient, A2ATask,
    skill_to_agent_card, registry_to_agent_card,
)
from .rag import (
    RAGTool, AgenticRAG, SimpleVectorStore, Document, SearchResult,
    create_rag,
)
# Paper III: Type & Effect System
from .types import (
    LamType, TypeTag, AgentType, AgentTypeError,
    T_ANY, T_NONE, T_STR, T_INT, T_FLOAT, T_BOOL, T_JSON, T_TUPLE, T_UNION,
    is_subtype, check_compose_types, parse_type_annotation, infer_type_from_value,
)
from .effects import (
    Effect, EffectKind, ComposedEffect,
    PURE, IO, LLM, STATE,
    serial, parallel, iterate,
    effect_leq, max_effect,
    parse_effect_annotation, infer_effect_for_term,
)
# Paper II: Store Independence Analysis
from .store_analysis import StoreConflictError
# Paper III §6: Algebraic Effect Handlers
from .handlers import (
    EffectHandler, ProductionHandler, TestHandler, TraceHandler,
    get_current_handler, set_current_handler, with_handler,
)
# Paper III §4.3: Graded Cost Prediction
from .cost_grade import (
    CostGrade, estimate_cost, format_cost_estimate,
    grade_serial, grade_parallel, grade_iterate, grade_guard,
)
# Paper II Theorems 36-41: Algebraic Law Rewrite Rules
from .rewrite import optimize as optimize_agent, RewriteLog, RewriteEntry
# Paper II: CEK Machine upgrades
from .cek_machine import AgentCEKMachine, CostVector, CostMonotonicityViolation
# Phase 1: P0 Engineering Improvements
from .cancellation import CancellationToken, CancelledError, NullCancellationToken
from .retry import RetryPolicy, CircuitBreaker, CircuitOpenError, with_retry, with_retry_sync
from .isolation import IsolationLevel, IsolatedWorkspace, WorkspaceManager
import lambdagent.async_core  # patches aapply() onto all Term types

# fromconfig (v2)
from .fromconfig import from_config, build_agent, describe_config
from .fromconfig import from_config as from_config_v2  # backwards compat alias
from .fromconfig import lint_config, to_lambda_expr

# Phase 6.5: Dual engine — FIX-07: export engine classes
from .agentruntime.engine import EngineMode, EngineResult, UnifiedTraceRecord
from .agentruntime.recursive_engine import RecursiveEngine
from .agentruntime.cek_engine import CEKEngine
from .agentruntime.runtime import Runtime, RuntimeResult

__all__ = [
    # 元层级
    "Term", "Context", "TraceEntry",
    # Enhanced trace system
    "TraceStore", "EnhancedTraceEntry", "Anomaly",
    "colorize_timeline", "detect_anomalies", "format_anomalies",
    "generate_flamegraph_html", "save_flamegraph",
    "replay", "diff_traces",
    # 异常
    "LambdagentError", "UnboundVariable", "RouteError", "ValidationError",
    # 11 个核心构造
    "Lam",          # 1.  λ 抽象
    "Compose",      # 3.  函数组合 (>> 操作符)
    "If",           # 4.  Church 条件
    "Loop",         # 5.  Y 组合子
    "Pair",         # 6.  Church 对
    "Fst", "Snd",   # 7.  投影
    "Tool",         # 8.  原语 / Oracle
    "Route",        # 9.  广义 Church 布尔
    "Guard",        # 10. 依赖类型
    # 安全沙盒 (Phase 8)
    "SandboxedTool",    # Tool + 进程隔离
    "SandboxPolicy",    # 安全策略
    "SecureExecutor",   # 沙盒感知求值器
    "sandboxed",        # @sandboxed 装饰器
    "SandboxViolation", "TimeoutViolation", "MemoryViolation", "OutputViolation",
    "ResourceLimiter",
    "Memory",       # 11. 环境扩展
    # 并行执行（Pair 的语法糖，| 操作符）
    "Par",
    # 多智能体扩展 (π-演算 + 并发)
    "Channel",      # 12. π-calculus 通道
    "Send",         #     通道发送 c!(v)
    "Receive",      #     通道接收 c?(x)
    "SharedMemory", # 13. 共享环境 Γ_shared
    "GroupChat",    # 14. 多Agent群组对话
    "Handoff",      # 15. 动态委派（运行时 Route）
    "AsyncPar",     # 16. 真并行执行
    # 多智能体异常
    "ChannelClosed", "HandoffError", "GroupChatError",
    # Skill 系统
    "Skill",          # 命名的可复用 Lambda 项
    "SkillSignature", # 技能类型签名
    "SkillPack",      # 技能集合
    "SkillRegistry",  # 全局技能注册表
    "SkillAgent",     # 自动发现技能的 Agent
    "skill",          # @skill 装饰器
    # MCP Client
    "MCPServer",    # MCP 服务器连接
    "MCPTool",      # MCP 工具 → lambdagent Term
    "mcp_tools",    # 便利: 一行获取所有 MCP 工具
    "mcp_tool",     # 便利: 一行获取单个 MCP 工具
    "MCPError", "MCPConnectionError", "MCPToolError",
    # Checkpoint
    "Checkpoint",          # 状态快照
    "CheckpointManager",   # 多 checkpoint 管理
    "save_context",        # 保存 Context
    "load_context",        # 恢复 Context
    "CheckpointError",
    # A2A Protocol
    "AgentCard",        # A2A Agent 能力描述
    "A2AServer",        # 发布 Agent 为 A2A 服务
    "A2AClient",        # 调用远程 A2A Agent
    "skill_to_agent_card", "registry_to_agent_card",
    # RAG
    "RAGTool",          # 检索增强工具
    "AgenticRAG",       # Agent 自主决定是否检索
    "SimpleVectorStore", # 零依赖向量存储
    "create_rag",       # 一行创建 RAG
    # Paper III: 类型系统
    "LamType", "TypeTag", "AgentType", "AgentTypeError",
    "T_ANY", "T_NONE", "T_STR", "T_INT", "T_FLOAT", "T_BOOL",
    "T_JSON", "T_TUPLE", "T_UNION",
    "is_subtype", "check_compose_types", "parse_type_annotation", "infer_type_from_value",
    # Paper III: 效果系统
    "Effect", "EffectKind", "ComposedEffect",
    "PURE", "IO", "LLM", "STATE",
    "serial", "parallel", "iterate",
    "effect_leq", "max_effect",
    "parse_effect_annotation", "infer_effect_for_term",
    # Paper II: 存储独立性分析
    "StoreConflictError",
    # Paper III §6: 代数效果处理器
    "EffectHandler", "ProductionHandler", "TestHandler", "TraceHandler",
    "get_current_handler", "set_current_handler", "with_handler",
    # Paper III §4.3: 分级成本预测
    "CostGrade", "estimate_cost", "format_cost_estimate",
    "grade_serial", "grade_parallel", "grade_iterate", "grade_guard",
    # Paper II: 代数定律重写
    "optimize_agent", "RewriteLog", "RewriteEntry",
    # Paper II: CEK 机器
    "AgentCEKMachine", "CostVector", "CostMonotonicityViolation",
    # 辅助设施
    "Dataset",
    "from_config", "build_agent", "describe_config",
    "from_config_v2",  # backwards compat alias
    "lint_config", "to_lambda_expr",
    # Phase 1: P0 Engineering
    "CancellationToken", "CancelledError", "NullCancellationToken",
    "RetryPolicy", "CircuitBreaker", "CircuitOpenError",
    "with_retry", "with_retry_sync",
    "IsolationLevel", "IsolatedWorkspace", "WorkspaceManager",
    # Phase 6.5: Dual engine (FIX-07)
    "EngineMode", "EngineResult", "UnifiedTraceRecord",
    "RecursiveEngine", "CEKEngine", "Runtime", "RuntimeResult",
]
