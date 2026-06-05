"""
lambdagent.multiagent — 多智能体构造

5 个新构造，扩展 Lambda 演算至 π-演算（进程演算）级别：

Channel      通道通信     π-calculus 的通道原语
SharedMemory 共享记忆     多 Agent 共享环境 Γ_shared
GroupChat    群组对话     Y_n(λself. scheduler >> speaker >> accumulate)
Handoff      动态委派     运行时确定的 Route（动态 CASE）
AsyncPar     真并行       concurrent.futures 并发 β-规约

Lambda 演算对应:
  Channel      → π-calculus: c!(v) / c?(x).P
  SharedMemory → Γ_shared = ∩ agents 的共享环境
  GroupChat    → Y_n(Loop + Route 组合)
  Handoff      → 动态 CASE (运行时确定路由表)
  AsyncPar     → λx. let (r₁,r₂) = concurrent(f(x), g(x)) in (r₁,r₂)
"""

from __future__ import annotations

import threading
import time
import queue
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .core import Term, Context, LambdagentError


# ════════════════════════════════════════════════════════════
# 异常
# ════════════════════════════════════════════════════════════

class ChannelClosed(LambdagentError):
    """通道已关闭"""
    pass


class HandoffError(LambdagentError):
    """委派目标不存在"""
    pass


class GroupChatError(LambdagentError):
    """群组对话错误"""
    pass


# ════════════════════════════════════════════════════════════
# 1. Channel: π-calculus 通道
# ════════════════════════════════════════════════════════════

class Channel:
    """
    π-演算通道: Agent 间的通信原语。

    形式语义:
        c = Channel()
        c.send(v)     ≡  c!(v)       输出到通道
        c.receive()   ≡  c?(x).P     从通道读取
        c.close()     ≡  通道终止

    通道是有类型的消息队列，支持阻塞读写。
    对应 π-calculus 的 name passing:
        ν(c). (P | Q)  — P 和 Q 通过私有通道 c 通信

    线程安全: 底层使用 queue.Queue，支持多生产者多消费者。
    """

    def __init__(self, name: str = "", capacity: int = 0,
                 allowed_agents: Optional[Set[str]] = None):
        """
        Args:
            name: 通道名称（调试用）
            capacity: 缓冲区大小。0=无缓冲（同步），>0=有缓冲（异步）
            allowed_agents: S15 — 允许访问此通道的 Agent 名称集合。None=不限制。
        """
        self.name = name or f"ch_{uuid.uuid4().hex[:6]}"
        self._queue: queue.Queue = queue.Queue(maxsize=capacity)
        self._closed = False
        self._lock = threading.Lock()
        self.history: List[Tuple[str, Any, float]] = []  # (direction, msg, timestamp)
        self.allowed_agents: Optional[Set[str]] = set(allowed_agents) if allowed_agents else None

    def _check_access(self, agent_name: Optional[str] = None) -> None:
        """S15: Check if agent is authorized for this channel."""
        if self.allowed_agents is not None and agent_name not in self.allowed_agents:
            raise PermissionError(f"Agent not authorized for channel '{self.name}'")

    def send(self, message: Any, timeout: Optional[float] = None,
             agent_name: Optional[str] = None) -> None:
        """
        c!(v) — 发送消息到通道。

        Args:
            message: 要发送的消息
            timeout: 超时秒数（None=永久阻塞）
            agent_name: S15 — 调用者 Agent 名称，用于权限检查

        Raises:
            ChannelClosed: 通道已关闭
            PermissionError: Agent 不在 allowed_agents 中
        """
        self._check_access(agent_name)
        if self._closed:
            raise ChannelClosed(f"Channel '{self.name}' is closed")
        try:
            self._queue.put(message, timeout=timeout)
            with self._lock:
                self.history.append(("send", message, time.time()))
        except queue.Full:
            raise ChannelClosed(f"Channel '{self.name}' send timeout")

    def receive(self, timeout: Optional[float] = None,
                agent_name: Optional[str] = None) -> Any:
        """
        c?(x) — 从通道接收消息。

        Args:
            timeout: 超时秒数（None=永久阻塞）
            agent_name: S15 — 调用者 Agent 名称，用于权限检查

        Returns:
            接收到的消息

        Raises:
            ChannelClosed: 通道已关闭且无剩余消息
            PermissionError: Agent 不在 allowed_agents 中
        """
        self._check_access(agent_name)
        try:
            msg = self._queue.get(timeout=timeout)
            with self._lock:
                self.history.append(("recv", msg, time.time()))
            return msg
        except queue.Empty:
            if self._closed:
                raise ChannelClosed(f"Channel '{self.name}' is closed and empty")
            raise ChannelClosed(f"Channel '{self.name}' receive timeout")

    def close(self) -> None:
        """关闭通道"""
        self._closed = True

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def pending(self) -> int:
        """待接收的消息数"""
        return self._queue.qsize()

    def __repr__(self):
        status = "closed" if self._closed else f"pending={self.pending}"
        return f"Channel({self.name!r}, {status})"


# ════════════════════════════════════════════════════════════
# Send / Receive: 通道通信的 Term 封装
# ════════════════════════════════════════════════════════════

class Send(Term):
    """
    向通道发送消息: c!(agent(x))

    Lambda + π 语义:
        Send(agent, channel) = λx. let v = agent(x) in c!(v); v

    agent 执行后，结果同时返回给调用者并发送到通道。
    """

    def __init__(self, agent: Term, channel: Channel):
        super().__init__(f"Send({agent._name}→{channel.name})")
        self.agent = agent
        self.channel = channel

    def apply(self, input: Any, ctx: Context | None = None) -> Any:
        ctx = ctx or Context()
        t0 = time.time()
        result = self.agent.apply(input, ctx)
        self.channel.send(result)
        elapsed = (time.time() - t0) * 1000
        ctx.log(self._name, self._trace_id, input, result, elapsed)
        return result


class Receive(Term):
    """
    从通道接收消息: c?(x).P(x)

    Lambda + π 语义:
        Receive(channel, handler) = λ_. let v = c?() in handler(v)
        Receive(channel)          = λ_. c?()

    阻塞等待通道消息，可选地传给 handler 处理。
    """

    def __init__(self, channel: Channel, handler: Optional[Term] = None,
                 timeout: Optional[float] = 30.0):
        name = f"Recv({channel.name})"
        if handler:
            name = f"Recv({channel.name})>>{handler._name}"
        super().__init__(name)
        self.channel = channel
        self.handler = handler
        self.timeout = timeout

    def apply(self, input: Any, ctx: Context | None = None) -> Any:
        ctx = ctx or Context()
        t0 = time.time()
        msg = self.channel.receive(timeout=self.timeout)
        if self.handler:
            result = self.handler.apply(msg, ctx)
        else:
            result = msg
        elapsed = (time.time() - t0) * 1000
        ctx.log(self._name, self._trace_id, f"channel={self.channel.name}", result, elapsed)
        return result


# ════════════════════════════════════════════════════════════
# 2. SharedMemory: 共享环境 Γ_shared
# ════════════════════════════════════════════════════════════

class SharedMemory:
    """
    多 Agent 共享记忆: Γ_shared = ∩ agents 的共享环境。

    形式语义:
        SharedMem(store) 创建共享环境 Γ_shared
        sm.wrap(agent)   = Memory(agent, Γ_shared)  — 多个 agent 绑定同一个 store
        sm.read(key)     = Γ_shared(key)
        sm.write(key, v) = Γ_shared[key ↦ v]        — append-only 或 可变

    线程安全: 使用 threading.Lock 保护读写。
    类型安全: append_only=True 时，已有 key 不可修改类型（对应 Σ' ⊇ Σ）。
    """

    def __init__(self, store: Optional[Dict[str, Any]] = None,
                 append_only: bool = False):
        """
        Args:
            store: 初始共享存储
            append_only: True 时对应 Preservation 定理的 Σ'⊇Σ 约束
        """
        self._store: Dict[str, Any] = dict(store) if store else {}
        self._type_registry: Dict[str, type] = {}  # key → 首次写入的类型
        self._lock = threading.RLock()
        self._append_only = append_only
        self._history: List[Tuple[str, str, Any, float]] = []  # (op, key, value, time)

    def read(self, key: str, default: Any = None) -> Any:
        """读取共享记忆"""
        with self._lock:
            val = self._store.get(key, default)
            self._history.append(("read", key, val, time.time()))
            return val

    def write(self, key: str, value: Any) -> None:
        """
        写入共享记忆。

        append_only=True 时:
            - 新 key: 写入并记录类型
            - 已有 key: 类型必须一致（Σ' ⊇ Σ），否则 TypeError
        """
        with self._lock:
            if self._append_only and key in self._type_registry:
                expected_type = self._type_registry[key]
                if not isinstance(value, expected_type):
                    raise TypeError(
                        f"SharedMemory type violation: key '{key}' expects "
                        f"{expected_type.__name__}, got {type(value).__name__}. "
                        f"(Store typing: Σ' ⊇ Σ requires type preservation)"
                    )
            self._store[key] = value
            if key not in self._type_registry:
                self._type_registry[key] = type(value)
            self._history.append(("write", key, value, time.time()))

    def read_all(self) -> Dict[str, Any]:
        """读取全部共享记忆"""
        with self._lock:
            return dict(self._store)

    def wrap(self, agent: Term) -> _SharedMemoryAgent:
        """
        将 Agent 绑定到共享记忆。

        Lambda 语义:
            wrap(agent) = λx. agent(x) [Γ ∪ Γ_shared]
        """
        return _SharedMemoryAgent(agent, self)

    def __repr__(self):
        mode = "append-only" if self._append_only else "mutable"
        return f"SharedMemory({len(self._store)} keys, {mode})"


class _SharedMemoryAgent(Term):
    """SharedMemory 包装的 Agent"""

    def __init__(self, agent: Term, shared: SharedMemory):
        super().__init__(f"Shared({agent._name})")
        self.agent = agent
        self.shared = shared

    def apply(self, input: Any, ctx: Context | None = None) -> Any:
        ctx = ctx or Context()
        # 注入共享记忆到输入
        mem = self.shared.read_all()
        if mem:
            mem_str = "\n".join(f"- {k}: {v}" for k, v in mem.items())
            augmented = f"[SharedMemory]\n{mem_str}\n\n[Input]\n{input}"
        else:
            augmented = str(input)

        t0 = time.time()
        result = self.agent.apply(augmented, ctx)
        elapsed = (time.time() - t0) * 1000
        ctx.log(self._name, self._trace_id, input, result, elapsed)
        return result


# ════════════════════════════════════════════════════════════
# 3. GroupChat: 群组对话
# ════════════════════════════════════════════════════════════

class GroupChat(Term):
    """
    多 Agent 群组对话: 多个 Agent 轮流发言直到达成共识。

    Lambda 语义:
        GroupChat([a,b,c], scheduler, n) =
            Y_n(λself.λstate.
                let speaker = scheduler(state) in
                let msg = speaker(state) in
                let state' = state ++ msg in
                IF done(state') THEN state' ELSE self(state')
            )

    这是 Loop + Route 的组合——不需要新的 Lambda 构造！

    调度策略:
        "round_robin"  → 固定顺序轮流
        "random"       → 随机选择
        Term           → LLM 分类器选择下一个发言者
    """

    def __init__(
        self,
        agents: List[Term],
        max_rounds: int = 10,
        scheduler: str | Term = "round_robin",
        termination: Optional[Callable[[str, int], bool]] = None,
        summary_agent: Optional[Term] = None,
    ):
        """
        Args:
            agents: 参与对话的 Agent 列表
            max_rounds: 最大轮数 (Y 组合子的界 n)
            scheduler: 调度策略或 LLM 分类器
            termination: 终止条件 (state, round) → bool
            summary_agent: 对话结束后的总结 Agent
        """
        names = ", ".join(a._name for a in agents)
        super().__init__(f"GroupChat([{names}])")
        self.agents = {a._name: a for a in agents}
        self.agent_list = list(agents)
        self.max_rounds = max_rounds
        self.scheduler = scheduler
        self.termination = termination or self._default_termination
        self.summary_agent = summary_agent

    def apply(self, input: Any, ctx: Context | None = None) -> Any:
        """
        执行群组对话。

        对应 Y_n 的展开:
            每轮 = 一次 β-规约
            state = 对话历史
            base case = termination 返回 True 或 达到 max_rounds
        """
        ctx = ctx or Context()
        state = str(input)
        conversation: List[Dict[str, str]] = []

        t0_total = time.time()

        for round_idx in range(self.max_rounds):
            # 选择发言者 = scheduler(state)
            speaker = self._select_speaker(round_idx, state, ctx)

            # 发言者执行 = speaker(state)
            t0 = time.time()
            # 构造带有对话历史的输入 (smart context window)
            speaker_input = self._build_speaker_input(speaker._name, conversation, input, round_idx)

            response = speaker.apply(speaker_input, ctx)
            elapsed = (time.time() - t0) * 1000

            # 记录对话
            conversation.append({
                "speaker": speaker._name,
                "content": str(response),
                "round": round_idx,
            })

            # 更新 state = state ++ msg
            state = f"{state}\n[{speaker._name}]: {response}"

            # 记录 β-规约
            ctx.log(
                f"GroupChat.round[{round_idx}]:{speaker._name}",
                self._trace_id, speaker_input[:100], str(response)[:100], elapsed
            )

            # 检查终止条件 = base case
            if self.termination(state, round_idx + 1):
                break

        # 可选: 总结
        if self.summary_agent:
            t0 = time.time()
            summary = self.summary_agent.apply(state, ctx)
            elapsed = (time.time() - t0) * 1000
            ctx.log("GroupChat.summary", self._trace_id, state[:100], str(summary)[:100], elapsed)
            return summary

        total_elapsed = (time.time() - t0_total) * 1000
        ctx.log(self._name, self._trace_id, str(input)[:100],
                f"{len(conversation)} rounds", total_elapsed)

        # 返回最后一条消息
        return conversation[-1]["content"] if conversation else state

    def _build_speaker_input(self, speaker_name: str, conversation: list,
                             original_input: Any, round_idx: int) -> str:
        """Build context-aware input for the next speaker.

        For short conversations (<= 6 messages), return full history.
        For longer ones, build a smart context window:
          1. First-round messages (topic establishment)
          2. This speaker's own previous messages (last 3)
          3. Most recent 3*len(agents) messages
        """
        if not conversation:
            return str(original_input)

        if len(conversation) <= 6:
            history = "\n".join(
                f"[{msg['speaker']}]: {msg['content']}"
                for msg in conversation
            )
            return (
                f"[Conversation History]\n{history}\n\n"
                f"[Original Task]\n{original_input}\n\nYour turn to speak:"
            )

        # Smart context: first-round messages
        first_round = [m for m in conversation if m["round"] == 0]

        # This speaker's own previous messages (last 3)
        own_msgs = [m for m in conversation if m["speaker"] == speaker_name][-3:]

        # Most recent messages (3 * number of agents)
        tail_size = 3 * len(self.agent_list)
        recent = conversation[-tail_size:]

        # Merge and deduplicate while preserving order
        seen_ids: set = set()
        merged: list = []
        for msg in first_round + own_msgs + recent:
            msg_id = id(msg)
            if msg_id not in seen_ids:
                seen_ids.add(msg_id)
                merged.append(msg)

        # Sort by round to maintain chronological order
        merged.sort(key=lambda m: (m["round"], conversation.index(m)))

        history = "\n".join(
            f"[{msg['speaker']}]: {msg['content']}"
            for msg in merged
        )
        omitted = len(conversation) - len(merged)
        note = f"  ... ({omitted} messages omitted) ...\n" if omitted > 0 else ""

        return (
            f"[Conversation History ({len(merged)}/{len(conversation)} messages)]\n"
            f"{note}{history}\n\n"
            f"[Original Task]\n{original_input}\n\nYour turn to speak:"
        )

    def _select_speaker(self, round_idx: int, state: str, ctx: Context) -> Term:
        """调度: 选择下一个发言者"""
        if isinstance(self.scheduler, str):
            if self.scheduler == "round_robin":
                return self.agent_list[round_idx % len(self.agent_list)]
            elif self.scheduler == "random":
                import random
                return random.choice(self.agent_list)
            else:
                raise GroupChatError(f"Unknown scheduler: {self.scheduler}")
        elif isinstance(self.scheduler, Term):
            # LLM 分类器选择 = Route 的动态版
            label = str(self.scheduler.apply(state, ctx)).strip()
            for name, agent in self.agents.items():
                if name.lower() in label.lower() or label.lower() in name.lower():
                    return agent
            # fallback: round robin
            return self.agent_list[round_idx % len(self.agent_list)]
        else:
            raise GroupChatError(f"Invalid scheduler type: {type(self.scheduler)}")

    @staticmethod
    def _default_termination(state: str, round_num: int) -> bool:
        """默认终止条件: 包含终止关键词"""
        terminators = ["CONSENSUS", "DONE", "TERMINATE", "FINAL ANSWER", "达成共识", "结束"]
        state_upper = state.upper()
        return any(t in state_upper for t in terminators)


# ════════════════════════════════════════════════════════════
# 4. Handoff: 动态委派
# ════════════════════════════════════════════════════════════

class Handoff(Term):
    """
    动态委派: 运行时确定路由目标。

    Lambda 语义:
        Handoff(selector, registry) =
            λx. let target = selector(x) in
                 let agent = registry[target] in
                 agent(x)

    与 Route 的区别:
        Route  = 编译时确定路由表（静态 CASE）
        Handoff = 运行时确定路由目标（动态 CASE）
        Handoff 的 selector 可以返回 registry 中尚未注册的名称
        → 支持运行时动态注册新 Agent
    """

    def __init__(
        self,
        selector: Term | Callable[[str], str],
        registry: Optional[Dict[str, Term]] = None,
        fallback: Optional[Term] = None,
    ):
        """
        Args:
            selector: 选择器（LLM Agent 或 Python 函数），返回目标 Agent 名称
            registry: Agent 注册表 {name: agent}
            fallback: 所有路由失败时的 fallback Agent
        """
        name = selector._name if isinstance(selector, Term) else "handoff_selector"
        super().__init__(f"Handoff({name})")
        self.selector = selector
        self.registry: Dict[str, Term] = dict(registry) if registry else {}
        self.fallback = fallback
        self._lock = threading.Lock()

    def register(self, name: str, agent: Term) -> None:
        """运行时注册新 Agent（动态扩展路由表）"""
        with self._lock:
            self.registry[name] = agent

    def unregister(self, name: str) -> None:
        """运行时注销 Agent"""
        with self._lock:
            self.registry.pop(name, None)

    def apply(self, input: Any, ctx: Context | None = None) -> Any:
        ctx = ctx or Context()
        t0 = time.time()

        # 选择目标 = selector(x)
        if isinstance(self.selector, Term):
            target_name = str(self.selector.apply(input, ctx)).strip()
        else:
            target_name = self.selector(str(input))

        # 查找目标 Agent
        with self._lock:
            agent = None
            # 精确匹配
            if target_name in self.registry:
                agent = self.registry[target_name]
            else:
                # 模糊匹配
                for name, a in self.registry.items():
                    if name.lower() in target_name.lower() or target_name.lower() in name.lower():
                        agent = a
                        target_name = name
                        break

        if agent is None:
            if self.fallback:
                agent = self.fallback
                target_name = "fallback"
            else:
                raise HandoffError(
                    f"Handoff target '{target_name}' not found. "
                    f"Available: {list(self.registry.keys())}"
                )

        # 执行目标 Agent
        result = agent.apply(input, ctx)
        elapsed = (time.time() - t0) * 1000
        ctx.log(f"Handoff→{target_name}", self._trace_id, str(input)[:100],
                str(result)[:100], elapsed)
        return result


# ════════════════════════════════════════════════════════════
# 5. AsyncPar: 真并行执行
# ════════════════════════════════════════════════════════════

class AsyncPar(Term):
    """
    真并行执行: 使用线程池并发运行多个 Agent。

    Lambda 语义:
        AsyncPar(f, g) = λx. let (r₁, r₂) = concurrent(f(x), g(x)) in (r₁, r₂)

    Paper II Proposition 30 (Pair Confluence):
        writes(f) ∩ writes(g) = ∅ → 结果与调度策略无关

    与 Par 的区别:
        Par      = 顺序执行（假并行）
        AsyncPar = 线程池并发（真并行）

    适用场景:
        多个 LLM 调用互不依赖 → 并发执行节省总时间
        例: AsyncPar(research, critique) — 研究和批评同时进行
    """

    def __init__(self, *agents: Term, max_workers: Optional[int] = None,
                 timeout: Optional[float] = 120.0,
                 check_store_independence: bool = True):
        """
        Args:
            agents: 要并行执行的 Agent
            max_workers: 线程池大小（默认=Agent 数量）
            timeout: 总超时时间（秒）
            check_store_independence: 是否在执行前检查存储独立性 (Paper II Prop. 30)
        """
        names = " ∥ ".join(a._name for a in agents)
        super().__init__(f"AsyncPar({names})")
        self.agents = list(agents)
        self.max_workers = max_workers or len(agents)
        self.timeout = timeout
        self._check_store_independence = check_store_independence

    def apply(self, input: Any, ctx: Context | None = None) -> tuple:
        """
        并发执行所有 Agent。

        Paper II Proposition 30:
            1. 检查 writes(f) ∩ writes(g) = ∅（存储独立性）
            2. 每个分支 fork 独立的 Context（防止竞态条件）
            3. 执行后合并 trace 到父 Context

        返回顺序与 agents 列表一致。
        """
        ctx = ctx or Context()
        t0 = time.time()

        # Paper II Prop. 30: 存储独立性检查
        if self._check_store_independence:
            from .store_analysis import check_store_independence
            check_store_independence(self.agents)

        results = [None] * len(self.agents)
        errors = [None] * len(self.agents)
        forked_ctxs = [ctx.fork() for _ in self.agents]  # Paper II: 独立上下文

        def _run_agent(idx: int, agent: Term, fork_ctx: Context) -> Tuple[int, Any]:
            try:
                result = agent.apply(input, fork_ctx)
                return (idx, result, None)
            except Exception as e:
                return (idx, None, e)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(_run_agent, i, agent, forked_ctxs[i]): i
                for i, agent in enumerate(self.agents)
            }
            for future in as_completed(futures, timeout=self.timeout):
                idx, result, error = future.result()
                results[idx] = result
                errors[idx] = error

        # 合并子 Context 的 trace 到父 Context
        for fork_ctx in forked_ctxs:
            ctx.merge_trace(fork_ctx)

        # 检查错误
        for i, err in enumerate(errors):
            if err is not None:
                raise LambdagentError(
                    f"AsyncPar: agent '{self.agents[i]._name}' failed: {err}"
                )

        elapsed = (time.time() - t0) * 1000
        ctx.log(self._name, self._trace_id, str(input)[:100],
                f"{len(results)} results in {elapsed:.0f}ms", elapsed)

        return tuple(results)

    def __or__(self, other: Term) -> AsyncPar:
        """展平: (f ∥ g) | h = AsyncPar(f, g, h)"""
        if isinstance(other, AsyncPar):
            return AsyncPar(*self.agents, *other.agents,
                           max_workers=self.max_workers, timeout=self.timeout)
        return AsyncPar(*self.agents, other,
                       max_workers=self.max_workers, timeout=self.timeout)
