"""
lambdagent.checkpoint — 状态序列化与恢复

长时运行 Agent 的断点续跑能力。

Lambda 语义:
    save(ctx)  = serialize(Γ, trace, Γ_shared) → JSON/文件
    load(path) = deserialize(JSON/文件) → (Γ, trace, Γ_shared)
    resume(agent, checkpoint) = agent(last_input) [Γ_restored]

核心理念:
    Checkpoint = (Context, Memory, SharedMemory, 执行位置) 的快照
    恢复 = 在保存时的环境中继续 β-规约

支持:
    - Context (bindings + trace + memory) 序列化
    - SharedMemory 状态序列化
    - SkillRegistry 序列化 (技能元数据，不含 Term)
    - JSON / 文件系统存储
    - 版本化 checkpoint（防止格式变更导致的不兼容）
"""

from __future__ import annotations

import json
import time
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .core import Context, TraceEntry, LambdagentError


# ════════════════════════════════════════════════════════════
# S13: AES-256-GCM 加密/解密
# ════════════════════════════════════════════════════════════


def _encrypt_aes_gcm(plaintext: bytes, key: bytes) -> bytes:
    """AES-256-GCM 加密。返回 nonce(12) + ciphertext + tag(16)。"""
    import secrets

    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError:
        raise ImportError(
            "Checkpoint encryption requires 'cryptography' package. "
            "Install: pip install cryptography"
        )
    if len(key) != 32:
        raise ValueError(f"AES-256 requires 32-byte key, got {len(key)}")
    nonce = secrets.token_bytes(12)  # 96-bit nonce (GCM standard)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, plaintext, None)
    return nonce + ciphertext  # nonce(12) + ciphertext + tag(16)


def _decrypt_aes_gcm(data: bytes, key: bytes) -> bytes:
    """AES-256-GCM 解密。输入 nonce(12) + ciphertext + tag(16)。"""
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError:
        raise ImportError(
            "Checkpoint decryption requires 'cryptography' package. "
            "Install: pip install cryptography"
        )
    if len(key) != 32:
        raise ValueError(f"AES-256 requires 32-byte key, got {len(key)}")
    if len(data) < 28:  # 12 nonce + 16 tag minimum
        raise ValueError("Encrypted data too short")
    nonce = data[:12]
    ciphertext = data[12:]
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ciphertext, None)


# ════════════════════════════════════════════════════════════
# 异常
# ════════════════════════════════════════════════════════════


class CheckpointError(LambdagentError):
    """Checkpoint 操作失败"""

    pass


class CheckpointVersionError(CheckpointError):
    """Checkpoint 版本不兼容"""

    pass


# ════════════════════════════════════════════════════════════
# Checkpoint 数据格式
# ════════════════════════════════════════════════════════════

CHECKPOINT_VERSION = "1.0.0"


def _serialize_value(v: Any) -> Any:
    """将值序列化为 JSON 兼容格式"""
    if isinstance(v, (str, int, float, bool, type(None))):
        return v
    if isinstance(v, (list, tuple)):
        return [_serialize_value(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _serialize_value(val) for k, val in v.items()}
    # 不可序列化的值转为字符串
    return str(v)


def _serialize_trace(trace: List[TraceEntry]) -> List[Dict]:
    """序列化 β-规约追踪"""
    return [
        {
            "term_name": e.term_name,
            "term_id": e.term_id,
            "input": _serialize_value(e.input),
            "output": _serialize_value(e.output),
            "duration_ms": e.duration_ms,
            "model": e.model,
            "tokens_used": e.tokens_used,
        }
        for e in trace
    ]


def _deserialize_trace(data: List[Dict]) -> List[TraceEntry]:
    """反序列化 β-规约追踪"""
    return [
        TraceEntry(
            term_name=d.get("term_name", ""),
            term_id=d.get("term_id", ""),
            input=d.get("input", ""),
            output=d.get("output", ""),
            duration_ms=d.get("duration_ms", 0.0),
            model=d.get("model", ""),
            tokens_used=d.get("tokens_used", 0),
        )
        for d in data
    ]


# ════════════════════════════════════════════════════════════
# Checkpoint: 核心类
# ════════════════════════════════════════════════════════════


class Checkpoint:
    """
    Agent 执行状态的快照。

    包含:
        - context: Context 的全部状态 (bindings, trace, memory)
        - shared_memories: SharedMemory 的状态
        - metadata: 时间戳、版本、描述等
        - last_input: 最后一次输入（用于 resume）
        - step_count: 已执行的 β-规约步数

    Lambda 语义:
        Checkpoint = (Γ, trace, Γ_shared, last_input, n_steps)
        save() = serialize 到 JSON
        load() = deserialize 从 JSON
    """

    def __init__(
        self,
        context: Optional[Context] = None,
        shared_data: Optional[Dict[str, Dict]] = None,
        last_input: str = "",
        description: str = "",
        metadata: Optional[Dict] = None,
    ):
        self.context = context
        self.shared_data = shared_data or {}
        self.last_input = last_input
        self.description = description
        self.metadata = metadata or {}
        self.timestamp = time.time()
        self.version = CHECKPOINT_VERSION

    @property
    def step_count(self) -> int:
        """已执行的 β-规约步数"""
        return len(self.context.trace) if self.context else 0

    @property
    def total_time_ms(self) -> float:
        """总执行时间"""
        if not self.context or not self.context.trace:
            return 0.0
        return sum(e.duration_ms for e in self.context.trace)

    def to_dict(self) -> Dict:
        """序列化为字典"""
        data = {
            "version": self.version,
            "timestamp": self.timestamp,
            "description": self.description,
            "last_input": self.last_input,
            "metadata": _serialize_value(self.metadata),
            "context": {
                "bindings": _serialize_value(self.context.bindings)
                if self.context
                else {},
                "memory": _serialize_value(self.context.memory) if self.context else {},
                "trace": _serialize_trace(self.context.trace) if self.context else [],
            },
            "shared_memories": {
                name: _serialize_value(store)
                for name, store in self.shared_data.items()
            },
            "stats": {
                "step_count": self.step_count,
                "total_time_ms": self.total_time_ms,
            },
        }
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> Checkpoint:
        """从字典反序列化"""
        version = data.get("version", "0.0.0")
        if version != CHECKPOINT_VERSION:
            # 尝试向后兼容
            if version.split(".")[0] != CHECKPOINT_VERSION.split(".")[0]:
                raise CheckpointVersionError(
                    f"Checkpoint version {version} incompatible with {CHECKPOINT_VERSION}"
                )

        ctx_data = data.get("context", {})
        context = Context(
            bindings=ctx_data.get("bindings", {}),
            memory=ctx_data.get("memory", {}),
        )
        context.trace = _deserialize_trace(ctx_data.get("trace", []))

        return cls(
            context=context,
            shared_data=data.get("shared_memories", {}),
            last_input=data.get("last_input", ""),
            description=data.get("description", ""),
            metadata=data.get("metadata", {}),
        )

    def save(self, path: str, encryption_key: Optional[bytes] = None) -> str:
        """
        保存到文件。支持可选 AES-256-GCM 加密 (S13)。

        Args:
            path: 文件路径 (.json 明文, .json.enc 加密)
            encryption_key: 32 字节 AES-256 密钥。传入时启用加密。

        Returns:
            实际保存的路径
        """
        path = str(path)

        # 确保目录存在
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        data = self.to_dict()
        payload = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")

        if encryption_key:
            # S13: AES-256-GCM 加密
            encrypted = _encrypt_aes_gcm(payload, encryption_key)
            if not path.endswith(".enc"):
                path = path.rstrip(".json") + ".json.enc"
            with open(path, "wb") as f:
                f.write(encrypted)
        else:
            if not path.endswith(".json"):
                path += ".json"
            with open(path, "w", encoding="utf-8") as f:
                f.write(payload.decode("utf-8"))

        return path

    @classmethod
    def load(cls, path: str, encryption_key: Optional[bytes] = None) -> Checkpoint:
        """
        从文件加载。支持自动检测加密格式 (S13)。

        Args:
            path: 文件路径 (.json 或 .json.enc)
            encryption_key: 32 字节 AES-256 密钥。加密文件必须提供。

        Returns:
            Checkpoint 对象
        """
        if path.endswith(".enc"):
            if not encryption_key:
                raise ValueError(
                    f"Encrypted checkpoint '{path}' requires encryption_key"
                )
            with open(path, "rb") as f:
                encrypted = f.read()
            payload = _decrypt_aes_gcm(encrypted, encryption_key)
            data = json.loads(payload.decode("utf-8"))
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        return cls.from_dict(data)

    def summary(self) -> str:
        """生成人类可读的摘要"""
        lines = [
            f"Checkpoint: {self.description or '(no description)'}",
            f"  Version:    {self.version}",
            f"  Timestamp:  {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.timestamp))}",
            f"  Steps:      {self.step_count} β-reductions",
            f"  Total time: {self.total_time_ms:.0f}ms",
            f"  Last input: {self.last_input[:80]}..."
            if len(self.last_input) > 80
            else f"  Last input: {self.last_input}",
            f"  Memory keys: {list(self.context.memory.keys()) if self.context else []}",
            f"  Shared mem: {list(self.shared_data.keys())}",
        ]
        return "\n".join(lines)

    def __repr__(self):
        return f"Checkpoint({self.step_count} steps, {self.description!r})"


# ════════════════════════════════════════════════════════════
# Context 扩展方法
# ════════════════════════════════════════════════════════════


def save_context(
    ctx: Context, path: str, last_input: str = "", description: str = "", **metadata
) -> str:
    """
    保存 Context 到 checkpoint 文件。

    用法:
        ctx = Context()
        agent("input", ctx)
        save_context(ctx, "checkpoint.json", last_input="input")
    """
    cp = Checkpoint(
        context=ctx,
        last_input=last_input,
        description=description,
        metadata=metadata,
    )
    return cp.save(path)


def load_context(path: str) -> Context:
    """
    从 checkpoint 文件恢复 Context。

    用法:
        ctx = load_context("checkpoint.json")
        agent("continue from here", ctx)  # 带着之前的 trace 继续
    """
    cp = Checkpoint.load(path)
    return cp.context


def save_context_with_shared(
    ctx: Context,
    shared_memories: Dict,
    path: str,
    last_input: str = "",
    description: str = "",
) -> str:
    """
    保存 Context + SharedMemory 到 checkpoint。

    用法:
        shared = SharedMemory({"key": "value"})
        save_context_with_shared(ctx, {"main": shared.read_all()}, "cp.json")
    """
    cp = Checkpoint(
        context=ctx,
        shared_data=shared_memories,
        last_input=last_input,
        description=description,
    )
    return cp.save(path)


# ════════════════════════════════════════════════════════════
# CheckpointManager: 管理多个 checkpoint
# ════════════════════════════════════════════════════════════


class CheckpointManager:
    """
    Checkpoint 管理器: 自动保存、版本管理、清理。

    用法:
        mgr = CheckpointManager("./checkpoints/my_agent")
        mgr.save(ctx, "after step 1")
        mgr.save(ctx, "after step 2")
        mgr.list()     # 列出所有 checkpoint
        mgr.latest()   # 获取最新的
        mgr.rollback()  # 回退到上一个
    """

    def __init__(self, directory: str, max_checkpoints: int = 10):
        self.directory = Path(directory)
        self.max_checkpoints = max_checkpoints
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(
        self, ctx: Context, description: str = "", last_input: str = "", **metadata
    ) -> str:
        """保存新 checkpoint"""
        # 生成文件名: cp_001.json, cp_002.json, ...
        existing = self._list_files()
        idx = len(existing) + 1
        filename = f"cp_{idx:03d}.json"
        path = str(self.directory / filename)

        cp = Checkpoint(
            context=ctx,
            last_input=last_input,
            description=description or f"Checkpoint #{idx}",
            metadata=metadata,
        )
        cp.save(path)

        # 清理旧 checkpoint
        self._cleanup()

        return path

    def list(self) -> List[Dict]:
        """列出所有 checkpoint 的摘要"""
        results = []
        for path in self._list_files():
            try:
                cp = Checkpoint.load(str(path))
                results.append(
                    {
                        "file": path.name,
                        "path": str(path),
                        "description": cp.description,
                        "steps": cp.step_count,
                        "time": time.strftime(
                            "%Y-%m-%d %H:%M:%S", time.localtime(cp.timestamp)
                        ),
                        "last_input": cp.last_input[:50],
                    }
                )
            except Exception:
                continue
        return results

    def latest(self) -> Optional[Checkpoint]:
        """获取最新的 checkpoint"""
        files = self._list_files()
        if not files:
            return None
        return Checkpoint.load(str(files[-1]))

    def load(self, index: int = -1) -> Checkpoint:
        """
        加载指定 checkpoint。

        Args:
            index: 索引（-1=最新，0=最早）
        """
        files = self._list_files()
        if not files:
            raise CheckpointError("No checkpoints found")
        return Checkpoint.load(str(files[index]))

    def rollback(self) -> Optional[Context]:
        """回退到上一个 checkpoint，返回恢复的 Context"""
        files = self._list_files()
        if len(files) < 2:
            raise CheckpointError("Not enough checkpoints to rollback")
        # 删除最新的
        files[-1].unlink()
        # 加载倒数第二个
        cp = Checkpoint.load(str(files[-2]))
        return cp.context

    def _list_files(self) -> List[Path]:
        """列出所有 checkpoint 文件（按名称排序）"""
        return sorted(self.directory.glob("cp_*.json"))

    def _cleanup(self):
        """清理超出限制的旧 checkpoint"""
        files = self._list_files()
        while len(files) > self.max_checkpoints:
            files[0].unlink()
            files = files[1:]

    def __repr__(self):
        n = len(self._list_files())
        return f"CheckpointManager({self.directory}, {n} checkpoints)"
