"""agentruntime.memory_backend — Environment Gamma persistence"""
from __future__ import annotations
import time
import sqlite3
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple


class MemoryBackend(ABC):
    """
    Memory backend abstract interface.
    Lambda: Gamma storage implementation.
        read()  = lookup binding
        write() = extend environment (Gamma' = Gamma union {k: v})
        evict() = reclaim bindings
    """

    @abstractmethod
    def read_recent(self, n: int) -> List[Tuple[str, Any, str]]:
        """Read n most recent entries. Returns [(key, value, age_str), ...]"""
        ...

    @abstractmethod
    def read(self, key: str) -> Optional[Any]:
        """Read specific key."""
        ...

    @abstractmethod
    def write(self, key: str, value: Any) -> None:
        """Write (auto TTL and LRU eviction)."""
        ...

    @abstractmethod
    def auto_save(self, key: str, thought: str, action: str, observation: str) -> None:
        """Auto-extract and save from one execution step."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Clear all memory."""
        ...

    @staticmethod
    def create(config, namespace: str = "") -> "MemoryBackend":
        """Factory: create backend from config.

        S14: namespace parameter provides tenant/agent isolation.
        Format: "tenant_id:agent_id" → all keys prefixed, preventing cross-agent access.
        """
        strategy = getattr(config, 'strategy', 'local') if hasattr(config, 'strategy') else config.get('strategy', 'local')
        size = getattr(config, 'size', 20) if hasattr(config, 'size') else config.get('size', 20)
        ttl = getattr(config, 'ttl', 3600) if hasattr(config, 'ttl') else config.get('ttl', 3600)

        # S14: Extract namespace from config if not provided
        if not namespace:
            if hasattr(config, 'namespace'):
                namespace = config.namespace
            elif isinstance(config, dict):
                namespace = config.get('namespace', '')

        if strategy == "sqlite":
            db_path = getattr(config, 'db_path', '') if hasattr(config, 'db_path') else config.get('db_path', '')
            return SQLiteMemory(size, ttl, db_path or ":memory:", namespace=namespace)
        elif strategy == "redis":
            try:
                redis_url = getattr(config, 'redis_url', '') if hasattr(config, 'redis_url') else config.get('redis_url', '')
                return RedisMemory(size, ttl, redis_url, namespace=namespace)
            except Exception:
                return LocalMemory(size, ttl, namespace=namespace)
        else:
            return LocalMemory(size, ttl, namespace=namespace)


class LocalMemory(MemoryBackend):
    """
    In-process memory backend. OrderedDict + LRU + TTL.
    Lambda: Gamma in Python heap. Lost on process exit.

    S14: namespace isolates memory by tenant/agent.
    """

    def __init__(self, size: int = 20, ttl: int = 3600, namespace: str = ""):
        self.size = size
        self.ttl = ttl
        self.namespace = namespace
        self._store = OrderedDict()  # key -> (value, timestamp)

    def _ns_key(self, key: str) -> str:
        """S14: Prefix key with namespace for isolation."""
        if self.namespace:
            return f"{self.namespace}:{key}"
        return key

    def read_recent(self, n: int) -> List[Tuple[str, Any, str]]:
        self._evict_expired()
        items = list(self._store.items())[-n:]
        result = []
        now = time.time()
        for key, (value, ts) in reversed(items):
            age = int(now - ts)
            if age < 60:
                age_str = f"{age}s ago"
            elif age < 3600:
                age_str = f"{age // 60}min ago"
            else:
                age_str = f"{age // 3600}h ago"
            result.append((key, value, age_str))
        return result

    def read(self, key: str) -> Optional[Any]:
        self._evict_expired()
        if key in self._store:
            self._store.move_to_end(key)
            return self._store[key][0]
        return None

    def write(self, key: str, value: Any) -> None:
        self._store[key] = (value, time.time())
        self._store.move_to_end(key)
        # LRU eviction
        while len(self._store) > self.size and self.size > 0:
            self._store.popitem(last=False)

    def auto_save(self, key: str, thought: str, action: str, observation: str) -> None:
        summary = f"[{action}] {observation[:200]}" if observation else f"[thought] {thought[:200]}"
        self.write(key, summary)

    def clear(self) -> None:
        self._store.clear()

    def _evict_expired(self):
        if self.ttl <= 0:
            return
        now = time.time()
        expired = [k for k, (v, ts) in self._store.items() if now - ts > self.ttl]
        for k in expired:
            del self._store[k]


class SQLiteMemory(MemoryBackend):
    """SQLite memory backend. Persistent on disk."""

    def __init__(self, size: int = 20, ttl: int = 3600, db_path: str = ":memory:"):
        self.size = size
        self.ttl = ttl
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS memory (
                key TEXT PRIMARY KEY,
                value TEXT,
                timestamp REAL
            )
        """)
        self.conn.commit()

    def read_recent(self, n: int) -> List[Tuple[str, Any, str]]:
        self._evict_expired()
        rows = self.conn.execute(
            "SELECT key, value, timestamp FROM memory ORDER BY timestamp DESC LIMIT ?", (n,)
        ).fetchall()
        result = []
        now = time.time()
        for key, value, ts in rows:
            age = int(now - ts)
            age_str = f"{age}s ago" if age < 60 else f"{age // 60}min ago" if age < 3600 else f"{age // 3600}h ago"
            result.append((key, value, age_str))
        return result

    def read(self, key: str) -> Optional[Any]:
        row = self.conn.execute("SELECT value FROM memory WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def write(self, key: str, value: Any) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO memory (key, value, timestamp) VALUES (?, ?, ?)",
            (key, str(value), time.time())
        )
        # LRU eviction
        count = self.conn.execute("SELECT COUNT(*) FROM memory").fetchone()[0]
        if count > self.size and self.size > 0:
            excess = count - self.size
            self.conn.execute(
                "DELETE FROM memory WHERE key IN (SELECT key FROM memory ORDER BY timestamp ASC LIMIT ?)",
                (excess,)
            )
        self.conn.commit()

    def auto_save(self, key, thought, action, observation):
        summary = f"[{action}] {observation[:200]}" if observation else f"[thought] {thought[:200]}"
        self.write(key, summary)

    def clear(self):
        self.conn.execute("DELETE FROM memory")
        self.conn.commit()

    def _evict_expired(self):
        if self.ttl <= 0:
            return
        cutoff = time.time() - self.ttl
        self.conn.execute("DELETE FROM memory WHERE timestamp < ?", (cutoff,))


class RedisMemory(MemoryBackend):
    """Redis memory backend. Cross-process, cross-machine."""

    def __init__(self, size: int = 20, ttl: int = 3600, redis_url: str = ""):
        self.size = size
        self.ttl = ttl
        try:
            import redis
            self.r = redis.from_url(redis_url or "redis://localhost:6379/0")
            self.r.ping()
        except Exception as e:
            raise RuntimeError(f"Redis connection failed: {e}")
        self.prefix = "lambdagent:memory:"

    def read_recent(self, n: int) -> List[Tuple[str, Any, str]]:
        keys = self.r.zrevrange(self.prefix + "index", 0, n - 1, withscores=True)
        result = []
        now = time.time()
        for key_bytes, ts in keys:
            key = key_bytes.decode() if isinstance(key_bytes, bytes) else key_bytes
            value = self.r.get(self.prefix + key)
            if value:
                value = value.decode() if isinstance(value, bytes) else value
                age = int(now - ts)
                age_str = f"{age}s ago" if age < 60 else f"{age // 60}min ago"
                result.append((key, value, age_str))
        return result

    def read(self, key: str) -> Optional[Any]:
        val = self.r.get(self.prefix + key)
        return val.decode() if val else None

    def write(self, key: str, value: Any) -> None:
        self.r.set(self.prefix + key, str(value))
        if self.ttl > 0:
            self.r.expire(self.prefix + key, self.ttl)
        self.r.zadd(self.prefix + "index", {key: time.time()})
        # LRU
        count = self.r.zcard(self.prefix + "index")
        if count > self.size:
            old_keys = self.r.zrange(self.prefix + "index", 0, count - self.size - 1)
            for k in old_keys:
                k_str = k.decode() if isinstance(k, bytes) else k
                self.r.delete(self.prefix + k_str)
            self.r.zremrangebyrank(self.prefix + "index", 0, count - self.size - 1)

    def auto_save(self, key, thought, action, observation):
        summary = f"[{action}] {observation[:200]}" if observation else f"[thought] {thought[:200]}"
        self.write(key, summary)

    def clear(self):
        keys = self.r.keys(self.prefix + "*")
        if keys:
            self.r.delete(*keys)
