"""
lambdagent.rate_limiter — Token bucket rate limiter for LLM API calls

Prevents API rate limit errors by throttling requests.
Supports both sync and async usage.
"""
from __future__ import annotations

import asyncio
import threading
import time


class RateLimiter:
    """Token bucket rate limiter.

    Usage:
        limiter = RateLimiter(requests_per_minute=60)
        limiter.acquire()  # blocks if rate exceeded
        # or
        await limiter.aacquire()  # async version
    """

    def __init__(self, requests_per_minute: int = 60):
        self.rpm = requests_per_minute
        self._tokens = float(requests_per_minute)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()
        self._async_lock = None  # Lazy init
        self._total_acquired = 0
        self._total_waited_ms = 0

    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self.rpm, self._tokens + elapsed * (self.rpm / 60.0))
        self._last_refill = now

    def acquire(self) -> float:
        """Acquire one token. Blocks if necessary. Returns wait time in seconds."""
        with self._lock:
            self._refill()
            if self._tokens >= 1:
                self._tokens -= 1
                self._total_acquired += 1
                return 0.0

            # Calculate wait time
            wait = (1 - self._tokens) / (self.rpm / 60.0)

        time.sleep(wait)

        with self._lock:
            self._refill()
            self._tokens = max(0, self._tokens - 1)
            self._total_acquired += 1
            self._total_waited_ms += wait * 1000

        return wait

    async def aacquire(self) -> float:
        """Async version of acquire."""
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()

        async with self._async_lock:
            self._refill()
            if self._tokens >= 1:
                self._tokens -= 1
                self._total_acquired += 1
                return 0.0

            wait = (1 - self._tokens) / (self.rpm / 60.0)

        await asyncio.sleep(wait)

        async with self._async_lock:
            self._refill()
            self._tokens = max(0, self._tokens - 1)
            self._total_acquired += 1
            self._total_waited_ms += wait * 1000

        return wait

    @property
    def available_tokens(self) -> float:
        with self._lock:
            self._refill()
            return self._tokens

    def summary(self) -> str:
        return (
            f"RateLimiter({self.rpm} rpm): "
            f"{self._total_acquired} acquired, "
            f"{self._total_waited_ms:.0f}ms total wait"
        )
