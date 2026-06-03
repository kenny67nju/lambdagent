"""
lambdagent.retry — Resilience primitives for β-reduction I/O

Provides retry with exponential backoff, configurable timeouts,
and circuit breaker pattern for LLM/MCP calls.
"""
from __future__ import annotations

import asyncio
import logging
import random
import threading
import time
from dataclasses import dataclass, field

from lambdagent.core import LambdagentError

logger = logging.getLogger(__name__)


class CircuitOpenError(LambdagentError):
    """Raised when a circuit breaker is open and rejecting calls."""


@dataclass
class RetryPolicy:
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    jitter: bool = True
    retryable_errors: tuple = (TimeoutError, ConnectionError, OSError)


def _backoff_delay(attempt: int, policy: RetryPolicy) -> float:
    delay = min(policy.base_delay * (2 ** attempt), policy.max_delay)
    if policy.jitter:
        delay = random.uniform(0, delay)
    return delay


async def with_retry(fn, policy: RetryPolicy | None = None, timeout: float | None = None):
    """Retry an async callable with exponential backoff and optional timeout."""
    if policy is None:
        policy = RetryPolicy()

    last_exc: Exception | None = None
    for attempt in range(policy.max_attempts):
        try:
            coro = fn()
            if timeout is not None:
                return await asyncio.wait_for(coro, timeout=timeout)
            return await coro
        except policy.retryable_errors as exc:
            last_exc = exc
            if attempt < policy.max_attempts - 1:
                delay = _backoff_delay(attempt, policy)
                logger.warning(
                    "Retry %d/%d after %.2fs: %s",
                    attempt + 1, policy.max_attempts, delay, exc,
                )
                await asyncio.sleep(delay)
        except asyncio.TimeoutError as exc:
            last_exc = exc
            if attempt < policy.max_attempts - 1:
                delay = _backoff_delay(attempt, policy)
                logger.warning(
                    "Retry %d/%d after timeout (%.2fs backoff): %s",
                    attempt + 1, policy.max_attempts, delay, exc,
                )
                await asyncio.sleep(delay)

    raise last_exc  # type: ignore[misc]


def with_retry_sync(fn, policy: RetryPolicy | None = None, timeout: float | None = None):
    """Synchronous retry with exponential backoff and optional timeout.

    When *timeout* is set, each individual attempt is bounded by a
    ``signal``-free wall-clock check (the callable itself must respect
    the timeout or be naturally short-lived).
    """
    if policy is None:
        policy = RetryPolicy()

    last_exc: Exception | None = None
    for attempt in range(policy.max_attempts):
        try:
            if timeout is not None:
                start = time.monotonic()
                result = fn()
                elapsed = time.monotonic() - start
                if elapsed > timeout:
                    raise TimeoutError(f"Call took {elapsed:.2f}s (limit {timeout}s)")
                return result
            return fn()
        except policy.retryable_errors as exc:
            last_exc = exc
            if attempt < policy.max_attempts - 1:
                delay = _backoff_delay(attempt, policy)
                logger.warning(
                    "Retry %d/%d after %.2fs: %s",
                    attempt + 1, policy.max_attempts, delay, exc,
                )
                time.sleep(delay)

    raise last_exc  # type: ignore[misc]


class CircuitBreaker:
    """Thread-safe circuit breaker for external service calls.

    States: closed (normal) -> open (failing) -> half_open (probing).
    """

    _CLOSED = "closed"
    _OPEN = "open"
    _HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
        name: str = "",
    ) -> None:
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.name = name or id(self)

        self._lock = threading.Lock()
        self._state = self._CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0.0

    @property
    def state(self) -> str:
        with self._lock:
            self._maybe_transition()
            return self._state

    def _maybe_transition(self) -> None:
        if self._state == self._OPEN:
            if time.monotonic() - self._last_failure_time >= self.reset_timeout:
                self._state = self._HALF_OPEN

    def _record_success(self) -> None:
        with self._lock:
            self._failure_count = 0
            self._state = self._CLOSED

    def _record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            if self._failure_count >= self.failure_threshold:
                self._state = self._OPEN
                logger.error("Circuit breaker '%s' opened after %d failures", self.name, self._failure_count)

    def _check_state(self) -> None:
        with self._lock:
            self._maybe_transition()
            if self._state == self._OPEN:
                raise CircuitOpenError(
                    f"Circuit breaker '{self.name}' is open — call rejected"
                )

    async def call(self, fn):
        """Execute *fn* (an async callable) through the circuit breaker."""
        self._check_state()
        try:
            result = await fn()
        except Exception:
            self._record_failure()
            raise
        self._record_success()
        return result

    def call_sync(self, fn):
        """Execute *fn* (a sync callable) through the circuit breaker."""
        self._check_state()
        try:
            result = fn()
        except Exception:
            self._record_failure()
            raise
        self._record_success()
        return result
