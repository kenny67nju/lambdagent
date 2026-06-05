"""
lambdagent.cancellation — Hierarchical cancellation for agent execution

Equivalent to AbortController in JavaScript. Supports parent-child
relationships with WeakRef to prevent memory leaks.
"""

from __future__ import annotations

import threading
import weakref
from typing import Callable, List, Optional

from lambdagent.core import LambdagentError


# ============================================================
# Exceptions
# ============================================================

class CancelledError(LambdagentError):
    """Raised when an operation is cancelled via its CancellationToken."""

    def __init__(self, reason: str = "cancelled") -> None:
        self.reason = reason
        super().__init__(reason)


# ============================================================
# CancellationToken
# ============================================================

class CancellationToken:
    """Hierarchical, thread-safe cancellation token.

    Create a root token directly, or derive child tokens via :meth:`child`.
    Cancelling a parent automatically cancels all living children.
    Children are stored as weak references so garbage-collected tokens
    do not keep the tree alive.
    """

    def __init__(self, parent: CancellationToken = None) -> None:
        self._lock = threading.Lock()
        self._cancelled = False
        self._reason: Optional[str] = None
        self._children: List[weakref.ref[CancellationToken]] = []
        self._callbacks: List[Callable] = []
        self._parent = parent

    # -- public API --------------------------------------------------

    def cancel(self, reason: str = "user_cancelled") -> None:
        """Cancel this token and propagate to all children.

        Idempotent: subsequent calls after the first are no-ops.
        """
        callbacks_to_run: List[Callable] = []
        children_to_cancel: List[CancellationToken] = []

        with self._lock:
            if self._cancelled:
                return
            self._cancelled = True
            self._reason = reason
            callbacks_to_run = list(self._callbacks)
            children_to_cancel = [
                ref() for ref in self._children if ref() is not None
            ]

        # Run callbacks and propagate outside the lock to avoid deadlocks.
        for cb in callbacks_to_run:
            cb()

        for child in children_to_cancel:
            child.cancel(reason)

    @property
    def is_cancelled(self) -> bool:
        """Return ``True`` if this token has been cancelled."""
        return self._cancelled

    @property
    def reason(self) -> Optional[str]:
        """Return the cancellation reason, or ``None`` if still active."""
        return self._reason

    def check(self) -> None:
        """Raise :class:`CancelledError` if this token has been cancelled.

        Intended to be called at each reduction step so that long-running
        agent computations can be interrupted promptly.
        """
        if self._cancelled:
            raise CancelledError(self._reason or "cancelled")

    def child(self) -> CancellationToken:
        """Create a child token whose lifetime is bound to this parent.

        If this token is already cancelled, the child is born cancelled.
        """
        child_token = CancellationToken(parent=self)
        with self._lock:
            self._children.append(weakref.ref(child_token))
            if self._cancelled:
                child_token.cancel(self._reason or "cancelled")
        return child_token

    def on_cancel(self, callback: Callable) -> None:
        """Register a callback to invoke when this token is cancelled.

        If the token is already cancelled the callback fires immediately.
        """
        fire_now = False
        with self._lock:
            if self._cancelled:
                fire_now = True
            else:
                self._callbacks.append(callback)
        if fire_now:
            callback()


# ============================================================
# NullCancellationToken
# ============================================================

class NullCancellationToken(CancellationToken):
    """A cancellation token that never cancels.

    Useful as a default parameter so callers don't need to ``None``-check.
    All mutation methods are harmless no-ops.
    """

    def __init__(self) -> None:  # noqa: D107
        # Intentionally skip parent __init__; this token carries no state.
        self._lock = threading.Lock()
        self._cancelled = False
        self._reason = None
        self._children = []
        self._callbacks = []
        self._parent = None

    def cancel(self, reason: str = "user_cancelled") -> None:  # noqa: D102
        pass

    @property
    def is_cancelled(self) -> bool:  # noqa: D102
        return False

    @property
    def reason(self) -> Optional[str]:  # noqa: D102
        return None

    def check(self) -> None:  # noqa: D102
        pass

    def child(self) -> CancellationToken:  # noqa: D102
        return NullCancellationToken()

    def on_cancel(self, callback: Callable) -> None:  # noqa: D102
        pass
