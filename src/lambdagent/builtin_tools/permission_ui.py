"""
lambdagent.builtin_tools.permission_ui — CLI permission approval

Implements ToolGateway confirm_callback for terminal interaction.
"""
from __future__ import annotations

import sys
import threading
import time
from typing import Optional


class PermissionUI:
    """CLI permission approval for high-risk tool calls.

    Usage:
        from lambdagent.builtin_tools.permission_ui import PermissionUI
        pui = PermissionUI()
        gateway = ToolGateway(policy, confirm_callback=pui.confirm)
    """

    def __init__(self, timeout: float = 30.0, auto_deny: bool = True):
        self.timeout = timeout
        self.auto_deny = auto_deny
        self._always_allow: set = set()  # Tool patterns always allowed this session
        self._lock = threading.Lock()

    def confirm(self, tool_name: str, tool_input: str, risk_level: str, reason: str) -> bool:
        """Called by ToolGateway when a HIGH-risk tool call needs confirmation.

        Returns True to allow, False to deny.
        """
        # Check if always-allowed
        with self._lock:
            if tool_name in self._always_allow:
                return True

        # Display prompt
        input_preview = tool_input[:120].replace("\n", " ")
        print(f"\n{'='*60}")
        print(f"  PERMISSION REQUEST")
        print(f"  Tool:  {tool_name}")
        print(f"  Risk:  {risk_level}")
        print(f"  Input: {input_preview}")
        if reason:
            print(f"  Reason: {reason}")
        print(f"{'='*60}")
        print(f"  [y] Allow  [n] Deny  [a] Always allow '{tool_name}'  [timeout={self.timeout}s]")

        try:
            response = _input_with_timeout("> ", self.timeout)
            if response is None:
                print(f"  [{'Denied' if self.auto_deny else 'Allowed'}] (timeout)")
                return not self.auto_deny

            choice = response.strip().lower()
            if choice in ("y", "yes"):
                return True
            elif choice in ("a", "always"):
                with self._lock:
                    self._always_allow.add(tool_name)
                print(f"  [Always allowed] '{tool_name}' for this session")
                return True
            else:
                return False
        except (EOFError, KeyboardInterrupt):
            print(f"\n  [Denied] (interrupted)")
            return False

    def reset(self):
        """Reset all 'always allow' entries."""
        with self._lock:
            self._always_allow.clear()


def _input_with_timeout(prompt: str, timeout: float) -> Optional[str]:
    """Read input with timeout. Returns None on timeout."""
    result = [None]

    def _read():
        try:
            result[0] = input(prompt)
        except (EOFError, KeyboardInterrupt):
            pass

    thread = threading.Thread(target=_read, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    return result[0]
