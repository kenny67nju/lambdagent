"""
lambdagent._shell_compat — Cross-platform shell helpers.

Two related concerns:

1. ``run_shell(command, **kwargs)`` — invoke a shell command portably. On
   POSIX uses ``shell=True`` (``/bin/sh``). On Windows uses ``shell=False``
   plus an explicit ``[bash.exe, "-c", command]`` argv when Git-for-Windows
   ``bash.exe`` is available — this is what lets bash-style commands
   (pipes, ``$VAR``, single-quote quoting, ``&&``, heredocs) work on
   Windows runners. When Git-Bash isn't installed, falls back to
   ``cmd.exe`` semantics — user-supplied bash syntax will then likely fail,
   so we recommend installing Git for Windows in the README.

   We don't pass ``executable=bash.exe`` with ``shell=True``: Python on
   Windows builds the command line as ``"<executable> <command>"`` without
   re-quoting, so a path with spaces ("C:\\Program Files\\...") gets split
   by CreateProcess and the exec fails with EXIT:127.

2. ``resolve_bash()`` — return the discovered ``bash.exe`` path on Windows
   (or ``None``) so callers that want to construct their own argv can do so.

Usage::

    from .._shell_compat import run_shell
    result = run_shell("echo hello", capture_output=True, text=True)
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
from typing import Any, Optional

_IS_WINDOWS = platform.system() == "Windows"
_CACHED_BASH: Optional[str] = None    # "" sentinel = searched, not found


def resolve_bash() -> Optional[str]:
    """Return the path to ``bash.exe`` on Windows, or ``None``.

    Looks at PATH first, then Git-for-Windows default install locations.
    Returns ``None`` on POSIX (where ``/bin/sh`` is always available).
    Result is cached after the first probe.
    """
    global _CACHED_BASH
    if not _IS_WINDOWS:
        return None
    if _CACHED_BASH is None:
        for candidate in [
            shutil.which("bash"),
            r"C:\Program Files\Git\bin\bash.exe",
            r"C:\Program Files (x86)\Git\bin\bash.exe",
        ]:
            if candidate and os.path.isfile(candidate):
                _CACHED_BASH = candidate
                break
        else:
            _CACHED_BASH = ""
    return _CACHED_BASH or None


def run_shell(command: str, **kwargs: Any) -> subprocess.CompletedProcess:
    """Run ``command`` portably via ``subprocess.run``.

    On POSIX: ``subprocess.run(command, shell=True, **kwargs)``.
    On Windows: ``subprocess.run([bash, "-c", command], shell=False, **kwargs)``
    when Git-Bash is available; else ``shell=True`` (cmd.exe).
    """
    bash = resolve_bash()
    if bash:
        return subprocess.run([bash, "-c", command], shell=False, **kwargs)
    return subprocess.run(command, shell=True, **kwargs)


def popen_shell(command: str, **kwargs: Any) -> subprocess.Popen:
    """Background variant of :func:`run_shell` — returns the live ``Popen``."""
    bash = resolve_bash()
    if bash:
        return subprocess.Popen([bash, "-c", command], shell=False, **kwargs)
    return subprocess.Popen(command, shell=True, **kwargs)
