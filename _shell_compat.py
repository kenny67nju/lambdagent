"""
lambdagent._shell_compat — Cross-platform shell resolution helper.

When ``shell=True`` is passed to ``subprocess.run``/``Popen``, Python picks
the platform's default shell:

  * POSIX (Linux / macOS): ``/bin/sh``
  * Windows: ``cmd.exe``

``cmd.exe`` does not understand bash-style continuations (``\``), variable
expansion (``$VAR``), single-quote quoting, ``&&`` short-circuiting in the
same way, or heredocs. To keep user-supplied shell commands portable, we
look for Git-for-Windows ``bash.exe`` first and pass it as the ``executable``
argument so the same command string works on all three OSes.

Usage::

    from .._shell_compat import resolve_shell

    subprocess.run(
        cmd, shell=True, executable=resolve_shell(),
        capture_output=True, text=True,
    )
"""
from __future__ import annotations

import os
import platform
import shutil
from typing import Optional

_IS_WINDOWS = platform.system() == "Windows"
_CACHED: Optional[str] = None    # "" sentinel = searched, not found


def resolve_shell() -> Optional[str]:
    """Return the path to pass as subprocess ``executable=`` for shell=True.

    Returns ``None`` on POSIX (let subprocess use ``/bin/sh``) and on
    Windows when bash is unavailable (fall back to ``cmd.exe``). Returns the
    absolute path to ``bash.exe`` on Windows when Git-Bash is installed.

    Result is cached after the first probe.
    """
    global _CACHED
    if not _IS_WINDOWS:
        return None
    if _CACHED is None:
        for candidate in [
            shutil.which("bash"),
            r"C:\Program Files\Git\bin\bash.exe",
            r"C:\Program Files (x86)\Git\bin\bash.exe",
        ]:
            if candidate and os.path.isfile(candidate):
                _CACHED = candidate
                break
        else:
            _CACHED = ""
    return _CACHED or None
