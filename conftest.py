"""
Pytest configuration for lambdagent.

The package uses a flat layout where the repo root IS the `lambdagent` package
(via package-dir = {"lambdagent" = "."}). This causes pytest to treat the root
__init__.py as a discoverable module, which fails with relative-import errors.

`collect_ignore` here tells pytest to skip top-level .py files (which are part
of the installed package, not tests) and only collect from `tests/`.
"""
import os

# Skip every top-level .py file — they are part of the lambdagent package
# itself (installed via pip install -e .), not standalone tests.
_top_dir = os.path.dirname(__file__)
collect_ignore = [
    f for f in os.listdir(_top_dir)
    if f.endswith(".py") and f != "conftest.py"
]
collect_ignore_glob = [
    "agentruntime/*",
    "builtin_tools/*",
    "cli/*",
    "extractors/*",
    "fromconfig/*",
    "providers/*",
    "skillpacks/*",
    "examples/*",
    "mcp_server_package/*",
]
