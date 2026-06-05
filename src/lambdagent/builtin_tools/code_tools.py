"""
lambdagent.builtin_tools.code_tools — Programming assistant tools

CodeSearch    λx. semantic_search(query, language)
ProjectMap    λx. directory_tree(path, depth)
RunTests      λx. test_framework(target)
"""
from __future__ import annotations

import json
import os
import subprocess
from typing import Any, Dict, Optional

from .._shell_compat import run_shell as _run_shell


# ════════════════════════════════════════════════════════════
# A11: CodeSearch — semantic code search
# ════════════════════════════════════════════════════════════

# Language-aware patterns for common constructs
_LANG_PATTERNS = {
    "python": {
        "class": r"^\s*class\s+{query}",
        "function": r"^\s*def\s+{query}\s*\(",
        "import": r"^\s*(?:from\s+\S+\s+)?import\s+.*{query}",
        "variable": r"^\s*{query}\s*=",
    },
    "typescript": {
        "class": r"(?:export\s+)?class\s+{query}",
        "function": r"(?:export\s+)?(?:async\s+)?function\s+{query}\s*\(",
        "interface": r"(?:export\s+)?interface\s+{query}",
        "type": r"(?:export\s+)?type\s+{query}\s*=",
        "const": r"(?:export\s+)?const\s+{query}\s*[=:]",
    },
    "javascript": {
        "class": r"(?:export\s+)?class\s+{query}",
        "function": r"(?:export\s+)?(?:async\s+)?function\s+{query}\s*\(",
        "const": r"(?:export\s+)?const\s+{query}\s*[=:]",
    },
    "go": {
        "function": r"func\s+(?:\([^)]*\)\s+)?{query}\s*\(",
        "type": r"type\s+{query}\s+(?:struct|interface)",
        "const": r"(?:const|var)\s+{query}\s",
    },
    "java": {
        "class": r"(?:public|private|protected)?\s*class\s+{query}",
        "function": r"(?:public|private|protected)\s+\S+\s+{query}\s*\(",
        "interface": r"interface\s+{query}",
    },
    "rust": {
        "function": r"(?:pub\s+)?fn\s+{query}\s*[<(]",
        "struct": r"(?:pub\s+)?struct\s+{query}",
        "enum": r"(?:pub\s+)?enum\s+{query}",
        "trait": r"(?:pub\s+)?trait\s+{query}",
        "impl": r"impl(?:<[^>]*>)?\s+{query}",
    },
}

_LANG_EXTENSIONS = {
    "python": "*.py", "typescript": "*.{ts,tsx}", "javascript": "*.{js,jsx}",
    "go": "*.go", "java": "*.java", "rust": "*.rs",
}


class CodeSearchSchema:
    def __init__(self, query: str, language: str = "", search_type: str = "",
                 path: str = ".", max_results: int = 20):
        if not query:
            raise ValueError("query is required")
        self.query = query
        self.language = language.lower()
        self.search_type = search_type  # class, function, import, etc.
        self.path = os.path.abspath(path)
        self.max_results = min(max(1, max_results), 100)

    def dict(self):
        return {"query": self.query, "language": self.language,
                "search_type": self.search_type, "path": self.path,
                "max_results": self.max_results}


def code_search(input_val: Any) -> str:
    """Semantic code search — finds definitions, references, imports."""
    params = _parse_input(input_val, CodeSearchSchema)
    query = params["query"]
    language = params.get("language", "")
    search_type = params.get("search_type", "")
    path = params.get("path", ".")
    max_results = params.get("max_results", 20)

    # Build search pattern
    if language and search_type and language in _LANG_PATTERNS:
        patterns = _LANG_PATTERNS[language]
        if search_type in patterns:
            pattern = patterns[search_type].format(query=_escape_regex(query))
        else:
            pattern = _escape_regex(query)
    elif language and language in _LANG_PATTERNS:
        # Try all patterns for this language
        results = []
        for stype, pat_template in _LANG_PATTERNS[language].items():
            pattern = pat_template.format(query=_escape_regex(query))
            hits = _do_search(pattern, path, language, max_results // 2)
            for h in hits:
                results.append(f"[{stype}] {h}")
        if results:
            return "\n".join(results[:max_results])
        # Fall back to plain search
        pattern = _escape_regex(query)
    else:
        pattern = _escape_regex(query)

    # Execute search
    results = _do_search(pattern, path, language, max_results)
    if not results:
        return f"[NO_MATCH] '{query}' not found in {path}"
    return "\n".join(results)


def _escape_regex(s: str) -> str:
    """Escape special regex chars but keep it usable."""
    import re
    return re.escape(s).replace(r"\ ", " ")


def _do_search(pattern: str, path: str, language: str, max_results: int) -> list:
    """Execute search with rg or Python fallback."""
    # Try ripgrep
    rg = _find_rg()
    if rg:
        cmd = [rg, "-n", "--no-heading", "-m", str(max_results)]
        if language and language in _LANG_EXTENSIONS:
            cmd.extend(["--glob", _LANG_EXTENSIONS[language]])
        cmd.extend([pattern, path])
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if result.stdout.strip():
                return result.stdout.strip().split("\n")[:max_results]
        except (subprocess.TimeoutExpired, Exception):
            pass

    # Python fallback
    import re
    import glob as glob_mod
    try:
        regex = re.compile(pattern)
    except re.error:
        regex = re.compile(re.escape(pattern))

    ext = _LANG_EXTENSIONS.get(language, "*.*")
    files = glob_mod.glob(os.path.join(path, "**", ext), recursive=True)
    ignore = {".git", "node_modules", "__pycache__", ".venv", "dist", "build"}
    results = []
    for fpath in files:
        if any(p in fpath.split(os.sep) for p in ignore):
            continue
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f, 1):
                    if regex.search(line):
                        rel = os.path.relpath(fpath, path)
                        results.append(f"{rel}:{i}:{line.rstrip()}")
                        if len(results) >= max_results:
                            return results
        except Exception:
            continue
    return results


def _find_rg() -> Optional[str]:
    """Locate the ripgrep binary across platforms.

    Order: PATH (works on Linux/macOS/Windows via shutil.which) → known
    homebrew/macports/winget locations. Returns None if rg is unavailable;
    callers fall back to a slower Python regex walk.
    """
    import shutil
    found = shutil.which("rg")
    if found:
        return found
    # Per-platform fallback locations (Brew, MacPorts, Chocolatey, scoop)
    candidates = [
        "/usr/local/bin/rg",            # Linux / Intel-mac Brew
        "/opt/homebrew/bin/rg",         # Apple-silicon Brew
        "/opt/local/bin/rg",            # MacPorts
        "C:/ProgramData/chocolatey/bin/rg.exe",   # Chocolatey
        "C:/tools/ripgrep/rg.exe",                # scoop default
    ]
    for cmd in candidates:
        try:
            subprocess.run([cmd, "--version"], capture_output=True, timeout=3)
            return cmd
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            continue
    return None


# ════════════════════════════════════════════════════════════
# A12: ProjectMap — project structure overview
# ════════════════════════════════════════════════════════════

_KEY_FILES = {"README.md", "README.rst", "readme.md", "package.json", "pyproject.toml",
              "setup.py", "Cargo.toml", "go.mod", "Makefile", "Dockerfile",
              "docker-compose.yml", ".github", "tsconfig.json", "requirements.txt"}

_IGNORE_DIRS_MAP = {".git", "node_modules", "__pycache__", ".venv", "venv",
                    ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
                    ".eggs", ".cache", "coverage", ".next", ".nuxt"}


class ProjectMapSchema:
    def __init__(self, path: str = ".", depth: int = 3, include_summary: bool = False):
        self.path = os.path.abspath(path)
        self.depth = min(max(1, depth), 6)
        self.include_summary = include_summary

    def dict(self):
        return {"path": self.path, "depth": self.depth, "include_summary": self.include_summary}


def project_map(input_val: Any) -> str:
    """Generate project structure overview."""
    params = _parse_input(input_val, ProjectMapSchema)
    path = params.get("path", ".")
    depth = params.get("depth", 3)

    if not os.path.isdir(path):
        return f"[ERROR] Not a directory: {path}"

    lines = []
    # Project name
    project_name = os.path.basename(path) or path
    lines.append(f"Project: {project_name}")
    lines.append("")

    # Key files
    key_found = []
    for name in sorted(_KEY_FILES):
        full = os.path.join(path, name)
        if os.path.exists(full):
            if os.path.isfile(full):
                size = os.path.getsize(full)
                key_found.append(f"  {name} ({size:,} bytes)")
            else:
                key_found.append(f"  {name}/")
    if key_found:
        lines.append("Key files:")
        lines.extend(key_found)
        lines.append("")

    # Directory tree
    lines.append("Structure:")
    _tree(path, "", depth, 0, lines)
    lines.append("")

    # File statistics
    stats = _count_files(path)
    if stats:
        lines.append("File statistics:")
        for ext, count in sorted(stats.items(), key=lambda x: -x[1])[:15]:
            lines.append(f"  {ext}: {count} files")

    return "\n".join(lines)


def _tree(path: str, prefix: str, max_depth: int, current_depth: int, lines: list):
    if current_depth >= max_depth:
        return
    try:
        entries = sorted(os.listdir(path))
    except PermissionError:
        return

    dirs = [e for e in entries if os.path.isdir(os.path.join(path, e)) and e not in _IGNORE_DIRS_MAP and not e.startswith(".")]
    files = [e for e in entries if os.path.isfile(os.path.join(path, e))]

    # Show files (limit to 10 per directory)
    for f in files[:10]:
        lines.append(f"{prefix}├── {f}")
    if len(files) > 10:
        lines.append(f"{prefix}├── ... ({len(files) - 10} more files)")

    # Show directories
    for i, d in enumerate(dirs):
        is_last = i == len(dirs) - 1
        connector = "└── " if is_last else "├── "
        child_prefix = "    " if is_last else "│   "
        lines.append(f"{prefix}{connector}{d}/")
        _tree(os.path.join(path, d), prefix + child_prefix, max_depth, current_depth + 1, lines)


def _count_files(path: str) -> dict:
    stats = {}
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in _IGNORE_DIRS_MAP]
        for f in files:
            ext = os.path.splitext(f)[1] or "(no ext)"
            stats[ext] = stats.get(ext, 0) + 1
    return stats


# ════════════════════════════════════════════════════════════
# A13: RunTests — test runner
# ════════════════════════════════════════════════════════════

_FRAMEWORK_DETECTORS = [
    ("pytest", ["pyproject.toml", "setup.cfg", "pytest.ini", "conftest.py"], "python -m pytest"),
    ("jest", ["jest.config.js", "jest.config.ts", "package.json"], "npx jest"),
    ("go", ["go.mod"], "go test ./..."),
    ("cargo", ["Cargo.toml"], "cargo test"),
]


class RunTestsSchema:
    def __init__(self, target: str = "", framework: str = "", verbose: bool = False,
                 working_dir: str = ""):
        self.target = target
        self.framework = framework
        self.verbose = verbose
        self.working_dir = os.path.abspath(working_dir) if working_dir else os.getcwd()

    def dict(self):
        return {"target": self.target, "framework": self.framework,
                "verbose": self.verbose, "working_dir": self.working_dir}


def run_tests(input_val: Any) -> str:
    """Auto-detect test framework and run tests."""
    params = _parse_input(input_val, RunTestsSchema)
    target = params.get("target", "")
    framework = params.get("framework", "")
    verbose = params.get("verbose", False)
    cwd = params.get("working_dir", os.getcwd())

    # Auto-detect framework
    if not framework:
        framework = _detect_framework(cwd)
    if not framework:
        return "[ERROR] No test framework detected. Specify --framework."

    # Build command
    cmd = _build_test_command(framework, target, verbose)

    # Execute
    try:
        result = _run_shell(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=300,
        )
        output = result.stdout
        if result.stderr:
            output += "\n[STDERR]\n" + result.stderr

        # Parse results
        summary = _parse_test_output(output, framework, result.returncode)
        return f"{summary}\n\n--- Full output ---\n{output[-5000:]}"

    except subprocess.TimeoutExpired:
        return f"[TIMEOUT] Tests timed out after 300s"
    except Exception as e:
        return f"[ERROR] {e}"


def _detect_framework(cwd: str) -> str:
    for name, markers, _ in _FRAMEWORK_DETECTORS:
        for marker in markers:
            if os.path.exists(os.path.join(cwd, marker)):
                if name == "jest" and marker == "package.json":
                    try:
                        with open(os.path.join(cwd, marker)) as f:
                            pkg = json.load(f)
                        if "jest" not in str(pkg.get("devDependencies", {})):
                            continue
                    except Exception:
                        continue
                return name
    return ""


def _build_test_command(framework: str, target: str, verbose: bool) -> str:
    base = {"pytest": "python -m pytest", "jest": "npx jest", "go": "go test",
            "cargo": "cargo test"}.get(framework, framework)
    parts = [base]
    if verbose:
        if framework == "pytest":
            parts.append("-v")
        elif framework == "jest":
            parts.append("--verbose")
    if target:
        parts.append(target)
    elif framework == "go":
        parts.append("./...")
    return " ".join(parts)


def _parse_test_output(output: str, framework: str, returncode: int) -> str:
    """Parse test output into structured summary."""
    status = "PASSED" if returncode == 0 else "FAILED"
    lines = output.split("\n")

    if framework == "pytest":
        for line in reversed(lines):
            if "passed" in line or "failed" in line or "error" in line:
                return f"[{status}] {line.strip()}"
    elif framework == "jest":
        for line in reversed(lines):
            if "Tests:" in line or "Test Suites:" in line:
                return f"[{status}] {line.strip()}"
    elif framework == "go":
        pass_count = sum(1 for l in lines if l.startswith("ok"))
        fail_count = sum(1 for l in lines if l.startswith("FAIL"))
        return f"[{status}] {pass_count} passed, {fail_count} failed"

    return f"[{status}] exit code {returncode}"


# ════════════════════════════════════════════════════════════
# Shared
# ════════════════════════════════════════════════════════════

def _parse_input(input_val: Any, schema_cls) -> dict:
    if isinstance(input_val, dict):
        data = input_val
    elif isinstance(input_val, str):
        try:
            data = json.loads(input_val)
        except (json.JSONDecodeError, ValueError):
            data = {"query": input_val} if "Search" in schema_cls.__name__ else {"path": input_val}
    else:
        data = {}
    try:
        validated = schema_cls(**data)
        return validated.dict()
    except (TypeError, ValueError) as e:
        raise ValueError(f"[VALIDATION_ERROR] {schema_cls.__name__}: {e}")
