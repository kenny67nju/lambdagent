"""
lambdagent.builtin_tools.file_tools — File operation tools

ReadFile   λx. read(path, offset, limit)
EditFile   λx. replace(path, old, new)
WriteFile  λx. write(path, content)
ListFiles  λx. glob(pattern)
SearchContent  λx. grep(pattern, path)
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from typing import Any, Dict, Optional


# ════════════════════════════════════════════════════════════
# A01: ReadFile
# ════════════════════════════════════════════════════════════

class ReadFileSchema:
    """Validate ReadFile input."""
    def __init__(self, file_path: str, offset: int = 0, limit: int = 2000, pages: str = ""):
        if not file_path or not isinstance(file_path, str):
            raise ValueError("file_path is required and must be a string")
        if not os.path.isabs(file_path):
            file_path = os.path.abspath(file_path)
        self.file_path = file_path
        self.offset = max(0, offset)
        self.limit = min(max(1, limit), 10000)
        self.pages = pages

    def dict(self):
        return {"file_path": self.file_path, "offset": self.offset,
                "limit": self.limit, "pages": self.pages}


def read_file(input_val: Any) -> str:
    """Read file contents with line numbers, offset/limit, encoding detection."""
    params = _parse_input(input_val, ReadFileSchema)
    path = params["file_path"]
    offset = params.get("offset", 0)
    limit = params.get("limit", 2000)

    if not os.path.exists(path):
        return f"[ERROR] File not found: {path}"

    if os.path.isdir(path):
        return f"[ERROR] {path} is a directory, not a file. Use ListFiles to list directory contents."

    # Binary file detection
    ext = os.path.splitext(path)[1].lower()

    # PDF
    if ext == ".pdf":
        return _read_pdf(path, params.get("pages", ""))

    # Image
    if ext in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".webp"):
        size = os.path.getsize(path)
        return f"[IMAGE] {path} ({ext}, {size:,} bytes). Use a multimodal LLM to describe this image."

    # Jupyter Notebook
    if ext == ".ipynb":
        return _read_notebook(path)

    # Text file
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        try:
            with open(path, "r", encoding="latin-1") as f:
                lines = f.readlines()
        except Exception:
            size = os.path.getsize(path)
            return f"[BINARY] {path} ({size:,} bytes). Cannot read as text."

    total = len(lines)
    if total == 0:
        return f"[EMPTY] {path} (0 lines)"

    selected = lines[offset:offset + limit]
    result_lines = []
    for i, line in enumerate(selected, start=offset + 1):
        result_lines.append(f"{i}\t{line.rstrip()}")

    result = "\n".join(result_lines)
    if offset + limit < total:
        result += f"\n\n... [{total - offset - limit} more lines. Use offset={offset + limit} to continue.]"

    return result


def _read_pdf(path: str, pages: str) -> str:
    """Read PDF file. Requires PyPDF2 or pdfplumber."""
    try:
        import PyPDF2
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            total_pages = len(reader.pages)

            if pages:
                page_range = _parse_page_range(pages, total_pages)
            else:
                page_range = range(min(total_pages, 10))

            text_parts = []
            for i in page_range:
                if 0 <= i < total_pages:
                    text = reader.pages[i].extract_text() or ""
                    text_parts.append(f"--- Page {i + 1} ---\n{text}")

            return "\n\n".join(text_parts) if text_parts else f"[PDF] {path} ({total_pages} pages, no text extracted)"
    except ImportError:
        return f"[PDF] {path}. Install PyPDF2 to read: pip install PyPDF2"
    except Exception as e:
        return f"[PDF_ERROR] {path}: {e}"


def _read_notebook(path: str) -> str:
    """Read Jupyter Notebook (.ipynb) — show all cells with outputs."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        cells = nb.get("cells", [])
        parts = []
        for i, cell in enumerate(cells):
            cell_type = cell.get("cell_type", "unknown")
            source = "".join(cell.get("source", []))
            parts.append(f"--- Cell {i + 1} [{cell_type}] ---\n{source}")

            outputs = cell.get("outputs", [])
            for out in outputs:
                if "text" in out:
                    parts.append("Output: " + "".join(out["text"])[:500])
                elif "data" in out and "text/plain" in out["data"]:
                    parts.append("Output: " + "".join(out["data"]["text/plain"])[:500])

        return "\n\n".join(parts) if parts else f"[NOTEBOOK] {path} (empty)"
    except Exception as e:
        return f"[NOTEBOOK_ERROR] {path}: {e}"


def _parse_page_range(pages: str, total: int) -> range:
    """Parse page range like '1-5', '3', '10-20'."""
    try:
        if "-" in pages:
            start, end = pages.split("-", 1)
            return range(int(start) - 1, min(int(end), total))
        else:
            p = int(pages) - 1
            return range(p, min(p + 1, total))
    except ValueError:
        return range(min(total, 10))


# ════════════════════════════════════════════════════════════
# A02: EditFile
# ════════════════════════════════════════════════════════════

class EditFileSchema:
    def __init__(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False):
        if not file_path:
            raise ValueError("file_path is required")
        if not old_string:
            raise ValueError("old_string is required")
        if old_string == new_string:
            raise ValueError("old_string and new_string must be different")
        if not os.path.isabs(file_path):
            file_path = os.path.abspath(file_path)
        self.file_path = file_path
        self.old_string = old_string
        self.new_string = new_string
        self.replace_all = replace_all

    def dict(self):
        return {"file_path": self.file_path, "old_string": self.old_string,
                "new_string": self.new_string, "replace_all": self.replace_all}


def edit_file(input_val: Any) -> str:
    """Precise string replacement in file."""
    params = _parse_input(input_val, EditFileSchema)
    path = params["file_path"]
    old = params["old_string"]
    new = params["new_string"]
    replace_all = params.get("replace_all", False)

    if not os.path.exists(path):
        return f"[ERROR] File not found: {path}"

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    count = content.count(old)
    if count == 0:
        return f"[ERROR] old_string not found in {path}. Read the file first to get the exact text."

    if count > 1 and not replace_all:
        return (f"[ERROR] old_string found {count} times in {path}. "
                f"Provide more context to make it unique, or set replace_all=true.")

    if replace_all:
        new_content = content.replace(old, new)
        replaced = count
    else:
        new_content = content.replace(old, new, 1)
        replaced = 1

    with open(path, "w", encoding="utf-8") as f:
        f.write(new_content)

    return f"[OK] Replaced {replaced} occurrence(s) in {path}"


# ════════════════════════════════════════════════════════════
# A03: WriteFile
# ════════════════════════════════════════════════════════════

class WriteFileSchema:
    SENSITIVE_NAMES = {".env", "credentials.json", "secrets.json", ".aws/credentials",
                       "id_rsa", "id_ed25519", ".npmrc", ".pypirc"}

    def __init__(self, file_path: str, content: str):
        if not file_path:
            raise ValueError("file_path is required")
        if not os.path.isabs(file_path):
            file_path = os.path.abspath(file_path)
        basename = os.path.basename(file_path)
        if basename in self.SENSITIVE_NAMES:
            raise ValueError(f"Refusing to write sensitive file: {basename}")
        self.file_path = file_path
        self.content = content

    def dict(self):
        return {"file_path": self.file_path, "content": self.content}


def write_file(input_val: Any) -> str:
    """Create or overwrite file."""
    params = _parse_input(input_val, WriteFileSchema)
    path = params["file_path"]
    content = params["content"]

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    existed = os.path.exists(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    action = "Overwrote" if existed else "Created"
    lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
    return f"[OK] {action} {path} ({lines} lines)"


# ════════════════════════════════════════════════════════════
# A04: ListFiles (Glob)
# ════════════════════════════════════════════════════════════

class ListFilesSchema:
    def __init__(self, pattern: str = "**/*", path: str = ".", max_results: int = 100):
        if not pattern:
            pattern = "**/*"
        self.pattern = pattern
        self.path = os.path.abspath(path) if path else os.getcwd()
        self.max_results = min(max(1, max_results), 1000)

    def dict(self):
        return {"pattern": self.pattern, "path": self.path, "max_results": self.max_results}


_IGNORE_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".tox",
                ".mypy_cache", ".pytest_cache", "dist", "build", ".eggs"}


def list_files(input_val: Any) -> str:
    """Find files matching glob pattern, sorted by modification time."""
    params = _parse_input(input_val, ListFilesSchema)
    pattern = params["pattern"]
    base = params["path"]
    max_results = params.get("max_results", 100)

    import glob as glob_mod
    full_pattern = os.path.join(base, pattern)
    matches = glob_mod.glob(full_pattern, recursive=True)

    # Filter out ignored directories
    filtered = []
    for m in matches:
        parts = m.replace(base, "").split(os.sep)
        if not any(p in _IGNORE_DIRS for p in parts):
            filtered.append(m)

    # Sort by modification time (newest first)
    filtered.sort(key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0, reverse=True)

    if not filtered:
        return f"[NO_MATCH] No files matching '{pattern}' in {base}"

    total = len(filtered)
    shown = filtered[:max_results]
    lines = [os.path.relpath(f, base) for f in shown]
    result = "\n".join(lines)

    if total > max_results:
        result += f"\n\n... [{total - max_results} more files]"

    return result


# ════════════════════════════════════════════════════════════
# A05: SearchContent (Grep)
# ════════════════════════════════════════════════════════════

class SearchContentSchema:
    def __init__(self, pattern: str, path: str = ".", glob: str = "",
                 file_type: str = "", context: int = 0,
                 output_mode: str = "content", max_results: int = 50):
        if not pattern:
            raise ValueError("pattern is required (regex)")
        self.pattern = pattern
        self.path = os.path.abspath(path) if path else os.getcwd()
        self.glob = glob
        self.file_type = file_type
        self.context = min(max(0, context), 10)
        self.output_mode = output_mode if output_mode in ("content", "files_only", "count") else "content"
        self.max_results = min(max(1, max_results), 500)

    def dict(self):
        return {"pattern": self.pattern, "path": self.path, "glob": self.glob,
                "file_type": self.file_type, "context": self.context,
                "output_mode": self.output_mode, "max_results": self.max_results}


def search_content(input_val: Any) -> str:
    """Search file contents using ripgrep (rg) or Python re fallback."""
    params = _parse_input(input_val, SearchContentSchema)
    pattern = params["pattern"]
    path = params["path"]
    glob_filter = params.get("glob", "")
    file_type = params.get("file_type", "")
    context = params.get("context", 0)
    mode = params.get("output_mode", "content")
    max_results = params.get("max_results", 50)

    # Try ripgrep first
    rg = _find_rg()
    if rg:
        return _search_with_rg(rg, pattern, path, glob_filter, file_type, context, mode, max_results)

    # Fallback to Python
    return _search_with_python(pattern, path, glob_filter, context, mode, max_results)


def _find_rg() -> Optional[str]:
    """Find ripgrep binary."""
    for cmd in ["rg", "/usr/local/bin/rg", "/opt/homebrew/bin/rg"]:
        try:
            subprocess.run([cmd, "--version"], capture_output=True, timeout=5)
            return cmd
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


def _search_with_rg(rg: str, pattern: str, path: str, glob_filter: str,
                    file_type: str, context: int, mode: str, max_results: int) -> str:
    cmd = [rg, "--no-heading", "-n"]
    if mode == "files_only":
        cmd.append("-l")
    elif mode == "count":
        cmd.append("-c")
    if context > 0 and mode == "content":
        cmd.extend(["-C", str(context)])
    if glob_filter:
        cmd.extend(["--glob", glob_filter])
    if file_type:
        cmd.extend(["--type", file_type])
    cmd.extend(["-m", str(max_results), pattern, path])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        output = result.stdout.strip()
        if not output:
            return f"[NO_MATCH] Pattern '{pattern}' not found in {path}"
        lines = output.split("\n")
        if len(lines) > max_results:
            lines = lines[:max_results]
            output = "\n".join(lines) + f"\n\n... [truncated at {max_results} results]"
        return output
    except subprocess.TimeoutExpired:
        return f"[TIMEOUT] Search timed out for '{pattern}'"
    except Exception as e:
        return f"[ERROR] rg failed: {e}"


def _search_with_python(pattern: str, path: str, glob_filter: str,
                        context: int, mode: str, max_results: int) -> str:
    """Fallback: pure Python regex search."""
    import glob as glob_mod
    if glob_filter:
        files = glob_mod.glob(os.path.join(path, glob_filter), recursive=True)
    else:
        files = glob_mod.glob(os.path.join(path, "**/*"), recursive=True)
    files = [f for f in files if os.path.isfile(f)]

    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"[ERROR] Invalid regex: {e}"

    results = []
    for fpath in files:
        parts = fpath.split(os.sep)
        if any(p in _IGNORE_DIRS for p in parts):
            continue
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            for i, line in enumerate(lines, 1):
                if regex.search(line):
                    if mode == "files_only":
                        results.append(os.path.relpath(fpath, path))
                        break
                    elif mode == "count":
                        results.append(f"{os.path.relpath(fpath, path)}:{i}")
                    else:
                        results.append(f"{os.path.relpath(fpath, path)}:{i}:{line.rstrip()}")
                    if len(results) >= max_results:
                        break
        except Exception:
            continue
        if len(results) >= max_results:
            break

    if mode == "files_only":
        results = list(dict.fromkeys(results))  # deduplicate

    if not results:
        return f"[NO_MATCH] Pattern '{pattern}' not found"
    return "\n".join(results)


# ════════════════════════════════════════════════════════════
# Shared helpers
# ════════════════════════════════════════════════════════════

def _parse_input(input_val: Any, schema_cls) -> dict:
    """Parse tool input: string→JSON→schema, or dict→schema."""
    if isinstance(input_val, dict):
        data = input_val
    elif isinstance(input_val, str):
        try:
            data = json.loads(input_val)
        except (json.JSONDecodeError, ValueError):
            data = {"file_path": input_val} if "file" in schema_cls.__name__.lower() else {"pattern": input_val}
    else:
        data = {"file_path": str(input_val)}

    # Extract nested "input" from ReAct format: {"action":"Tool","input":{...}}
    if isinstance(data, dict) and "input" in data and ("action" in data or "tool" in data):
        inner = data["input"]
        if isinstance(inner, str):
            try:
                data = json.loads(inner)
            except (json.JSONDecodeError, ValueError):
                data = {"file_path": inner} if "file" in schema_cls.__name__.lower() else {"pattern": inner}
        elif isinstance(inner, dict):
            data = inner
        else:
            data = {"file_path": str(inner)}

    # Normalize common parameter aliases from LLM output
    if isinstance(data, dict):
        # Only alias 'path' → 'file_path' for schemas that use file_path (not ListFiles which uses 'path')
        _ALIASES = {
            "filepath": "file_path",
            "file": "file_path",
            "old": "old_string",
            "new": "new_string",
            "old_text": "old_string",
            "new_text": "new_string",
            "cmd": "command",
            "query": "pattern",
        }
        # 'path' → 'file_path' only when schema has file_path param (ReadFile, WriteFile, EditFile)
        if "path" in data and "file_path" not in data:
            import inspect
            params = inspect.signature(schema_cls.__init__).parameters
            if "file_path" in params and "path" not in params:
                _ALIASES["path"] = "file_path"

        normalized = {}
        for k, v in data.items():
            key = _ALIASES.get(k, k)
            if key not in normalized:
                normalized[key] = v
            elif k not in _ALIASES:
                normalized[key] = v
        data = normalized

    try:
        validated = schema_cls(**data)
        return validated.dict()
    except (TypeError, ValueError) as e:
        raise ValueError(f"[VALIDATION_ERROR] {schema_cls.__name__}: {e}")
