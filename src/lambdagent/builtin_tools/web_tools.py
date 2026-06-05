"""
lambdagent.builtin_tools.web_tools — Web + Notebook tools

NotebookEdit  λx. edit_notebook(path, cell, action, content)
WebSearch     λx. search(query)
WebFetch      λx. fetch(url) → markdown
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import urllib.request
import urllib.error
from typing import Any, Dict, Optional


# ════════════════════════════════════════════════════════════
# A18: NotebookEdit
# ════════════════════════════════════════════════════════════


class NotebookEditSchema:
    def __init__(
        self,
        path: str,
        cell_index: int = -1,
        action: str = "replace",
        content: str = "",
        cell_type: str = "code",
    ):
        if not path:
            raise ValueError("path is required")
        if action not in ("replace", "insert", "delete", "append"):
            raise ValueError(
                f"action must be replace/insert/delete/append, got '{action}'"
            )
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        self.path = path
        self.cell_index = cell_index
        self.action = action
        self.content = content
        self.cell_type = (
            cell_type if cell_type in ("code", "markdown", "raw") else "code"
        )

    def dict(self):
        return {
            "path": self.path,
            "cell_index": self.cell_index,
            "action": self.action,
            "content": self.content,
            "cell_type": self.cell_type,
        }


def notebook_edit(input_val: Any) -> str:
    """Edit Jupyter Notebook at cell level."""
    params = _parse_input(input_val, NotebookEditSchema)
    path = params["path"]
    cell_index = params["cell_index"]
    action = params["action"]
    content = params.get("content", "")
    cell_type = params.get("cell_type", "code")

    if not os.path.exists(path):
        return f"[ERROR] File not found: {path}"

    try:
        with open(path, "r", encoding="utf-8") as f:
            nb = json.load(f)
    except (json.JSONDecodeError, Exception) as e:
        return f"[ERROR] Invalid notebook: {e}"

    cells = nb.get("cells", [])

    if action == "append":
        new_cell = _make_cell(content, cell_type)
        cells.append(new_cell)
        nb["cells"] = cells
        _save_notebook(path, nb)
        return f"[OK] Appended {cell_type} cell (now {len(cells)} cells)"

    if cell_index < 0 or cell_index >= len(cells):
        if action != "insert" or cell_index != len(cells):
            return f"[ERROR] cell_index {cell_index} out of range (0-{len(cells) - 1})"

    if action == "replace":
        cells[cell_index] = _make_cell(content, cell_type)
        nb["cells"] = cells
        _save_notebook(path, nb)
        return f"[OK] Replaced cell {cell_index}"

    elif action == "insert":
        new_cell = _make_cell(content, cell_type)
        cells.insert(cell_index, new_cell)
        nb["cells"] = cells
        _save_notebook(path, nb)
        return f"[OK] Inserted {cell_type} cell at index {cell_index} (now {len(cells)} cells)"

    elif action == "delete":
        deleted = cells.pop(cell_index)
        nb["cells"] = cells
        _save_notebook(path, nb)
        deleted_type = deleted.get("cell_type", "?")
        return (
            f"[OK] Deleted cell {cell_index} ({deleted_type}, now {len(cells)} cells)"
        )

    return f"[ERROR] Unknown action: {action}"


def _make_cell(content: str, cell_type: str) -> dict:
    """Create a notebook cell dict."""
    source = content.split("\n") if content else [""]
    # Ensure each line except last ends with \n
    source = [
        line + "\n" if i < len(source) - 1 else line for i, line in enumerate(source)
    ]
    cell = {
        "cell_type": cell_type,
        "source": source,
        "metadata": {},
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def _save_notebook(path: str, nb: dict):
    """Save notebook preserving format."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")


# ════════════════════════════════════════════════════════════
# A19: WebSearch
# ════════════════════════════════════════════════════════════


class WebSearchSchema:
    def __init__(self, query: str, max_results: int = 5):
        if not query:
            raise ValueError("query is required")
        self.query = query
        self.max_results = min(max(1, max_results), 20)

    def dict(self):
        return {"query": self.query, "max_results": self.max_results}


def web_search(input_val: Any) -> str:
    """Web search via DuckDuckGo HTML (no API key needed)."""
    params = _parse_input(input_val, WebSearchSchema)
    query = params["query"]
    max_results = params.get("max_results", 5)

    try:
        # Use DuckDuckGo HTML search (no API key needed)
        encoded = urllib.request.quote(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded}"
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0 (compatible; lambdagent/1.0)"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="ignore")

        # Parse results from HTML
        results = _parse_ddg_html(html, max_results)
        if not results:
            return f"[NO_RESULTS] No results for '{query}'"

        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. **{r['title']}**")
            lines.append(f"   {r['url']}")
            if r.get("snippet"):
                lines.append(f"   {r['snippet'][:200]}")
            lines.append("")

        return "\n".join(lines)

    except urllib.error.URLError as e:
        return f"[ERROR] Search failed: {e}"
    except Exception as e:
        return f"[ERROR] {e}"


def _parse_ddg_html(html: str, max_results: int) -> list:
    """Parse DuckDuckGo HTML results."""
    results = []
    # Find result links
    pattern = r'class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>'
    matches = re.findall(pattern, html, re.DOTALL)

    # Find snippets
    snippet_pattern = r'class="result__snippet"[^>]*>(.*?)</(?:a|div|span)'
    snippets = re.findall(snippet_pattern, html, re.DOTALL)

    for i, (url, title) in enumerate(matches[:max_results]):
        # Clean HTML tags
        title = re.sub(r"<[^>]+>", "", title).strip()
        # Decode URL (DuckDuckGo wraps in redirect)
        if "uddg=" in url:
            url_match = re.search(r"uddg=([^&]+)", url)
            if url_match:
                url = urllib.request.unquote(url_match.group(1))
        elif url.startswith("//"):
            url = "https:" + url

        snippet = ""
        if i < len(snippets):
            snippet = re.sub(r"<[^>]+>", "", snippets[i]).strip()

        results.append({"title": title, "url": url, "snippet": snippet})

    return results


# ════════════════════════════════════════════════════════════
# A20: WebFetch
# ════════════════════════════════════════════════════════════


class WebFetchSchema:
    def __init__(self, url: str, max_length: int = 5000, selector: str = ""):
        if not url:
            raise ValueError("url is required")
        if not url.startswith(("http://", "https://")):
            raise ValueError("url must start with http:// or https://")
        self.url = url
        self.max_length = min(max(100, max_length), 50000)
        self.selector = selector

    def dict(self):
        return {
            "url": self.url,
            "max_length": self.max_length,
            "selector": self.selector,
        }


def web_fetch(input_val: Any) -> str:
    """Fetch URL content, convert HTML to readable text/markdown."""
    params = _parse_input(input_val, WebFetchSchema)
    url = params["url"]
    max_length = params.get("max_length", 5000)

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; lambdagent/1.0)",
                "Accept": "text/html,application/xhtml+xml,text/plain,application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            content_type = resp.headers.get("Content-Type", "")
            raw = resp.read()

        # JSON response
        if "json" in content_type:
            try:
                data = json.loads(raw.decode("utf-8"))
                text = json.dumps(data, indent=2, ensure_ascii=False)
            except Exception:
                text = raw.decode("utf-8", errors="ignore")

        # Plain text
        elif "text/plain" in content_type:
            text = raw.decode("utf-8", errors="ignore")

        # HTML → simplified text
        else:
            html = raw.decode("utf-8", errors="ignore")
            text = _html_to_text(html)

        # Truncate
        if len(text) > max_length:
            text = text[:max_length] + f"\n\n... [truncated, {len(text)} chars total]"

        return text

    except urllib.error.HTTPError as e:
        return f"[HTTP_ERROR] {e.code}: {url}"
    except urllib.error.URLError as e:
        return f"[URL_ERROR] {e}: {url}"
    except Exception as e:
        return f"[ERROR] {e}"


def _html_to_text(html: str) -> str:
    """Convert HTML to readable text. Best-effort without dependencies."""
    # Try markdownify if available
    try:
        import markdownify

        return markdownify.markdownify(html, strip=["img", "script", "style"])
    except ImportError:
        pass

    # Try html2text if available
    try:
        import html2text

        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = True
        return h.handle(html)
    except ImportError:
        pass

    # Fallback: regex-based stripping
    # Remove script/style blocks
    text = re.sub(
        r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE
    )
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Convert common tags
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<p[^>]*>", "\n\n", text, flags=re.IGNORECASE)
    text = re.sub(
        r"<h[1-6][^>]*>(.*?)</h[1-6]>",
        r"\n\n## \1\n",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(r"<li[^>]*>", "\n- ", text, flags=re.IGNORECASE)
    # Strip remaining tags
    text = re.sub(r"<[^>]+>", "", text)
    # Decode entities
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
    # Clean whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


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
            # Heuristic: if it looks like a URL, treat as url; if looks like a query, treat as query
            if input_val.startswith(("http://", "https://")):
                data = {"url": input_val}
            elif (
                "/" in input_val or "." in input_val.split()[-1]
                if input_val.split()
                else False
            ):
                data = {"path": input_val}
            else:
                data = {"query": input_val}
    else:
        data = {}
    try:
        validated = schema_cls(**data)
        return validated.dict()
    except (TypeError, ValueError) as e:
        raise ValueError(f"[VALIDATION_ERROR] {schema_cls.__name__}: {e}")
