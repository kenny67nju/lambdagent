"""Tests for Tier 3: A18-A23 (Notebook, Web, Fallback, Full UI, Integration)."""
from __future__ import annotations
import json
import os
import tempfile
import pytest


# ════════════════════════════════════════════════════════════
# A18: NotebookEdit
# ════════════════════════════════════════════════════════════

class TestNotebookEdit:
    def _make_notebook(self, cells_data):
        """Helper: create temp notebook file."""
        nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {},
              "cells": []}
        for ct, src in cells_data:
            cell = {"cell_type": ct, "source": [src], "metadata": {}}
            if ct == "code":
                cell["execution_count"] = None
                cell["outputs"] = []
            nb["cells"].append(cell)
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".ipynb", delete=False)
        json.dump(nb, f)
        f.close()
        return f.name

    def test_append_cell(self):
        from lambdagent.builtin_tools.web_tools import notebook_edit
        path = self._make_notebook([("code", "x = 1")])
        try:
            result = notebook_edit({"path": path, "action": "append", "content": "print(x)"})
            assert "OK" in result
            assert "2 cells" in result
            with open(path) as f:
                nb = json.load(f)
            assert len(nb["cells"]) == 2
        finally:
            os.unlink(path)

    def test_replace_cell(self):
        from lambdagent.builtin_tools.web_tools import notebook_edit
        path = self._make_notebook([("code", "old"), ("markdown", "# Title")])
        try:
            result = notebook_edit({"path": path, "cell_index": 0, "action": "replace", "content": "new"})
            assert "OK" in result
            with open(path) as f:
                nb = json.load(f)
            assert "new" in nb["cells"][0]["source"][0]
        finally:
            os.unlink(path)

    def test_delete_cell(self):
        from lambdagent.builtin_tools.web_tools import notebook_edit
        path = self._make_notebook([("code", "a"), ("code", "b"), ("code", "c")])
        try:
            result = notebook_edit({"path": path, "cell_index": 1, "action": "delete"})
            assert "OK" in result
            assert "2 cells" in result
        finally:
            os.unlink(path)

    def test_insert_cell(self):
        from lambdagent.builtin_tools.web_tools import notebook_edit
        path = self._make_notebook([("code", "first"), ("code", "last")])
        try:
            result = notebook_edit({"path": path, "cell_index": 1, "action": "insert",
                                    "content": "middle", "cell_type": "markdown"})
            assert "OK" in result
            with open(path) as f:
                nb = json.load(f)
            assert len(nb["cells"]) == 3
            assert nb["cells"][1]["cell_type"] == "markdown"
        finally:
            os.unlink(path)

    def test_invalid_index(self):
        from lambdagent.builtin_tools.web_tools import notebook_edit
        path = self._make_notebook([("code", "only")])
        try:
            result = notebook_edit({"path": path, "cell_index": 99, "action": "replace", "content": "x"})
            assert "ERROR" in result
        finally:
            os.unlink(path)


# ════════════════════════════════════════════════════════════
# A19: WebSearch (offline-safe test)
# ════════════════════════════════════════════════════════════

class TestWebSearch:
    def test_schema_validation(self):
        from lambdagent.builtin_tools.web_tools import WebSearchSchema
        schema = WebSearchSchema(query="python tutorial", max_results=3)
        assert schema.query == "python tutorial"
        assert schema.max_results == 3

    def test_empty_query_rejected(self):
        from lambdagent.builtin_tools.web_tools import WebSearchSchema
        with pytest.raises(ValueError):
            WebSearchSchema(query="")

    def test_search_returns_string(self):
        """WebSearch should return a string (may fail if offline, that's OK)."""
        from lambdagent.builtin_tools.web_tools import web_search
        result = web_search({"query": "python lambda calculus", "max_results": 2})
        assert isinstance(result, str)
        # Either results or an error message
        assert len(result) > 0


# ════════════════════════════════════════════════════════════
# A20: WebFetch (offline-safe test)
# ════════════════════════════════════════════════════════════

class TestWebFetch:
    def test_schema_validation(self):
        from lambdagent.builtin_tools.web_tools import WebFetchSchema
        schema = WebFetchSchema(url="https://example.com", max_length=1000)
        assert schema.url == "https://example.com"

    def test_invalid_url_rejected(self):
        from lambdagent.builtin_tools.web_tools import WebFetchSchema
        with pytest.raises(ValueError):
            WebFetchSchema(url="not-a-url")

    def test_fetch_returns_string(self):
        from lambdagent.builtin_tools.web_tools import web_fetch
        result = web_fetch({"url": "https://example.com", "max_length": 500})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_html_to_text(self):
        from lambdagent.builtin_tools.web_tools import _html_to_text
        html = "<html><body><h1>Title</h1><p>Hello <b>world</b></p></body></html>"
        text = _html_to_text(html)
        assert "Title" in text
        assert "Hello" in text
        assert "<h1>" not in text


# ════════════════════════════════════════════════════════════
# A21: Model Fallback
# ════════════════════════════════════════════════════════════

class TestModelFallback:
    def test_fallback_list_stored(self):
        from lambdagent.agentruntime.llm_adapter import LLMAdapter
        adapter = LLMAdapter(fallback_models=["gpt-4o", "qwen-max"])
        assert adapter.fallback_models == ["gpt-4o", "qwen-max"]

    def test_no_fallback_by_default(self):
        from lambdagent.agentruntime.llm_adapter import LLMAdapter
        adapter = LLMAdapter()
        assert adapter.fallback_models == []

    def test_call_with_fallback_signature(self):
        """call_with_fallback method exists and has correct signature."""
        from lambdagent.agentruntime.llm_adapter import LLMAdapter
        adapter = LLMAdapter()
        assert hasattr(adapter, "call_with_fallback")
        import inspect
        sig = inspect.signature(adapter.call_with_fallback)
        assert "model" in sig.parameters
        assert "system" in sig.parameters
        assert "user" in sig.parameters


# ════════════════════════════════════════════════════════════
# A22: Registry update check
# ════════════════════════════════════════════════════════════

class TestRegistryTier3:
    def test_web_tools_registered(self):
        from lambdagent.builtin_tools.registry import BUILTIN_TOOLS
        assert "NotebookEdit" in BUILTIN_TOOLS
        assert "WebSearch" in BUILTIN_TOOLS
        assert "WebFetch" in BUILTIN_TOOLS

    def test_total_tool_count(self):
        from lambdagent.builtin_tools.registry import BUILTIN_TOOLS
        assert len(BUILTIN_TOOLS) >= 21  # 18 previous + 3 new


# ════════════════════════════════════════════════════════════
# A23: E2E Integration
# ════════════════════════════════════════════════════════════

class TestE2ETier3:
    def test_notebook_create_and_read(self):
        """E2E: WriteFile → ReadFile notebook → NotebookEdit."""
        from lambdagent.builtin_tools.file_tools import read_file, write_file
        from lambdagent.builtin_tools.web_tools import notebook_edit

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test.ipynb")
            nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {},
                  "cells": [{"cell_type": "code", "source": ["x = 1"],
                             "metadata": {}, "execution_count": None, "outputs": []}]}
            write_file({"file_path": path, "content": json.dumps(nb)})

            # Read
            content = read_file({"file_path": path})
            assert "x = 1" in content

            # Edit
            result = notebook_edit({"path": path, "action": "append", "content": "print(x)"})
            assert "OK" in result

    def test_full_tool_pipeline(self):
        """E2E: Write → Search → Edit → Read to verify."""
        from lambdagent.builtin_tools.file_tools import write_file, read_file, edit_file, search_content

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "app.py")
            write_file({"file_path": path, "content": "def main():\n    print('hello')\n"})

            # Search
            found = search_content({"pattern": "def main", "path": d})
            assert "main" in found

            # Edit
            edit_file({"file_path": path, "old_string": "print('hello')", "new_string": "print('world')"})

            # Verify
            content = read_file({"file_path": path})
            assert "world" in content
            assert "hello" not in content


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
