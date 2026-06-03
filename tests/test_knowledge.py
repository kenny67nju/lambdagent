"""Tests for Tier 5: A34-A38 Knowledge base pipeline."""
from __future__ import annotations
import json
import os
import tempfile
import pytest


class TestChunkSplitter:
    def test_paragraph_split(self):
        from lambdagent.builtin_tools.knowledge_tools import ChunkSplitter
        text = "Paragraph one with some content here.\n\nParagraph two with different content.\n\nParagraph three with more."
        chunks = ChunkSplitter.split(text, "paragraph", 50)  # small chunk size to force splits
        assert len(chunks) >= 2

    def test_heading_split(self):
        from lambdagent.builtin_tools.knowledge_tools import ChunkSplitter
        text = "# Title\nIntro\n## Section 1\nContent 1\n## Section 2\nContent 2"
        chunks = ChunkSplitter.split(text, "heading")
        assert len(chunks) >= 2
        assert any("Title" in c for c in chunks)

    def test_fixed_split(self):
        from lambdagent.builtin_tools.knowledge_tools import ChunkSplitter
        text = "a" * 1000
        chunks = ChunkSplitter.split(text, "fixed", 200, 50)
        assert len(chunks) >= 5
        assert all(len(c) <= 200 for c in chunks)

    def test_tool_function(self):
        from lambdagent.builtin_tools.knowledge_tools import chunk_split
        result = chunk_split({"text": "Hello world with some content.\n\nSecond paragraph with more.\n\nThird paragraph here.", "strategy": "paragraph", "chunk_size": 40})
        data = json.loads(result)
        assert data["chunks"] >= 2

    def test_file_input(self):
        from lambdagent.builtin_tools.knowledge_tools import chunk_split
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("# Title\nContent here.\n## Section\nMore content.")
            path = f.name
        try:
            result = chunk_split({"file_path": path, "strategy": "heading"})
            data = json.loads(result)
            assert data["chunks"] >= 1
        finally:
            os.unlink(path)


class TestOCR:
    def test_no_backend_message(self):
        from lambdagent.builtin_tools.knowledge_tools import ocr_extract
        # Create a dummy image file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x89PNG\r\n\x1a\n")  # PNG header
            path = f.name
        try:
            result = ocr_extract({"file_path": path})
            # Either succeeds or reports "no backend"
            assert isinstance(result, str)
        finally:
            os.unlink(path)

    def test_missing_file(self):
        from lambdagent.builtin_tools.knowledge_tools import ocr_extract
        result = ocr_extract({"file_path": "/nonexistent.png"})
        assert "ERROR" in result


class TestDocGen:
    def test_html_generation(self):
        from lambdagent.builtin_tools.knowledge_tools import doc_generate
        with tempfile.TemporaryDirectory() as d:
            md_path = os.path.join(d, "test.md")
            with open(md_path, "w") as f:
                f.write("# Hello\n\nThis is a test.\n\n## Section\n\n- Item 1\n- Item 2")
            html_path = os.path.join(d, "test.html")
            result = doc_generate({"source": md_path, "format": "html", "output": html_path})
            assert "OK" in result
            assert os.path.exists(html_path)
            with open(html_path) as f:
                html = f.read()
            assert "Hello" in html
            assert "<html" in html

    def test_html_from_text(self):
        from lambdagent.builtin_tools.knowledge_tools import doc_generate
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "out.html")
            result = doc_generate({"source": "# Title\n\nContent", "format": "html", "output": out})
            assert "OK" in result

    def test_unsupported_format(self):
        from lambdagent.builtin_tools.knowledge_tools import doc_generate
        result = doc_generate({"source": "text", "format": "pptx"})
        assert "ERROR" in result


class TestKBManager:
    def test_create_and_list(self):
        from lambdagent.builtin_tools.knowledge_tools import kb_manage
        name = f"test_kb_{os.getpid()}"
        try:
            result = kb_manage({"action": "create", "name": name, "description": "Test KB"})
            assert "OK" in result
            result = kb_manage({"action": "list"})
            assert name in result
            result = kb_manage({"action": "info", "name": name})
            assert "Test KB" in result
        finally:
            kb_manage({"action": "delete", "name": name})

    def test_add_and_search(self):
        from lambdagent.builtin_tools.knowledge_tools import kb_manage
        name = f"test_kb_search_{os.getpid()}"
        try:
            kb_manage({"action": "create", "name": name})
            # Create temp file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                f.write("Lambda calculus is the foundation of functional programming.\n\n"
                        "Python is a popular programming language.\n\n"
                        "Machine learning uses neural networks.")
                path = f.name
            try:
                result = kb_manage({"action": "add", "name": name, "file_path": path})
                assert "OK" in result
                result = kb_manage({"action": "search", "name": name, "query": "Lambda calculus"})
                assert "Lambda" in result or "lambda" in result
            finally:
                os.unlink(path)
        finally:
            kb_manage({"action": "delete", "name": name})

    def test_delete_nonexistent(self):
        from lambdagent.builtin_tools.knowledge_tools import kb_manage
        result = kb_manage({"action": "delete", "name": "nonexistent_kb_xyz"})
        assert "ERROR" in result


class TestRegistryKnowledge:
    def test_tools_registered(self):
        from lambdagent.builtin_tools.registry import BUILTIN_TOOLS
        assert "ChunkSplit" in BUILTIN_TOOLS
        assert "OCR" in BUILTIN_TOOLS
        assert "DocGen" in BUILTIN_TOOLS
        assert "KBCreate" in BUILTIN_TOOLS
        assert "KBSearch" in BUILTIN_TOOLS

    def test_total_count(self):
        from lambdagent.builtin_tools.registry import BUILTIN_TOOLS
        assert len(BUILTIN_TOOLS) >= 28  # 21 + 7 new


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
