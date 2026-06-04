"""Tests for built-in tools (A01-A10, A11-A16)."""
from __future__ import annotations
import json
import os
import tempfile
import time
import pytest


# ════════════════════════════════════════════════════════════
# A01: ReadFile
# ════════════════════════════════════════════════════════════

class TestReadFile:
    def test_read_basic(self):
        from lambdagent.builtin_tools.file_tools import read_file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("line1\nline2\nline3\n")
            path = f.name
        try:
            result = read_file({"file_path": path})
            assert "1\tline1" in result
            assert "2\tline2" in result
            assert "3\tline3" in result
        finally:
            os.unlink(path)

    def test_read_offset_limit(self):
        from lambdagent.builtin_tools.file_tools import read_file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for i in range(100):
                f.write(f"line {i}\n")
            path = f.name
        try:
            result = read_file({"file_path": path, "offset": 10, "limit": 5})
            assert "11\tline 10" in result
            assert "15\tline 14" in result
            assert "more lines" in result
        finally:
            os.unlink(path)

    def test_read_not_found(self):
        from lambdagent.builtin_tools.file_tools import read_file
        result = read_file({"file_path": "/nonexistent/file.txt"})
        assert "ERROR" in result

    def test_read_directory(self):
        from lambdagent.builtin_tools.file_tools import read_file
        result = read_file({"file_path": tempfile.gettempdir()})
        assert "directory" in result.lower()

    def test_read_empty_file(self):
        from lambdagent.builtin_tools.file_tools import read_file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            path = f.name
        try:
            result = read_file({"file_path": path})
            assert "EMPTY" in result
        finally:
            os.unlink(path)

    def test_read_notebook(self):
        from lambdagent.builtin_tools.file_tools import read_file
        nb = {"cells": [{"cell_type": "code", "source": ["print('hello')"], "outputs": [{"text": ["hello\n"]}]}]}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ipynb", delete=False) as f:
            json.dump(nb, f)
            path = f.name
        try:
            result = read_file({"file_path": path})
            assert "print('hello')" in result
            assert "hello" in result
        finally:
            os.unlink(path)

    def test_read_string_input(self):
        from lambdagent.builtin_tools.file_tools import read_file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("content")
            path = f.name
        try:
            result = read_file(path)  # plain string
            assert "content" in result
        finally:
            os.unlink(path)


# ════════════════════════════════════════════════════════════
# A02: EditFile
# ════════════════════════════════════════════════════════════

class TestEditFile:
    def test_edit_basic(self):
        from lambdagent.builtin_tools.file_tools import edit_file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def hello():\n    return 'world'\n")
            path = f.name
        try:
            result = edit_file({"file_path": path, "old_string": "return 'world'", "new_string": "return 'hello'"})
            assert "OK" in result
            with open(path) as f2:
                assert "return 'hello'" in f2.read()
        finally:
            os.unlink(path)

    def test_edit_not_found(self):
        from lambdagent.builtin_tools.file_tools import edit_file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("abc")
            path = f.name
        try:
            result = edit_file({"file_path": path, "old_string": "xyz", "new_string": "123"})
            assert "ERROR" in result
            assert "not found" in result
        finally:
            os.unlink(path)

    def test_edit_non_unique(self):
        from lambdagent.builtin_tools.file_tools import edit_file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("aaa\naaa\naaa\n")
            path = f.name
        try:
            result = edit_file({"file_path": path, "old_string": "aaa", "new_string": "bbb"})
            assert "ERROR" in result
            assert "3 times" in result
        finally:
            os.unlink(path)

    def test_edit_replace_all(self):
        from lambdagent.builtin_tools.file_tools import edit_file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("aaa\naaa\naaa\n")
            path = f.name
        try:
            result = edit_file({"file_path": path, "old_string": "aaa", "new_string": "bbb", "replace_all": True})
            assert "OK" in result
            assert "3 occurrence" in result
            with open(path) as f2:
                assert f2.read() == "bbb\nbbb\nbbb\n"
        finally:
            os.unlink(path)


# ════════════════════════════════════════════════════════════
# A03: WriteFile
# ════════════════════════════════════════════════════════════

class TestWriteFile:
    def test_write_new(self):
        from lambdagent.builtin_tools.file_tools import write_file
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "new.txt")
            result = write_file({"file_path": path, "content": "hello\nworld\n"})
            assert "Created" in result
            assert os.path.exists(path)
            with open(path) as f:
                assert f.read() == "hello\nworld\n"

    def test_write_overwrite(self):
        from lambdagent.builtin_tools.file_tools import write_file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("old")
            path = f.name
        try:
            result = write_file({"file_path": path, "content": "new"})
            assert "Overwrote" in result
        finally:
            os.unlink(path)

    def test_write_creates_dirs(self):
        from lambdagent.builtin_tools.file_tools import write_file
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sub", "dir", "file.txt")
            result = write_file({"file_path": path, "content": "deep"})
            assert "Created" in result
            assert os.path.exists(path)

    def test_write_sensitive_rejected(self):
        from lambdagent.builtin_tools.file_tools import write_file
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, ".env")
            with pytest.raises(ValueError, match="sensitive"):
                write_file({"file_path": path, "content": "SECRET=bad"})


# ════════════════════════════════════════════════════════════
# A04: ListFiles
# ════════════════════════════════════════════════════════════

class TestListFiles:
    def test_glob_basic(self):
        from lambdagent.builtin_tools.file_tools import list_files
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "a.py"), "w").close()
            open(os.path.join(d, "b.py"), "w").close()
            open(os.path.join(d, "c.txt"), "w").close()
            result = list_files({"pattern": "*.py", "path": d})
            assert "a.py" in result
            assert "b.py" in result
            assert "c.txt" not in result

    def test_glob_recursive(self):
        from lambdagent.builtin_tools.file_tools import list_files
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "src"))
            open(os.path.join(d, "src", "main.py"), "w").close()
            result = list_files({"pattern": "**/*.py", "path": d})
            assert "main.py" in result

    def test_glob_no_match(self):
        from lambdagent.builtin_tools.file_tools import list_files
        with tempfile.TemporaryDirectory() as d:
            result = list_files({"pattern": "*.xyz", "path": d})
            assert "NO_MATCH" in result

    def test_glob_ignores_git(self):
        from lambdagent.builtin_tools.file_tools import list_files
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, ".git"))
            open(os.path.join(d, ".git", "config"), "w").close()
            open(os.path.join(d, "real.txt"), "w").close()
            result = list_files({"pattern": "**/*", "path": d})
            assert "config" not in result
            assert "real.txt" in result


# ════════════════════════════════════════════════════════════
# A05: SearchContent
# ════════════════════════════════════════════════════════════

class TestSearchContent:
    def test_search_basic(self):
        from lambdagent.builtin_tools.file_tools import search_content
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "test.py"), "w") as f:
                f.write("def hello():\n    return 42\n")
            result = search_content({"pattern": "def hello", "path": d})
            assert "hello" in result

    def test_search_no_match(self):
        from lambdagent.builtin_tools.file_tools import search_content
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "test.py"), "w") as f:
                f.write("nothing here\n")
            result = search_content({"pattern": "nonexistent_xyz", "path": d})
            assert "NO_MATCH" in result

    def test_search_files_only(self):
        from lambdagent.builtin_tools.file_tools import search_content
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "a.py"), "w") as f:
                f.write("match here\n")
            with open(os.path.join(d, "b.py"), "w") as f:
                f.write("no match\n")
            result = search_content({"pattern": "match here", "path": d, "output_mode": "files_only"})
            assert "a.py" in result


# ════════════════════════════════════════════════════════════
# A06: Bash
# ════════════════════════════════════════════════════════════

class TestBash:
    def test_basic_command(self):
        from lambdagent.builtin_tools.shell_tools import run_bash
        result = run_bash({"command": "echo hello"})
        assert "hello" in result

    def test_command_failure(self):
        from lambdagent.builtin_tools.shell_tools import run_bash
        result = run_bash({"command": "false"})
        assert "EXIT:" in result or "no output" in result

    def test_timeout(self):
        from lambdagent.builtin_tools.shell_tools import run_bash
        result = run_bash({"command": "sleep 10", "timeout": 1})
        assert "TIMEOUT" in result

    def test_cd_persistent(self):
        import tempfile
        from lambdagent.builtin_tools.shell_tools import run_bash, _get_cwd
        original = _get_cwd()
        # Use platform's actual tempdir instead of hardcoded /tmp so the test
        # works on Windows (where /tmp doesn't exist).
        target = tempfile.gettempdir()
        run_bash({"command": f"cd {target}"})
        assert _get_cwd() == target or _get_cwd() == os.path.realpath(target)
        # Restore
        run_bash({"command": f"cd {original}"})

    def test_portable_command_works_across_platforms(self):
        """`echo` works in /bin/sh, bash, cmd.exe — true cross-shell smoke test."""
        from lambdagent.builtin_tools.shell_tools import run_bash
        # No quoting / glob / variable expansion / pipe — works the same in
        # every shell on Linux, macOS, and Windows (both Git-Bash and cmd.exe).
        result = run_bash({"command": "echo lambdagent-portable-579"})
        assert "lambdagent-portable-579" in result, f"got: {result!r}"

    def test_resolve_shell_returns_sensible_value(self):
        """resolve_shell() returns None on POSIX, bash path on Windows when available."""
        import platform
        from lambdagent._shell_compat import resolve_shell
        result = resolve_shell()
        if platform.system() == "Windows":
            # CI runner has Git-Bash; if user installs without it, result is None.
            assert result is None or result.lower().endswith("bash.exe"), result
        else:
            assert result is None, f"POSIX should return None, got {result!r}"

    def test_interactive_rejected(self):
        from lambdagent.builtin_tools.shell_tools import run_bash
        with pytest.raises(ValueError, match="Interactive"):
            run_bash({"command": "vim file.txt"})

    def test_background(self):
        from lambdagent.builtin_tools.shell_tools import run_bash
        result = run_bash({"command": "sleep 0.1", "run_in_background": True})
        assert "BACKGROUND" in result
        assert "PID=" in result

    def test_string_input(self):
        from lambdagent.builtin_tools.shell_tools import run_bash
        result = run_bash("echo world")
        assert "world" in result


# ════════════════════════════════════════════════════════════
# A07: Git tools
# ════════════════════════════════════════════════════════════

class TestGitTools:
    def test_git_status(self):
        from lambdagent.builtin_tools.shell_tools import git_status
        result = git_status("")
        # Should not error (we're in a git repo)
        assert "GIT_ERROR" not in result or "not a git" not in result

    def test_git_log(self):
        from lambdagent.builtin_tools.shell_tools import git_log
        result = git_log({"n": 3, "oneline": True})
        assert "GIT_ERROR" not in result

    def test_git_branch(self):
        from lambdagent.builtin_tools.shell_tools import git_branch
        result = git_branch({"action": "list"})
        assert "main" in result or "master" in result or "GIT_ERROR" not in result

    def test_git_diff(self):
        from lambdagent.builtin_tools.shell_tools import git_diff
        result = git_diff("")
        # May be empty diff, that's OK
        assert isinstance(result, str)

    def test_git_commit_no_message(self):
        from lambdagent.builtin_tools.shell_tools import git_commit
        result = git_commit({"message": ""})
        assert "ERROR" in result


# ════════════════════════════════════════════════════════════
# A08: Tool Registry
# ════════════════════════════════════════════════════════════

class TestToolRegistry:
    def test_all_tools_registered(self):
        from lambdagent.builtin_tools.registry import BUILTIN_TOOLS
        expected = {"ReadFile", "EditFile", "WriteFile", "ListFiles", "SearchContent",
                    "Bash", "GitStatus", "GitDiff", "GitLog", "GitCommit", "GitBranch",
                    "terminate"}
        assert expected.issubset(set(BUILTIN_TOOLS.keys()))

    def test_get_builtin_tool(self):
        from lambdagent.builtin_tools.registry import get_builtin_tool
        assert get_builtin_tool("ReadFile") is not None
        assert get_builtin_tool("nonexistent") is None

    def test_resolve_tools(self):
        from lambdagent.builtin_tools.registry import resolve_tools
        result = resolve_tools(["ReadFile", "Bash", "terminate", "fake_tool"])
        assert "ReadFile" in result
        assert "Bash" in result
        assert "terminate" in result
        assert "fake_tool" not in result

    def test_from_config_integration(self):
        """Test that from_config resolves built-in tool names."""
        from lambdagent.builtin_tools.registry import get_builtin_tool
        tool = get_builtin_tool("ReadFile")
        assert tool is not None
        # Tool should work
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test content\n")
            path = f.name
        try:
            result = tool.apply(json.dumps({"file_path": path}))
            assert "test content" in result
        finally:
            os.unlink(path)


# ════════════════════════════════════════════════════════════
# A09: Terminal UI
# ════════════════════════════════════════════════════════════

class TestTerminalUI:
    def test_ui_creation(self):
        from lambdagent.builtin_tools.terminal_ui import TerminalUI
        ui = TerminalUI(verbose=True)
        assert ui.verbose is True

    def test_ui_start(self, capsys):
        from lambdagent.builtin_tools.terminal_ui import TerminalUI
        ui = TerminalUI(color=False)
        ui.start("test-agent", "react")
        captured = capsys.readouterr()
        assert "test-agent" in captured.out

    def test_ui_result(self, capsys):
        from lambdagent.builtin_tools.terminal_ui import TerminalUI
        ui = TerminalUI(color=False)
        ui._start_time = time.time()
        ui.result("Hello world", 3)
        captured = capsys.readouterr()
        assert "Hello world" in captured.out
        assert "3 β-reductions" in captured.out

    def test_ui_tool_cycle(self, capsys):
        from lambdagent.builtin_tools.terminal_ui import TerminalUI
        ui = TerminalUI(color=False)
        ui.tool_start("search", "query text")
        ui.tool_end("search", "found 5 results", 150.0)
        captured = capsys.readouterr()
        assert "search" in captured.out
        assert "150ms" in captured.out


# ════════════════════════════════════════════════════════════
# A10: Integration — end-to-end with built-in tools
# ════════════════════════════════════════════════════════════

class TestE2EBuiltinTools:
    def test_read_edit_verify(self):
        """End-to-end: ReadFile → EditFile → ReadFile to verify."""
        from lambdagent.builtin_tools.file_tools import read_file, edit_file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def greet():\n    return 'hello'\n")
            path = f.name
        try:
            # Read
            content = read_file({"file_path": path})
            assert "hello" in content
            # Edit
            result = edit_file({"file_path": path, "old_string": "return 'hello'", "new_string": "return 'world'"})
            assert "OK" in result
            # Verify
            content2 = read_file({"file_path": path})
            assert "world" in content2
            assert "hello" not in content2
        finally:
            os.unlink(path)

    def test_write_list_search(self):
        """End-to-end: WriteFile → ListFiles → SearchContent."""
        from lambdagent.builtin_tools.file_tools import write_file, list_files, search_content
        with tempfile.TemporaryDirectory() as d:
            # Write
            path = os.path.join(d, "app.py")
            write_file({"file_path": path, "content": "def main():\n    print('hello')\n"})
            # List
            found = list_files({"pattern": "*.py", "path": d})
            assert "app.py" in found
            # Search
            matches = search_content({"pattern": "def main", "path": d})
            assert "main" in matches

    def test_bash_git_workflow(self):
        """End-to-end: Bash + Git in temp repo."""
        from lambdagent.builtin_tools.shell_tools import run_bash, git_status, git_log
        with tempfile.TemporaryDirectory() as d:
            run_bash({"command": "git init", "working_dir": d})
            run_bash({"command": "git config user.email 'test@test.com'", "working_dir": d})
            run_bash({"command": "git config user.name 'Test'", "working_dir": d})
            run_bash({"command": "echo hello > test.txt", "working_dir": d})
            run_bash({"command": "git add . && git commit -m 'init'", "working_dir": d})

            # Status should be clean
            from lambdagent.builtin_tools.shell_tools import _set_cwd, _get_cwd
            original_cwd = _get_cwd()
            _set_cwd(d)
            try:
                status = git_status("")
                assert "nothing to commit" in status.lower() or status.strip() == "[OK] (no output)"
            finally:
                _set_cwd(original_cwd)


# ════════════════════════════════════════════════════════════
# A11: CodeSearch
# ════════════════════════════════════════════════════════════

class TestCodeSearch:
    def test_search_function(self):
        from lambdagent.builtin_tools.code_tools import code_search
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "app.py"), "w") as f:
                f.write("def hello_world():\n    return 42\n\ndef other():\n    pass\n")
            result = code_search({"query": "hello_world", "language": "python", "path": d})
            assert "hello_world" in result

    def test_search_class(self):
        from lambdagent.builtin_tools.code_tools import code_search
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "model.py"), "w") as f:
                f.write("class MyModel:\n    pass\n")
            result = code_search({"query": "MyModel", "language": "python",
                                  "search_type": "class", "path": d})
            assert "MyModel" in result

    def test_search_no_match(self):
        from lambdagent.builtin_tools.code_tools import code_search
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "empty.py"), "w") as f:
                f.write("pass\n")
            result = code_search({"query": "nonexistent_xyz", "path": d})
            assert "NO_MATCH" in result


# ════════════════════════════════════════════════════════════
# A12: ProjectMap
# ════════════════════════════════════════════════════════════

class TestProjectMap:
    def test_basic_map(self):
        from lambdagent.builtin_tools.code_tools import project_map
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "src"))
            open(os.path.join(d, "README.md"), "w").close()
            open(os.path.join(d, "src", "main.py"), "w").close()
            result = project_map({"path": d})
            assert "README.md" in result
            assert "src/" in result

    def test_file_statistics(self):
        from lambdagent.builtin_tools.code_tools import project_map
        with tempfile.TemporaryDirectory() as d:
            for i in range(5):
                open(os.path.join(d, f"file{i}.py"), "w").close()
            open(os.path.join(d, "data.json"), "w").close()
            result = project_map({"path": d})
            assert ".py" in result
            assert "5 files" in result


# ════════════════════════════════════════════════════════════
# A13: RunTests
# ════════════════════════════════════════════════════════════

class TestRunTests:
    def test_detect_pytest(self):
        from lambdagent.builtin_tools.code_tools import _detect_framework
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "conftest.py"), "w").close()
            assert _detect_framework(d) == "pytest"

    def test_detect_go(self):
        from lambdagent.builtin_tools.code_tools import _detect_framework
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "go.mod"), "w").close()
            assert _detect_framework(d) == "go"

    def test_no_framework(self):
        from lambdagent.builtin_tools.code_tools import _detect_framework
        with tempfile.TemporaryDirectory() as d:
            assert _detect_framework(d) == ""


# ════════════════════════════════════════════════════════════
# A14: TaskManager
# ════════════════════════════════════════════════════════════

class TestTaskManager:
    def test_create_and_list(self):
        from lambdagent.builtin_tools.task_manager import TaskManager
        mgr = TaskManager()
        t1 = mgr.create("Fix bug")
        t2 = mgr.create("Add feature")
        tasks = mgr.list_all()
        assert len(tasks) == 2
        assert tasks[0].subject == "Fix bug"

    def test_update_status(self):
        from lambdagent.builtin_tools.task_manager import TaskManager
        mgr = TaskManager()
        t = mgr.create("Test task")
        mgr.update(t.id, status="in_progress")
        assert mgr.get(t.id).status == "in_progress"
        mgr.update(t.id, status="completed")
        assert mgr.get(t.id).status == "completed"

    def test_tool_functions(self):
        from lambdagent.builtin_tools.task_manager import task_create, task_list, task_update, _default_manager
        _default_manager._tasks.clear()
        _default_manager._counter = 0

        result = task_create({"subject": "Write tests"})
        assert "OK" in result
        result = task_list("")
        assert "Write tests" in result

    def test_persistence(self):
        from lambdagent.builtin_tools.task_manager import TaskManager
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            mgr1 = TaskManager(persist_path=path)
            mgr1.create("Persistent task")
            mgr2 = TaskManager(persist_path=path)
            assert len(mgr2.list_all()) == 1
            assert mgr2.list_all()[0].subject == "Persistent task"
        finally:
            os.unlink(path)


# ════════════════════════════════════════════════════════════
# A15: Permission UI
# ════════════════════════════════════════════════════════════

class TestPermissionUI:
    def test_always_allow(self):
        from lambdagent.builtin_tools.permission_ui import PermissionUI
        ui = PermissionUI(timeout=1)
        ui._always_allow.add("safe_tool")
        assert ui.confirm("safe_tool", "input", "HIGH", "") is True

    def test_reset(self):
        from lambdagent.builtin_tools.permission_ui import PermissionUI
        ui = PermissionUI()
        ui._always_allow.add("tool1")
        ui.reset()
        assert len(ui._always_allow) == 0


# ════════════════════════════════════════════════════════════
# A16: Project config (.lambdagent.md)
# ════════════════════════════════════════════════════════════

class TestProjectConfig:
    def test_load_project_config(self):
        from lambdagent.fromconfig.compiler import _load_project_config
        # Create temp .lambdagent.md in a temp dir
        with tempfile.TemporaryDirectory() as d:
            config_path = os.path.join(d, ".lambdagent.md")
            with open(config_path, "w") as f:
                f.write("Always use TypeScript.\nPrefer functional style.\n")
            original_cwd = os.getcwd()
            try:
                os.chdir(d)
                result = _load_project_config()
                assert "TypeScript" in result
                assert "functional" in result
            finally:
                os.chdir(original_cwd)

    def test_no_config(self):
        from lambdagent.fromconfig.compiler import _load_project_config
        with tempfile.TemporaryDirectory() as d:
            original_cwd = os.getcwd()
            try:
                os.chdir(d)
                result = _load_project_config()
                assert result == ""
            finally:
                os.chdir(original_cwd)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
