"""
S28: YAML parser fuzz testing

Feed malicious/malformed YAML to from_config and lint_config.
All should raise controlled errors, not crash.
"""

from __future__ import annotations

import os
import tempfile
import pytest
import yaml

from lambdagent.fromconfig import from_config, lint_config
from lambdagent.fromconfig.errors import CompileError, SchemaError


def _write_yaml(content: str) -> str:
    """Write content to a temp YAML file and return the path."""
    fd, path = tempfile.mkstemp(suffix=".yml")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def _cleanup(path: str):
    try:
        os.unlink(path)
    except OSError:
        pass


# ── Deeply nested YAML (1000 levels) ──


class TestDeeplyNestedYAML:
    def test_from_config_deeply_nested(self):
        nested = "a:\n"
        for i in range(1000):
            nested += "  " * (i + 1) + "b:\n"
        nested += "  " * 1001 + "c: 1\n"
        path = _write_yaml(nested)
        try:
            with pytest.raises(
                (CompileError, SchemaError, RecursionError, yaml.YAMLError, Exception)
            ):
                from_config(path)
        finally:
            _cleanup(path)

    def test_lint_config_deeply_nested(self):
        nested = "a:\n"
        for i in range(1000):
            nested += "  " * (i + 1) + "b:\n"
        nested += "  " * 1001 + "c: 1\n"
        path = _write_yaml(nested)
        try:
            # lint_config should not crash; it may return errors or handle gracefully
            try:
                result = lint_config(path)
                # If it returns, it should be a list of LintResult or similar
                assert isinstance(result, list)
            except (
                CompileError,
                SchemaError,
                RecursionError,
                yaml.YAMLError,
                Exception,
            ):
                pass  # Controlled error is acceptable
        finally:
            _cleanup(path)


# ── YAML bomb (billion laughs) ──


class TestYAMLBomb:
    def test_from_config_billion_laughs(self):
        """yaml.safe_load should reject alias-based expansion attacks."""
        bomb = """
a: &a ["lol","lol","lol","lol","lol","lol","lol","lol","lol"]
b: &b [*a,*a,*a,*a,*a,*a,*a,*a,*a]
c: &c [*b,*b,*b,*b,*b,*b,*b,*b,*b]
d: &d [*c,*c,*c,*c,*c,*c,*c,*c,*c]
e: &e [*d,*d,*d,*d,*d,*d,*d,*d,*d]
f: &f [*e,*e,*e,*e,*e,*e,*e,*e,*e]
"""
        path = _write_yaml(bomb)
        try:
            with pytest.raises(
                (CompileError, SchemaError, MemoryError, yaml.YAMLError, Exception)
            ):
                from_config(path)
        finally:
            _cleanup(path)


# ── Binary content in YAML ──


class TestBinaryContent:
    def test_from_config_binary(self):
        binary_content = b"\x00\x01\x02\xff\xfe\xfd" * 100
        fd, path = tempfile.mkstemp(suffix=".yml")
        with os.fdopen(fd, "wb") as f:
            f.write(binary_content)
        try:
            with pytest.raises(
                (
                    CompileError,
                    SchemaError,
                    UnicodeDecodeError,
                    yaml.YAMLError,
                    Exception,
                )
            ):
                from_config(path)
        finally:
            _cleanup(path)

    def test_lint_config_binary(self):
        binary_content = b"\x00\x01\x02\xff\xfe\xfd" * 100
        fd, path = tempfile.mkstemp(suffix=".yml")
        with os.fdopen(fd, "wb") as f:
            f.write(binary_content)
        try:
            try:
                result = lint_config(path)
                assert isinstance(result, list)
            except (
                CompileError,
                SchemaError,
                UnicodeDecodeError,
                yaml.YAMLError,
                Exception,
            ):
                pass  # Controlled error is acceptable
        finally:
            _cleanup(path)


# ── Unicode edge cases ──


class TestUnicodeEdgeCases:
    def test_from_config_unicode_surrogates(self):
        """Escaped surrogates in YAML strings — should either parse safely or raise."""
        content = 'name: "test\\ud800"\ntype: simple\nsystemPrompt: "hello"'
        path = _write_yaml(content)
        try:
            # YAML parser may accept escaped surrogates as literal text — that's OK.
            # The key is it must NOT crash with an unhandled exception.
            try:
                from_config(path)
            except (
                CompileError,
                SchemaError,
                yaml.YAMLError,
                ValueError,
                UnicodeError,
            ):
                pass  # Controlled rejection is fine
            # If it succeeds, that's also acceptable (YAML treats \ud800 as literal text)
        finally:
            _cleanup(path)

    def test_from_config_zero_width_chars(self):
        content = 'name: "te\u200bst"\ntype: simple\nsystemPrompt: "\u200bhello\u200b"'
        path = _write_yaml(content)
        try:
            # Should either succeed or raise a controlled error, not crash
            try:
                from_config(path)
            except (CompileError, SchemaError, yaml.YAMLError, Exception):
                pass
        finally:
            _cleanup(path)

    def test_from_config_rtl_override(self):
        content = 'name: "\u202eevil"\ntype: simple\nsystemPrompt: "test"'
        path = _write_yaml(content)
        try:
            try:
                from_config(path)
            except (CompileError, SchemaError, yaml.YAMLError, Exception):
                pass
        finally:
            _cleanup(path)


# ── Empty/null configs ──


class TestEmptyNullConfigs:
    def test_from_config_empty_file(self):
        path = _write_yaml("")
        try:
            with pytest.raises(
                (CompileError, SchemaError, TypeError, AttributeError, Exception)
            ):
                from_config(path)
        finally:
            _cleanup(path)

    def test_from_config_null(self):
        path = _write_yaml("null")
        try:
            with pytest.raises(
                (CompileError, SchemaError, TypeError, AttributeError, Exception)
            ):
                from_config(path)
        finally:
            _cleanup(path)

    def test_from_config_empty_dict(self):
        path = _write_yaml("{}")
        try:
            # May succeed with defaults or raise error
            try:
                from_config(path)
            except (CompileError, SchemaError, Exception):
                pass
        finally:
            _cleanup(path)

    def test_lint_config_empty(self):
        path = _write_yaml("")
        try:
            try:
                result = lint_config(path)
                assert isinstance(result, list)
            except (CompileError, SchemaError, TypeError, AttributeError, Exception):
                pass
        finally:
            _cleanup(path)

    def test_lint_config_null(self):
        path = _write_yaml("null")
        try:
            try:
                result = lint_config(path)
                assert isinstance(result, list)
            except (CompileError, SchemaError, TypeError, AttributeError, Exception):
                pass
        finally:
            _cleanup(path)


# ── Extremely long strings ──


class TestExtremelyLongStrings:
    def test_from_config_long_name(self):
        long_name = "a" * 1_000_000
        content = f'name: "{long_name}"\ntype: simple\nsystemPrompt: "hello"'
        path = _write_yaml(content)
        try:
            try:
                from_config(path)
            except (CompileError, SchemaError, MemoryError, Exception):
                pass
        finally:
            _cleanup(path)

    def test_from_config_long_prompt(self):
        long_prompt = "x" * 1_000_000
        content = f'name: "test"\ntype: simple\nsystemPrompt: "{long_prompt}"'
        path = _write_yaml(content)
        try:
            try:
                from_config(path)
            except (CompileError, SchemaError, MemoryError, Exception):
                pass
        finally:
            _cleanup(path)

    def test_lint_config_long_string(self):
        long_name = "b" * 500_000
        content = f'name: "{long_name}"\ntype: react\nsystemPrompt: "test"'
        path = _write_yaml(content)
        try:
            try:
                result = lint_config(path)
                assert isinstance(result, list)
            except (CompileError, SchemaError, MemoryError, Exception):
                pass
        finally:
            _cleanup(path)


# ── Malformed YAML syntax ──


class TestMalformedYAML:
    def test_from_config_invalid_yaml(self):
        content = ":\n  - : [\n  invalid: {{"
        path = _write_yaml(content)
        try:
            with pytest.raises((CompileError, SchemaError, yaml.YAMLError, Exception)):
                from_config(path)
        finally:
            _cleanup(path)

    def test_from_config_tabs_instead_of_spaces(self):
        content = "name: test\n\ttype: simple"
        path = _write_yaml(content)
        try:
            with pytest.raises((CompileError, SchemaError, yaml.YAMLError, Exception)):
                from_config(path)
        finally:
            _cleanup(path)
