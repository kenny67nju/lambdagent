"""
lambdagent.validated_tool — Tool input validation via Pydantic

Wraps Tool with Pydantic BaseModel schema validation.
Invalid inputs return error messages instead of crashing.
"""
from __future__ import annotations
import json
import time
from typing import Any, Callable, Optional, Type
from .core import Term, Context


class ToolSchema:
    """Marker base for tool input schemas. Use pydantic.BaseModel in practice."""
    pass


class ValidatedTool(Term):
    """Tool with Pydantic input schema validation."""

    def __init__(self, name: str, fn: Callable, schema: Optional[Type] = None,
                 description: str = ""):
        super().__init__(name)
        self.fn = fn
        self.schema = schema
        self.description = description

    def apply(self, input: Any, ctx: Context | None = None) -> Any:
        ctx = ctx or Context()
        t0 = time.time()

        # Validate input if schema provided
        if self.schema:
            validated_input, error = self._validate(input)
            if error:
                duration = (time.time() - t0) * 1000
                ctx.log(self._name, self._trace_id, input, error, duration)
                return error
            input = validated_input

        result = self.fn(input)
        duration = (time.time() - t0) * 1000
        ctx.log(self._name, self._trace_id, input, result, duration)
        return result

    def _validate(self, input: Any) -> tuple:
        """Returns (validated_input, None) or (None, error_string)."""
        try:
            # Try to parse as JSON if string
            if isinstance(input, str):
                try:
                    input = json.loads(input)
                except (json.JSONDecodeError, ValueError):
                    input = {"input": input}

            if isinstance(input, dict):
                # Extract nested "input" field if present (from ReAct JSON format)
                if "input" in input and ("action" in input or "tool" in input):
                    input = input["input"]
                    if isinstance(input, str):
                        try:
                            input = json.loads(input)
                        except (json.JSONDecodeError, ValueError):
                            input = {"command": input} if "Bash" in self._name or "Shell" in self._name else {"input": input}
                validated = self.schema(**input) if isinstance(input, dict) else self.schema(input)
            else:
                validated = self.schema(input=input)

            return validated.dict() if hasattr(validated, 'dict') else validated.model_dump(), None
        except Exception as e:
            return None, f"[VALIDATION_ERROR] Tool '{self._name}': {e}"


# Built-in schemas for common tools
class ShellToolInput:
    """Schema for shell tool. Usable with or without Pydantic."""
    def __init__(self, command: str, timeout: int = 30, working_dir: str = "."):
        if not command or not isinstance(command, str):
            raise ValueError("command must be a non-empty string")
        if timeout < 1 or timeout > 600:
            raise ValueError("timeout must be between 1 and 600")
        self.command = command
        self.timeout = timeout
        self.working_dir = working_dir

    def dict(self):
        return {"command": self.command, "timeout": self.timeout, "working_dir": self.working_dir}
