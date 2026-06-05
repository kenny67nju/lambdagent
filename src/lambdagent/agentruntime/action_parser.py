"""agentruntime.action_parser — Extract structured actions from LLM output"""

from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class ParseError(Exception):
    pass


@dataclass
class Action:
    """Structured action extracted from LLM output."""

    tool: str
    input: Dict[str, Any] = field(default_factory=dict)
    thought: str = ""
    raw: str = ""


class ActionParser:
    """
    Parse LLM free-text output into structured Actions.

    Lambda: parse : String -> Action
    Priority: JSON block > inline JSON > XML > keyword > implicit terminate
    """

    TERMINATE_SIGNALS = [
        "final answer:",
        "task complete",
        "task is done",
        "i have completed",
        "here is the result:",
        "in conclusion,",
    ]

    def __init__(self, tool_names: List[str], tool_schemas: Dict[str, dict] = None):
        self.tool_names = tool_names
        self.tool_schemas = tool_schemas or {}

    def parse(self, llm_output: str) -> Action:
        """Parse LLM output into Action."""
        # Priority 1: JSON block ```json ... ```
        result = self._try_json_block(llm_output)
        if result:
            return self._validate_action(result)

        # Priority 2: Inline JSON
        result = self._try_inline_json(llm_output)
        if result:
            return self._validate_action(result)

        # Priority 3: XML tags
        result = self._try_xml(llm_output)
        if result:
            return self._validate_action(result)

        # Priority 4: Keyword match
        result = self._try_keyword(llm_output)
        if result:
            return result

        # Priority 5: Implicit terminate
        result = self._try_implicit_terminate(llm_output)
        if result:
            return result

        raise ParseError(
            f"Could not parse action from LLM output: {llm_output[:200]}..."
        )

    def _try_json_block(self, text: str) -> Optional[Action]:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return self._parse_json_to_action(match.group(1), text)
        return None

    def _try_inline_json(self, text: str) -> Optional[Action]:
        # Find JSON-like objects with action/tool key
        candidates = re.findall(r'\{[^{}]*"(?:action|tool)"[^{}]*\}', text)
        for candidate in candidates:
            result = self._parse_json_to_action(candidate, text)
            if result:
                return result
        return None

    def _try_xml(self, text: str) -> Optional[Action]:
        action_match = re.search(r"<action>\s*(\w+)\s*</action>", text)
        if not action_match:
            return None
        tool_name = action_match.group(1)
        input_match = re.search(r"<input>(.*?)</input>", text, re.DOTALL)
        inp_str = input_match.group(1).strip() if input_match else ""
        try:
            inp = json.loads(inp_str)
        except (json.JSONDecodeError, ValueError):
            inp = {"query": inp_str} if inp_str else {}

        answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if tool_name == "terminate" and answer_match:
            inp["answer"] = answer_match.group(1).strip()

        return Action(tool=tool_name, input=inp, thought=text, raw=text)

    def _try_keyword(self, text: str) -> Optional[Action]:
        text_lower = text.lower()
        for name in self.tool_names:
            if name != "terminate" and name.lower() in text_lower:
                return Action(tool=name, input={"query": text}, thought=text, raw=text)
        return None

    def _try_implicit_terminate(self, text: str) -> Optional[Action]:
        text_lower = text.lower()
        for signal in self.TERMINATE_SIGNALS:
            if signal in text_lower:
                # Extract answer after the signal
                idx = text_lower.find(signal)
                answer = text[idx + len(signal) :].strip()
                return Action(
                    tool="terminate",
                    input={"answer": answer or text},
                    thought=text,
                    raw=text,
                )

        # If "terminate" is in tool_names and text mentions it
        if "terminate" in self.tool_names and "terminate" in text_lower:
            return Action(
                tool="terminate", input={"answer": text}, thought=text, raw=text
            )

        return None

    def _parse_json_to_action(self, json_str: str, full_text: str) -> Optional[Action]:
        try:
            data = json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            return None

        tool_name = data.get("action") or data.get("tool") or data.get("name")
        if not tool_name:
            return None

        tool_input = (
            data.get("input") or data.get("args") or data.get("arguments") or {}
        )
        if isinstance(tool_input, str):
            tool_input = {"query": tool_input}
        if not isinstance(tool_input, dict):
            tool_input = {"value": tool_input}

        # For terminate, also capture "answer" field
        if tool_name == "terminate":
            answer = data.get("answer", data.get("result", ""))
            if answer:
                tool_input["answer"] = answer

        return Action(
            tool=tool_name, input=tool_input, thought=full_text, raw=full_text
        )

    def _validate_action(self, action: Action) -> Action:
        if action.tool not in self.tool_names:
            raise ParseError(
                f"Unknown tool: '{action.tool}'. Available: {self.tool_names}"
            )
        return action
