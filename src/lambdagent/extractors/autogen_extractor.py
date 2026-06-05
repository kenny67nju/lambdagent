"""
extractors.autogen_extractor — Extract config from AutoGen GroupChatManager (I09).

Maps:
  - GroupChatManager → type: parallel (multi-agent loop)
  - AssistantAgent → type: react
  - UserProxyAgent → type: react (with code execution)

Detects termination risks:
  - is_termination_msg using exact string match → flagged as fragile (AutoGen #108)
"""

from __future__ import annotations
from typing import Any, Dict
from .base import FrameworkExtractor, ExtractionError


class AutoGenExtractor(FrameworkExtractor):
    @property
    def framework_name(self) -> str:
        return "autogen"

    def detect(self, obj: Any) -> bool:
        cls_name = type(obj).__name__
        module = type(obj).__module__ or ""

        # GroupChatManager
        if cls_name == "GroupChatManager" or (
            hasattr(obj, "groupchat") and hasattr(obj, "_oai_messages")
        ):
            return True

        # AssistantAgent / UserProxyAgent
        if cls_name in ("AssistantAgent", "UserProxyAgent") or (
            "autogen" in module
            and hasattr(obj, "llm_config")
            and hasattr(obj, "system_message")
        ):
            return True

        return False

    def extract(self, obj: Any) -> Dict[str, Any]:
        cls_name = type(obj).__name__

        if cls_name == "GroupChatManager" or hasattr(obj, "groupchat"):
            return self._extract_group_chat_manager(obj)

        if hasattr(obj, "llm_config") and hasattr(obj, "system_message"):
            return self._extract_agent(obj)

        raise ExtractionError(f"Unsupported AutoGen object: {cls_name}")

    def _extract_group_chat_manager(self, manager) -> Dict:
        """GroupChatManager → type: parallel"""
        chat = getattr(manager, "groupchat", None)
        if chat is None:
            raise ExtractionError("GroupChatManager has no groupchat attribute")

        agents = getattr(chat, "agents", [])
        agent_configs = [self._extract_agent(a) for a in agents]

        max_round = getattr(chat, "max_round", 10) or 10

        # Detect termination condition
        termination_msg = getattr(chat, "is_termination_msg", None)
        termination_info = self._analyze_termination(termination_msg)

        config = {
            "agentId": "autogen-groupchat",
            "name": "AutoGen GroupChat",
            "type": "parallel",
            "parallel": {"agents": agent_configs},
            "multiagent": {
                "maxRounds": max_round,
                "terminationCondition": termination_info["type"],
            },
            "_source": "autogen",
            "_class": "GroupChatManager",
        }

        if termination_info.get("warning"):
            config["_termination_warning"] = termination_info["warning"]

        return config

    def _extract_agent(self, agent) -> Dict:
        """AssistantAgent/UserProxyAgent → type: react"""
        name = getattr(agent, "name", type(agent).__name__)
        system_message = getattr(agent, "system_message", "") or ""

        # Extract model from llm_config
        llm_config = getattr(agent, "llm_config", {}) or {}
        model_name = "unknown"
        if isinstance(llm_config, dict):
            config_list = llm_config.get("config_list", [])
            if config_list and isinstance(config_list, list):
                model_name = config_list[0].get("model", "unknown")
            elif "model" in llm_config:
                model_name = llm_config["model"]

        # Extract tools/functions
        func_map = getattr(agent, "_function_map", {}) or {}
        tool_names = list(func_map.keys())

        max_reply = getattr(agent, "max_consecutive_auto_reply", 10) or 10

        # Detect if this is a code-executing agent
        code_config = getattr(agent, "code_execution_config", None)
        is_code_agent = code_config is not None and code_config is not False

        config = {
            "agentId": f"autogen-{name.lower().replace(' ', '-')[:30]}",
            "name": str(name),
            "type": "react",
            "model": {"name": str(model_name)},
            "systemPrompt": str(system_message)[:5000],
            "react": {"maxSteps": max_reply},
            "mcp": {"localTools": tool_names},
            "_source": "autogen",
            "_class": type(agent).__name__,
        }

        if is_code_agent:
            config["_code_execution"] = True
            config["mcp"]["localTools"].append("execute_code")

        return config

    def _analyze_termination(self, termination_fn) -> Dict:
        """Analyze termination condition for fragility risks."""
        if termination_fn is None:
            return {
                "type": "none",
                "warning": (
                    "No termination condition set. GroupChat will run until "
                    "max_round is reached. This is equivalent to a Y combinator "
                    "without a base case — forced truncation, not clean exit."
                ),
            }

        # Check if it's a simple string matcher (AutoGen #108 risk)
        # Most AutoGen termination functions check for exact "TERMINATE" string
        source = ""
        try:
            import inspect

            source = inspect.getsource(termination_fn)
        except Exception:
            pass

        if "TERMINATE" in source and ("==" in source or "in " in source):
            return {
                "type": "string_match",
                "warning": (
                    "Termination uses exact string matching for 'TERMINATE'. "
                    "This is fragile: LLM may write '(TERMINATE)' or 'Terminate.' "
                    "which won't match. (cf. AutoGen issue #108, #391). "
                    "Recommend case-insensitive substring matching."
                ),
            }

        return {"type": "function"}
