"""
extractors.crewai_extractor — Extract config from CrewAI Crew (I08).

Maps:
  - Crew(process=sequential) → type: chain
  - Crew(process=hierarchical) → type: router
  - Individual Agent → type: react (role/goal/backstory → systemPrompt)
"""

from __future__ import annotations
from typing import Any, Dict, List
from .base import FrameworkExtractor, ExtractionError


class CrewAIExtractor(FrameworkExtractor):

    @property
    def framework_name(self) -> str:
        return "crewai"

    def detect(self, obj: Any) -> bool:
        cls_name = type(obj).__name__
        # Crew class
        if cls_name == "Crew" or (
            hasattr(obj, 'agents') and hasattr(obj, 'tasks') and
            hasattr(obj, 'kickoff')
        ):
            return True
        # Single Agent
        if cls_name == "Agent" and hasattr(obj, 'role') and hasattr(obj, 'goal'):
            return True
        return False

    def extract(self, obj: Any) -> Dict[str, Any]:
        cls_name = type(obj).__name__

        if cls_name == "Crew" or (hasattr(obj, 'kickoff') and hasattr(obj, 'agents')):
            return self._extract_crew(obj)

        if hasattr(obj, 'role') and hasattr(obj, 'goal'):
            return self._extract_agent(obj)

        raise ExtractionError(f"Unsupported CrewAI object: {cls_name}")

    def _extract_crew(self, crew) -> Dict:
        """Crew → chain or router based on process type."""
        agents = getattr(crew, 'agents', [])
        agent_configs = [self._extract_agent(a) for a in agents]

        process = str(getattr(crew, 'process', 'sequential')).lower()
        # CrewAI Process enum: "Process.sequential" → extract "sequential"
        if "sequential" in process:
            crew_type = "chain"
        elif "hierarchical" in process:
            crew_type = "router"
        else:
            crew_type = "parallel"

        config = {
            "agentId": f"crewai-crew",
            "name": getattr(crew, 'name', None) or "CrewAI Crew",
            "type": crew_type,
            "_source": "crewai",
            "_class": "Crew",
        }

        if crew_type == "chain":
            config["chain"] = {"steps": agent_configs}
        elif crew_type == "router":
            config["router"] = {
                "classifier": agent_configs[0] if agent_configs else {},
                "routes": {a.get("name", f"agent_{i}"): a
                           for i, a in enumerate(agent_configs[1:])},
            }
        else:
            config["parallel"] = {"agents": agent_configs}

        # Extract tasks info
        tasks = getattr(crew, 'tasks', [])
        if tasks:
            config["_tasks"] = [
                {
                    "description": getattr(t, 'description', '')[:500],
                    "agent": getattr(t, 'agent', None) and getattr(t.agent, 'role', ''),
                }
                for t in tasks
            ]

        return config

    def _extract_agent(self, agent) -> Dict:
        """Single CrewAI Agent → type: react"""
        role = getattr(agent, 'role', 'Agent')
        goal = getattr(agent, 'goal', '')
        backstory = getattr(agent, 'backstory', '')

        # Build system prompt from role/goal/backstory
        prompt = f"Role: {role}\nGoal: {goal}"
        if backstory:
            prompt += f"\nBackstory: {backstory}"

        # Extract tools
        tools = getattr(agent, 'tools', []) or []
        tool_names = [getattr(t, 'name', type(t).__name__) for t in tools]

        # Extract model
        llm = getattr(agent, 'llm', None)
        model_name = "unknown"
        if isinstance(llm, dict):
            model_name = llm.get('model', llm.get('model_name', 'unknown'))
        elif llm is not None:
            model_name = getattr(llm, 'model_name', getattr(llm, 'model', str(llm)))

        max_iter = getattr(agent, 'max_iter', 25) or 25

        return {
            "agentId": f"crewai-{role.lower().replace(' ', '-')[:30]}",
            "name": str(role),
            "type": "react",
            "model": {"name": str(model_name)},
            "systemPrompt": prompt,
            "react": {"maxSteps": max_iter},
            "mcp": {"localTools": tool_names},
            "_source": "crewai",
            "_class": "Agent",
        }
