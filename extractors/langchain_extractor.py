"""
extractors.langchain_extractor — Extract config from LangChain AgentExecutor (I07).

Handles:
  - AgentExecutor (ReAct agent with tools)
  - RunnableSequence (LCEL chain)
  - LLMChain (legacy chain)

Maps to lambdagent types:
  - AgentExecutor → type: react
  - RunnableSequence → type: chain
  - LLMChain → type: simple
"""

from __future__ import annotations
from typing import Any, Dict
from .base import FrameworkExtractor, ExtractionError


class LangChainExtractor(FrameworkExtractor):

    @property
    def framework_name(self) -> str:
        return "langchain"

    def detect(self, obj: Any) -> bool:
        """Detect LangChain objects by class name and attributes."""
        cls_name = type(obj).__name__
        module = type(obj).__module__ or ""

        # AgentExecutor
        if cls_name == "AgentExecutor" or (
            hasattr(obj, 'agent') and hasattr(obj, 'tools') and
            hasattr(obj, 'max_iterations')
        ):
            return True

        # RunnableSequence (LCEL)
        if cls_name == "RunnableSequence" or (
            "langchain" in module and hasattr(obj, 'first') and hasattr(obj, 'last')
        ):
            return True

        # LLMChain
        if cls_name == "LLMChain" or (
            hasattr(obj, 'llm') and hasattr(obj, 'prompt') and
            not hasattr(obj, 'tools')
        ):
            return True

        return False

    def extract(self, obj: Any) -> Dict[str, Any]:
        cls_name = type(obj).__name__

        if cls_name == "AgentExecutor" or (
            hasattr(obj, 'agent') and hasattr(obj, 'tools')
        ):
            return self._extract_agent_executor(obj)

        if cls_name == "RunnableSequence" or (
            hasattr(obj, 'first') and hasattr(obj, 'last')
        ):
            return self._extract_runnable_sequence(obj)

        if hasattr(obj, 'llm') and hasattr(obj, 'prompt'):
            return self._extract_llm_chain(obj)

        raise ExtractionError(f"Unsupported LangChain object: {cls_name}")

    def _extract_agent_executor(self, executor) -> Dict:
        """AgentExecutor → type: react"""
        # Extract model info
        llm = self._safe_getattr(executor, 'agent', 'llm_chain', 'llm') or \
              self._safe_getattr(executor, 'agent', 'llm')
        model_name = (
            getattr(llm, 'model_name', None) or
            getattr(llm, 'model', None) or
            "unknown"
        ) if llm else "unknown"
        temperature = getattr(llm, 'temperature', 0.0) if llm else 0.0

        # Extract tools
        tools = getattr(executor, 'tools', [])
        tool_names = [getattr(t, 'name', str(t)) for t in tools]

        # Check for terminate equivalent
        has_terminate = any(
            n in ('terminate', 'final_answer', 'human', '_Exception')
            for n in tool_names
        )

        # Extract prompt
        prompt = ""
        chain = self._safe_getattr(executor, 'agent', 'llm_chain')
        if chain and hasattr(chain, 'prompt'):
            prompt = getattr(chain.prompt, 'template', str(chain.prompt))

        # Detect provider
        provider = self._detect_provider(llm, model_name)

        return {
            "agentId": f"langchain-{type(executor).__name__}",
            "name": getattr(executor, 'name', None) or "LangChain Agent",
            "type": "react",
            "model": {
                "provider": provider,
                "name": str(model_name),
                "temperature": float(temperature),
            },
            "systemPrompt": str(prompt)[:5000] if prompt else "",
            "react": {
                "maxSteps": getattr(executor, 'max_iterations', 15) or 15,
            },
            "mcp": {
                "localTools": tool_names + (["terminate"] if has_terminate else []),
            },
            "_source": "langchain",
            "_class": type(executor).__name__,
        }

    def _extract_runnable_sequence(self, seq) -> Dict:
        """RunnableSequence → type: chain"""
        steps = []
        if hasattr(seq, 'first'):
            steps.append(self._step_info(seq.first))
        if hasattr(seq, 'middle'):
            for s in seq.middle:
                steps.append(self._step_info(s))
        if hasattr(seq, 'last'):
            steps.append(self._step_info(seq.last))

        return {
            "agentId": "langchain-chain",
            "name": "LangChain Chain",
            "type": "chain",
            "chain": {"steps": steps},
            "_source": "langchain",
            "_class": "RunnableSequence",
        }

    def _extract_llm_chain(self, chain) -> Dict:
        """LLMChain → type: simple"""
        llm = getattr(chain, 'llm', None)
        model_name = getattr(llm, 'model_name', 'unknown') if llm else "unknown"

        prompt = ""
        if hasattr(chain, 'prompt'):
            prompt = getattr(chain.prompt, 'template', str(chain.prompt))

        return {
            "agentId": "langchain-simple",
            "name": "LangChain LLMChain",
            "type": "simple",
            "model": {
                "name": str(model_name),
            },
            "systemPrompt": str(prompt)[:5000],
            "_source": "langchain",
        }

    def _step_info(self, step) -> Dict:
        """Extract basic info from a chain step."""
        return {
            "name": getattr(step, 'name', type(step).__name__),
            "type": "simple",
            "systemPrompt": str(getattr(step, 'template', ''))[:1000],
        }

    def _detect_provider(self, llm, model_name: str) -> str:
        """Detect LLM provider from model object."""
        if llm is None:
            return "unknown"
        module = type(llm).__module__ or ""
        cls = type(llm).__name__

        if "anthropic" in module.lower() or "claude" in str(model_name).lower():
            return "anthropic"
        if "openai" in module.lower() or "gpt" in str(model_name).lower():
            return "openai"
        if "ollama" in module.lower():
            return "ollama"
        return "openai"  # default
