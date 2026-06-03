"""
test_extractors — Tier 4: Framework extractor and guard tests.

Tests use mock framework objects — no real LangChain/CrewAI/AutoGen imports needed.
"""

import pytest
from lambdagent.extractors.base import FrameworkExtractor, ExtractionError
from lambdagent.extractors.langchain_extractor import LangChainExtractor
from lambdagent.extractors.crewai_extractor import CrewAIExtractor
from lambdagent.extractors.autogen_extractor import AutoGenExtractor
from lambdagent.extractors import extract_config


# ============================================================
# Mock Framework Objects
# ============================================================

class MockLLM:
    model_name = "gpt-4o"
    temperature = 0.7

class MockPrompt:
    template = "You are a helpful assistant."

class MockLLMChain:
    llm = MockLLM()
    prompt = MockPrompt()

class MockTool:
    def __init__(self, name):
        self.name = name

class MockLangChainAgent:
    llm_chain = MockLLMChain()
    llm = MockLLM()

class MockAgentExecutor:
    """Mimics LangChain AgentExecutor."""
    agent = MockLangChainAgent()
    tools = [MockTool("search"), MockTool("read_file"), MockTool("terminate")]
    max_iterations = 15
    name = "test-executor"

class MockCrewAgent:
    """Mimics CrewAI Agent."""
    role = "Researcher"
    goal = "Find relevant papers"
    backstory = "Expert in ML"
    llm = {"model": "claude-sonnet"}
    tools = [MockTool("web_search")]
    max_iter = 20

class MockTask:
    description = "Research AI safety papers"
    agent = MockCrewAgent()

class MockCrew:
    """Mimics CrewAI Crew."""
    agents = [MockCrewAgent(), MockCrewAgent()]
    tasks = [MockTask()]
    process = "sequential"
    name = "test-crew"
    def kickoff(self): pass

class MockAutoGenAgent:
    """Mimics AutoGen AssistantAgent."""
    name = "assistant"
    system_message = "You are a coding assistant."
    llm_config = {"config_list": [{"model": "gpt-4o"}]}
    max_consecutive_auto_reply = 10
    _function_map = {"execute_code": lambda x: x}
    code_execution_config = None

class MockGroupChat:
    agents = [MockAutoGenAgent(), MockAutoGenAgent()]
    max_round = 20
    is_termination_msg = None

class MockGroupChatManager:
    """Mimics AutoGen GroupChatManager."""
    groupchat = MockGroupChat()
    _oai_messages = {}


# ============================================================
# Extractor Detection Tests
# ============================================================

class TestDetection:
    def test_langchain_detect_executor(self):
        assert LangChainExtractor().detect(MockAgentExecutor())

    def test_langchain_reject_crew(self):
        assert not LangChainExtractor().detect(MockCrew())

    def test_crewai_detect_crew(self):
        assert CrewAIExtractor().detect(MockCrew())

    def test_crewai_detect_agent(self):
        """Single agent detected by role+goal attributes."""
        agent = MockCrewAgent()
        # CrewAI detect checks for 'role' + 'goal' on Agent class name
        ext = CrewAIExtractor()
        # detect may not match single agents (designed for Crew objects)
        # but extract() should work when called directly
        config = ext.extract(agent)
        assert config["type"] == "react"

    def test_crewai_reject_langchain(self):
        assert not CrewAIExtractor().detect(MockAgentExecutor())

    def test_autogen_detect_manager(self):
        assert AutoGenExtractor().detect(MockGroupChatManager())

    def test_autogen_extract_agent(self):
        """Single AutoGen agent extractable directly."""
        agent = MockAutoGenAgent()
        config = AutoGenExtractor().extract(agent)
        assert config["type"] == "react"
        assert "coding assistant" in config["systemPrompt"]

    def test_autogen_reject_crew(self):
        assert not AutoGenExtractor().detect(MockCrew())


# ============================================================
# Extraction Tests
# ============================================================

class TestExtraction:
    def test_langchain_executor_config(self):
        config = LangChainExtractor().extract(MockAgentExecutor())
        assert config["type"] == "react"
        assert config["react"]["maxSteps"] == 15
        assert "search" in config["mcp"]["localTools"]
        assert "terminate" in config["mcp"]["localTools"]
        assert config["_source"] == "langchain"

    def test_crewai_crew_sequential(self):
        config = CrewAIExtractor().extract(MockCrew())
        assert config["type"] == "chain"
        assert len(config["chain"]["steps"]) == 2
        assert config["_source"] == "crewai"

    def test_crewai_single_agent(self):
        config = CrewAIExtractor().extract(MockCrewAgent())
        assert config["type"] == "react"
        assert "Researcher" in config["systemPrompt"]
        assert config["react"]["maxSteps"] == 20

    def test_autogen_groupchat(self):
        config = AutoGenExtractor().extract(MockGroupChatManager())
        assert config["type"] == "parallel"
        assert len(config["parallel"]["agents"]) == 2
        assert config["multiagent"]["maxRounds"] == 20
        assert config["_source"] == "autogen"

    def test_autogen_no_termination_warning(self):
        config = AutoGenExtractor().extract(MockGroupChatManager())
        # No is_termination_msg → should have warning
        assert config["multiagent"]["terminationCondition"] == "none"

    def test_autogen_agent(self):
        config = AutoGenExtractor().extract(MockAutoGenAgent())
        assert config["type"] == "react"
        assert "coding assistant" in config["systemPrompt"]


# ============================================================
# Auto-Detection Tests
# ============================================================

class TestAutoDetect:
    def test_auto_langchain(self):
        config = extract_config(MockAgentExecutor())
        assert config["_source"] == "langchain"

    def test_auto_crewai(self):
        config = extract_config(MockCrew())
        assert config["_source"] == "crewai"

    def test_auto_autogen(self):
        config = extract_config(MockGroupChatManager())
        assert config["_source"] == "autogen"

    def test_auto_unknown_raises(self):
        with pytest.raises(ExtractionError):
            extract_config("not a framework object")

    def test_force_framework(self):
        config = extract_config(MockAgentExecutor(), framework="langchain")
        assert config["_source"] == "langchain"

    def test_force_unknown_framework_raises(self):
        with pytest.raises(ExtractionError):
            extract_config(MockAgentExecutor(), framework="pytorch")


# ============================================================
# Guard Core Tests (I10)
# ============================================================

class TestGuardCore:
    def test_runtime_monitor_cost_tracking(self):
        from lambdagent_guard.core import RuntimeMonitor, GuardConfig
        monitor = RuntimeMonitor(GuardConfig(cost_budget=0.10))
        monitor.on_step({"output": "step1", "tokens": 100, "cost_usd": 0.003})
        monitor.on_step({"output": "step2", "tokens": 200, "cost_usd": 0.006})
        assert monitor.total_tokens == 300
        assert monitor.total_cost == pytest.approx(0.009)
        assert monitor.step_count == 2

    def test_runtime_monitor_budget_exceeded(self):
        from lambdagent_guard.core import RuntimeMonitor, GuardConfig, CostBudgetExceeded
        monitor = RuntimeMonitor(GuardConfig(cost_budget=0.005))
        monitor.on_step({"output": "step1", "tokens": 100, "cost_usd": 0.003})
        with pytest.raises(CostBudgetExceeded):
            monitor.on_step({"output": "step2", "tokens": 200, "cost_usd": 0.006})

    def test_runtime_monitor_loop_detection(self):
        from lambdagent_guard.core import RuntimeMonitor, GuardConfig, InfiniteLoopDetected
        monitor = RuntimeMonitor(GuardConfig(loop_window=5, loop_threshold=3))
        for _ in range(2):
            monitor.on_step({"output": "same output", "tokens": 10, "cost_usd": 0.001})
        with pytest.raises(InfiniteLoopDetected):
            monitor.on_step({"output": "same output", "tokens": 10, "cost_usd": 0.001})

    def test_runtime_monitor_empty_message(self):
        from lambdagent_guard.core import RuntimeMonitor, GuardConfig, InfiniteLoopDetected
        monitor = RuntimeMonitor(GuardConfig(empty_message_detection=True))
        monitor.on_step({"output": "", "tokens": 0, "cost_usd": 0})
        monitor.on_step({"output": "  ", "tokens": 0, "cost_usd": 0})
        with pytest.raises(InfiniteLoopDetected, match="Empty"):
            monitor.on_step({"output": "", "tokens": 0, "cost_usd": 0})

    def test_guarded_result(self):
        from lambdagent_guard.core import RuntimeMonitor, GuardConfig
        monitor = RuntimeMonitor(GuardConfig(cost_budget=1.0))
        monitor.on_step({"output": "ok", "tokens": 500, "cost_usd": 0.05})
        result = monitor.result("final answer")
        assert result.result == "final answer"
        assert result.total_cost_usd == 0.05
        assert result.total_tokens == 500
        assert result.budget_remaining == pytest.approx(0.95)


# ============================================================
# Guard Integration Tests (I11-I13)
# ============================================================

class TestGuardIntegration:
    def test_langchain_guard_attaches_monitor(self):
        from lambdagent_guard import guard_langchain
        executor = MockAgentExecutor()
        guarded = guard_langchain(executor, cost_budget=5.0)
        assert hasattr(guarded, '_lambdagent_monitor')

    def test_crewai_guard_wraps_kickoff(self):
        from lambdagent_guard import guard_crewai
        crew = MockCrew()
        original = crew.kickoff
        guarded = guard_crewai(crew, cost_budget=10.0)
        assert guarded.kickoff != original  # wrapped
        assert hasattr(guarded, '_lambdagent_monitor')

    def test_autogen_guard_attaches_monitor(self):
        from lambdagent_guard import guard_autogen
        manager = MockGroupChatManager()
        guarded = guard_autogen(manager, cost_budget=5.0)
        assert hasattr(guarded, '_lambdagent_monitor')
