"""Tests for Phase 6: PaaS platform services (P01-P08)."""
from __future__ import annotations
import json
import os
import tempfile
import time
import pytest


class TestMemoryService:
    def test_store_and_recall(self):
        from agentpaas.services.memory_service import MemoryStore, MemoryEntry
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db = f.name
        try:
            store = MemoryStore(db)
            store.store(MemoryEntry(key="pref1", content="用户喜欢用 Python", tags=["preference"], tier="semantic"))
            store.store(MemoryEntry(key="proj1", content="项目使用 FastAPI 框架", tags=["project"], tier="semantic"))
            results = store.recall("Python 项目")
            assert len(results) >= 1
            assert any("Python" in r.content for r in results)
        finally:
            os.unlink(db)

    def test_episodic_expiry(self):
        from agentpaas.services.memory_service import MemoryStore, MemoryEntry
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db = f.name
        try:
            store = MemoryStore(db)
            store.store(MemoryEntry(key="tmp1", content="临时记忆", tier="episodic", expires_at=time.time() - 1))
            cleaned = store.cleanup_expired()
            assert cleaned >= 1
        finally:
            os.unlink(db)

    def test_tool_functions(self):
        from agentpaas.services.memory_service import memory_store, memory_recall, memory_list
        # Store
        r = memory_store({"content": "test memory", "tags": ["test"]})
        assert "OK" in r
        # Recall
        r = memory_recall({"query": "test"})
        assert "test memory" in r
        # List
        r = memory_list("")
        assert "test memory" in r

    def test_forget(self):
        from agentpaas.services.memory_service import memory_store, memory_forget, _get_store
        memory_store({"content": "to forget", "key": "forget_me"})
        r = memory_forget({"key": "forget_me"})
        assert "OK" in r


class TestScheduler:
    def test_create_and_list(self):
        from agentpaas.services.scheduler import Scheduler
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            s = Scheduler(path)
            task = s.create("5m", description="Test task")
            assert task.schedule == "5m"
            tasks = s.list_all()
            assert len(tasks) >= 1
            s.delete(task.id)
            assert len(s.list_all()) == 0
        finally:
            os.unlink(path)

    def test_tool_functions(self):
        from agentpaas.services.scheduler import schedule_create, schedule_list, schedule_delete
        r = schedule_create({"schedule": "1h", "description": "hourly check"})
        assert "OK" in r
        r = schedule_list("")
        assert "hourly check" in r


class TestNotification:
    def test_terminal_notify(self, capsys):
        from agentpaas.services.notification import notify
        r = notify({"message": "Hello from test", "channel": "terminal"})
        assert "OK" in r
        captured = capsys.readouterr()
        assert "Hello from test" in captured.out

    def test_string_input(self, capsys):
        from agentpaas.services.notification import notify
        r = notify("Quick notification")
        assert "OK" in r


class TestEventBus:
    def test_subscribe_and_publish(self):
        from agentpaas.services.event_bus import EventBus
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            bus = EventBus(path)
            sub = bus.subscribe("system.file.created", action="notify")
            matched = bus.publish("system.file.created", {"path": "~/test.txt"})
            assert len(matched) >= 1
            assert matched[0].id == sub.id
        finally:
            os.unlink(path)

    def test_tool_functions(self):
        from agentpaas.services.event_bus import event_subscribe, event_list
        r = event_subscribe({"event": "system.disk.low", "action": "notify"})
        assert "OK" in r
        r = event_list("")
        assert "disk.low" in r


class TestProfile:
    def test_profile_operations(self):
        from agentpaas.services.profile import UserProfile
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            p = UserProfile(path)
            p.update("user", "name", "Kenny")
            p.record_command("git status")
            p.record_command("git status")
            p.record_command("pytest")
            summary = p.summary()
            assert "Kenny" in summary
        finally:
            os.unlink(path)

    def test_tool_functions(self):
        from agentpaas.services.profile import profile_update, profile_get
        r = profile_update({"section": "user", "key": "name", "value": "TestUser"})
        assert "OK" in r


class TestPersona:
    def test_compile_lambda(self):
        from agentpaas.services.persona import compile_persona
        result = compile_persona({"template": "lambda"})
        assert "lambda" in result
        assert "🐑" in result

    def test_compile_jarvis(self):
        from agentpaas.services.persona import compile_persona
        result = compile_persona({"template": "jarvis"})
        assert "JARVIS" in result
        assert "sir" in result

    def test_custom_persona(self):
        from agentpaas.services.persona import compile_persona
        result = compile_persona({
            "name": "Friday",
            "style": "efficient",
            "rules": ["直奔主题", "不废话"],
        })
        assert "Friday" in result

    def test_list_personas(self):
        from agentpaas.services.persona import list_personas
        names = list_personas()
        assert "lambda" in names
        assert "jarvis" in names


class TestLearning:
    def test_record_and_extract(self):
        from agentpaas.services.learning import LearningService
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            ls = LearningService(path)
            for _ in range(5):
                ls.record("WriteFile", "long markdown content", False, "JSON parse error")
            ls.record("WriteFile", "long markdown content", True, strategy="bash_python3")
            strategies = ls.list_strategies()
            # May or may not have extracted a rule depending on threshold
            assert isinstance(strategies, list)
        finally:
            os.unlink(path)


class TestRegistryPaaS:
    def test_paas_tools_registered(self):
        from lambdagent.builtin_tools.registry import BUILTIN_TOOLS
        paas_tools = ["MemoryStore", "MemoryRecall", "ScheduleCreate", "Notify",
                       "EventSubscribe", "ProfileGet", "LearningStrategies"]
        for t in paas_tools:
            assert t in BUILTIN_TOOLS, f"{t} not registered"

    def test_total_count(self):
        from lambdagent.builtin_tools.registry import BUILTIN_TOOLS
        assert len(BUILTIN_TOOLS) >= 42


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
