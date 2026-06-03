"""
lambdagent.builtin_tools.task_manager — Task management for agent sessions

Simple in-memory task tracking with JSON persistence.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class Task:
    id: str
    subject: str
    description: str = ""
    status: str = "pending"  # pending, in_progress, completed, deleted
    created_at: float = 0
    updated_at: float = 0
    blocked_by: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.created_at == 0:
            self.created_at = time.time()
            self.updated_at = self.created_at


class TaskManager:
    """In-memory task manager with optional JSON persistence."""

    def __init__(self, persist_path: str = ""):
        self._tasks: Dict[str, Task] = {}
        self._counter = 0
        self._persist_path = persist_path
        if persist_path and os.path.exists(persist_path):
            self._load()

    def create(self, subject: str, description: str = "", **metadata) -> Task:
        self._counter += 1
        task_id = str(self._counter)
        task = Task(id=task_id, subject=subject, description=description, metadata=metadata)
        self._tasks[task_id] = task
        self._save()
        return task

    def get(self, task_id: str) -> Optional[Task]:
        return self._tasks.get(task_id)

    def update(self, task_id: str, **kwargs) -> Optional[Task]:
        task = self._tasks.get(task_id)
        if not task:
            return None
        for key, val in kwargs.items():
            if hasattr(task, key) and key not in ("id", "created_at"):
                setattr(task, key, val)
        task.updated_at = time.time()
        if kwargs.get("status") == "deleted":
            del self._tasks[task_id]
        self._save()
        return task

    def list_all(self, status: str = "") -> List[Task]:
        tasks = list(self._tasks.values())
        if status:
            tasks = [t for t in tasks if t.status == status]
        return sorted(tasks, key=lambda t: t.created_at)

    def _save(self):
        if not self._persist_path:
            return
        os.makedirs(os.path.dirname(self._persist_path) or ".", exist_ok=True)
        data = {tid: asdict(t) for tid, t in self._tasks.items()}
        data["_counter"] = self._counter
        with open(self._persist_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _load(self):
        try:
            with open(self._persist_path) as f:
                data = json.load(f)
            self._counter = data.pop("_counter", 0)
            for tid, td in data.items():
                self._tasks[tid] = Task(**td)
        except Exception:
            pass


# Tool functions for agent use
_default_manager = TaskManager()

def task_create(input_val: Any) -> str:
    params = _parse(input_val)
    subject = params.get("subject", "")
    if not subject:
        return "[ERROR] subject is required"
    task = _default_manager.create(subject, params.get("description", ""))
    return f"[OK] Task #{task.id} created: {task.subject}"

def task_update(input_val: Any) -> str:
    params = _parse(input_val)
    task_id = params.get("task_id", params.get("id", ""))
    if not task_id:
        return "[ERROR] task_id is required"
    status = params.get("status", "")
    kwargs = {}
    if status:
        kwargs["status"] = status
    if "subject" in params:
        kwargs["subject"] = params["subject"]
    if "description" in params:
        kwargs["description"] = params["description"]
    task = _default_manager.update(task_id, **kwargs)
    if not task:
        return f"[ERROR] Task #{task_id} not found"
    return f"[OK] Task #{task_id} updated: status={task.status if task_id in _default_manager._tasks else 'deleted'}"

def task_list(input_val: Any) -> str:
    params = _parse(input_val) if input_val else {}
    status = params.get("status", "")
    tasks = _default_manager.list_all(status)
    if not tasks:
        return "[EMPTY] No tasks."
    lines = []
    for t in tasks:
        marker = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}.get(t.status, "[?]")
        lines.append(f"  #{t.id} {marker} {t.subject}")
    return "\n".join(lines)

def _parse(input_val: Any) -> dict:
    if isinstance(input_val, dict):
        return input_val
    if isinstance(input_val, str):
        try:
            return json.loads(input_val)
        except (json.JSONDecodeError, ValueError):
            return {"subject": input_val}
    return {}
