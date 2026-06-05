"""
lambdagent.builtin_tools.registry — Built-in tool registry

Provides BUILTIN_TOOLS dict mapping tool names to (fn, schema) pairs.
Used by from_config compiler to resolve localTools references.
"""
from __future__ import annotations

from lambdagent.primitives import Tool
from lambdagent.validated_tool import ValidatedTool

from .file_tools import (
    read_file, ReadFileSchema,
    edit_file, EditFileSchema,
    write_file, WriteFileSchema,
    list_files, ListFilesSchema,
    search_content, SearchContentSchema,
)
from .shell_tools import (
    run_bash, BashSchema,
    git_status, git_diff, git_log, git_commit, git_branch,
)
from .code_tools import (
    code_search, CodeSearchSchema,
    project_map, ProjectMapSchema,
    run_tests, RunTestsSchema,
)
from .task_manager import task_create, task_update, task_list
from .web_tools import (
    notebook_edit, NotebookEditSchema,
    web_search, WebSearchSchema,
    web_fetch, WebFetchSchema,
)
from .knowledge_tools import chunk_split, ocr_extract, doc_generate, kb_manage
from .qa_tools import ingest_files, query_knowledge, list_knowledge, remove_knowledge, deep_analysis
from .wiki_tools import wiki_ingest, wiki_query, wiki_lint, wiki_search, wiki_status, wiki_grow

# PaaS services (P01-P08) — optional, only available when agentpaas is installed
try:
    from agentpaas.services.memory_service import memory_store, memory_recall, memory_list, memory_forget
    from agentpaas.services.scheduler import schedule_create, schedule_list, schedule_delete
    from agentpaas.services.notification import notify
    from agentpaas.services.event_bus import event_subscribe, event_list
    from agentpaas.services.profile import profile_get, profile_update
    from agentpaas.services.learning import learning_feedback, learning_strategies
    _HAS_PAAS_SERVICES = True
except ImportError:
    _HAS_PAAS_SERVICES = False


def _make_tool(name: str, fn, schema=None, description: str = "") -> ValidatedTool:
    """Create a ValidatedTool with optional schema."""
    return ValidatedTool(name, fn, schema=schema, description=description)


# ════════════════════════════════════════════════════════════
# Master registry
# ════════════════════════════════════════════════════════════

BUILTIN_TOOLS = {
    # File operations (A01-A05)
    "ReadFile":       _make_tool("ReadFile",       read_file,       ReadFileSchema,    "Read file contents with line numbers"),
    "EditFile":       _make_tool("EditFile",       edit_file,       EditFileSchema,    "Precise string replacement in files"),
    "WriteFile":      _make_tool("WriteFile",      write_file,      WriteFileSchema,   "Create or overwrite files"),
    "ListFiles":      _make_tool("ListFiles",      list_files,      ListFilesSchema,   "Find files by glob pattern"),
    "SearchContent":  _make_tool("SearchContent",  search_content,  SearchContentSchema, "Search file contents with regex"),

    # Shell execution (A06)
    "Bash":           _make_tool("Bash",           run_bash,        BashSchema,        "Execute shell commands"),

    # Git operations (A07)
    "GitStatus":      _make_tool("GitStatus",      git_status,      None, "Show git status"),
    "GitDiff":        _make_tool("GitDiff",        git_diff,        None, "Show git diff"),
    "GitLog":         _make_tool("GitLog",         git_log,         None, "Show git log"),
    "GitCommit":      _make_tool("GitCommit",      git_commit,      None, "Stage and commit changes"),
    "GitBranch":      _make_tool("GitBranch",      git_branch,      None, "List/create/switch branches"),

    # Code tools (A11-A13)
    "CodeSearch":     _make_tool("CodeSearch",     code_search,   CodeSearchSchema,  "Semantic code search"),
    "ProjectMap":     _make_tool("ProjectMap",     project_map,   ProjectMapSchema,  "Project structure overview"),
    "RunTests":       _make_tool("RunTests",       run_tests,     RunTestsSchema,    "Run tests (auto-detect framework)"),

    # Task management (A14)
    "TaskCreate":     _make_tool("TaskCreate",     task_create,   None, "Create a task"),
    "TaskUpdate":     _make_tool("TaskUpdate",     task_update,   None, "Update task status"),
    "TaskList":       _make_tool("TaskList",       task_list,     None, "List all tasks"),

    # Web + Notebook (A18-A20)
    "NotebookEdit":   _make_tool("NotebookEdit",   notebook_edit,  NotebookEditSchema, "Edit Jupyter Notebook cells"),
    "WebSearch":      _make_tool("WebSearch",       web_search,     WebSearchSchema,    "Web search via DuckDuckGo"),
    "WebFetch":       _make_tool("WebFetch",        web_fetch,      WebFetchSchema,     "Fetch URL content as text"),

    # Knowledge base (A34-A37)
    "ChunkSplit":     _make_tool("ChunkSplit",    chunk_split,   None, "Split documents into chunks"),
    "OCR":            _make_tool("OCR",           ocr_extract,   None, "Extract text from images/scanned PDFs"),
    "DocGen":         _make_tool("DocGen",        doc_generate,  None, "Generate PDF/Word/HTML from Markdown"),
    "KBCreate":       _make_tool("KBCreate",      kb_manage,     None, "Create knowledge base"),
    "KBAdd":          _make_tool("KBAdd",         kb_manage,     None, "Add documents to knowledge base"),
    "KBSearch":       _make_tool("KBSearch",      kb_manage,     None, "Search knowledge base"),
    "KBList":         _make_tool("KBList",        kb_manage,     None, "List knowledge bases"),

    # QA Agent tools (qaagent67)
    "IngestFiles":     _make_tool("IngestFiles",     ingest_files,     None, "Ingest files into knowledge base (batch, supports dir/glob/single file)"),
    "QueryKnowledge":  _make_tool("QueryKnowledge",  query_knowledge,  None, "Answer questions from knowledge base with source citations"),
    "ListKnowledge":   _make_tool("ListKnowledge",   list_knowledge,   None, "List indexed files and knowledge base status"),
    "RemoveKnowledge": _make_tool("RemoveKnowledge", remove_knowledge, None, "Remove files from knowledge base"),
    "DeepAnalysis":    _make_tool("DeepAnalysis",    deep_analysis,    None, "Deep analysis with parallel sub-agent retrieval (for complex queries: compare, summarize, evaluate)"),

    # LLM Wiki tools (qaagent67wiki — Karpathy pattern)
    "WikiIngest":     _make_tool("WikiIngest",  wiki_ingest,  None, "Ingest source files into wiki: extract → cross-reference → update related pages → rebuild index"),
    "WikiQuery":      _make_tool("WikiQuery",   wiki_query,   None, "Answer from compiled wiki: multi-page synthesis with tag routing and link following"),
    "WikiLint":       _make_tool("WikiLint",    wiki_lint,    None, "Health-check + auto-fix: contradictions, orphans, broken links, stale content (auto_fix=true to repair)"),
    "WikiSearch":     _make_tool("WikiSearch",  wiki_search,  None, "Search wiki pages by keyword"),
    "WikiStatus":     _make_tool("WikiStatus",  wiki_status,  None, "Wiki statistics + growth metrics + health score"),
    "WikiGrow":       _make_tool("WikiGrow",    wiki_grow,    None, "Grow wiki: find missing concepts, create synthesis pages (dry_run=true to preview)"),

    # Base case (always available)
    "terminate":      Tool("terminate", fn=lambda x: x),
}

# PaaS Services (P01-P08) — register only when agentpaas is importable
if _HAS_PAAS_SERVICES:
    BUILTIN_TOOLS.update({
        "MemoryStore":        _make_tool("MemoryStore",        memory_store,        None, "Store a memory (persistent)"),
        "MemoryRecall":       _make_tool("MemoryRecall",       memory_recall,       None, "Recall memories by query"),
        "MemoryList":         _make_tool("MemoryList",         memory_list,         None, "List all memories"),
        "MemoryForget":       _make_tool("MemoryForget",       memory_forget,       None, "Forget a memory"),
        "ScheduleCreate":     _make_tool("ScheduleCreate",     schedule_create,     None, "Create scheduled task"),
        "ScheduleList":       _make_tool("ScheduleList",       schedule_list,       None, "List scheduled tasks"),
        "ScheduleDelete":     _make_tool("ScheduleDelete",     schedule_delete,     None, "Delete scheduled task"),
        "Notify":             _make_tool("Notify",             notify,              None, "Send notification"),
        "EventSubscribe":     _make_tool("EventSubscribe",     event_subscribe,     None, "Subscribe to system events"),
        "EventList":          _make_tool("EventList",          event_list,          None, "List event subscriptions"),
        "ProfileGet":         _make_tool("ProfileGet",         profile_get,         None, "Get user profile"),
        "ProfileUpdate":      _make_tool("ProfileUpdate",      profile_update,      None, "Update user profile"),
        "LearningFeedback":   _make_tool("LearningFeedback",   learning_feedback,   None, "Record learning feedback"),
        "LearningStrategies": _make_tool("LearningStrategies", learning_strategies, None, "List learned strategies"),
    })


def get_builtin_tool(name: str) -> Tool | None:
    """Look up a built-in tool by name. Returns None if not found."""
    return BUILTIN_TOOLS.get(name)


def list_builtin_tools() -> list[str]:
    """Return sorted list of all built-in tool names."""
    return sorted(BUILTIN_TOOLS.keys())


def resolve_tools(names: list[str]) -> dict[str, Tool]:
    """Resolve a list of tool names to Tool instances.

    Returns only tools that exist in the registry.
    Unknown names are silently skipped.
    """
    result = {}
    for name in names:
        tool = BUILTIN_TOOLS.get(name)
        if tool:
            result[name] = tool
    return result
