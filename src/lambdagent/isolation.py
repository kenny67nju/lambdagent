"""
lambdagent.isolation — Agent file isolation (Git Worktree)

Three-level isolation for multi-agent file operations:
  NONE      — shared CWD (single agent)
  DIRECTORY — temp directory copy (no Git)
  WORKTREE  — Git Worktree per agent (recommended)
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from lambdagent.core import LambdagentError

logger = logging.getLogger(__name__)


# ============================================================
# Exceptions
# ============================================================

class IsolationError(LambdagentError):
    """Raised when workspace creation or management fails."""


class UncommittedChangesError(IsolationError):
    """Raised when cleanup is attempted with uncommitted changes."""


# ============================================================
# IsolationLevel
# ============================================================

class IsolationLevel(Enum):
    NONE = "none"            # Shared CWD (single agent only)
    DIRECTORY = "directory"  # Temp directory isolation (no Git)
    WORKTREE = "worktree"    # Git Worktree isolation (recommended)


# ============================================================
# IsolatedWorkspace
# ============================================================

@dataclass
class IsolatedWorkspace:
    """Represents an isolated file-system workspace for one agent."""

    agent_id: str
    workspace_path: str
    isolation_level: IsolationLevel
    branch_name: Optional[str] = None
    original_cwd: Optional[str] = None
    git_root: Optional[str] = None
    _cleanup_registered: bool = field(default=False, repr=False)

    # ----------------------------------------------------------
    # helpers
    # ----------------------------------------------------------

    def _run_git(self, *args: str, cwd: Optional[str] = None) -> subprocess.CompletedProcess[str]:
        """Run a git command inside the workspace."""
        return subprocess.run(
            ["git", *args],
            cwd=cwd or self.workspace_path,
            capture_output=True,
            text=True,
            check=True,
        )

    # ----------------------------------------------------------
    # public API
    # ----------------------------------------------------------

    def has_changes(self) -> bool:
        """Detect whether the agent produced any file changes."""
        if self.isolation_level == IsolationLevel.WORKTREE:
            result = self._run_git("status", "--porcelain")
            return bool(result.stdout.strip())

        if self.isolation_level == IsolationLevel.DIRECTORY:
            # Without git history there is no cheap baseline comparison;
            # conservatively assume the agent wrote *something*.
            return True

        # NONE — no isolation, cannot reliably detect per-agent changes
        return False

    def get_diff(self) -> str:
        """Return all changes as a unified diff string."""
        if self.isolation_level == IsolationLevel.WORKTREE:
            # Staged + unstaged against the branch root
            self._run_git("add", "-A")
            result = self._run_git("diff", "HEAD")
            return result.stdout

        if self.isolation_level == IsolationLevel.DIRECTORY:
            return "(directory isolation — git diff unavailable)"

        return ""

    def commit(self, message: str) -> Optional[str]:
        """Commit all changes in the isolated branch. Returns the commit hash or *None*."""
        if self.isolation_level != IsolationLevel.WORKTREE:
            logger.warning("commit() is only supported in WORKTREE isolation")
            return None

        if not self.has_changes():
            logger.info("No changes to commit for agent %s", self.agent_id)
            return None

        self._run_git("add", "-A")
        self._run_git("commit", "-m", message)
        result = self._run_git("rev-parse", "HEAD")
        commit_hash = result.stdout.strip()
        logger.info("Agent %s committed %s on branch %s", self.agent_id, commit_hash, self.branch_name)
        return commit_hash

    def merge_back(self, target_branch: str = "main") -> bool:
        """Merge the agent's branch back into *target_branch*.

        Returns ``True`` on success, ``False`` on merge conflict (the caller
        must resolve it).
        """
        if self.isolation_level != IsolationLevel.WORKTREE:
            logger.warning("merge_back() is only supported in WORKTREE isolation")
            return False

        if self.git_root is None:
            raise IsolationError("git_root is not set — cannot merge")

        # Commit any outstanding changes before merging
        if self.has_changes():
            self.commit(f"agent/{self.agent_id}: auto-commit before merge")

        try:
            self._run_git("checkout", target_branch, cwd=self.git_root)
            self._run_git("merge", self.branch_name or "", "--no-edit", cwd=self.git_root)
            logger.info("Merged branch %s into %s", self.branch_name, target_branch)
            return True
        except subprocess.CalledProcessError as exc:
            logger.error("Merge conflict for agent %s: %s", self.agent_id, exc.stderr)
            # Abort the failed merge so the repo is left in a clean state
            try:
                self._run_git("merge", "--abort", cwd=self.git_root)
            except subprocess.CalledProcessError:
                pass
            return False


# ============================================================
# WorkspaceManager
# ============================================================

class WorkspaceManager:
    """Create, track, and tear-down isolated workspaces for agents."""

    WORKTREE_BASE: str = ".lambdagent/worktrees"
    SYMLINK_DIRS: List[str] = ["node_modules", ".venv", "__pycache__", ".tox"]

    def __init__(self, base_dir: str = ".") -> None:
        self.base_dir: str = os.path.abspath(base_dir)
        self._workspaces: Dict[str, IsolatedWorkspace] = {}
        self._git_root: Optional[str] = self._find_git_root()

    # ----------------------------------------------------------
    # internal helpers
    # ----------------------------------------------------------

    def _find_git_root(self) -> Optional[str]:
        """Return the git repository root, or *None* if not inside a repo."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=self.base_dir,
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    @staticmethod
    def _validate_slug(slug: str) -> None:
        """Raise if *slug* contains path-traversal characters or is too long."""
        if len(slug) > 64:
            raise IsolationError(f"Slug exceeds 64 characters: {slug!r}")
        if ".." in slug or "/" in slug or "\\" in slug:
            raise IsolationError(f"Slug contains illegal characters: {slug!r}")
        if not re.match(r"^[A-Za-z0-9_\-]+$", slug):
            raise IsolationError(
                f"Slug must be alphanumeric (plus '_' and '-'): {slug!r}"
            )

    def _symlink_large_dirs(self, target: str) -> None:
        """Create symlinks in *target* pointing back to large dirs in base_dir."""
        for dirname in self.SYMLINK_DIRS:
            src = os.path.join(self.base_dir, dirname)
            dst = os.path.join(target, dirname)
            if os.path.isdir(src) and not os.path.exists(dst):
                os.symlink(src, dst)
                logger.debug("Symlinked %s -> %s", dst, src)

    def _create_worktree(self, agent_id: str, slug: str) -> IsolatedWorkspace:
        """Create an isolated workspace via ``git worktree add``."""
        if self._git_root is None:
            raise IsolationError("No git root found — cannot create worktree")

        ts = int(time.time())
        branch_name = f"agent-{slug}-{ts}"
        worktree_dir = os.path.join(self._git_root, self.WORKTREE_BASE, slug)

        os.makedirs(os.path.dirname(worktree_dir), exist_ok=True)

        subprocess.run(
            ["git", "worktree", "add", "-b", branch_name, worktree_dir],
            cwd=self._git_root,
            capture_output=True,
            text=True,
            check=True,
        )

        self._symlink_large_dirs(worktree_dir)

        ws = IsolatedWorkspace(
            agent_id=agent_id,
            workspace_path=worktree_dir,
            isolation_level=IsolationLevel.WORKTREE,
            branch_name=branch_name,
            original_cwd=self.base_dir,
            git_root=self._git_root,
        )
        return ws

    def _create_directory(self, agent_id: str, slug: str) -> IsolatedWorkspace:
        """Create an isolated workspace by copying the project tree."""
        tmp_root = os.path.join(tempfile.gettempdir(), "lambdagent", slug)
        if os.path.exists(tmp_root):
            shutil.rmtree(tmp_root)

        ignore = shutil.ignore_patterns(
            ".git",
            *self.SYMLINK_DIRS,
            self.WORKTREE_BASE,
        )
        shutil.copytree(self.base_dir, tmp_root, ignore=ignore, dirs_exist_ok=False)

        self._symlink_large_dirs(tmp_root)

        ws = IsolatedWorkspace(
            agent_id=agent_id,
            workspace_path=tmp_root,
            isolation_level=IsolationLevel.DIRECTORY,
            original_cwd=self.base_dir,
        )
        return ws

    # ----------------------------------------------------------
    # public API
    # ----------------------------------------------------------

    def create(
        self,
        agent_id: str,
        level: IsolationLevel = IsolationLevel.WORKTREE,
        slug: Optional[str] = None,
    ) -> IsolatedWorkspace:
        """Create an isolated workspace for *agent_id*.

        If *level* is ``WORKTREE`` but no git root is detected the manager
        gracefully degrades to ``DIRECTORY`` isolation.
        """
        if agent_id in self._workspaces:
            raise IsolationError(f"Workspace already exists for agent {agent_id!r}")

        slug = slug or re.sub(r"[^A-Za-z0-9_\-]", "_", agent_id)[:64]
        self._validate_slug(slug)

        if level == IsolationLevel.NONE:
            ws = IsolatedWorkspace(
                agent_id=agent_id,
                workspace_path=self.base_dir,
                isolation_level=IsolationLevel.NONE,
                original_cwd=self.base_dir,
                git_root=self._git_root,
            )
        elif level == IsolationLevel.WORKTREE:
            if self._git_root is None:
                logger.warning(
                    "No git root detected — degrading from WORKTREE to DIRECTORY for agent %s",
                    agent_id,
                )
                ws = self._create_directory(agent_id, slug)
            else:
                ws = self._create_worktree(agent_id, slug)
        elif level == IsolationLevel.DIRECTORY:
            ws = self._create_directory(agent_id, slug)
        else:
            raise IsolationError(f"Unknown isolation level: {level!r}")

        self._workspaces[agent_id] = ws
        logger.info(
            "Created %s workspace for agent %s at %s",
            ws.isolation_level.value,
            agent_id,
            ws.workspace_path,
        )
        return ws

    def cleanup(self, agent_id: str, force: bool = False) -> None:
        """Remove the workspace for *agent_id*.

        Raises :class:`UncommittedChangesError` if the workspace has
        uncommitted changes and *force* is ``False``.
        """
        ws = self._workspaces.get(agent_id)
        if ws is None:
            raise IsolationError(f"No workspace found for agent {agent_id!r}")

        if not force and ws.has_changes():
            raise UncommittedChangesError(
                f"Agent {agent_id!r} has uncommitted changes. "
                "Pass force=True to discard them."
            )

        if ws.isolation_level == IsolationLevel.WORKTREE and self._git_root:
            try:
                subprocess.run(
                    ["git", "worktree", "remove", "--force", ws.workspace_path],
                    cwd=self._git_root,
                    capture_output=True,
                    text=True,
                    check=True,
                )
            except subprocess.CalledProcessError as exc:
                logger.warning("Failed to remove worktree: %s", exc.stderr)
                # Fallback: manual removal
                if os.path.isdir(ws.workspace_path):
                    shutil.rmtree(ws.workspace_path)

            # Optionally delete the branch
            if ws.branch_name:
                try:
                    subprocess.run(
                        ["git", "branch", "-D", ws.branch_name],
                        cwd=self._git_root,
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                except subprocess.CalledProcessError:
                    pass  # branch may have been merged already

        elif ws.isolation_level == IsolationLevel.DIRECTORY:
            if os.path.isdir(ws.workspace_path):
                shutil.rmtree(ws.workspace_path)

        # NONE — nothing to clean up on disk

        del self._workspaces[agent_id]
        logger.info("Cleaned up workspace for agent %s", agent_id)

    def cleanup_all(self, force: bool = False) -> None:
        """Remove every tracked workspace."""
        # Iterate over a snapshot of the keys since cleanup mutates the dict
        for agent_id in list(self._workspaces):
            self.cleanup(agent_id, force=force)

    def get(self, agent_id: str) -> Optional[IsolatedWorkspace]:
        """Return the workspace for *agent_id*, or ``None``."""
        return self._workspaces.get(agent_id)

    def list_active(self) -> List[IsolatedWorkspace]:
        """Return all currently active workspaces."""
        return list(self._workspaces.values())
