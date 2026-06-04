# Changelog

All notable changes to this project will be documented in this file.
Format inspired by [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed
- **`pip install -e .` is now unblocked.** `pyproject.toml::project.license` migrated from the deprecated `{text = "BUSL-1.1"}` dict form to the SPDX string form (`"BUSL-1.1"`) required by setuptools ≥ 77.
- **Stop shadowing stdlib `types` and `trace`.** Renamed top-level `types.py → lam_types.py` and `trace.py → tracing.py`. All public symbols (`LamType`, `T_STR`, `TraceStore`, …) are re-exported unchanged from `lambdagent`; only direct sub-module imports need updating.
- **Closed YAML→eval RCE vector** in `fromconfig.compiler._compile_guard`. The `guard.validator` expression now runs through an AST-validated `_safe_eval` that blocks attribute access, dunders, and non-whitelisted calls.
- README inconsistencies: stale stats (lines/files/symbols), `your-org` placeholder URLs (3 places), nonexistent `serve` CLI command, missing `trace`/`tools`/`version` subcommands, install instruction pointing at PyPI before publishing, broken `nl2agent.py` path, under-documented architecture tree (13 → 60+ entries).

### Added
- 10 new optional-dependency extras matching the actual feature set: `[rag]`, `[knowledge]`, `[ocr]`, `[pdfgen]`, `[sandbox]`, `[checkpoint-crypto]`, `[redis]`, `[otel]`, `[tui]`, `[dev]`. Use `pip install -e ".[knowledge,rag]"` etc.
- `py.typed` marker (PEP 561) so downstream type checkers consume the 92%-covered annotations.
- `SECURITY.md` with disclosure policy and scope.
- `CONTRIBUTING.md` with setup, workflow, and PR expectations.
- `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1).
- GitHub Actions CI: pytest matrix on Python 3.10 / 3.11 / 3.12.
- `conftest.py` at repo root for flat-layout test discovery.
- `[tool.pytest.ini_options]` block: `importlib` import mode, deprecation-warning filters.

### Changed
- `providers.ClaudeLam` is **deprecated** (DeprecationWarning, removed in 0.3.0). Use `lambdagent.Lam(provider=ClaudeCodeProvider(...))` instead.
- `fromconfig.compiler` no longer silently registers a fake `ToolSearch` when the optional `agentexample` sibling is absent — calls now KeyError loudly.
- Maintainer email surfaced consistently: **qinliu@nju.edu.cn** in `pyproject.toml`, `SECURITY.md`, `CONTRIBUTING.md`, README.

### Removed
- `mcp_server_package/` placeholder directory (empty stub with a leaked `smail.nju.edu.cn` email and a conflicting MIT license claim). The MCP server setup snippet now lives in the main README.
- `tests/__init__.py` (interfered with pytest collection in the flat layout).

## [0.1.0] — 2026-06-04

- Initial public release on GitHub.
- Paper I + II + III implementations: 11 core λ-constructs, 5 multi-agent π-constructs, skill system, MCP/A2A/RAG/sandbox, type+effect system, CEK machine, graded cost prediction, algebraic-law rewriting, store-independence analysis.
- ~35k lines, 125 source files, 152 exported symbols.
- Business Source License 1.1, converting to Apache 2.0 on 2031-04-05.

[Unreleased]: https://github.com/kenny67nju/lambdagent/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/kenny67nju/lambdagent/releases/tag/v0.1.0
