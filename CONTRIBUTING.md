# Contributing to lambdagent

Thanks for considering a contribution!

## Quick Setup

```bash
git clone https://github.com/kenny67nju/lambdagent.git
cd lambdagent
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,all]"
```

Confirm the install:

```bash
pytest tests/ -q
```

You should see `~517 passed, 1 skipped` in ~25 seconds.

## Development Workflow

1. **Fork** the repo on GitHub.
2. **Branch**: `git checkout -b feat/short-description` or `fix/short-description`.
3. **Code** + **test**: add or update tests in `tests/`. A change without tests is rarely merged.
4. **Lint**: `ruff check .` (config inherits sensible defaults).
5. **Format**: keep imports grouped, prefer `from __future__ import annotations` in new files for forward-ref types.
6. **PR**: open against `main` with a clear title and a one-paragraph summary linking any related issue.

## What I Look For

- **Mathematical correctness** for any change in `primitives.py`, `extensions.py`, `multiagent.py`, `lam_types.py`, `effects.py`, `cek_machine.py`, `cost_grade.py`, `rewrite.py`. These trace back to the three lambdagent papers — please cite the paper section affected.
- **Stability** of the public API in `__init__.py`. Adding symbols is fine; renaming or removing requires a deprecation cycle.
- **Tests pass** locally and in CI on Python 3.10 / 3.11 / 3.12.
- **No new top-level module names that shadow stdlib** (`types`, `trace`, `io`, `json`, …).
- **Optional dependencies stay optional** — gate them with `try / except ImportError` and add an entry in `[project.optional-dependencies]`.

## Commit Style

Loosely Conventional Commits is preferred but not enforced:

```
feat(skills): add SkillRegistry.discover() with tag filtering
fix(react): handle empty action sequence in step 7
docs(readme): correct Loop signature in table
chore(ci): bump pytest to 8.x
```

Keep the subject under 72 chars. Body explains the *why*, not just the *what*.

## License of Contributions

By submitting a PR you agree your contribution is licensed under the same
[BUSL-1.1](./LICENSE) as the rest of the project, and that you grant the maintainer the right to relicense the project as needed (e.g. to apply a Change License earlier than 2031-04-05 if appropriate).

## Questions

- **Bugs / feature requests**: [Issues](https://github.com/kenny67nju/lambdagent/issues)
- **Open discussion**: [Discussions](https://github.com/kenny67nju/lambdagent/discussions)
- **Security**: see [SECURITY.md](./SECURITY.md) — do NOT open a public issue
- **Commercial licensing (BUSL-1.1)** or anything else: **qinliu@nju.edu.cn**
