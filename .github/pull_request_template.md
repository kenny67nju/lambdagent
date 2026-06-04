<!-- Thanks for the PR. Keep this template short — delete sections that don't apply. -->

## Summary
<!-- 1-2 sentences: what does this change do? -->

## Why
<!-- Link the issue / paper section / discussion that motivated this -->
Closes #

## Test plan
<!-- Bulleted: what tests were added/modified; how to manually verify -->
- [ ] Added tests in `tests/`
- [ ] `pytest tests/ -q` passes locally
- [ ] No new top-level module names shadow stdlib
- [ ] If a public symbol moved or was renamed, added a re-export shim with `DeprecationWarning`

## Breaking changes?
<!-- If yes, describe the migration path -->
- [ ] No
- [ ] Yes — described in CHANGELOG and below:
