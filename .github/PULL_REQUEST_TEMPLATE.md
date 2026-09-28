## Summary

<!-- Work package (docs/PLAN.md), what was ported, upstream equivalents, deviations. -->

## Gas

<!-- Net gas table of the main operations, with the losing variants measured. -->

## Deviations and deferred items

<!-- Anything that differs from upstream or DESIGN.md, and what is explicitly deferred. -->

## Escalations

<!-- Re-exports, shared files, missing kernels in simba/base — for the orchestrator. -->

## Test plan

- [ ] Crate-scoped checks green locally (fmt, `api_parity.py` / `shapegen.py --check`, build, lint, tests and gas snapshots of the touched packages); CI is the full gate
- [ ] Oracle vectors for the ported operations
- [ ] Package-split move PRs only: `tools/split/gas_compare.py --base origin/main --head gas/` reports 0 changed / missing / added, `tools/split/public_paths.py --check` green (docs/ORCHESTRATOR.md, move-PR checklist)
- [ ] CI green (`gh pr checks --watch`)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
