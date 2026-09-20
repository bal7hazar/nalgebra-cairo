## Summary

<!-- Work package (docs/PLAN.md), what was ported, upstream equivalents, deviations. -->

## Gas

<!-- Net gas table of the main operations, with the losing variants measured. -->

## Deviations and deferred items

<!-- Anything that differs from upstream or DESIGN.md, and what is explicitly deferred. -->

## Escalations

<!-- Re-exports, shared files, missing kernels in simba/base — for the orchestrator. -->

## Test plan

- [ ] `./scripts/check.sh --update` green (fmt, lint, build, tests, gas snapshots)
- [ ] Oracle vectors for the ported operations
- [ ] CI green (`gh pr checks --watch`)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
