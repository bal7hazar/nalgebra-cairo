# CLAUDE.md

Follow [AGENTS.md](AGENTS.md) — it is the single source of truth. Context in one glance:

- Plan: [docs/PLAN.md](docs/PLAN.md) · Decisions: [docs/DESIGN.md](docs/DESIGN.md) ·
  Evidence: [docs/BENCHMARK.md](docs/BENCHMARK.md)
- Orchestration: [docs/ORCHESTRATOR.md](docs/ORCHESTRATOR.md) · Gate: `./scripts/check.sh` (`--update` to refresh `gas/`)
- Design-time benchmarks: `cd benchmarks && snforge test -p <pkg> | python3 scripts/gas_report.py`
