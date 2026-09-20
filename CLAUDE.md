# CLAUDE.md

Follow [AGENTS.md](AGENTS.md) — it is the single source of truth. Context in one glance:

- Plan: [docs/ROADMAP.md](docs/ROADMAP.md) · Decisions: [docs/DESIGN.md](docs/DESIGN.md) ·
  Evidence: [docs/BENCHMARK.md](docs/BENCHMARK.md)
- Gate: `./scripts/check.sh` (`--update` to refresh the gas snapshot)
- Design-time benchmarks: `cd benchmarks && snforge test -p <pkg> | python3 scripts/gas_report.py`
