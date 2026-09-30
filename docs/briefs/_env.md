# Environment of every brief (nalgebra-cairo)

Every brief points here; read it before starting. It describes the machine, not the task.

- `scarb` / `snforge` on your PATH are shims that serialise heavy builds through a lock shared with
  other projects: a command may wait silently for minutes, which is normal. Your unit is capped at
  14 GB.
- Shared, CPU-capped machine: never two `scarb` / `snforge` commands at once; run EVERY `scarb` /
  `snforge` command in the FOREGROUND with `timeout: 3600000`; never background a command and end
  your turn (headless mode would end your session); prefix python commands with `timeout 600`. If a
  run dies with "Killed" / exit 137, wait a minute and re-run.
- NEVER run `./scripts/check.sh` nor any `--workspace` command locally: the pull-request CI is the
  full gate. Local checks are crate-scoped: `scarb fmt --check`, `python3 scripts/api_parity.py
  --check`, `python3 tools/shapegen/shapegen.py --check`, `scarb build -p <pkg>`, `scarb lint -p
  <pkg> --deny-warnings`, `snforge test -p <pkg> [filter]` while iterating, and at the end an
  UNFILTERED `snforge test -p <pkg> | python3 scripts/gas_report.py --update gas/` for every package
  you touched (refreshes its gas snapshot).
- Push early, then `gh pr checks --watch` (wait until the checks are registered) and fix from the CI
  output (a gas difference in a package you did not re-run: re-run that package unfiltered with
  `--update`). Commit coherent states early and often (every hour at least).
- Steps criterion (owner): stay as close as possible to the Rust API when it costs no Cairo steps;
  when a faithful formulation costs steps, the cheaper one wins (within the oracle tolerance) and
  the deviation is documented; a parity item that can only be ported with a costlier formulation is
  reported in `REPORT.md` (Escalations) instead of being ported as is.
