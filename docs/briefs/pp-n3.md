# PP-N3 — nalgebra pre-push: the Cairo compile capped on the VPS

Lot of the nalgebra track of Slingfall (orchestrator: herdr project `slingfall-nalgebra`), 2026-10-03.
Light lot on the VPS. Its own push touches no manifest. Commit this brief verbatim as
`docs/briefs/pp-n3.md` (first commit). Allowlist: the brief and `scripts/prepush.sh` (and its header).

## Ruling (project manager, 2026-10-03)

The pre-push hook must never compile uncapped on the VPS, for any lot. A nalgebra workspace build peaks
near 11 GB; the VPS memory rule caps Cairo work at `prlimit --as=8589934592`. CI remains the gate: a hook
that defers to CI when the cap is reached weakens no merge check.

## Goal

1. Where the heavy-build lock exists (the VPS: `~/orchestrator/heavy-build.lock`), every `scarb` and
   `snforge` command of the Cairo block runs under `prlimit --as=8589934592 -- …`, **inside** the flock as
   today (the lock rules stay: the script holds the lock with `flock -E 75 -w 90`, exports
   `HEAVY_BUILD_LOCK_HELD=1` inside, signals only its own process groups, never probes the shared lock).
2. If the cap kills a step (an allocation failure: e.g. Rust's "memory allocation of N bytes failed" and an
   abort, `Cannot allocate memory`, or a signal exit such as 134/137 — find what scarb/snforge actually do
   under an address-space cap, by testing with a deliberately low cap such as `--as=300000000` on a small
   package), the hook prints exactly one line `memory cap reached: Cairo compile left to CI` and **passes**,
   like the lock-busy case. A real compile or test failure (a Cairo error, a failing test, a gas mismatch)
   still fails the push: the detection must not swallow those — show both cases.
3. fmt, the Python self-tests and the other light checks run as now.
4. Where the lock does not exist (the Mac), the compile is uncapped, as now.
5. Header comment: the cap, where it applies, and the "left to CI" behaviour.

## Verification and report

Runs at the new head, exit codes in the PR body: prose docs only (0); a passing fixed check (0); a Cairo
change on a private `HEAVY_BUILD_LOCK=$(mktemp)` file with the normal cap (0, compiles); the same with a
deliberately low cap injected for the test (0, the "memory cap reached" line); a Cairo compile error under
the normal cap (1); the lock busy on a private lock held by your own `flock <file> sleep` (0, the lock-busy
line). Push through the hook. Conventional commits; one push per review loop; never merge, never launch a
review or any agent; foreground only; `REPORT.md` at the worktree root (not committed), `## Summary` first.

Reading CI logs (programme rule): wait for the run to complete, then `gh run view <id> -R <owner>/<repo> --log-failed`, alone in its call (if your thread cannot run it, name the job and run id in your report and end your turn: the orchestrator relays the log). Never `gh api …/logs`. To read a sibling repository (simba-cairo, fixed-cairo), use the Read tool on its checkout. A bare command refused: report its exact text, no other attempt. At most one GitHub call per PR every 5 minutes. Never probe the shared heavy-build lock. Mac repositories by absolute path (/Users/bal7hazar/git/<repo>).
Signals (programme rule after an incident): a thread, a test or a script signals only processes it started itself, by pid or pgid taken from `$!` or ids it recorded. Never find a pid by searching (`ps | grep`, `pgrep`, `pkill`, `killall`).
Picking up main: after any push (and until nexus #83 is installed), run `git merge origin/main`, alone in its call. `git rebase origin/main` in exactly that form is allowed only before your first push, once #83 is installed; every other rebase form and every force push stay refused.
Files: to remove an untracked file of your own worktree, use `git clean -f -- <exact path>` (nexus #92); a bare `rm` stays refused. NEVER run `git clean` with `-d`, `-x` or `-X`: it deletes your own .herdr-project folder (brief, report). `gh pr edit` works for a PR title and body (nexus #91).
Memory (organisation rule): the 8 GB cap (`prlimit --as=8589934592 -- /usr/bin/time -v <command>`) applies to Cairo builds and measures (scarb, snforge) on the VPS; every peak-memory measure stays capped; a Node suite, or a whole pre-push hook, runs uncapped only when each of its steps was measured well under 8 GB (`prlimit --as` kills Node at start-up). On the Mac there is no cap. Long pushes: if your pre-push hook may run long (several minutes), push with `git -c core.sshCommand='ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=40' push ...` (writes no config), otherwise GitHub drops the SSH connection (exit 141) and nothing lands. Real peak memory: measure it uncapped only on the Mac (64 GB, no lock), never uncapped on the VPS; on the VPS, `prlimit --as=8589934592` (an address-space cap, which can kill a Cairo build at ~6.6 GB of real memory) stays the cap, used only for Cairo work known to fit well under it.

Work autonomously, do not ask questions, do not widen the scope.
