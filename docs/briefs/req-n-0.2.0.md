# REQ-N — release request for nalgebra 0.2.0 (documents only)

Lot of the nalgebra track of Slingfall (orchestrator: herdr project `slingfall-nalgebra`), 2026-10-03.
Runs on the VPS. Commit this brief verbatim as `docs/briefs/req-n-0.2.0.md` (first commit).

## Goal

The batch release request of nalgebra 0.2.0 for the project manager's go (slingfall `OPERATIONS.md` §7:
one request, one go naming every row: package, version, commit, sha256 of the archive built from a checkout
detached exactly at that commit). **Never run `scarb publish`.**

1. Release commit: efa49dd78ff614e5c74989e3c647fa2992a546d0 (nalgebra-cairo main after #106).
2. Run `scripts/release.py request --commit efa49dd78ff614e5c74989e3c647fa2992a546d0` (the request mode merged in #105), under the memory cap:
   `prlimit --as=8589934592 -- python3 scripts/release.py request …` (every `scarb package` it starts inherits
   the cap). It builds each published package's archive with `scarb package --no-verify -p` in a clean
   checkout detached at that commit and writes `docs/releases/0.2.0.md` (rows in publication order, sha256,
   sizes, the ordered `scarb publish -p` + verify commands). The push to main runs the full CI on that commit (Consumer cost included): wait until that run has completed and succeeded before the real request (`gh run list -R bal7hazar/nalgebra-cairo --commit <sha>`, one call every 5 minutes). Read its own refusals (detached HEAD, commit on
   GitHub, Consumer cost succeeded on that commit, version not yet published).
3. Reproducibility: run it twice; every row's sha256 must be equal between the two runs (show the diff of the
   two tables: empty).
4. The facade `nalgebra` is the heaviest package. **If the cap kills its packaging** (an allocation failure),
   stop there and report it with the exact output; do not retry another way: the orchestrator will build that
   row on the Mac.
5. Spot proof the project manager also runs: for 2-3 packages with no 0.2.0 dependency (e.g. `nalgebra_core`),
   build the archive WITH verification too (`scarb package -p`) and show the sha256 is the same.

Allowlist: the brief and `docs/releases/0.2.0.md`. One PR `docs(release): request for nalgebra 0.2.0`
("Audit: none needed — release request, documents only"). Never merge, never publish, never launch a review
or any agent; foreground only; `REPORT.md` at the worktree root (not committed), with both runs' sha256
tables, the wall times and the PR number and head.

Reading CI logs (programme rule): wait for the run to complete, then `gh run view <id> -R <owner>/<repo> --log-failed`, alone in its call (if your thread cannot run it, name the job and run id in your report and end your turn: the orchestrator relays the log). Never `gh api …/logs`. To read a sibling repository (simba-cairo, fixed-cairo), use the Read tool on its checkout. A bare command refused: report its exact text, no other attempt. At most one GitHub call per PR every 5 minutes. Never probe the shared heavy-build lock. Mac repositories by absolute path (/Users/bal7hazar/git/<repo>).
Signals (programme rule after an incident): a thread, a test or a script signals only processes it started itself, by pid or pgid taken from `$!` or ids it recorded. Never find a pid by searching (`ps | grep`, `pgrep`, `pkill`, `killall`).
Picking up main: after any push (and until nexus #83 is installed), run `git merge origin/main`, alone in its call. `git rebase origin/main` in exactly that form is allowed only before your first push, once #83 is installed; every other rebase form and every force push stay refused.
Files: to remove an untracked file of your own worktree, use `git clean -f -- <exact path>` (nexus #92); a bare `rm` stays refused. NEVER run `git clean` with `-d`, `-x` or `-X`: it deletes your own .herdr-project folder (brief, report). `gh pr edit` works for a PR title and body (nexus #91).
Memory (organisation rule): the 8 GB cap (`prlimit --as=8589934592 -- /usr/bin/time -v <command>`) applies to Cairo builds and measures (scarb, snforge) on the VPS; every peak-memory measure stays capped; a Node suite, or a whole pre-push hook, runs uncapped only when each of its steps was measured well under 8 GB (`prlimit --as` kills Node at start-up). On the Mac there is no cap. Long pushes: if your pre-push hook may run long (several minutes), push with `git -c core.sshCommand='ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=40' push ...` (writes no config), otherwise GitHub drops the SSH connection (exit 141) and nothing lands. Real peak memory: measure it uncapped only on the Mac (64 GB, no lock), never uncapped on the VPS; on the VPS, `prlimit --as=8589934592` (an address-space cap, which can kill a Cairo build at ~6.6 GB of real memory) stays the cap, used only for Cairo work known to fit well under it.

Work autonomously, do not ask questions, do not widen the scope.

