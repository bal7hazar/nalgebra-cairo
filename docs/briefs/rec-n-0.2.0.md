# REC-N — record of the nalgebra 0.2.0 publication (documents only)

Lot of the nalgebra track of Slingfall (orchestrator: herdr project `slingfall-nalgebra`), 2026-10-04.
Light lot on the VPS. Commit this brief verbatim as `docs/briefs/rec-n-0.2.0.md` (first commit).

nalgebra 0.2.0 was published by the orchestrator on the project manager's go (request
`docs/releases/0.2.0.md` at main ab3095f, file sha256 a8eff298679932aeb0cbbb8dd7d49cf39b9a67f52092947328418b8af1a46395):
all 54 packages from commit efa49dd78ff614e5c74989e3c647fa2992a546d0; every registry checksum equals its row (checked
54/54 after the last); annotated tag `v0.2.0` on efa49dd; GitHub release `v0.2.0`. Rows 1-52 were published on
2026-10-03 22:20-22:33Z under `prlimit --as` 8 GiB. Rows 53-54 under a one-off ruling of the Overseer (the facade's
publish verification exceeded the 8 GiB cap: "memory allocation of 114688 bytes failed"), each with
`flock heavy-build.lock env RAYON_NUM_THREADS=1 prlimit --as=25769803776 -- /usr/bin/time -v scarb publish -p <pkg>`,
started at 20 GB free: `nalgebra` (row 53) 2026-10-03 23:23:00Z, wall 2:28.09, maximum resident set size 9154636 kB;
`nalgebra_glam` (row 54) 2026-10-04 00:31:56Z, wall 0:22.71, maximum resident set size 1912516 kB.
Verify each fact you can yourself (registry index `https://scarbs.xyz/api/v1/index/na/lg/<package>.json` for a sample
including `nalgebra` and `nalgebra_glam`, `git ls-remote --tags origin v0.2.0`, `gh release view v0.2.0 -R
bal7hazar/nalgebra-cairo`).

Edit, allowlist exactly these files:
1. `docs/releases/0.2.0.md`: status "published" with dates, the 54/54 read-back, the tag and release links (the tag link
   to the tree at v0.2.0, the release link to the release), and a short section on the one-off for rows 53-54 with the
   two measured peaks and wall times.
2. `CHANGELOG.md`: the 0.2.0 heading dated (2026-10-03, the day of the first publication) instead of "unreleased".
3. `docs/PLAN.md`: rows OPT-0, OPT-1, OPT-2, API-1, REL done with their PRs; a row for NAL-S3 (nalgebra on simba 0.3.0 /
   fixed 0.5.0, waiting for glam_core on fixed 0.5.0) and OPT-3 (in progress).
4. `docs/SPLIT.md` §21: one new dated entry (2026-10-04) summarising 0.2.0; earlier entries untouched.

One PR `docs(release): record nalgebra 0.2.0`; "Audit: none needed — documents". Never merge, never publish, never launch
a review or any agent; foreground only; `REPORT.md` at the worktree root (not committed).

Reading CI logs (programme rule): wait for the run to complete, then `gh run view <id> -R <owner>/<repo> --log-failed`, alone in its call (if your thread cannot run it, name the job and run id in your report and end your turn: the orchestrator relays the log). Never `gh api …/logs`. To read a sibling repository (simba-cairo, fixed-cairo), use the Read tool on its checkout. A bare command refused: report its exact text, no other attempt. At most one GitHub call per PR every 5 minutes. Never probe the shared heavy-build lock. Mac repositories by absolute path (/Users/bal7hazar/git/<repo>).
Signals (programme rule after an incident): a thread, a test or a script signals only processes it started itself, by pid or pgid taken from `$!` or ids it recorded. Never find a pid by searching (`ps | grep`, `pgrep`, `pkill`, `killall`).
Picking up main: after any push (and until nexus #83 is installed), run `git merge origin/main`, alone in its call. `git rebase origin/main` in exactly that form is allowed only before your first push, once #83 is installed; every other rebase form and every force push stay refused.
Files: to remove an untracked file of your own worktree, use `git clean -f -- <exact path>` (nexus #92); a bare `rm` stays refused. NEVER run `git clean` with `-d`, `-x` or `-X`: it deletes your own .herdr-project folder (brief, report). `gh pr edit` works for a PR title and body (nexus #91).
Memory (organisation rule): the 8 GB cap (`prlimit --as=8589934592 -- /usr/bin/time -v <command>`) applies to Cairo builds and measures (scarb, snforge) on the VPS; every peak-memory measure stays capped; a Node suite, or a whole pre-push hook, runs uncapped only when each of its steps was measured well under 8 GB (`prlimit --as` kills Node at start-up). On the Mac there is no cap. Long pushes: if your pre-push hook may run long (several minutes), push with `git -c core.sshCommand='ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=40' push ...` (writes no config), otherwise GitHub drops the SSH connection (exit 141) and nothing lands. Real peak memory: measure it uncapped only on the Mac (64 GB, no lock), never uncapped on the VPS; on the VPS, `prlimit --as=8589934592` (an address-space cap, which can kill a Cairo build at ~6.6 GB of real memory) stays the cap, used only for Cairo work known to fit well under it.

Work autonomously, do not ask questions, do not widen the scope.

