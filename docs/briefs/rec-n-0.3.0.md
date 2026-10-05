# REC-N3 — record of the nalgebra 0.3.0 publication and the track's final status (documents)

Lot of the nalgebra track of Slingfall (orchestrator: herdr project `slingfall-nalgebra`), 2026-10-05. Light lot on the
VPS. Commit this brief verbatim as `docs/briefs/rec-n-0.3.0.md` (first commit).

nalgebra 0.3.0 was published by the orchestrator on the project manager's go (request `docs/releases/0.3.0.md` at main
e2bbc09, file sha256 9a8d33deb5a2288365e3c0549fc5c6faa6c06963c035d2aa226707f6cf59dd2c): all 54 packages from commit
8891748dec2555823532e9b584c9bdf70b5a0879; every registry checksum equals its row (checked 54/54 after the last); annotated
tag `v0.3.0` on 8891748; GitHub release `v0.3.0`. Rows 1-52 on 2026-10-05 10:32-10:44Z and row 54 (`nalgebra_glam`) at
14:54Z, under `prlimit --as` 8 GiB. Row 53 (`nalgebra`, the facade) under the Overseer's standing heavy-publication rule,
whose threshold the Overseer lowered on 2026-10-05 from 20 GB to 18 GB free (the VPS stayed at 18-19 GB for ~4 h):
`flock heavy-build.lock env RAYON_NUM_THREADS=1 prlimit --as=25769803776 -- /usr/bin/time -v scarb publish -p nalgebra`,
started at 14:49:05Z with 18 GB free; wall 4:41.57; maximum resident set size 8985020 kB.
Verify the facts you can (`python3 scripts/release.py verify --request docs/releases/0.3.0.md`; `git ls-remote --tags
origin v0.3.0`; `gh release view v0.3.0 -R bal7hazar/nalgebra-cairo`).

Edit, allowlist exactly these files:
1. `docs/releases/0.3.0.md`: status "published" with dates, the 54/54 read-back, the tag link (tree at v0.3.0) and the
   release link, and a section on row 53 (the standing rule, the 18 GB ruling, the start reading, peak and wall). The
   request table stays untouched.
2. `CHANGELOG.md`: the 0.3.0 heading dated (2026-10-05) instead of "unreleased".
3. `docs/PLAN.md`: replace the top status section by a final one, "Status 2026-10-05 — final (programme paused; the
   nalgebra track finished its planned versions)": published nalgebra 0.2.0 (2026-10-03/04) and 0.3.0 (2026-10-05, on
   simba 0.3.0 / fixed 0.5.0 / glam_core 0.5.0, no result moved), simba 0.3.0 (2026-10-03); merged lots (OPT-0..3, API-1,
   release tooling, pre-push and CI gating, NAL-S3, REL-DOC); the track is idle; open rows only if the owner resumes the
   programme (TC2, the deferred pre-push notes). Row NAL-S3 and REL done with their PRs.
4. `docs/SPLIT.md` §21: one new dated entry (2026-10-05) for 0.3.0; earlier entries untouched.

One PR `docs(release): record nalgebra 0.3.0 and the final status`. Do NOT write any "Review:" line. Never merge, never
publish, never launch a review or any agent; foreground only; `REPORT.md` at the worktree root (not committed) with the PR
number and head.

Reading CI logs (programme rule): wait for the run to complete, then `gh run view <id> -R <owner>/<repo> --log-failed`, alone in its call (if your thread cannot run it, name the job and run id in your report and end your turn: the orchestrator relays the log). Never `gh api …/logs`. To read a sibling repository (simba-cairo, fixed-cairo), use the Read tool on its checkout. A bare command refused: report its exact text, no other attempt. At most one GitHub call per PR every 5 minutes. Never probe the shared heavy-build lock. Mac repositories by absolute path (/Users/bal7hazar/git/<repo>).
Signals (programme rule after an incident): a thread, a test or a script signals only processes it started itself, by pid or pgid taken from `$!` or ids it recorded. Never find a pid by searching (`ps | grep`, `pgrep`, `pkill`, `killall`).
Picking up main: before your first push, `git rebase origin/main` in exactly that form, alone in its call (nexus #95; no option, no -i/--exec/--abort, no `$`, backticks, `<` or `>`); a rebase that hits a conflict is reported to the orchestrator, not driven (--abort is refused). After any push: `git merge origin/main`, alone in its call. Every force push stays refused.
Files: to remove an untracked file of your own worktree, use `git clean -f -- <exact path>` (nexus #92); a bare `rm` stays refused. NEVER run `git clean` with `-d`, `-x` or `-X`: it deletes your own .herdr-project folder (brief, report). `gh pr edit` works for a PR title and body (nexus #91).
Memory (organisation rule): the 8 GB cap (`prlimit --as=8589934592 -- /usr/bin/time -v <command>`) applies to Cairo builds and measures (scarb, snforge) on the VPS; every peak-memory measure stays capped; a Node suite, or a whole pre-push hook, runs uncapped only when each of its steps was measured well under 8 GB (`prlimit --as` kills Node at start-up). On the Mac there is no cap. Long pushes: if your pre-push hook may run long (several minutes), push with `git -c core.sshCommand='ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=40' push ...` (writes no config), otherwise GitHub drops the SSH connection (exit 141) and nothing lands. Real peak memory: measure it uncapped only on the Mac (64 GB, no lock), never uncapped on the VPS; on the VPS, `prlimit --as=8589934592` (an address-space cap, which can kill a Cairo build at ~6.6 GB of real memory) stays the cap, used only for Cairo work known to fit well under it.

Work autonomously, do not ask questions, do not widen the scope.
