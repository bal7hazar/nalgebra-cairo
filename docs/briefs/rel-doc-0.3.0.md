# REL-DOC — documents before the nalgebra 0.3.0 request

Lot of the nalgebra track of Slingfall (orchestrator: herdr project `slingfall-nalgebra`), 2026-10-05. Light lot on the
VPS. Documents only: nothing that runs changes. Commit this brief verbatim as `docs/briefs/rel-doc-0.3.0.md` (first
commit).

nalgebra 0.3.0 is prepared on main (60dc9bf, #110): simba 0.3.0, fixed 0.5.0, glam_core 0.5.0, version 0.3.0. Before the
release request, the documents that ship or describe the packages must state those versions:

1. **README version snippets**: the root `README.md` (the dependency snippets near lines 26-28 and 91-93, which still say
   `nalgebra = "0.1.1"`, `fixed = "0.4.0"`, `simba = "0.2.0"`) and `crates/nalgebra_glam/README.md` (lines ~16-27:
   `glam_core` 0.4.1, `fixed` 0.4.0 — this README is packaged into `nalgebra_glam`'s published archive), and any other
   crate README that names a dependency version (find them: `grep -rn '0\.1\.1\|0\.2\.0\|0\.4\.[01]' crates/*/README.md
   README.md`): update to nalgebra 0.3.0, simba 0.3.0, fixed 0.5.0, glam_core 0.5.0. Change version numbers only, plus a
   sentence only where a snippet would otherwise be wrong.
2. **`docs/PACKAGES.md`**: regenerate it from the `consumer-cost` artifact of main's CI run on 60dc9bf (the push to main
   runs the full CI: wait for it to complete with Consumer cost succeeded — `gh run list -R bal7hazar/nalgebra-cairo
   --commit 60dc9bf…`, one call every 5 minutes), with `scripts/packages_table.py`, as its header says
   (`gh run download <id> -R bal7hazar/nalgebra-cairo -n consumer-cost`). The closure `glam_0_5_0_alone` replaces
   `glam_0_4_1_alone`.

Allowlist: the brief, `README.md`, `crates/*/README.md` (version snippets only), `docs/PACKAGES.md`. Not: CHANGELOG
(0.3.0 stays "unreleased" until the publication), manifests, code.

One PR `docs: README versions and PACKAGES.md for 0.3.0`. Do NOT write any "Review:" line (the orchestrator does). Never
merge, never publish, never launch a review or any agent; foreground only; `REPORT.md` at the worktree root (not
committed) with the PR number and head.

Reading CI logs (programme rule): wait for the run to complete, then `gh run view <id> -R <owner>/<repo> --log-failed`, alone in its call (if your thread cannot run it, name the job and run id in your report and end your turn: the orchestrator relays the log). Never `gh api …/logs`. To read a sibling repository (simba-cairo, fixed-cairo), use the Read tool on its checkout. A bare command refused: report its exact text, no other attempt. At most one GitHub call per PR every 5 minutes. Never probe the shared heavy-build lock. Mac repositories by absolute path (/Users/bal7hazar/git/<repo>).
Signals (programme rule after an incident): a thread, a test or a script signals only processes it started itself, by pid or pgid taken from `$!` or ids it recorded. Never find a pid by searching (`ps | grep`, `pgrep`, `pkill`, `killall`).
Picking up main: before your first push, `git rebase origin/main` in exactly that form, alone in its call (nexus #95; no option, no -i/--exec/--abort, no `$`, backticks, `<` or `>`); a rebase that hits a conflict is reported to the orchestrator, not driven (--abort is refused). After any push: `git merge origin/main`, alone in its call. Every force push stays refused.
Files: to remove an untracked file of your own worktree, use `git clean -f -- <exact path>` (nexus #92); a bare `rm` stays refused. NEVER run `git clean` with `-d`, `-x` or `-X`: it deletes your own .herdr-project folder (brief, report). `gh pr edit` works for a PR title and body (nexus #91).
Memory (organisation rule): the 8 GB cap (`prlimit --as=8589934592 -- /usr/bin/time -v <command>`) applies to Cairo builds and measures (scarb, snforge) on the VPS; every peak-memory measure stays capped; a Node suite, or a whole pre-push hook, runs uncapped only when each of its steps was measured well under 8 GB (`prlimit --as` kills Node at start-up). On the Mac there is no cap. Long pushes: if your pre-push hook may run long (several minutes), push with `git -c core.sshCommand='ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=40' push ...` (writes no config), otherwise GitHub drops the SSH connection (exit 141) and nothing lands. Real peak memory: measure it uncapped only on the Mac (64 GB, no lock), never uncapped on the VPS; on the VPS, `prlimit --as=8589934592` (an address-space cap, which can kill a Cairo build at ~6.6 GB of real memory) stays the cap, used only for Cairo work known to fit well under it.

Work autonomously, do not ask questions, do not widen the scope.
