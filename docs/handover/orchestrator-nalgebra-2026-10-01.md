# Handover: orchestrator of the nalgebra track (slingfall), 2026-10-01

Written at the owner's soft stop (2026-10-01, relayed by the project manager), at 15:40 UTC, by the
session "[Opus 5.5] Orchestrateur nalgebra — slingfall" (claude-opus-5-5). It stays on standby and
answers only its successor once the successor has announced itself.

## Who you are

- The orchestrator of the **nalgebra** track of project `slingfall` (repositories simba-cairo and
  nalgebra-cairo), in the Nexus standard role. You own the track's briefs, its agents, the review
  and the merge of its pull requests, and its status (`docs/SPLIT.md`, `docs/PLAN.md`).
- Above you: the project manager, the session "[Fable 5.1] Chef de projet Slingfall" (also shown as
  "[Fable 5.1] PM - Slingfall"). It reports to the Overseer, and through it to the owner.
- Below you: nobody runs. The agents' worktrees under `~/orchestrator/nalgebra-cairo/wt/`
  (`wp-9-r1`, `wp-9-r2`, `wp-9-r3`, `wp-9-ns12b`) are finished; their reports are archived under
  `~/orchestrator/nalgebra-cairo/reports/`.
- Two earlier orchestrator sessions are on standby and must not be used: "[Retired] Orchestrateur
  nalgebra.cairo (fork)", and this one once you have announced yourself.
- Read first: `slingfall/OPERATIONS.md`, then in this repository `AGENTS.md`, `docs/ORCHESTRATOR.md`,
  `docs/SPLIT.md` §20-§22 (status and release readiness), and the M9 rows of `docs/PLAN.md`.

## State

`nexus progress --project slingfall` at 15:36 UTC: no nalgebra agent is running, and every nalgebra
review has ended (`succeeded`).

**Done, the five objectives of the project manager's context of 2026-09-30**
- **1. Documents aligned with the standard:** `docs/ORCHESTRATOR.md` and committed briefs under
  `docs/briefs/` (#81, #83, #84, #88).
- **2. R1:** per-dimension types, methods, geometry, transforms, statistics (#80).
- **3. R2 and R3:** R2, the 21 decomposition packages (#85). R3, READMEs, closures with their
  budgets, `docs/PACKAGES.md`, CHANGELOG 0.1.1 (#82).
- **4. NS12b:** `nalgebra_dynamic` cut under gate 2 (the fixed-size edition moved to
  `nalgebra_blocks`, no forwarding of `closures`), and the marginals are gated in CI (#86).
- **5. Release 0.1.1:** the release PR (#87). **nalgebra 0.1.1 is published**, 54 packages on
  scarbs.xyz, each verified against the index:
  - 12 packages from `3e5e4ba` (the first run stopped on a keyword limit, see Traps);
  - the 42 others from `b3915c7`, after #89 (the keywords fixed and the continuation check of
    `scripts/release.py`);
  - tag `v0.1.1` on `b3915c7`, and a GitHub release whose notes are CHANGELOG 0.1.1.

**Open**
- **PR #90** (`docs/post-release-0.1.1`), documents only:
  - content: `docs/PACKAGES.md` regenerated from the release commit's CI run 36841314574, CHANGELOG
    0.1.1 dated 2026-10-01, the release status in `docs/SPLIT.md` §21-§22, and `docs/PLAN.md`
    (0.1.1 done, the `REL-FU` and `TC` rows);
  - reviews: `slingfall/review-9-docs-post-0-1-1` and `-2` (claude-sonnet-5-5), both PASS WITH
    FINDINGS, the second with a note only;
  - CI on its head `97e75d4` was running when this note was written;
  - not merged because of the soft stop.
- **This handover note's own pull request**, if the session's guard refused its merge (it is
  documents only, with no review because of the soft stop).

**Nothing else** is open on the track. No unit `nalgebra-*` runs, and no release is in flight. The
state file of the release, `target/release/state-0.1.1.json` in the Desktop worktree of the
predecessor session, records 54 published packages.

## Decisions

**Taken, and recorded in the repository**
- **Briefs:** committed under `docs/briefs/` before each launch, which `nexus review --brief` needs.
- **R3 in two phases**, so that its files never overlap R2's. Order: R2 → NS12b phase 2 → R3 phase 2,
  because the cut changed the content of `nalgebra_blocks` and `nalgebra_dynamic`.
- **NS12b, cut A + D1 (ii):** the fixed-size edition moved to `nalgebra_blocks`, no new package, and
  `nalgebra_dynamic`'s `closures` no longer forwarded. Measured in CI at 2.5 s, which confirms the
  phase-1 inference.
- **Release continuation (project manager's option A):** keywords limited to 20 characters and 8 per
  package, and `release.py` accepts a later commit only when no file of a published package nor any
  workspace file changed.

**Pending, with my recommendation**
1. **Merge #90** when work resumes, if its CI is green. The merge body names the reviews above.
   Recommended: yes.
2. **The TC lots** (owner D-180):
   - scope: Scarb 2.20.1 / starknet-foundry 0.64.0, simba-cairo first and nalgebra-cairo second;
     `.tool-versions`, CI, scripts, gas snapshots (zero change), path proof, Consumer cost and
     `docs/PACKAGES.md` re-measured; no release for the bump;
   - recommended: Sonnet for simba-cairo (mechanical), Opus 5.5 for nalgebra-cairo (the
     re-measurement and its comparison).
   - Every brief pins `RAYON_NUM_THREADS=1` for every measurement (see Traps). Its report says
     whether the 0.1.1 figures were stable under the shims' four threads. They were not identical
     between the two CI runs of the same code: `nalgebra_linalg_svd_eigen6` 2.3 s against 3.5 s, the
     facade 29.8 s against 46.0 s, line counts identical. SPLIT §21 records it.
3. **`REL-FU`** (PLAN), from the Fable review notes on #89:
   - `python3 scripts/release.py --self-test` in the CI Workspace job;
   - the continuation check also for ignore files in intermediate directories and an inherited
     `readme = true`;
   - a lagging index for the package in flight.

   Small; it can ride with the nalgebra TC lot or be its own lot.
4. **Authorisation of publications.**
   - The project manager wrote that the owner told them its written go is the owner's authorisation
     for publications.
   - This session nevertheless asked the owner in its own chat before each `--publish`, and the
     owner approved both runs: a relayed authorisation is not the owner's own words in the session.
   - Recommended: keep asking the owner in your session before any publication, unless the owner
     tells you directly there that the project manager's go suffices.
5. **Unit-test placement** (owner, 2026-09-30, `AGENTS.md`): a module's unit tests go in its file
   under `#[cfg(test)] mod tests`, from the lots after 0.1.1. Put it in every code brief.

## Threads

- **Project manager:** it has the five objective messages, the release links, the authorisation
  stance (item 4 above), and the note about the run-to-run spread. It expects one message with this
  note's path and the first message's path. After that, a message only for a decision, a blocker or
  an objective done.
- **Owner:** approved the first publication run (07:27 UTC) and the continuation from `b3915c7`, in
  the predecessor's chat.

## Next, in order

1. Announce yourself to the project manager (and the owner, if they are there).
2. Confirm the soft stop is lifted.
3. Merge #90, and this note's PR if it is still open, after their CI.
4. Write the TC briefs: simba-cairo, then nalgebra-cairo (with `REL-FU`, or after it). Commit them
   and launch under the capacity rule.
5. Keep `docs/SPLIT.md` §21 dated at each step.

## Traps

- **scarbs.xyz keywords.**
  - The registry refuses a package with a keyword over 20 characters, and answers with something
    that is not JSON (`error decoding response body: expected value at line 2 column 1`). This was
    `symmetric-tridiagonal` in 0.1.1, and earlier `procedural-generation` in dojoengine/origami#137.
  - 8 keywords per package were accepted; Scarb recommends at most 5.
  - Check every manifest before a release.
- **`scripts/release.py` continuation.**
  - A release begun at an older commit continues only when that commit is an ancestor of HEAD, every
    recorded package is still published, and `git diff --name-only --no-renames OLD HEAD` touches
    none of the published packages' files nor `Scarb.toml` / `Scarb.lock` / `.gitignore` /
    `.scarbignore`.
  - It prints the check; read that line before the go.
  - Run it with `python3 -u`, detached with `setsid -f`, and follow it by PID.
- **Single-thread pin.** The Cairo compiler's Sierra output differs between multi-threaded builds
  (grimworld SPK-13, Scarb 2.19.4 and 2.20.1); it is identical with `RAYON_NUM_THREADS=1`. Every gas,
  steps, size or hash measurement (`gas_compare.py`, `consumer_cost.py`, `docs/PACKAGES.md`, the path
  proof if it reads Sierra) runs with `RAYON_NUM_THREADS=1`, and so do the measurement scripts.
- **Compile drift** (owner D-180): reproduce a drift's minimal case on the latest Scarb before any
  upstream issue.
- **systemd from a Desktop session.**
  - Prefix `systemctl --user` / `systemd-run --user` with
    `DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus`.
  - The drop-in `~/.config/systemd/user/nalgebra-.service.d/50-agent-limits.conf` sets
    `MemoryMax=20G`, which wins over `-p MemoryMax`. Run
    `systemctl --user set-property --runtime nalgebra-<wp> MemoryMax=14G` after each launch
    (`docs/ORCHESTRATOR.md` recipe).
- **Watch loops.**
  - Never `pgrep -f '<pattern>'`: it matches the waiting loop's own command line. On 2026-10-01 a
    publication that had stopped looked "still running" for an hour.
  - Record the PID and test it with `kill -0`.
  - Python in the background buffers its output: use `-u`.
- **Reviews.**
  - While Codex has no quota, `nexus review` falls back to Claude Sonnet and audits to Claude Opus
    5.5, with no `--model` needed.
  - `nexus review --brief` takes a path that is inside the repository and present on the pull
    request's branch.
  - The session's auto-mode classifier refused documents-only merges without a review ("Merge
    Without Review"): get the review even when the standard would allow `Codex review: none`.
- **Models.** Launch with explicit ids (`claude-opus-5-5`, `claude-sonnet-5-5`), and read the model
  that ran from the agent's transcript before titling anything.
