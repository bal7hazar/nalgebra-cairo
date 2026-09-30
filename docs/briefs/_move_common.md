# Rules of every package-split move (nalgebra-cairo M9)

Every move brief points here; its own sections say what moves, its proofs beyond these, and its
allowlist.

## Read first

- `docs/SPLIT.md` §18 in full: the per-dimension map §18.1, the closures §18.2, the knot §18.3, the
  base families §18.4, the move plan §18.5, the owner's decisions §18.7. Then §12-§17, the rules of
  the moves: explicit facade lists where moved impls land, `internal::` for former `pub(crate)`
  items, descriptions / keywords / README per package, the `[internal]` table, in-crate tests, the
  anchor check, the strict path proof, the facade features.
- `docs/ORCHESTRATOR.md`, the move-PR checklist (steps 1-7).
- `tools/split/crates.recut.toml` (the approved map) and `tools/split/crates.toml` (the live map),
  `cratemap.py`, `public_paths.py`, `gas_compare.py`, `rewrite_imports.py`.
- The report of the previous move, `~/orchestrator/nalgebra-cairo/reports/wp-9-r1.md`, and its
  PR #80: the precedent, follow it. The older moves (`wp-9-ns3.md` .. `wp-9-ns11a.md`, PRs #66-#76)
  when a rule is unclear.

## Rules

- Switch the map rows of the moved crates to the approved map, generators with the map,
  hand-written files / items moved verbatim, module paths kept, nothing else moved; `[internal]`
  entries for what higher packages still need; facade re-exports (explicit lists where moved impls
  land); the anchor check (0 impl outside its trait's or argument types' modules).
- Each new package: workspace version; `description` = the §18.1 one-line content; `keywords`
  including the upstream modules it covers; its own `README.md` (what it holds, when to depend on
  it, the facade, `internal`); `license`, `repository`; registry-only dev-dependencies; features
  forwarded by the facade when a moved family is gated (as `closures` in the static crates).
- Proofs, all of them: `gas_compare.py --base origin/main --head gas/` 0 changed / 0 missing /
  0 added; `public_paths.py --check` strict (9,289 / 0 / 0); `cratemap.py --anchors` 0 findings;
  gate 1 (every crate at most 40,000 library lines); `api_parity.py --check` (report unchanged); the
  generators' `--check`; build / lint of the moved crates and of `nalgebra` (defaults and
  `--no-default-features`); the `Facade features` test; unfiltered tests of every package whose
  snapshots you regenerate; `consumer_cost.py --lines-only --report-only` for the new crates; the
  CI `Consumer cost` figures in the report; a `docs/SPLIT.md` §18.5 progress line.
- Forbidden in every move: behaviour changes, test code of test packages (imports only if forced,
  reported), docs other than the progress line, publishing.
- Definition of done: every proof green; conventional commits with the trailer; `git push -u origin
  <branch>`; `gh pr create` following the template (move-PR checkbox, `gas_compare.py`, path-proof
  and anchors outputs pasted); `gh pr checks --watch` until green; NEVER merge. `REPORT.md` as R1's:
  what moved where (per crate: lines, dependencies), `internal::` items, facade changes, proofs, the
  CI Consumer cost figures, deviations, deferred items, escalations, PR URL.
- Work autonomously, do not ask questions, do not widen the scope. FOREGROUND ONLY.
