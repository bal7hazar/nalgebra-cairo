# WP 9-R3 — Release plumbing on the final package names (two phases)

Branch: `feat/wp-9-r3`, already checked out in this worktree, cut from `main` after R1 (#80). ONE
pull request, opened as a draft at the end of phase 1. Model: Sonnet (documents and configuration,
precisely framed).

## 0. Environment and phases

Read `docs/briefs/_env.md` (the machine) first and follow it. This lot is not a move: no code, no
public path and no result changes; most of the work is Markdown and TOML. Build only what a change
can break (`scarb build -p <pkg>` after a manifest or doc-comment change, crate-scoped).

WP 9-R2 (the decomposition crates, `crates/linalg*`) runs at the same time. So that the two lots
never edit the same file, this lot has two phases:
- **Phase 1, now**: only the files of §3.1. At its end: push, open the PR as a draft, write
  `REPORT.md` (section "Phase 1"), end your turn.
- **Phase 2**: the orchestrator resumes you after R2 and WP 9-NS12b (the cut of `nalgebra_dynamic`,
  which adds packages to §18.1's 54) are merged, with the merge commit and the final package list;
  only then the files of §3.2.

## 1. Read first

- `docs/SPLIT.md`: §17 (the facade features, the README's feature table, the CHANGELOG's Deprecated
  section); §18 in full (the map §18.1 with each crate's one-line content, lines and direct
  dependencies; the closures §18.2 with their runner figures; the base families §18.4; R3's row in
  §18.5; the owner's decisions §18.7); §19 (the dimension 5-6 budgets, the facade README section
  "Dimensions 5 and 6", `docs/PACKAGES.md`); §20 (what R1 leaves to R3); §12.1 and §13-§16 for the
  older README conditions, keeping those that still hold on the new names (the facade README opens
  with a table "what you need → which crates to depend on → measured cost") and dropping what the
  re-cut made obsolete (`static3`'s geometry sentence, `shapes5` / `shapes6`, the `geometry6` and
  `static6_wide` exceptions now covered by the dimension 5-6 budget).
- R1's report `~/orchestrator/nalgebra-cairo/reports/wp-9-r1.md`: Deviation 8 (module docs of split
  modules), "Deferred", the `ci.yml` stale comment.
- `CHANGELOG.md` (the 0.1.1 skeleton and its TODOs); the root `README.md`, which is the facade's
  README (`crates/nalgebra/Scarb.toml`: `readme = "../../README.md"`); `crates/blas/README.md`, the
  style of a full sub-crate README (what it holds, upstream module paths kept, its dependencies
  linked, when to depend on it, `internal`, license); `crates/types5/README.md`, R1's minimal README
  to replace.
- `scripts/consumer_cost.py` (its docstring: closures, `report_only`, `budget`), `consumer_cost.toml`,
  `scripts/packages_table.py`, `scripts/release.py` (docstring), `tools/split/crates.recut.toml`
  `[closures]` (the declared closures in short names).

## 2. Goal

### Phase 1

1. A full README, in `crates/blas/README.md`'s style, for every published package that is not a
   decomposition crate: `core`, `types2` .. `types6`, `static_core`, `static2` .. `static5`,
   `static6_tall`, `static6_wide`, `geometry2` .. `geometry6`, `transform2`, `transform3`,
   `statistics2` .. `statistics6`, `blocks`, `views`, `norm`, `blas`, `dynamic`, `sparse`,
   `nalgebra_glam`. Each one: the title; §18.1's one-line content (the manifest's `description`);
   what it holds (types, traits, the upstream module paths); its direct dependencies (the
   manifest's, linked to their READMEs); when to depend on it, with the declared closure it belongs
   to and that closure's runner cost from §18.2 where one exists; the facade; its `internal`
   modules ("no stability promise") or the statement that it has none; license. No README names a
   crate that no longer exists (`nalgebra_shapes5`, `nalgebra_shapes6`, `nalgebra_statistics`...).
2. The facade README (root `README.md`), keeping its other sections:
   - "Packages": the table "what you need → which crates to depend on → measured cost" on the new
     names (the costs are §18.2's runner medians; the decomposition rows name the linalg crates
     exactly as §18.1 lists them, which is what R2 creates);
   - the feature table of §17: the features that still save compile work (`closures` and the linalg
     families) apart from the five no-ops (`statistics`, `blas`, `dynamic`, `sparse`, `io`);
   - a section "Dimensions 5 and 6" (§19): why (dimension k builds on every dimension below it), the
     runner cost of each dimension 5-6 closure (§18.2), the combined static 6 + SVD / eigen 6 and
     static 6 + spectral 6 figures (documented, not gated: §18.7.1), the comparison with 0.1.0
     (97 s / 10.4 GB);
   - `blocks`, `views`, `norm` as advanced use, with their "static 2-4 + family" figures (§18.4,
     §18.7.3), and the everyday methods that do not need them.
3. `CHANGELOG.md`, 0.1.1: what does not depend on the final CI run: the list of the 54 packages
   (names and one-line contents from §18.1), the Deprecated section's replacements (the sub-crates
   that hold each no-op feature's code), `nalgebra_glam`'s dependencies (glam ≥ 0.4.1), the
   dimension 5-6 note (§19). Every figure that comes from the final run stays `TODO` for phase 2.
4. Push, `gh pr create --draft` following the template, `gh pr checks --watch` until green,
   `REPORT.md` with a "Phase 1" section, end your turn.

### Phase 2 (when resumed after R2 and WP 9-NS12b)

1. `git fetch origin && git merge origin/main` (no rebase, no force push); resolve conflicts in your
   own files only.
2. The READMEs of the 21 decomposition packages (`linalg_core`, `linalg2` .. `6`,
   `linalg_svd_eigen2` .. `6`, `linalg_pivot2` .. `6`, `linalg_spectral2` .. `6`) and of the packages
   WP 9-NS12b made of `nalgebra_dynamic`, same style; the phase-1 files updated for the final package
   list (the CHANGELOG list, the facade README's rows, the closure sentences of the READMEs).
3. The module docs (`//!` lines only, no code) of the modules split over several packages: each
   package's part says what that package holds of the module, instead of its lowest part's doc (R1
   Deviation 8, and the same after R2).
4. `consumer_cost.toml`: every closure of §18.2 declared on the full names, with its budget: 15 s /
   3 GB up to dimension 4, `budget = { seconds = 20, gb = 4.5 }` with dimension 5 or 6 (§19). The
   dimension-6 decomposition closures are the decomposition crate with its types, without the
   dimension-6 method crates (`svd_eigen6`, `pivot6`, as `crates.recut.toml`; §18.7.1); `static6_svd`
   and `static6_spectral` are `report_only` (documented in the README, not gated); `static4_blocks`,
   `static4_views`, `static4_norm` are `report_only` (advanced use, §18.7.3); `glam_0_4_1_alone`
   (`glam_core@0.4.1`) is `report_only` as the reference; `facades = ["nalgebra"]` and
   `[crates.nalgebra] report_only = true` stay; `nalgebra_with_glam` and
   `facade_no_default_features` stay `report_only`. Keep the comments' history style.
5. `scripts/release.py` (dry run, no argument): the order lists every package of the final list,
   dependencies first, then the facade, then `nalgebra_glam`; paste the order in the report. On a branch the dry run
   lists refusals (HEAD is not `origin/main`, the version is not bumped yet): expected, list them.
   Edit the script only if the order is wrong (report why).
6. Push, wait for the checks, then `docs/PACKAGES.md` from this PR's CI run: `gh run download <run>
   -n consumer-cost` (the `Consumer cost` job's artifact holds `consumer_cost.json`), then
   `python3 scripts/packages_table.py consumer_cost.json --runner "<the runner the job names>"
   --output docs/PACKAGES.md`; commit it.
7. `CHANGELOG.md`: every `TODO` filled from that run (the facade, the closures, `nalgebra_glam`),
   with the run's link.
8. `.github/workflows/ci.yml`: only the stale comment that names `nalgebra_shapes6` and its 9.5 %
   margin (now `nalgebra_types6`, with the margin of that run).
9. Push, `gh pr ready`, `gh pr checks --watch` until green, `REPORT.md` with a "Phase 2" section.

## 3. Scope

### 3.1 Phase 1

Allowed: `crates/<package>/README.md` for the non-decomposition packages of phase 1 item 1; the
root `README.md`; `CHANGELOG.md`. Nothing else.

### 3.2 Phase 2

Allowed, in addition: `crates/linalg*/README.md` and the READMEs of the packages WP 9-NS12b
created; the `//!` module-doc lines of `crates/*/src/**`
(no other line of a `.cairo` file); `consumer_cost.toml`; `docs/PACKAGES.md` (new);
`scripts/release.py` (only if the order is wrong); `.github/workflows/ci.yml` (the one comment).

Forbidden in both phases: code, manifests, the maps and generators, `gas/**`, `docs/` other than
`docs/PACKAGES.md`, test packages, publishing, version bumps.

## 4. Definition of done

Each phase: the checks it can run locally (`scarb fmt --check` when a `.cairo` doc line changed,
`scarb build -p` of each package whose doc lines changed, `python3 scripts/consumer_cost.py
--dry-run --report-only` after the `consumer_cost.toml` change, `python3 scripts/packages_table.py
--self-test`), conventional commits with the trailer, push, `gh pr checks --watch` until green, NEVER
merge. `REPORT.md`: per phase, the files changed and why, the facade README's tables as written, the
closures declared (name, members, budget, report-only or gated), the release order, the CI
`Consumer cost` verdict and figures, deviations, escalations, PR URL.

## 5. Work autonomously, do not ask questions, do not widen the scope. FOREGROUND ONLY.
