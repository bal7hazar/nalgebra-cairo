# WP 9-R2 — Re-cut move 2: the decompositions per dimension

Branch: `feat/wp-9-r2`, already checked out in this worktree, cut from `main` after R1 (#80). ONE
pull request (large; commit family by family as coherent states). Model: Opus 5.5 (a
generator-driven move of about 185k lines with placement decisions).

## 0. Environment and shared rules

Read `docs/briefs/_env.md` (the machine) and `docs/briefs/_move_common.md` (the reading list, rules,
proofs and definition of done of every move) first, and follow them: this brief adds to them.
Other lots run at the same time on files you do not touch: WP 9-R3 (the READMEs of the
non-decomposition packages, the root `README.md`, `CHANGELOG.md`) and the plan of WP 9-NS12b (only
the new proposal map `tools/split/crates.dyncut.toml`, which you never create); no other agent
touches the decomposition crates, the live and re-cut maps (`crates.toml`, `crates.recut.toml`) or
the generators during R2.

## 1. Read first (beyond `_move_common.md`)

- `docs/SPLIT.md` §18.1: the rows of `linalg_core`, `linalg2` .. `linalg6`, `linalg_svd_eigen2` ..
  `6`, `linalg_pivot2` .. `6`, `linalg_spectral2` .. `6` (content, lines, direct dependencies,
  runner marginal); the R2 row of §18.5 and R1's progress paragraph there; §17 (the `linalg6`
  decision, superseded below); §20 (what R1 leaves to R2).
- R1's report `~/orchestrator/nalgebra-cairo/reports/wp-9-r1.md`, in particular Deviations 2, 3 and 6
  (`GivensRotation` in `types2`, `Perm1Trait` .. `Perm5Trait` left for R2, R1's decisions mirrored
  into `crates.recut.toml` so that R2 switches only the decomposition rows).
- `tools/linalggen/` (how the decompositions are generated and routed through the map);
  `tools/split/crates.recut.toml`: the decomposition rows, `[band_split]` (`SvdRightTrait`), the rules
  `Cholesky2UpdateTrait` / `Cholesky3UpdateTrait`, `[closures]`.

## 2. Goal (§18.5, move R2)

1. Switch the live map `tools/split/crates.toml` to `crates.recut.toml` for the decompositions, so
   that the live map is the approved map for every crate (report any remaining difference and why):
   the 21 packages `nalgebra_linalg_core`, `nalgebra_linalg2` .. `6`, `nalgebra_linalg_svd_eigen2` ..
   `6`, `nalgebra_linalg_pivot2` .. `6`, `nalgebra_linalg_spectral2` .. `6` replace the 10 current
   decomposition crates (`linalg4`, `linalg5`, `linalg6`, `linalg_svd_eigen4`, `linalg_pivot4` ..
   `6`, `linalg_spectral4` .. `6`). The dimension-1 decompositions fold into the `*2` crates.
2. In particular: `SvdRightTrait` split per dimension 2 .. 6 (today ≤ 4 / 5 / 6; `[band_split]`);
   `Cholesky2UpdateTrait` in `linalg3`, `Cholesky3UpdateTrait` in `linalg4`; `Perm1Trait` ..
   `Perm5Trait` from their types crates to `linalg2` .. `linalg5` as the approved map places them
   (their fields are `pub`); the shared kernels of `linalg_core` (§18.1); the `tools/linalggen`
   switch (the generator emits into the new packages through the map, `--check` clean).
3. Packages: `git mv` where a crate continues (say which directory became which), create the new
   ones, delete the emptied ones. Each package: workspace version, §18.1's one-line `description`,
   keywords with the upstream modules it covers (`linalg::svd`, `linalg::schur`...), a minimal README
   as R1's (`crates/types5/README.md`; the full READMEs come with R3), registry-only
   dev-dependencies; the facade's linalg features (`eigen`, `svd`, `qr`, `cholesky_update`,
   `full_piv_lu`, `col_piv_qr`, `lblt`, `hessenberg`, `bidiagonal`, `schur`, `exp`) keep gating the
   same code, forwarded as today.
4. The facade `nalgebra` re-exports every 0.1.0 path; every package's direct dependencies are
   §18.1's (report any difference with the reason).
5. Proofs: all those of `_move_common.md`, plus **every package's marginal ≤ 5 s / 1 GB** on the CI
   `Consumer cost` medians (gate 2; the job still reports marginals only, so read them in its
   output). A crate over the gate is reported with its figures and a proposed cut, never left
   silently: the gate is not relaxed (this supersedes §17's `linalg6` decision). Report the
   facade's build figures too (R1: 43.8 s / 9.92 GB over the baseline; `main` before R1: 36.9 s /
   10.03 GB).
6. `consumer_cost.toml`: only the three closures that name decomposition crates, re-expressed on the
   new names with §18.2's content and the current budgets (members as `crates.recut.toml`
   `[closures]`, prefixed `nalgebra_`): `static3_svd` (static 2-3 + SVD / eigen 2-3), `core_pivot`
   (pivoting 2-4), `static4_factor` (static 2-4 + LU / Cholesky / QR 2-4). The other closures and the
   per-closure budgets are R3's.
7. `docs/SPLIT.md`: the §18.5 progress line of R2.

## 3. Scope

Allowed: `crates/linalg*/**` (the old and the new decomposition packages); `crates/nalgebra/**`
(manifest, re-exports; not the root `README.md`); `crates/core/**` and `crates/types2/**` ..
`crates/types5/**` only for the `Perm*Trait` moves and the `[internal]` or kernel entries the move
needs (list each file in the report); the manifests `crates/dynamic/Scarb.toml`,
`crates/sparse/Scarb.toml`, `crates/nalgebra_glam/Scarb.toml` only if a dependency name changes;
test packages' imports only if forced (`rewrite_imports.py`, reported); `tools/split/**` (except
`crates.dyncut.toml`), `tools/linalggen/**`, `tools/shapegen/**` (routing only); workspace
`Scarb.toml`, `Scarb.lock`;
`.github/workflows/ci.yml` (shards only, if a package with in-crate tests appears);
`consumer_cost.toml` (the three closures of item 6 only); `gas/**`; `docs/SPLIT.md` (§18.5 progress
line); `scripts/api_parity.py` (`CROSS_FILE_TRAITS` only, if a trait's implementors land in another
package than its declaration; R1's precedent; reported).

Forbidden: every README except the minimal ones of the decomposition packages (R3 owns the others,
the root `README.md` included), `CHANGELOG.md`, `docs/` beyond the progress line, the other closures
and the budgets of `consumer_cost.toml`, behaviour changes, test code beyond forced imports,
publishing.

## 4. Definition of done

As `_move_common.md`. `REPORT.md` as R1's: per package its lines and direct dependencies; what moved
where (`SvdRightTrait` split, `Perm*Trait`, the Cholesky update traits, `linalg_core`'s contents,
which old directory became which); facade changes; the proofs' outputs; the CI `Consumer cost`
figures (the three closures, the marginals of the 21 decomposition packages, the facade); deviations;
deferred items; escalations; PR URL.

## 5. Work autonomously, do not ask questions, do not widen the scope. FOREGROUND ONLY.
