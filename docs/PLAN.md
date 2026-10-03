# Execution plan

> **State (2026-09-30): `nalgebra` 0.1.0 and `nalgebra_glam` 0.1.0 are published** (M8 done, parity
> 99.9 %). Active: **M9, package split behind a facade** (owner decision 2026-09-28), now the
> per-dimension re-cut (R1-R3, `docs/SPLIT.md` §18); its dated status is `docs/SPLIT.md` §20 and the
> sections after it. Process: [ORCHESTRATOR.md](ORCHESTRATOR.md).

## M9 — Package split under the size rule (owner, 2026-09-28)

Scarb compiles a dependency crate whole: nalgebra 0.1.0 costs a consumer that never names it 115 s and
11.6 GB (38 s / 6.1 GB with `default-features = false`), against 3.4 s / 0.8 GB for `fixed` (programme
study `pm/research/R7-package-granularity.md`). Owner-validated rule (`pm/decisions/2026-09-28-package-
granularity-rule.md`): a published crate has at most **40,000 library lines** (inline tests excluded);
an empty consumer of the crate alone adds at most **5 s and 1 GB** to the no-dependency build (cold);
a typical product closure stays under 15 s / 3 GB; scopes follow the upstream module tree; a **facade
crate `nalgebra`** re-exports every sub-crate so names and paths do not change (no breaking change,
parity kept); the split costs **zero Cairo steps** (gas snapshots); CI checks the gates with a shared
script. Model: keep-starknet-strange/alexandria (one `nalgebra_` prefix, one shared version, tests
outside `src/`, one README per package).

| WP | Content | Depends on |
|---|---|---|
| NS0 | `scripts/consumer_cost.py` (repository-agnostic: glam and rapier copy it): an empty consumer per published crate from its local path, cold wall time, peak RSS, library lines, gates; a CI job | — |
| NS1 | Research, no code moved: inventory and cut plan (sub-crates with lines and an acyclic dependency graph, impls pinned by Cairo's coherence rules, what the features become, `nalgebra_glam`'s dependencies, measured cost per planned crate, release order); sent to the programme session before any move | NS0 |
| NS0b | `consumer_cost.py`: `marginal` column (gate 2), registry crates in closures, `--manifest-path` before the subcommand | NS0 |
| NS1 ✅ | Plan `docs/SPLIT.md` (#62), approved by the programme session with conditions (SPLIT §12) | NS0 |
| NS2 | Tooling: generators emit per crate (names from a configuration), `gas_compare.py`, `path_proof`, test-import rewriter, `api_parity.py` over several crates | NS1 |
| NS1b | Final crate names (dimension + content, no bare suffix), small-crate merges, GitHub-runner measurement of the declared closures (median < 15 s); list approved by the programme session and shown to the owner | NS2 |
| NS3..NS11 ✅ | The moves, bottom-up, one family band per PR, each with zero step change (`gas_compare.py`) and unchanged public paths (`path_proof`): the 27-crate split (#66-#76); NS11b (#77) no-op facade features, `scripts/release.py`, enforcing `Consumer cost` | NS1b |
| NS12a ✅ | Per-closure budgets in `consumer_cost.py`, `scripts/packages_table.py`, the PACKAGES.md artifact (#78) | NS11b |
| NS13 ✅ | Re-cut plan (owner, 2026-09-29: a number in a crate name means exactly that dimension): `docs/SPLIT.md` §18, `tools/split/crates.recut.toml`, 54 packages (#79) | NS12a |
| R1 ✅ | Re-cut move 1: types, methods, geometry, transforms, statistics per dimension, the 8 type-level kernels (#80) | NS13 |
| R2 ✅ | Re-cut move 2: `linalg_core`, `linalg2..6`, `linalg_svd_eigen2..6`, `linalg_pivot2..6`, `linalg_spectral2..6`, `SvdRightTrait` per dimension, `Perm*Trait` to `linalg2..5` | R1 |
| R3 ✅ | Release plumbing on the final names (phase 2 after NS12b): closures with the §18.2 / §19 budgets, release order, one README per package, facade README ("Dimensions 5 and 6"), `docs/PACKAGES.md`, CHANGELOG 0.1.1 | phase 1: R1; phase 2: R2 and NS12b |
| NS12b ✅ | `nalgebra_dynamic` cut under gate 2: plan measured on the runner (phase 1 done: the fixed-size edition to `nalgebra_blocks`, no new package; `dynamic`'s `closures` no longer forwarded), then the move after R2, `--report-only-marginals` dropped, `split-measure.yml` repointed or retired | phase 1: R1; phase 2: R2 |
| 0.1.1 ✅ | Released 2026-10-01: the 54 packages on scarbs.xyz, tag `v0.1.1` (on b3915c7), GitHub release; 12 packages from 3e5e4ba, the 42 others from b3915c7 after the keyword fix (#89); `docs/PACKAGES.md` regenerated from the release commit's CI run | R3, NS12b |
| REL-FU ✅ | Follow-ups of the release: `scripts/release.py --self-test` in the CI Workspace job; the continuation check also for ignore files in intermediate directories and an inherited `readme = true`; an in-flight package the index lists late (Fable review of #89, notes); merged 2026-10-02 (#92, `9dc3360`) | 0.1.1 |
| TC ✅ | Toolchain lots (owner, 2026-10-01): Scarb 2.20.1 / starknet-foundry 0.64.0, simba-cairo then nalgebra-cairo; `.tool-versions`, CI, scripts, gas snapshots (the gas moves are not all the snforge 0.64.0 harness offset: −7830 typical, other group offsets from setup code under Cairo 2.20.0 and the +100 builtin charge; no test result changed), path proof, Consumer cost and PACKAGES.md re-measured; merged 2026-10-02: simba-cairo #4 (`33fba0d`), nalgebra-cairo #93 (`d87c4e5`), `benchmarks/libs` kept on 2.19.4 / 0.61.0; every measurement with `RAYON_NUM_THREADS=1` (Sierra output is not deterministic across threads); no release for the bump | 0.1.1 |
| PP-N ✅ | nalgebra-cairo's own pre-push lot (WP 10-PP, WP 10-PP-FU): a pre-push check and hook, retries on the 12 setup steps, cancel-in-progress for pull-request runs only; merged 2026-10-02/03: #95 (`0e478c4`), then the follow-ups #96 (`7ee554d`): untracked build files refused, fixed checks gated on their inputs, the lock wait overlaps the checks, an interrupt stops the Cairo block, an ancestor-held lock, `realpath`, the whole gas comparison when `gas/**` changes with a package (simba-cairo's equivalent: #5 `811e3ab`, follow-ups #7 `d2a80f8`); the brief's rule "kills nothing" is read as "kills nothing the script did not start" (review of #96) | TC |
| CI-N ✅ | CI path gating, both repositories: a `changes` job and the test jobs gated on their files (prose `.md` triggers nothing, `docs/API_PARITY.md` triggers the workspace job; a push to main runs everything in a per-commit concurrency group; `CI result` always runs); merged 2026-10-02: nalgebra-cairo #97 (`7111163`), simba-cairo #6 (`9b7c065`) | PP-N |
| PP-FU2 | Notes of the reviews of the pre-push lots, being fixed in lot PP-FU2 (in progress 2026-10-03) | PP-N |
| REL-FU2 | Deferred notes of #92's review, none blocking: the `scripts/release.py` docstring says 45 index checks where the code makes 46; `--verify-timeout` no longer caps wall-clock time; a small window between a successful `scarb publish` and saving the `submitted` marker. Folded into REL-a (2026-10-03): the publishing mode, its state file and the `submitted` marker are gone; the index wait of the verify mode is a wall-clock cap, its docstring and help say so | REL-FU |
| PP-S2 | simba-cairo's pre-push hook, deferred notes of the review of its #5 (a row here, since simba's plan is this track's): a tag of a tree or blob gives a garbled message; the tag exemption does not check that the tag points at an already-checked commit; a stale "only deletions" comment | PP-S |
| TC2 | Left on the old toolchain by TC: `tools/split/probes`, `tools/shapegen/proto` and `tools/shapegen/compare.py` still pin Cairo 2.19.4 / snforge 0.61.0 (no CI job builds them); the curated figures of `benchmarks/*/README.md` and `GAS.md` keep their 0.61.0 values; the `--md` option those READMEs name does not exist in `benchmarks/scripts/gas_report.py`; `benchmarks/libs` stays on 2.19.4 / 0.61.0 until its vendored libraries (cubit, orion; corelib 2.20.0 removed `u32_as_non_zero` / `u64_as_non_zero`) are updated or retired | TC |
| PUB | Publishing is done by the orchestrator by hand, one `scarb publish -p <package>` per package, on the project manager's go naming package, version, commit and archive sha256 (organisation standard, 2026-10-02). In REL-a (2026-10-03): `scripts/release.py` no longer publishes; `request --commit SHA` builds each published package's archive from a clean checkout detached at the commit and writes `docs/releases/<version>.md` (package, version, commit, sha256, sizes, in publication order, with the ordered `scarb publish -p` commands); `verify` reads the registry index back against it | 0.1.1 |
| REL | nalgebra 0.2.0 with simba 0.3.0. REL-a (2026-10-03, no manifest change): the release tooling (PUB, REL-FU2), the slerp and test doc lines, and the exact-zero analysis of the six `R::is_sign_positive` sites (`docs/research/rel-zero.md`: one public result moves, `from_rotation_matrix` at a zero trace, away from nalgebra-rs unless its test becomes `tr > 0`). REL-b (2026-10-03): version 0.2.0 for every package, `from_rotation_matrix` on upstream's `tr > 0` with its zero-trace test, CHANGELOG 0.2.0. simba 0.3.0 / fixed 0.5.0 **not taken**: Scarb cannot resolve them with `nalgebra_glam`, whose `glam_core` 0.4.1 (the latest) requires `fixed ^0.4.0`; it waits for a `glam_core` on fixed 0.5.0 or a decision on `nalgebra_glam` | OPT-2, simba 0.3.0 |
| TC3 | A future Scarb release carrying the upstream fix that makes closure type names path-free changes the Sierra and class hashes of anything holding a closure: such a bump gets its own lot (row only; our libraries declare no class) | TC |

Gates, precisely (programme session, 2026-09-28): gate 2 (5 s / 1 GB) is a crate's **marginal** cost,
cost(empty consumer of the crate) − cost(empty consumer of its direct dependencies together); gate 3
(15 s / 3 GB) is a **declared closure**'s cost over the no-dependency baseline. Facades and products are
judged on gate 3 only: the facade `nalgebra` (it re-exports everything) cannot pass it, so the plan
declares the closures that must (e.g. core + geometry, core + one decomposition family, `nalgebra_glam`'s
closure, the first consumer to fix: +37 s / +5.5 GB today).

No public path may change (glam-cairo's cut, PK-G section 9: Cairo finds a core-trait impl without an
import only in the type's own module, and has no supertraits, so types and the methods returning them stay
with the type's crate); target: a non-breaking **0.1.x** release through the facade, like glam 0.4.1.
Releases: no publication without the project manager's written go (`docs/ORCHESTRATOR.md`,
"Releases"); sub-crates share the repository version.

Execution model: one **orchestrator** session owns the plan, the workspace manifests, CI, the gas
snapshot and the merges. Work packages are delegated to **sub-agents running in parallel**, each
owning a disjoint set of files (one module directory per agent) and delivering through a PR that
must be green (fmt, build, tests, gas snapshot). Packages inside a milestone are independent
unless a dependency is stated; milestones are sequential only where an arrow says so.

```
M0 ─► M1 ─► M2 ─┬─► M3 ─┬─► M6
                └─► M4 ─┴─► M5
```

Priority inside each milestone follows what a game physics engine actually calls
([research/01](research/01-nalgebra-analysis.md) §3): vectors → unit quaternion / complex →
isometries → 2x2/3x3 matrices → scalar helpers → symmetric eigen → 6D → small factorizations →
dynamic → SVD.

## M0 — Foundations ✅

Research reports, four benchmark suites, design decisions, CI, agent conventions.

## M1 — `simba`: the scalar ✅

| WP | Content | Depends on |
|---|---|---|
| 1.1 | `Fixed` Q32.32: storage, constants, conversions, add/sub/neg/cmp, mul/div/rem, abs/signum/min/max/clamp/floor/ceil/round/recip, fused kernels (`sum_prod2/3/4`, `diff_prod`, `mul_add`, `lerp`, `norm2/3/4`), wide accumulator, `Real` trait, approx-eq in ulp, Python bit-exact model | — |
| 1.2 | Transcendentals: sqrt, inv_sqrt, sin/cos/sin_cos/tan, atan2/atan, acos/asin, exp/ln/pow (low priority) + `tools/` polynomial generator | 1.1 (API only) |
| 1.3 | `tools/oracle`: Rust generator of golden vectors from upstream nalgebra (f64, inputs quantised to Q32.32) | — |

## M2 — `nalgebra::base`: static vectors and matrices ✅

| WP | Content | Depends on |
|---|---|---|
| 2.1 | `Vector2/3/4`: constructors, ops, dot, cross, perp, norms, normalize, lerp, component-wise, min/max, angle, orthonormal basis | M1 |
| 2.2 | `Matrix2/3/4`: constructors (row-major `new`, from columns/rows/diagonal), ops, `mul_vec`, transpose, trace, determinant, `try_inverse`, `cross_matrix`, outer product | 2.1 |
| 2.3 | `SymMatrix2/3`, structured kernels (`quadform`, `mul_transpose`, sym inverse) | 2.1 |
| 2.4 | `Point2/3`, `Unit<V>` | 2.1 |
| 2.5 | `Vector6`, `Matrix6` (blocks of 3), rectangular `Matrix3x2`-style blocks only where M4/M5 need them | 2.2 |

## M3 — `nalgebra::geometry` ✅ (Similarity included; `Scale`, `Reflection`, `Transform` family deferred)

| WP | Content | Depends on |
|---|---|---|
| 3.1 | `UnitComplex`, `Rotation2` | 2.2, 2.4 |
| 3.2 | `Quaternion`, `UnitQuaternion` (axis-angle, `from_rotation_matrix`, `append_axisangle_linearized`, `renormalize_fast`, nlerp/slerp, rotation between vectors), `Rotation3` (euler angles, look_at) | 2.2, 2.4 |
| 3.3 | `Translation2/3`, `Isometry2/3` (`inv_mul`, `transform_point/vector`, inverse forms, lerp_slerp) | 3.1, 3.2 |
| 3.4 | `Similarity2/3` (in progress); `Scale`, `Reflection`, `Transform`/`Projective`/`Perspective`/`Orthographic` are deferred (rendering-oriented, unused by the physics stack) | 3.3 |

## M4 — `nalgebra::linalg`: small static decompositions ✅

| WP | Content | Depends on |
|---|---|---|
| 4.1 | `Cholesky` and `LDLᵀ` (upstream `UDU`) for 2/3/4/6, `solve`, `inverse` | 2.2, 2.5 |
| 4.2 | `LU` with partial pivoting for 2/3/4/6, `solve`, `determinant` | 2.2, 2.5 |
| 4.3 | `SymmetricEigen` 2x2 (closed form) and 3x3 (fixed-sweep Jacobi) | 2.3 |
| 4.4 | `SVD` 2x2/3x3, polar decomposition, `pseudo_inverse`; `QR` 2/3/4 | 4.3 |

## M5 — Dynamic algebra (scoped by multibody needs) — superseded by M8 (P13 `dynamic`, P14-P17)

| WP | Content | Depends on |
|---|---|---|
| 5.1 | `DVector`, `DMatrix`: construction, element-wise ops, `gemv`, `gemm`, `axpy`, `tr_mul`, `quadform`, static-kernel dispatch | M2 |
| 5.2 | Dynamic `LU` / `Cholesky` solve | 5.1, M4 |

Gate: rapier-cairo v1 explicitly cuts multibody joints, IK and soft bodies (confirmed 2026-09-20),
so M5 stays deferred until a consumer exists.

## M4b — Consolidation

| WP | Content | Depends on |
|---|---|---|
| 4.5 ✅ (A-B #19, C #18, E; `Real::div`/`rem` split out as 4.6) | Promote `base::matrix_test_utils` to `pub(crate)` and delete the duplicated builders in `linalg`/`geometry`; fused `conj_mul` quaternion kernel (saves the 3 negations of `Isometry3::inv_mul`); `Wide × Fixed` accumulator op in `simba` for exact triple products (4x4 / 6x6 determinants); `Real::div` / `Real::rem` so that `normalize`/`unscale`/`new_normalize` no longer depend on the foreign scalar's `/` semantics (closes the one `simba_fixed` conformance gap); add `crates/simba_fixed/tools/gen_vectors.py --check` to the gate | M3, M4, 6.1 |
| 4.6 ✅ (#20) | `Real::div` / `Real::rem`: nalgebra's generic code no longer reaches the scalar's `/` operator; `simba_fixed` integration tests become bit-identical (closes the D8 gap) | 4.5 |

## M7 — One scalar for the stack (mirror of the Rust dependency model)

| WP | Content | Depends on |
|---|---|---|
| 7.1 ✅ (#21) | Delete `simba::fixed` (the second Q32.32), `simba` = `Real` / `Transcendental` over fixed-cairo's `fixed` (registry, pinned 0.3.0: `wide::Acc` and the `f64`-like nearest division were obtained by escalation), `simba_fixed` merged into `simba`, goldens and gas regenerated, prepared divisors (`Real::div3..div16`) | owner decision 2026-09-22 |
| 7.2 ✅ (#22) | Re-rank the measured variants under `fixed` 0.3 (oracle tolerance > upstream formula > gas): no variant switched (each cheaper loser fails an oracle case or is not upstream's formula); bit-identical `divN` refactors (`Lu6::try_inverse` −3.4 %) | 7.1 |
| 7.3 ✅ (#23) | API parity inventory against nalgebra-rs 0.35.0: `scripts/api_parity.py` + generated `docs/API_PARITY.md` (21.9 % coverage, 23 proposed packages) | 7.1 |

## Repositories (2026-09-25)

The stack mirrors the Rust ecosystem repository by repository (owner's decision; precedent:
glam-cairo `docs/SPLIT.md`): bal7hazar/fixed-cairo (`fixed`, the scalar), bal7hazar/simba-cairo
(`simba`, the scalar traits), bal7hazar/nalgebra-cairo (this repository, `nalgebra`),
bal7hazar/glam-cairo, bal7hazar/glamx-cairo, bal7hazar/rapier-cairo (renamed from `*.cairo`; old
URLs redirect). Escalations (scalar included) and cross-repository questions go to the project
manager of `slingfall` (notebook `/home/claude/projects/pm/`), which relays them to the sibling
tracks.

| step | content | state |
|---|---|---|
| S1 | extract `simba-cairo` from this repository's history (prune commit, own gate, CI, docs, version 0.1.0, branch protection) | ✅ `6fd8a43` in simba-cairo |
| S1b | publish `simba` 0.1.0 on scarbs.xyz from simba-cairo (tag `v0.1.0`; `simba_testing` folded into a test-only module, simba-cairo #1) | ✅ 2026-09-25 |
| S2 | remove `crates/simba` here, depend on `simba = "0.1.0"` (registry), drop the `Library (simba)` shard; `api_parity.py` reads simba's sources from the registry package | ✅ |

## M8 — Complete coverage of nalgebra-rs 0.35.0, then release 0.1.0

Target (owner, 2026-09-23/24): publish `nalgebra` 0.1.0 (`simba` 0.1.0 is out, from simba-cairo) on scarbs.xyz once the public API
is **strictly nalgebra-rs 0.35.0's, neither more nor less** — every non-excluded item of
[API_PARITY.md](API_PARITY.md) `ported`, and no Cairo-only public item beyond the renames Cairo
imposes (its operator traits are homogeneous: `mul_vec` for `M * v`, …). Progress is measured by
`scripts/api_parity.py` (gate: `--check`). Started at 21.9 % (413 / 1,888); 22.9 % after 8.0 (extras 401 → 72, all the scalar-kernel exception); 25.6 % after 8.1b-3; 31.2 % after P08; 47.0 % after P09a, P02 and P09b; 53.0 % after P03 and P10; 64.0 % after P04 / P05 (8.2c) and P12; 68.9 % after P11b and P07; 72.7 % after P11a; 74.7 % after P06; 76.8 % after WP 8.4-R (geometry residuals); 80.6 % after P14a; 83.3 % after P13; 85.2 % after P20 + P18; 86.5 % after P14b; 89.1 % after P15; 91.2 % after P21; 96.1 % after P19; 99.6 % after P16; **99.9 % after P17 + polish (#59): 1,885 ported, 0 partial, 1 missing (the owner-ruled `cs_matrix_from_matrix_market(path)`), 550 excluded — the M8 coverage target is reached**; 0 undocumented extras. WP 8.0b aligns `simba::Real` names on simba-rs (`is_sign_negative`, `T::pi()`, …).

| WP | Content (parity packages) | Depends on |
|---|---|---|
| 8.0 ✅ (#25) | Strict removal of the Cairo-only public API (`SymMatrix2/3`, `conj_mul`, fused-kernel helpers, undocumented decomposition extras…), keeping implementation kernels private; ruling needed on `simba::Real`'s fused-kernel hooks (with gas figures) | 7.3 |
| 8.1 (8.1a ✅ #26: design + prototype; 8.1d spike #49: compile memory → DESIGN D9, features per family; 8.1e ✅ #51: features `statistics` / `blas` / `closures`, one Workspace CI job) | `tools/shapegen`: generator of the 54 static shapes (`Matrix1..6`, `MatrixRxC`, `Vector1..6`, `RowVector1..6`) from templates, committed output + `--check`; existing shapes migrated bit-identically, gas not worse (P01) | 8.0 |
| 8.2 (P02 ✅ #34, P03 ✅ #36, P04 + P05 ✅ #39) | Static base completion through the generator: P02 (norms, component-wise, construction, conversions), P03 (`map` / `zip` / in-place), P04 (swizzles), P05 (rows, columns, blocks) | 8.1 |
| 8.3 ✅ (P06 #45, P07 #43) | P06 (statistics, BLAS-like), P07 (homogeneous / cg helpers; the 6-D homogeneous conversions are out of scope, issue #41) | 8.2 |
| 8.4 (P08 ✅ #31, P09a ✅ #33, P09b ✅ #35, P10 ✅ #37, P11a ✅ #44, P11b ✅ #42, P12 ✅ #38, R ✅ #46 in-place / `Unit`, H ✅ #48 simba 0.2.0 / fixed 0.4.0) | Geometry: P08 (quaternions, unit complex) → P09a (rotation, translation, point) → P09b (isometry, similarity, `*Matrix` variants) → P10 (scale, reflection) → P11a/b (transform family, perspective, orthographic) → P12 (dual quaternions) | 8.0 (parallel with 8.1-8.3 where files are disjoint) |
| 8.5 (P13 ✅ #52 `dynamic` feature, P14a ✅ #50, P14b ✅ #54, P15 ✅ #56 + linalg features, P16 ✅ #58, P17 ✅ #59) | Dynamic: P13 (`DMatrix` / `DVector`, macros) → P14 (decomposition API, triangular solves) → P15 (full-pivot LU, col-pivot QR, LBLᵀ) → P16 (Schur, Hessenberg, bidiagonal, tridiagonal, general eigen) → P17 (exp, pow), P18 (convolution) | 8.2 |
| 8.6 (P18 + P20 ✅ #53 `sparse` / `io`, P21 ✅ #55 `macros`, P19 ✅ #57 `nalgebra_glam`) | P19 (glam-cairo conversions, `glam = "0.3.0"`), P20 (sparse `CsMatrix`, Matrix Market from strings), P21 (crate-root functions and macros) | 8.5 |
| 8.7 ✅ (8.7a #60: packaging, CHANGELOG, README; `nalgebra` 0.1.0 and `nalgebra_glam` 0.1.0 published 2026-09-26, tag `v0.1.0` on `adf981b`) | Release: parity 100 %, `scarb doc`, CHANGELOG, versioning policy (numeric change = MINOR), tag-driven publication of `nalgebra` 0.1.0 (`scarb package` refuses path-only dependencies, dev ones included: `nalgebra_testing` must become a test-only module or get a registry version first, as simba-cairo #1 did) | all |

Owner rulings of 2026-09-26: `nalgebra::io::cs_matrix_from_matrix_market(path)` (reads a file; Cairo
has no file system; the `_str` form is ported) stays a **missing** item: the 0.1.0 target is every
other item. glam conversions (P19) go to the separate package `nalgebra_glam` (DESIGN D11).

Owner rulings for the shapes (2026-09-24): one heterogeneous-product name, `mul_mat` (every
conformable product, `M * v` included; `mul_vec` / `tr_mul_vec` removed; `*` stays on square
shapes); `Vector6` / `Matrix6` flat like upstream (the 3D-block layout goes). 8.1b sequence:
8.1b-1 migrate the hand-written 2/3/4 shapes into the generator (bit- and gas-identical) → 8.1b-2
flatten `Vector6` / `Matrix6` → 8.1b-3 the 28 new shapes + `mul_mat` everywhere + parity script →
8.1b-4 generated test packages + CI jobs. Done: 8.1b-1 #28, 8.1b-2 #29, 8.1b-3 #30 (36 shapes, `mul_mat`,
`shapes_tests_core`), 8.1c #32 (public-API tests moved to `tests_base` / `tests_geometry` /
`tests_linalg`: the `nalgebra` test build fell from 14.7 GB to 5.3 GB, CI shards ≈ 3 min).

Fidelity rules settled by the orchestrator under the owner's "same as the Rust reference" rule
(applied by a later sweep, WP 8.4-fix): approximate comparisons of quaternions / unit quaternions
accept `−q` like upstream's `approx` impls; an upstream result that would be `NaN` (e.g. `ln` of a
negative real quaternion) panics instead of returning a Cairo-only convention, like `fixed`'s
`sqrt` of a negative. Applied in P09a, P09b and P10.

Steps criterion (owner guideline, restated 2026-09-25 for every repository): be as close as possible
to the Rust API when it does not cost Cairo steps; where a faithful formulation costs steps, the
cheaper form wins and the deviation is documented (doc comment + report). Variant ranking: oracle
tolerance first (a variant outside the reference tolerance never ships), then steps / gas, then
closeness to upstream's formula. A parity item that can only be ported with a costlier formulation
is reported to the orchestrator rather than ported as is. The strict-parity target of 0.1.0 stands.

Execution: models, machine budget and launchers are those of `slingfall/OPERATIONS.md` §2-§4, the
track's recipes are in [ORCHESTRATOR.md](ORCHESTRATOR.md); one PR per WP; parity figures reported
per PR.

## M6 — Interop and release — superseded by M8 (P19 `nalgebra_glam`, 8.7 release)

| WP | Content | Depends on |
|---|---|---|
| 6.1 ✅ | `simba_fixed`: `Real` + `Transcendental` for glam-cairo's shared `fixed::Fixed` (git-pinned), bit-for-bit conformance suite between the two scalars, integration tests on `Vector3`/`SymMatrix3`/`UnitQuaternion`/`Isometry3<fixed::Fixed>` | M3, M4 |
| 6.2 | Conversions with glam-cairo types, `scarb doc`, publication on scarbs.xyz | 6.1 |

Conversions with glam-cairo, conformance suite against the oracle, `scarb doc`, publication of
`simba` and `nalgebra` on scarbs.xyz (tag-driven), upgrade policy (toolchain bumps are separate
PRs that re-run `benchmarks/` and review every ranking).

## Out of scope

Only what is not part of the `nalgebra` crate 0.35.0 (the separate crates `nalgebra-lapack`,
`nalgebra-glm`, `nalgebra-sparse`, `nalgebra-macros` internals) and the closed list of exclusion
reasons of [API_PARITY.md](API_PARITY.md) (SIMD, rayon, unsafe storage, borrowed views, formatting,
zero-copy glue, random generators, foreign-crate interop, generic-dimension machinery). Everything
else in nalgebra-rs, including its optional features (`sparse`, `io`), its deprecated items and
what this plan used to list as out of scope (Schur, Hessenberg, matrix exponential, convolution,
macros), is in the 0.1.0 target (owner, 2026-09-24).

## Definition of done (every WP)

See [DESIGN.md](DESIGN.md) §D7 and [AGENTS.md](../AGENTS.md): doc comments, unit tests, oracle
vectors, gas benchmarks (with variants when the optimum is ambiguous), regenerated snapshot,
`scripts/check.sh` green.
