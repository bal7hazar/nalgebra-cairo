# AGENTS.md

Canonical instructions for AI agents (and humans) working on nalgebra-cairo. Read
[docs/DESIGN.md](docs/DESIGN.md) and [docs/PLAN.md](docs/PLAN.md) before writing code.

## Mission

A Cairo port of the Rust `nalgebra` crate on fixed-cairo's Q32.32 `fixed::Fixed`, as a building block of a
**provable physics engine for games** (siblings: fixed-cairo, simba-cairo, glam-cairo, glamx-cairo, rapier-cairo). Every Cairo step is
proven, so gas is a first-class requirement, on par with correctness.

## Toolchain

`.tool-versions` pins scarb 2.19.4 and snforge 0.61.0 (asdf). Gate: the pull-request CI (fmt,
lint, build, every test shard, gas snapshots). Locally, work packages run crate-scoped checks only:
`scarb build -p <pkg>`, `scarb lint -p <pkg> --deny-warnings`, `snforge test -p <pkg>` (unfiltered,
piped into `python3 scripts/gas_report.py --update gas/` to refresh that package's snapshot), plus
the cheap `scarb fmt --check`, `api_parity.py --check`, `shapegen.py --check`. `./scripts/check.sh`
is the whole-workspace gate, for orchestrator-driven runs only (release, toolchain bump): the shared
machine is CPU-capped (programme rule, 2026-09-25).

## Efficiency rules (measured, see docs/BENCHMARK.md)

1. **No loops** in static code: unroll. A loop iteration costs ~1,100 gas before doing any work.
2. Prefer, in order: felt252 / `BoundedInt` arithmetic → corelib libfuncs (`Sqrt`, `wide_mul`,
   `DivRem`) and const tables → small-integer math → bitwise → loops. Bitwise beats math only on
   u128/u256 and for data-parallel bit tricks.
3. **One `DivRem` instead of `/` and `%`**; divide by constants, never by variables when a
   reciprocal can be stored.
4. **Every sum of products goes through a fused kernel** (`sum_prod*`, `diff_prod`, `mul_add`,
   `norm*`): accumulate unscaled, rescale once per output scalar.
5. Named-field `Copy` structs, passed by value. No `Array`, `Span`, `Felt252Dict`, `Box` in static types.
6. `#[inline(always)]` on small ops; default on big kernels. Methods of generic impls, not generic
   free functions.
7. Never use u256 as a working type, `pow()`, sign-magnitude numbers, or `try_into().unwrap()`
   chains inside unrolled code (checks belong at the end of a kernel).
8. **When the cheapest implementation is not obvious, implement the candidates (math / bitwise /
   table / loop), benchmark them side by side, keep the cheapest, and leave the benchmark in place.**
   Intuition has been wrong several times in this project; only measurements count.

## Numeric rules

- The scalar is `fixed` (fixed-cairo, pinned version); its rounding is the spec: products and fused
  kernels floor once per output scalar, `/` / `recip` round to nearest ties-to-even like `f64 /`.
  Overflow panics; nothing may wrap silently.
- A kernel or rounding mode `fixed` lacks is an ESCALATION to the fixed-cairo orchestrator (in the
  report), never a local reimplementation.
- Same inputs must give bit-identical outputs forever: changing the result of a function in its last
  bit is a breaking change.
- Tolerances are expressed in raw units (ulp), not float epsilons.

## Definition of done (every public function)

- `///` doc comment: what it computes, rounding/overflow behaviour, upstream nalgebra equivalent.
- `test_<fn>_<scenario>` unit tests: exact cases, oracle vectors, identities, `#[should_panic]` cases.
- `bench_<group>__<variant>` gas tests, all `#[inline(never)]`, inputs through
  `nalgebra_testing::black_box`, results asserted, one `bench_<group>__baseline` per group.
- `gas/<module>.json` + `.md` regenerated for the packages touched (`snforge test -p <pkg> |
  python3 scripts/gas_report.py --update gas/`); the PR explains any gas increase.
- Crate-scoped checks run in the foreground, conventional commits with the trailer, push, PR following
  `.github/PULL_REQUEST_TEMPLATE.md`, `gh pr checks --watch` until green, never merge. One work
  package per PR, plus a `REPORT.md` (git-ignored) at the worktree root: summary, API, gas table,
  deviations, deferred items, requested re-exports, escalations, PR URL.
- Compile budget: keep generated test files small (a few hundred cases per op at most); oversized
  test crates are the first cause of CI failures. Pass test operands through `black_box` too: the
  compiler specialises a function per set of constant arguments (WP 8.5-P14a: 16.9 GB for one test
  package with constant operands). Test packages depend on `nalgebra` with `default-features =
  false` and the features they test (DESIGN D9); `scarb build -p nalgebra --no-default-features`
  must pass.

## Conventions

- Upstream names and semantics wherever the operation exists upstream (`try_inverse`,
  `transform_point`, `Matrix3::new` row-major). Deviations are documented in the doc comment.
- One type per file. Tests of the PUBLIC API live in the test-only package of the module
  (`crates/tests_base`, `crates/tests_geometry`, `crates/tests_linalg`, generated shape tests in
  `crates/shapes_tests_*`; helpers in `crates/tests_utils`), mirroring the module tree; only tests
  of crate-internal items stay in-crate (`#[cfg(test)] mod tests;`). Keep each test package's
  `scarb build --test` peak under ~7 GB (split it and add a CI shard when it grows).
- Errors are `felt252` constants in an `errors` module; panic messages are stable API.
- Pure library: no `starknet` dependency, no storage, no proc macros; the only dependency is
  fixed-cairo's `fixed` (registry, pinned version).
- No `core::internal::bounded_int` in this repository: the bounded-int kernels live in `fixed`.

## Boundaries

- Agent-safe: `crates/**`, `tools/**`, `benchmarks/**`, docs.
- Orchestrator only (see `docs/ORCHESTRATOR.md`): workspace `Scarb.toml`, `.tool-versions`,
  `.github/**`, `scripts/**`, `docs/**`, re-export lines of `lib.cairo` / module roots beyond
  what a brief allows, gas snapshots of other modules, merging PRs, toolchain bumps (always a
  dedicated PR that re-runs `benchmarks/`). Needs on those files go in the report's
  "Escalations" section.
- Never edit `benchmarks/libs/vendor/**` (pristine third-party code, see each `NOTICE`).
- Sub-agents own a disjoint set of files per work package (allowlist in the brief), commit on
  their own `feat/<module>` branch in their own worktree, push and open the PR, never merge.
