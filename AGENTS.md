# AGENTS.md

Canonical instructions for AI agents (and humans) working on nalgebra.cairo. Read
[docs/DESIGN.md](docs/DESIGN.md) and [docs/PLAN.md](docs/PLAN.md) before writing code.

## Mission

A Cairo port of the Rust `nalgebra` crate on glam.cairo's Q32.32 `fixed::Fixed`, as a building block of a
**provable physics engine for games** (siblings: glam.cairo, rapier.cairo). Every Cairo step is
proven, so gas is a first-class requirement, on par with correctness.

## Toolchain

`.tool-versions` pins scarb 2.19.4 and snforge 0.61.0 (asdf). Gate: `./scripts/check.sh`
(fmt, lint, build, tests, gas snapshot). Refresh the snapshot with `./scripts/check.sh --update`.

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

- The scalar is `fixed` (glam.cairo, pinned version); its rounding is the spec: products and fused
  kernels floor once per output scalar, `/` / `recip` round to nearest ties-to-even like `f64 /`.
  Overflow panics; nothing may wrap silently.
- A kernel or rounding mode `fixed` lacks is an ESCALATION to the glam.cairo orchestrator (in the
  report), never a local reimplementation.
- Same inputs must give bit-identical outputs forever: changing the result of a function in its last
  bit is a breaking change.
- Tolerances are expressed in raw units (ulp), not float epsilons.

## Definition of done (every public function)

- `///` doc comment: what it computes, rounding/overflow behaviour, upstream nalgebra equivalent.
- `test_<fn>_<scenario>` unit tests: exact cases, oracle vectors, identities, `#[should_panic]` cases.
- `bench_<group>__<variant>` gas tests, all `#[inline(never)]`, inputs through
  `nalgebra_testing::black_box`, results asserted, one `bench_<group>__baseline` per group.
- `gas/<module>.json` + `.md` regenerated (`./scripts/check.sh --update`); the PR explains any gas increase.
- Gate run in the foreground, conventional commits with the trailer, push, PR following
  `.github/PULL_REQUEST_TEMPLATE.md`, `gh pr checks --watch` until green, never merge. One work
  package per PR, plus a `REPORT.md` (git-ignored) at the worktree root: summary, API, gas table,
  deviations, deferred items, requested re-exports, escalations, PR URL.
- Compile budget: keep generated test files small (a few hundred cases per op at most); oversized
  test crates are the first cause of CI failures.

## Conventions

- Upstream names and semantics wherever the operation exists upstream (`try_inverse`,
  `transform_point`, `Matrix3::new` row-major). Deviations are documented in the doc comment.
- One type per file, tests in an inline `#[cfg(test)] mod tests` at the bottom of the file.
- Errors are `felt252` constants in an `errors` module; panic messages are stable API.
- Pure library: no `starknet` dependency, no storage, no proc macros; the only dependency is
  glam.cairo's `fixed` (registry, pinned version).
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
