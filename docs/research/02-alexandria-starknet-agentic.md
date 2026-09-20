# Benchmark report: Alexandria + starknet-agentic

Scope: architecture and best practices to adopt for `nalgebra.cairo` (Cairo port of Rust `nalgebra`,
base of a provable physics engine).

Sources (read-only shallow clones, paths below are relative to `scratchpad/refs/`):

- `alexandria/` — keep-starknet-strange/alexandria @ `6d2cfcc` (2026-03-05), Cairo/Scarb 2.16.0, snforge 0.56.0
- `starknet-agentic/` — keep-starknet-strange/starknet-agentic @ `c7e1c9e` (2026-09-14)

User guidance used as the evaluation lens:

> G1. Prefer simple math (add, mul, divmod) over bitwise ops (and/or/xor/shift), and bitwise ops over loops.
> G2. Every feature ships with unit tests that track gas, so alternative implementations can be compared.

---

## 1. Alexandria

### 1.1 Workspace layout

`alexandria/Scarb.toml` is a pure virtual workspace (no root package); 17 members under `packages/`.

```toml
[workspace]
members = ["packages/data_structures", "packages/linalg", "packages/math", "packages/numeric", ...]
cairo-version = "2.16.0"

[workspace.dependencies]
starknet = "2.16.0"
cairo_test = "2.16.0"
snforge_std = "0.56.0"

[workspace.tool.fmt]
sort-module-level-items = true

[workspace.package]
version = "0.10.0"

[profile.coverage]
sierra = true

[scripts]
all = "scarb build && snforge test"
```

Observations:

- One version for the whole workspace (`[workspace.package] version`), inherited with
  `version.workspace = true` in each member. All packages are released in lockstep.
- Shared dependency versions are centralised in `[workspace.dependencies]` and consumed with
  `snforge_std.workspace = true`.
- Formatter config is centralised (`[workspace.tool.fmt]`) and each member opts in with
  `[tool] fmt.workspace = true`. The only fmt option used is `sort-module-level-items = true`.
- Tool versions pinned in `alexandria/.tool-versions` (`scarb 2.16.0`, `starknet-foundry 0.56.0`).
  CI's `setup-scarb` reads this file; snforge version is duplicated by hand in the workflow.
- `Scarb.lock` is committed.
- Sloppiness to avoid: stray `name = "alexandria"` / `version = "0.8.0"` keys sit inside the
  `[workspace]` table and disagree with `[workspace.package] version = "0.10.0"`; the README version
  is maintained by hand.

Per-package manifest (`alexandria/packages/linalg/Scarb.toml`):

```toml
[package]
name = "alexandria_linalg"
version.workspace = true
description = "A set of linear algebra libraries and algorithms"
homepage = "https://github.com/keep-starknet-strange/alexandria/tree/main/packages/linalg"
edition = "2023_11"
cairo-version = "2.16.0"

[tool]
fmt.workspace = true

[dependencies]
alexandria_math = { path = "../math", version = "0.10.0" }

[dev-dependencies]
snforge_std.workspace = true
```

- Package names are prefixed (`alexandria_<pkg>`) because scarbs.xyz is a flat namespace.
- Intra-workspace deps carry BOTH `path` and `version` so `scarb publish` can rewrite them to
  registry deps. This is required for publishing; keep it.
- Pure libraries do not depend on `starknet` at all (math, linalg, numeric have no `starknet` dep and no
  `[[target.starknet-contract]]`). Only `storage`, `utils`, `evm`, ... pull it in.
- Edition is still `2023_11` for most packages (`2024_07` only for `macros`). This is legacy; the
  2023_11 edition has a wider implicit prelude. New code should start on `2024_07`.
- Each package has a `snfoundry.toml` with `[snforge.default] sierra = true / casm = true`
  and its own `README.md` listing the modules.

Each package directory: `Scarb.toml`, `snfoundry.toml`, `README.md`, `src/`, `tests/`.

### 1.2 Module / trait conventions

- `lib.cairo` is a flat list of `pub mod x;` with a `//!` crate doc header
  (`alexandria/packages/linalg/src/lib.cairo`, `packages/math/src/lib.cairo`). One algorithm per file.
- Two API styles coexist:
  1. Free generic functions (math/linalg/numeric):
     `pub fn dot<T, +Mul<T>, +AddAssign<T, T>, +Zero<T>, +Copy<T>, +Drop<T>>(mut xs: Span<T>, mut ys: Span<T>) -> T`
     (`packages/linalg/src/dot.cairo`). Anonymous `+Trait<T>` bounds everywhere.
  2. Trait + impl for types: `pub trait StackTrait<S, T>` / `pub trait QueueTrait<T>` / `pub trait VecTrait<V, T>`
     (`packages/data_structures/src/*.cairo`), or `#[generate_trait] pub impl I257Impl of I257Trait`
     (`packages/math/src/i257.cairo`, `decimal.cairo`; 17 uses of `#[generate_trait]` overall).
- Naming: traits are `XxxTrait`; impls are `XxxImpl` or `<Type><Trait>` (`U8BitShift of BitShift<u8>`,
  `U256OptWrappingImpl of OptWrapping<u256>`, `i257Add of Add<i257>`). Inconsistencies exist
  (`U64OptWrappingAddImpl`, typo `U16OptWrappingmpl`, `I128Default of Default<i257>`), showing there is no
  lint/naming gate.
- Strategy pattern through impl selection: `packages/sorting/src/interface.cairo` defines
  `pub trait Sortable { fn sort<T, ...>(array: Span<T>) -> Array<T>; }` and each algorithm is
  `pub impl MergeSort of Sortable`, re-exported from `lib.cairo`
  (`pub use merge_sort::MergeSort;`). This is a good template for "several implementations of one
  operation, benchmark them side by side".
- Visibility: everything public is `pub`; internal consts use `pub(crate)` (`wad_ray_math.cairo`);
  operator impls are left private (`impl i257Add of Add<i257>`), relying on trait-impl resolution.
  Re-exports are rare (sorting, and `pub use core::num::traits::{Bounded, WideMul, ...}` in math `lib.cairo`).
- Errors: mixed. `dot` panics with a felt short-string (`assert(xs.len() == ys.len(), 'Arrays must have the same len')`),
  `kron` returns `Result<Array<T>, KronError>`, `interpolate` uses `assert!` with ByteArray messages.
  No convention — pick one for nalgebra.cairo (see section 3).
- Docs: `///` with a house format `#### Arguments` / `#### Returns` / `#### Panics`; harvested by `scarb doc`.
- `#[inline(always)]` is used selectively and almost exclusively in hot numeric helpers
  (71 occurrences in `math/src/opt_math.cairo`, 8 in `mod_arithmetics.cairo`, ~20 each in `bytes`), with the
  explicit caveat at the top of `opt_math.cairo`: "Runtime optimized math utils ... Might increase contract size".

### 1.3 Test organisation

- All tests live in `packages/<pkg>/tests/*.cairo`, one file per source module (`dot_test.cairo`,
  `norm_test.cairo`), with NO `tests/lib.cairo`. Scarb therefore compiles them into a single
  `<pkg>_integrationtest` target, which is why gas report keys look like
  `alexandria_linalg_integrationtest::norm_test::norm_test_1`.
- Consequence: tests only see the public API (black-box). No inline `#[cfg(test)]` unit tests in math
  packages (only 3 files in the whole repo use `cfg(test)`).
- Leftovers: `packages/*/src/tests.cairo` files still exist (e.g. `packages/linalg/src/tests.cairo`
  containing `mod dot_test; ...`) but are not referenced from `lib.cairo` — dead files from the
  pre-migration layout. Anti-pattern: stale files nobody compiles.
- Runner: `snforge` everywhere (`snforge_std` dev-dep) even for pure libraries; `cairo_test` is only a
  dev-dep of the Rust proc-macro package. Reason: snforge prints per-test gas, which feeds the gas report.
- Test style: plain `#[test]` + `assert!(expr == value)`, `#[should_panic(expected: ('msg',))]`.
  No `#[fuzzer]` anywhere (0 occurrences), no `#[available_gas]` (0), no property tests.
  Coverage is shallow for numeric code: e.g. `dot` has 2 tests, `norm` 4 tests all on `[3, 4]`.

### 1.4 Gas tracking: `gas_report.json` + `scripts/generate_gas_report.sh`

Mechanism (`alexandria/scripts/generate_gas_report.sh`):

1. Run `snforge test` (optionally `-p alexandria_<pkg>`), grep the PASS lines and keep `l2_gas` only:
   ```bash
   snforge test $1 | grep -E '\[PASS\] .* \(l1_gas: ~[0-9]+, l1_data_gas: ~[0-9]+, l2_gas: ~[0-9]+\)' \
     | sed -E 's/\[PASS\] ([^(]*) \(l1_gas: ~[0-9]+, l1_data_gas: ~[0-9]+, l2_gas: ~([0-9]+)\)/\1 \2/' | sort -k2 -nr
   ```
2. Load the committed root `gas_report.json` (1,425 entries, `"<target>::<module>::<test>": <l2_gas>`),
   line-parse it with bash string ops.
3. For each test, print `INCREASE:` / `DECREASE:` lines, overwrite `gas_report.json`, and write a
   git-ignored `gas_report_diff.json` (`"test": "+123"`).
4. CI job `gas-report` in `.github/workflows/test.yml` (`needs: test`) just runs the script.

Assessment — the idea is right, the implementation is weak:

- The script ALWAYS `exit 0`, even when gas increased. CI never fails and never comments on the PR;
  the diff is only visible in job logs. It is a log, not a gate.
- CI does not check that the committed `gas_report.json` is up to date (no `git diff --exit-code`),
  so the baseline drifts unless a contributor remembers to run the script locally.
- The generated file is not valid JSON (trailing comma before `}`), so no standard tool (`jq`) can read it;
  hence the hand-rolled bash parser.
- `sed -i ''` is macOS-only; on the Ubuntu runner that line errors (no `set -e`, so silently ignored).
- Sorted by gas descending, so any change reshuffles the file and produces noisy diffs. Sort by key instead.
- Measures the whole test body (array construction + asserts + the op). The floor for a trivial test is
  13,840 l2_gas (e.g. `kron_product_test_check_len`, `fast_nr_optimize_test_2`), so small ops are dominated
  by overhead. There are no dedicated benchmark tests, no input-size sweeps, and no A/B variants.
- Only `l2_gas`; steps / builtin counts (range_check vs bitwise) are not recorded, although they are
  what explains G1.

Still, the committed numbers are informative for us (from `alexandria/gas_report.json`):

| Test | l2_gas | Note |
|---|---|---|
| `linalg::dot_test::dot_product_test` (3 elems) | 27,940 | ~14k over the floor |
| `linalg::kron_test::kron_product_test` | 60,770 | nested loops + array appends |
| `linalg::norm_test::norm_test_1` (L2 of [3,4]) | 240,340 | generic `pow` + 10 fixed Newton iterations |
| `linalg::norm_test::norm_test_2` (L1) | 26,950 | same loop, no root |
| `math::fast_root_test::fast_sqrt_test_1` (sqrt(100), 10 iters) | 206,810 | loop-based sqrt is ~15x a dot product |
| `math::fast_root_test::fast_nr_optimize_test_1` (4th root, 30 iters) | 923,310 | |
| `math::math_test::pow_test` | 200,960 | recursive generic pow |
| `math::fast_power_test::fast_power_test` | 1,119,740 | u256 square-and-multiply loop |
| `math::trigonometry_test::sin_positive_test_1` | 20,930 | table lookup + mul/div, no loop |

This is direct evidence for G1: the loop/iteration-based routines (sqrt, pow, norm) cost 10-50x the
closed-form arithmetic ones (table-driven sin at ~7k over the floor).

### 1.5 CI workflows (`alexandria/.github/workflows/`)

- `test.yml` (on push + pull_request): `build` (`scarb build`) -> `test` (`snforge test`, snfoundry 0.56.0
  via `foundry-rs/setup-snfoundry@v3`) and `check-format` (`scarb fmt --check`) -> `gas-report`
  (`./scripts/generate_gas_report.sh`). Uses `software-mansion/setup-scarb@v1.3.2` (reads `.tool-versions`).
  Actions are tag-pinned, not SHA-pinned; no caching; no concurrency group.
- `macros.yml`: path-filtered job for the Rust proc-macro package (cargo build, `cargo fmt --check`,
  clippy with `-Dwarnings`, `scarb fmt --check` on `macros_tests`).
- `mdbook.yml`: manual (`workflow_dispatch`) docs deploy: `scripts/generate_doc.sh` runs
  `scarb doc --workspace --exclude ... --remote-base-url`, injects `docs/intro.md`, strips the `core`
  crate from `SUMMARY.md`, copies `docs/book.toml`, `mdbook build`, then deploys to GitHub Pages.
- `labels.yml`, `lock.yml`, `stale.yml`: repo hygiene.
- No release workflow. Publishing to scarbs.xyz is a manual local script,
  `scripts/update_registry.sh`, which loops `scarb publish --package <name>` over a hand-maintained,
  dependency-ordered list (data_structures, ascii, math, linalg, ...). No tag trigger, no changelog, no
  `--dry-run`/`scarb package` verification in CI.
- No `scarb lint` / cairo-lint anywhere (grep finds nothing), no coverage job despite the
  `[profile.coverage]` stanza.

### 1.6 Docs and contribution guidelines

- `README.md`: package index, per-package `scarb add alexandria_<pkg>@0.10.0`, build/test/format commands,
  "not audited" security notice, all-contributors table.
- `docs/CONTRIBUTING.md` is a half-filled template ("2. TODO" in the dev-setup section); only real rules:
  discuss first, branch `feat/<name>`, conventional commit message (`feat: add ...`).
- `.github/PULL_REQUEST_TEMPLATE.md`: PR type checklist, current/new behaviour, breaking-change flag.
  `CODEOWNERS`: single owner.
- No AGENTS.md / CLAUDE.md, no ADRs, no changelog.

### 1.7 Math-related packages: what is there, quality, reusability

`alexandria_linalg` (`packages/linalg/src/`, 90 lines total):

- `dot.cairo`: generic dot over two `Span<T>`, loop with `pop_front`, panics on length mismatch.
- `kron.cairo`: despite the name, requires equal lengths and returns the flattened outer product
  (`Result<Array<T>, KronError>`), nested `for` loops.
- `norm.cairo`: `norm<T, +Into<T, u128>, ...>(xs: Span<T>, ord: u128, iter: usize) -> u128`; branches on
  `ord == 0` inside the loop, uses generic recursive `pow`, then `fast_nr_optimize(norm, ord, iter)` with
  a caller-supplied fixed iteration count.
- Verdict: a toy. Unsigned-only (`u128` result), no fixed-point, no vector/matrix types, no cross product,
  no matrix multiply, no transforms. Everything is `Span`-based and therefore loop-based by construction.
  Not reusable for nalgebra.cairo. Useful as an anti-pattern: for a physics engine the hot types are
  fixed-dimension (`Vector2/3/4`, `Matrix2/3/4`, quaternions) and must be structs with fully unrolled
  field arithmetic — zero loops, zero array allocation, zero length checks.

`alexandria_math` (`packages/math/src/`, selected):

- `lib.cairo`: generic recursive `pow<T>` (O(log n), uses separate `%` and `/` on the same value — violates
  the DivRem rule), `BitShift`/`BitRotate` traits with per-type impls. `shl` is
  `(WideMul::wide_mul(x, pow(2, n)) & Bounded::<u8>::MAX.into()).try_into().unwrap()` — shift emulated by
  mul (good) but the power of two is computed with a recursive loop and the truncation with a bitwise AND
  (bad on both G1 axes). `rotate_left` is better: `DivRem::div_rem(word, 0x100)` then `quotient + remainder`
  (pure arithmetic).
- `opt_math.cairo` (881 lines): the "optimized" rewrite. Const lookup tables
  (`const SHIFT_TABLE128: [u128; 128]`), `shr128(a, b) = b / *SHIFT_TABLE128.span()[a.into()]`,
  `shl128 = overflowing_mul128(b, table[a])` via `wide_mul(...).low`, all `#[inline(always)]`.
  This is exactly G1 applied (loop -> table, shift -> mul/div). Remaining bitwise residue:
  `overflowing_mul64` uses `(res & 0xffffffffffffffff)` and `rotl128` combines with `|` where `+` on
  disjoint bits (or a single DivRem) would do.
- `const_pow.cairo`: `pow2(exponent: u32) -> u128`, `pow2_felt252`, `pow10`, `pow10_u256` via fixed-size
  const arrays indexed with `*hardcoded_results.span()[exponent]`. Reusable idea (fixed-point scale
  factors, shifts). Note starknet-agentic recommends a `match`-based table instead; benchmark both.
- `fast_power.cairo`: square-and-multiply in a `loop`, always promoted to `u256` (expensive), separate
  `%` and `/`. `fast_root.cairo`: Newton-Raphson `fast_nr_optimize(x, r, iter)` with a caller-chosen fixed
  iteration count (no convergence test, no initial-guess scaling: starts from `x / r`), rounding to nearest.
  `fast_sqrt` = 206,810 gas for sqrt(100). Do not reuse: corelib already exposes `core::num::traits::Sqrt`
  (hint-assisted `u128_sqrt`/`u256_sqrt` libfuncs, verification by arithmetic) which should be the baseline
  to benchmark against.
- `trigonometry.cairo`: `fast_sin/fast_cos/fast_tan` on `i64` degrees scaled by 1e8, using a 10-entry sin
  table (10 degree steps) + 10-entry cos table (1 degree steps) and the angle-addition formula with a
  linear term. Loop-free and cheap (~21k gas per test), but low precision, degree-based, decimal scale, and
  buggy on untested paths:
  - `fast_cos`: `let mut a = x + FAST_I90; if x < 0 { a = -x; }` drops the +90 degree offset for negative
    inputs, so `fast_cos(-x)` returns `sin(x)`. The test named `cos_negative_test_1` only checks a negative
    result, never a negative input.
  - `fast_tan`: sign is `sig_sin || sig_cos`, which is wrong in quadrant II (sin > 0, cos < 0 gives "positive").
  Lesson: table + arithmetic is the right shape; but tests must sweep all quadrants/signs, preferably
  with fuzzing against a reference.
- `wad_ray_math.cairo`: Aave-style 1e18/1e27 fixed point on `u256` (`(a * b + HALF_WAD) / WAD`). Unsigned,
  u256 (2 limbs, costly), decimal scale, overflow on `a * b` — unsuitable for physics.
- `i257.cairo`, `decimal.cairo`: sign-magnitude structs (`{ abs: u256, is_negative: bool }`,
  `{ int_part: u64, frac_part: u64, is_negative: bool }`) with operator trait impls
  (`Add`, `Sub`, `Mul`, `Div`, `Rem`, `*Assign`, `PartialOrd`, `Neg`, `Zero`, `Display`). Reusable as a
  checklist of which core traits a numeric type must implement; not reusable as representation
  (branchy sign handling, negative-zero hazard — there is a dedicated `i257_assert_no_negative_zero`).
  `decimal.cairo::from_parts` computes digit counts with a `while temp > 0` loop.
- `u512_arithmetics.cairo`, `mod_arithmetics.cairo`, `karatsuba.cairo`, hashes, ed25519: not relevant, except
  as examples of `#[inline(always)]` thin wrappers over corelib (`u512_safe_div_rem_by_u256`, `u256_inv_mod`).

`alexandria_numeric`: `cumsum`, `cumprod`, `diff`, `trapezoidal_rule`, `interpolate` (linear scan in a `loop`,
with a separate `interpolate_fast` using binary search from `alexandria_searching`), `integers.cairo`.
Generic over `T` with long `+Trait<T>` bound lists; all Span/loop based. Nothing to reuse; the
"slow vs fast variant + separate test file per variant" pairing (`interpolate_test.cairo` /
`interpolate_fast_test.cairo`) is a small precedent for A/B benchmarking.

`alexandria_macros` (Rust proc-macros): derive `Add/Sub/Mul/Div`, `AddAssign/...`, `Zero` for structs whose
fields support the op, plus a compile-time `pow!` (deprecated in favour of a declarative
`pub macro pow_inline` in `math/src/pow_macro.cairo`, which requires
`experimental-features = ["user_defined_inline_macros"]`). Component-wise derive is attractive for
`Vector3 { x, y, z }`, but a proc-macro dependency forces every consumer to have a Rust toolchain or
prebuilt plugin; hand-write the impls (they are 5 lines each and need `#[inline]` control anyway).

Overall quality: breadth over depth, community-grade, explicitly unaudited, thin tests, no fuzzing, at
least two latent sign bugs in trig. Reuse ideas (workspace wiring, gas report concept, const tables,
impl-as-strategy), not code. Do not take an `alexandria_*` runtime dependency.

How Alexandria scores against the user guidance:

- G1: mixed. `opt_math`, `const_pow`, `trigonometry`, `BitRotate` support it (tables + mul/div instead of
  loops/shifts). `lib.cairo::BitShift`, `fast_root`, `fast_power`, `norm`, `decimal::from_parts` contradict it
  (loops where tables or closed forms exist; `&` masks where DivRem/try_into would do).
- G2: partially. Gas is recorded for every test automatically, but there are no benchmark-purpose tests,
  no A/B comparison, and no enforcement.

---

## 2. starknet-agentic

### 2.1 Where the Cairo guidance lives

- `starknet-agentic/skills/cairo-optimization/` — `SKILL.md`, `references/legacy-full.md` (12 rules + BoundedInt
  deep dive, imported from feltroidprime/cairo-skills), `references/anti-pattern-pairs.md`,
  `references/profiling.md`, `workflows/default.md`, `scripts/profile.py`, `scripts/bounded_int_calc.py`.
- `skills/cairo-testing/` — `SKILL.md`, `references/legacy-full.md` (snforge reference), `workflows/default.md`.
- `skills/cairo-contract-authoring/` — `SKILL.md`, `references/language.md` (language primer),
  `references/legacy-full.md`, `references/anti-pattern-pairs.md` (security pairs).
- `skills/cairo-auditor/` — 170+ attack vectors (`references/attack-vectors/attack-vectors-{1..4}.md`),
  `references/vulnerability-db/*.md` cards, semgrep rules, release-gate checklist.
- `evals/contracts/{secure,insecure}_math_patterns/src/lib.cairo` — paired fixtures that make each
  optimization rule deterministically checkable.
- Agent conventions: `AGENTS.md` (canonical), `CLAUDE.md` (context adapter), `CONTRIBUTING.md`,
  `VERSIONING.md`, `.github/workflows/ci.yml`.

### 2.2 Optimization rules (from `skills/cairo-optimization/references/legacy-full.md`)

| # | Instead of | Use | Stated reason |
|---|---|---|---|
| 1 | `x / m` and `x % m` | `DivRem::div_rem(x, m)` | one operation yields both; separate ops double the cost |
| 2 | `while i < n` | `while i != n` (exact-trip loops only) | equality is cheaper than comparison (range check) |
| 3 | `2_u32.pow(k)` | `match`-based lookup table | `pow()` is a loop |
| 4 | `*data.at(i)` in an index loop | `pop_front` / `for` / `multi_pop_front::<N>()` | bounds check per access |
| 5 | `.len()` in the loop condition | cache `let n = data.len();` | recomputed each iteration |
| 6 | manual copy loop | `span.slice(start, len)` | pointer arithmetic, no copies |
| 7 | `index & 1`, `index / 2` | `DivRem::div_rem(index, 2)` | "Bitwise AND is more expensive than div_rem in Cairo" |
| 8 | `u256` when range < 2^128 | smallest integer type that fits | fewer limbs, type encodes the constraint |
| 9 | one storage slot per field | `StorePacking` (pack with `a + b * POW_2_128`, unpack via u256 split) | n/a for a pure lib |
| 10 | bitwise ops / raw u128 math for limb split/assembly | `bounded_int::{div_rem, mul, add}` | bounds tracked at compile time, no overflow checks; measured 28,340 -> 13,840 gas for 4 x u32 -> u128 |
| 11 | `poseidon_hash_span([x, y])` | `hades_permutation(x, y, 2)` | n/a for us |
| 12 | bulk `try_into`/`downcast` felt252 -> BoundedInt | `u128s_from_felt252` + `upcast` | 2 vs 4 steps per conversion |

BoundedInt specifics (same file):

- "The #1 optimization pitfall" is converting between `u16/u32/u64` and `BoundedInt` at function
  boundaries: `downcast` costs a range check that can exceed the savings. Use BoundedInt types as function
  inputs AND outputs; downcast only at system entry points (deserialization).
- `upcast` is free only when the target range is a superset; otherwise it is a `downcast`.
- Negative dividends: `bounded_int_div_rem` does not accept negative lower bounds; add
  `SHIFT = ceil(|min| / modulus) * modulus` first.
- Bounds are hard-capped at 2^128; compute bounds with `scripts/bounded_int_calc.py`, never by hand
  (sub bounds are `[a_lo - b_hi, a_hi - b_lo]`).
- Measured libfunc costs: `u128s_from_felt252` 2 steps, `downcast` 4 steps, `bounded_int_div_rem` 7 steps.
- `try_into().unwrap()` in unrolled/generated code causes O(N^2) Sierra bloat (each panic path drops all
  live variables). Highly relevant to fully unrolled matrix code: avoid panicking conversions inside
  unrolled bodies; convert once at the boundary.
- Imports come from the `corelib_imports` package (`corelib_imports = "0.1.2"`), since bounded_int is
  not part of the public prelude.

Process rules (`skills/cairo-optimization/SKILL.md`, `workflows/default.md`):

- Optimize only after correctness: tests must pass before AND after each optimization.
- Profile first (`scripts/profile.py profile --mode snforge --package <p> --test <t> --metric steps|rc|sierra-gas|l2-gas`),
  which wraps `snforge test --save-trace-data` -> `cairo-profiler` -> `pprof`; rank hotspots from the text
  summary. Never reuse a stale trace.
- One optimization class per commit; re-profile after each; revert if no measurable gain
  ("Reject changes that reduce readability without measurable gains").
- Record before/after step deltas (absolute and %) in the PR description.
- "Rationalizations to Reject": optimizing before tests exist; skipping re-profiling because "obviously
  better"; avoiding BoundedInt as "too complex"; hand-computed bounds.
- `l2-gas` profiling requires `[cairo] enable-gas = true` and code running inside a deployed contract via a
  dispatcher; use `steps` for pure-library hot paths.
- Code quality: `scarb fmt` before every commit; pin Scarb + snforge in `.tool-versions`; keep toolchain
  updated ("Newer Scarb/Foundry versions include gas optimizations").

What it does NOT cover: `#[inline(always)]` policy, const vs function, felt252-vs-integer arithmetic
trade-offs beyond rules 8/10/12, fixed-point math, signed numbers. These must come from our own benchmarks.

### 2.3 Testing rules (`skills/cairo-testing/SKILL.md`, `references/legacy-full.md`)

- Structure: `#[cfg(test)] mod tests { ... }` for unit tests in the same file; `tests/` directory for
  integration tests; shared `helpers` module (`tests/helpers.cairo`); split `test_unit` / `test_integration` /
  `test_fuzz` files.
- Naming: `test_<function>_<scenario>`.
- Plan before writing: list functions with positive/negative paths, edge cases (zero, max, duplicates),
  and fuzz targets ("identify functions with numeric inputs that should get `#[fuzzer]` tests").
- Fuzz: `#[fuzzer(runs: 256, seed: 12345)]` with a fixed seed; constrain inputs via bounded types /
  custom `Fuzzable` impl (`generate_arg(lo, hi)`), not by silently skipping invalid inputs.
- Failure tests use `#[should_panic(expected: '...')]` with the exact message.
- Regression mode: every bug becomes a failing-before / passing-after test pair.
- Rationalizations to reject: "happy-path only", "fuzz tests are overkill".
- Gas: only `snforge test --detailed-resources > gas-report.txt` and "diff manually" — there is no gas
  regression mechanism here; Alexandria's script is the more advanced of the two.
- In the repo's own contracts, tests live in `contracts/*/tests/` with a `tests/lib.cairo` and separate
  `*_fuzz.cairo` files (`contracts/erc8004-cairo/tests/test_reputation_registry_fuzz.cairo`,
  `#[fuzzer(runs: 64)]` on `i128` inputs).

### 2.4 Security patterns relevant to a math library

From `skills/cairo-auditor/references/attack-vectors/attack-vectors-3.md` and `vulnerability-db/`:

- #43 Rounding bias: define and document rounding direction; test invariants.
- #44 / #145 felt/int boundary misuse: never do accounting-grade math in `felt252` without range
  assertions (wrap-around modulo P is silent).
- #45 Underflow guarded only by assumptions: explicit precondition or typed bounds.
- #53 Division before multiplication: multiply first, divide last, with overflow-safe widening.
- #54 Scale mismatch (WAD vs RAY): one scale per type; conversions explicit.
- #55 Signed/unsigned cast hazards; `UNSAFE-TYPE-CONVERSION`: `try_into().expect()` on a runtime path is a
  DoS vector — pre-validate or prove bounds.
- #108 Multiply-chain overflow before the safe divide: widen (`WideMul`) before dividing.
- `PRECISION-LOSS`: low scale factors accumulate drift; long-horizon regression tests required.
- `UNBOUNDED-LOOP`: iteration bound tied to caller/state size.
- CONTRIBUTING "Security Guardrails": never ship stubbed success behind a TODO — panic explicitly.

For a deterministic physics engine these translate into: documented rounding mode per op, overflow
behaviour as part of the API contract (panic vs saturate vs wrap), and fuzz/property tests for them.

### 2.5 Repo layout, CI, agent conventions

- Instruction hierarchy: `AGENTS.md` is "the single canonical instruction file"; `CLAUDE.md` must be "a
  thin adapter or reference layer, not a conflicting instruction source". AGENTS.md holds mission, scope,
  operating principles (single source of truth; small testable changes; reproducibility; explicit
  handoffs "authoring -> testing -> optimization -> auditor"), roles, task lifecycle, parallelization
  rules (serialize shared interface / `scripts/**` / CI changes), "Required Validation by Change Type"
  (contracts: `scarb build` and `snforge test` in impacted packages), and an escalation template.
- `CLAUDE.md` is structured in XML-like tagged sections: `<identity>`, `<stack>` (pinned versions table),
  `<structure>` (annotated tree), `<commands>` (task / command / working dir table), `<conventions>`,
  `<workflows>` ("Adding a new Cairo contract" as numbered steps), `<boundaries>` (DO NOT modify /
  Require human review / Safe for agents), `<references>`, `<implementation_status>`, `<troubleshooting>`.
  Notable boundaries: lockfiles are do-not-touch; dependency version bumps require human review.
- Skill anatomy: `SKILL.md` with YAML frontmatter (`name`, `description` with trigger words, `allowed-tools`),
  "When to Use / When NOT to Use", "Rationalizations to Reject", mode selection, a fixed 4-turn
  orchestration (understand/baseline -> plan, wait for confirmation -> implement -> verify), numbered
  "Security-Critical Rules", error-code table with recovery, `references/` loaded on demand,
  `workflows/default.md` checklist, deterministic helper `scripts/`. Anti-pattern/"good pattern" pairs are the
  preferred teaching format and are backed by eval fixtures.
- CI (`.github/workflows/ci.yml`): `dorny/paths-filter` change detection gates per-area jobs; all actions
  pinned by commit SHA with a version comment; top-level `permissions: contents: read`; separate
  `cairo-check` (`scarb build`) and per-package `snforge test` jobs with explicit tool versions; gitleaks
  secret scan, CodeQL, dependency review, OpenSSF scorecard. No `scarb fmt --check`, no `scarb lint`, no gas job
  (Alexandria is ahead on fmt + gas; agentic is ahead on supply-chain hygiene).
- Release: `VERSIONING.md` (SemVer, pre-1.0: PATCH = fixes/refactors, MINOR = any externally visible
  change), `CHANGELOG.md` with `Unreleased`, annotated tags `v0.y.z`, changesets for npm packages.
  Conventional commits (`feat|fix|docs|chore|test|refactor|ci`). PR checklist demands an acceptance test
  and "no unrelated refactors".

### 2.6 Extracted checklist (paraphrased rules)

Arithmetic and types
- [ ] Need quotient and remainder of the same value -> one `DivRem::div_rem`.
- [ ] Parity / halving / bit extraction -> `div_rem(x, 2)`, not `& 1` / shifts.
- [ ] Limb split/assembly, byte cutting -> mul/add/div_rem (ideally BoundedInt), never `&`, `|`, shifts.
- [ ] Known power of two/ten -> const or `match` lookup table, never `pow()` at runtime.
- [ ] Smallest integer type that fits; avoid `u256` unless the range needs it.
- [ ] BoundedInt in and out of hot functions; downcast only at the boundary; bounds via the calculator.
- [ ] No `try_into().unwrap()` inside unrolled hot code (Sierra bloat); convert at the edge.
- [ ] Multiply before divide; widen before multiply chains; state the rounding direction.
- [ ] No unchecked felt252 arithmetic for values with integer semantics.

Loops and collections
- [ ] Exact-trip loops use `!=`; keep `<` when overshoot is possible.
- [ ] Iterate with `for` / `pop_front` / `multi_pop_front`, not `at(i)`.
- [ ] Cache `.len()`; use `span.slice()` instead of copy loops.
- [ ] Loop bounds must not be attacker/state-sized without a cap.

Process
- [ ] Tests green before and after; profile before optimizing; one optimization class per commit.
- [ ] Re-measure after every change; record deltas in the PR; revert non-gains.
- [ ] Fixed-seed fuzz tests for every numeric function; regression test for every bug.
- [ ] `test_<function>_<scenario>` names; `#[should_panic(expected: ...)]` with exact messages.
- [ ] `scarb fmt` before commit; `.tool-versions` pins scarb + starknet-foundry; conventional commits.
- [ ] AGENTS.md canonical, CLAUDE.md thin; explicit do-not-touch / human-review / agent-safe boundaries.

Against the user guidance:

- G1 is explicitly supported: rules 7 and 10 say arithmetic beats bitwise (bitwise goes through the
  bitwise builtin; div_rem through range checks), rule 3 says tables beat loops, and rules 2/4/5/6 are
  about making unavoidable loops cheaper. One nuance: `anti-pattern-pairs.md` justifies DivRem-over-`&`
  partly on reviewability rather than cost, and no numbers are given for rule 7 — we should measure it
  ourselves rather than take it on faith. Nothing in the repo contradicts G1.
- G2 is only weakly supported: the skill insists on measured before/after deltas, but measurement is an
  ad hoc profiling session, not a committed per-test gas baseline. Alexandria's `gas_report.json` is the
  closer match; neither repo has A/B implementation comparison built in.

---

## 3. Best practices to adopt in nalgebra.cairo

### 3.1 Workspace structure

Scarb virtual workspace, single lockstep version, pure Cairo (no `starknet` dependency in library crates).

```
nalgebra.cairo/
  Scarb.toml                # [workspace], [workspace.package], [workspace.dependencies], [workspace.tool.fmt]
  Scarb.lock                # committed
  .tool-versions            # scarb X, starknet-foundry Y (single source of truth, read by CI)
  AGENTS.md  CLAUDE.md  CONTRIBUTING.md  CHANGELOG.md  VERSIONING.md  README.md
  gas/                      # committed baselines, one JSON per package (valid JSON, sorted by key)
  scripts/gas_report.py     # generate + compare + gate (see 3.3)
  scripts/publish.sh        # dependency-ordered scarb publish
  packages/
    core/        -> nalgebra_core      # scalar layer: fixed-point type(s), signed handling, sqrt, trig, consts/tables
    vector/      -> nalgebra_vector    # Vector2/3/4 structs, fully unrolled ops
    matrix/      -> nalgebra_matrix    # Matrix2/3/4, unrolled mul/transpose/det/inverse
    geometry/    -> nalgebra_geometry  # Rotation2/3, UnitQuaternion, Isometry, Transform
    bench/       -> nalgebra_bench     # NOT published: A/B candidate implementations + benchmark tests
  docs/                     # mdBook intro + scarb doc output (generated)
  .github/workflows/        # ci.yml, docs.yml, release.yml
```

Rules:

- `edition = "2024_07"` everywhere; `version.workspace = true`; `[tool] fmt.workspace = true`;
  internal deps as `{ path = "../core", version = "<workspace version>" }` (both keys, for publishing).
- Package names prefixed `nalgebra_` (flat registry namespace). Optionally an umbrella `nalgebra` package that
  only re-exports, mirroring Rust `nalgebra`'s single-crate UX.
- Start with few packages (core + vector + matrix can even be one crate initially); split only when a
  consumer benefits. Alexandria's 17-package sprawl costs a manual ordered publish list.
- No proc-macro dependency (keeps consumers free of a Rust toolchain). No `alexandria_*` runtime dependency.
- Fixed-dimension types are `#[derive(Copy, Drop, Serde, PartialEq, Debug)]` structs with named fields;
  dynamic `Span`-based `DVector/DMatrix` is a later, separate, explicitly loop-based module.

### 3.2 Naming and module conventions

- One concept per file; `lib.cairo` = `//!` crate doc + `pub mod` list + curated `pub use` re-exports so users
  write `use nalgebra_vector::{Vector3, Vector3Trait}`.
- Types `PascalCase` matching Rust nalgebra names (`Vector3`, `Matrix3`, `UnitQuaternion`, `Isometry3`).
  Traits `XxxTrait`; inherent-style API via `#[generate_trait] pub impl Vector3Impl of Vector3Trait`.
  Operator/core trait impls named `<Type><Trait>`: `Vector3Add`, `Vector3Mul`, `Vector3Zero`, `Vector3Neg`.
  Cross-type ops spell both: `Matrix3MulVector3`.
- Required trait surface per numeric type (checklist taken from `i257.cairo`): `Add Sub Mul Div Neg`,
  `AddAssign ...`, `PartialEq`, `PartialOrd` (scalars), `Zero`, `One`, `Default`, `Into/TryInto` conversions,
  `Debug`/`Display` behind test-only usage.
- Generic surface: define a small scalar trait bundle in `core` and make vector/matrix code generic over it only if
  benchmarks show monomorphisation is free; otherwise specialise on the chosen fixed-point type. Decide with
  data from `bench`.
- Visibility: `pub` only for API; helpers private or `pub(crate)`; consts `SCREAMING_SNAKE_CASE`, `pub(crate)`
  unless part of the API.
- Errors: one policy — hot-path ops panic with short-string felt messages prefixed by module
  (`'vec3: div by zero'`); fallible constructors/inversions return `Option<T>` (`try_inverse`, `try_normalize`),
  mirroring Rust nalgebra. Document overflow behaviour and rounding mode in every `///` doc
  (`#### Arguments / #### Returns / #### Panics` format, which `scarb doc` renders well).
- `#[inline(always)]` only on small leaf functions and only when the benchmark shows a gain; note the
  code-size trade-off as Alexandria does.
- Implementation preference order (G1), to be stated in AGENTS.md: closed-form arithmetic
  (add/mul/`DivRem`/`WideMul`) > const or `match` lookup tables > bitwise builtin > loops. Fixed-size ops are
  always unrolled. Any loop in a hot path needs a benchmark-backed justification comment.
- Specific starting points to benchmark: corelib `Sqrt` vs Newton with table-seeded guess; power-of-two
  (binary) fixed-point scale so rescaling is a `div_rem` by a const `NonZero` vs decimal scale; sign-magnitude
  struct vs offset/two's-complement-in-felt vs native `i64/i128`; BoundedInt end-to-end vs plain integers;
  `match` table vs const-array-span table.

### 3.3 Tests and gas tracking (G2)

Test layout

- Unit tests inline (`#[cfg(test)] mod tests`) only for private helpers; everything else in
  `packages/<pkg>/tests/<module>_test.cairo` without `tests/lib.cairo` (single `_integrationtest` target,
  black-box, same as Alexandria). No stale `src/tests.cairo` files.
- snforge for everything (it reports gas); `snfoundry.toml` per package.
- Names: `test_<fn>_<scenario>`; for every public function at least: nominal, zero/identity, extreme
  magnitude (overflow policy), sign combinations/all quadrants, exact expected panics.
- Fuzz every numeric function with `#[fuzzer(runs: 256, seed: <fixed>)]` on algebraic properties
  (commutativity, `a + (-a) == 0`, `|normalize(v)| ~ 1` within epsilon, `M * M^-1 ~ I`, `sin^2 + cos^2 ~ 1`),
  with bounded `Fuzzable` inputs. Alexandria's trig bugs are exactly what this would have caught.
- Golden vectors generated from Rust `nalgebra` (script in `scripts/`), so parity with the reference
  crate is tested, including rounding.

Gas tracking — keep Alexandria's concept, fix its flaws

- Dedicated benchmark tests, separate from correctness tests: `bench_<op>__<variant>` (double underscore
  separates the variant), e.g. `bench_vec3_dot__unrolled`, `bench_sqrt__corelib`, `bench_sqrt__newton_table`.
  Each does the minimum: fixed inputs, one call (or a fixed N calls to amortise the ~13.8k l2_gas test floor),
  one assert. Include a `bench_baseline__empty` test to subtract the floor.
- Candidate implementations live in `packages/bench` (unpublished) behind a common trait with one impl per
  variant (Alexandria's `Sortable`/`MergeSort`/`QuickSort` pattern). The winner is promoted into the
  library; losers stay in `bench` so the comparison remains reproducible when the compiler changes.
- `scripts/gas_report.py` (Python, not bash): runs `snforge test` per package, parses
  `l2_gas` (and, with `--detailed-resources`, steps + range_check + bitwise builtin counts, which directly
  evidences G1), writes `gas/<package>.json` as valid JSON sorted by test name, and prints a Markdown delta
  table grouped by `<op>` with variants side by side.
- CI gate: (a) regenerate and `git diff --exit-code gas/` so the committed baseline can never be stale;
  (b) fail on any increase above a threshold (say 1%) unless the PR carries a `gas-regression-ok` label;
  (c) post the delta table as a PR comment / job summary. Every new public function must add at least one
  `bench_` entry (checked by the script: public fn list vs bench names, or simply by review checklist).
- Toolchain bumps are their own PR that only updates `.tool-versions` + `gas/*.json`, so compiler-induced
  deltas are never mixed with code changes.
- For deep dives use the agentic profiling pipeline (`snforge test --save-trace-data` -> `cairo-profiler` ->
  `pprof`, metric `steps`); commit nothing from it, paste the summary in the PR.

### 3.4 CI pipeline

`ci.yml` on push to main + pull_request, `permissions: contents: read`, concurrency-cancel per ref,
actions pinned by SHA (agentic style), tool versions read from `.tool-versions`:

1. `fmt`: `scarb fmt --check` (workspace; `sort-module-level-items = true`).
2. `lint`: `scarb lint` with warnings denied (neither reference repo does this; we should).
3. `build`: `scarb build` (workspace).
4. `test`: `snforge test --workspace` (fuzz seeds fixed, so deterministic).
5. `gas`: `scripts/gas_report.py --check` (needs `test`), uploads the table to the job summary / PR comment.
6. `package`: `scarb package` for each publishable member (catches manifest/publish breakage early).
7. `docs` (main only or manual): `scarb doc --workspace` + mdBook -> GitHub Pages (reuse the approach in
   Alexandria's `scripts/generate_doc.sh`, including stripping the `core` crate from `SUMMARY.md`).

Path filters are unnecessary while the repo is Cairo-only; add `dorny/paths-filter` if Rust/TS tooling appears.

### 3.5 Release process

- SemVer with the agentic pre-1.0 policy: PATCH = fixes/internal; MINOR = any API or numerically observable
  behaviour change (results/rounding are part of the contract for a deterministic engine). Additionally note
  gas-significant changes in the changelog.
- `CHANGELOG.md` (Keep a Changelog, `Unreleased` section), conventional commits, PR template with type +
  breaking-change + gas-delta sections.
- `release.yml` triggered by annotated tag `v*`: verify tag == `[workspace.package] version`, run the full CI,
  `scarb publish` each package in dependency order (`scripts/publish.sh`, using `SCARB_REGISTRY_AUTH_TOKEN`
  secret), create the GitHub Release from the changelog section, deploy docs. This replaces Alexandria's
  manual local `update_registry.sh`.
- README install lines and version strings are generated/checked from `Scarb.toml`, never hand-edited.

### 3.6 Agent-development conventions (AGENTS.md / CLAUDE.md)

Follow the agentic split: AGENTS.md is normative, CLAUDE.md is a thin pointer + context.

AGENTS.md should contain:

- Mission and scope (provable linear algebra for a deterministic physics engine; parity target = Rust nalgebra API).
- Operating principles: correctness first, then measured optimization; small single-purpose diffs; one
  optimization class per commit; no unrelated refactors.
- The implementation preference order (G1) verbatim, plus the extracted checklist of section 2.6.
- Definition of done for any feature: public API documented (`Arguments/Returns/Panics`, rounding, overflow);
  unit tests (nominal, edge, panic); fuzz property test with fixed seed; `bench_` test(s); `gas/*.json`
  regenerated; if alternatives were considered, each exists as a `bench` variant and the PR body shows the
  comparison table; `scarb fmt`, `scarb lint`, `snforge test` all green.
- Workflow per task (4 turns, as in the skills): understand + baseline -> short plan -> implement -> verify
  and report measured deltas. "Rationalizations to reject": optimizing without tests; "obviously faster"
  without numbers; adding a loop where a closed form or table exists; hand-waving bounds/overflow.
- Required validation by change type (commands to run), and an escalation template for API/representation
  decisions (scalar representation, scale factor, rounding mode, panic vs Option).
- Boundaries: do not edit `Scarb.lock`, `.tool-versions`, `gas/*.json` by hand (only via the script);
  human review required for dependency/toolchain bumps, public API changes, numeric-behaviour changes,
  CI/release workflow edits; safe: source, tests, benches, docs.

CLAUDE.md should contain (tagged sections like agentic's): `<identity>`, `<stack>` with pinned versions,
`<structure>` annotated tree, `<commands>` table (build, test one package, run one test, fmt, lint, gas report,
profile), `<conventions>` summary pointing to AGENTS.md, `<workflows>` ("adding a new operation", "adding an
alternative implementation and benchmarking it", "bumping the toolchain"), `<troubleshooting>`
(snforge/scarb version mismatch, stale traces, BoundedInt bound errors).

Optionally vendor or reference the three Cairo skills (`cairo-optimization`, `cairo-testing`,
`cairo-contract-authoring/references/language.md`) under `.claude/skills/` and add one project skill,
`cairo-bench`, that encodes the A/B benchmark + gas-report workflow of section 3.3.
