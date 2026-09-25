# Benchmark synthesis

What we learned from studying nalgebra, the Cairo ecosystem, and ~2,100 gas micro-benchmarks.
Everything here is backed by a detailed report or a reproducible measurement:

| Source | Where |
|---|---|
| nalgebra / rapier / parry analysis | [research/01-nalgebra-analysis.md](research/01-nalgebra-analysis.md) |
| alexandria + starknet-agentic (architecture, best practices) | [research/02-alexandria-starknet-agentic.md](research/02-alexandria-starknet-agentic.md) |
| origami (architecture, code efficiency) | [research/03-origami.md](research/03-origami.md) |
| cubit + orion (fixed point, linear algebra) | [research/04-cubit-orion-fixed-point.md](research/04-cubit-orion-fixed-point.md) |
| Cairo primitives (879 tests) | [../benchmarks/primitives](../benchmarks/primitives/README.md) |
| Fixed-point scalar candidates (407 tests) | [../benchmarks/scalar](../benchmarks/scalar/README.md) |
| Vector / matrix layout and abstractions (648 tests) | [../benchmarks/layout](../benchmarks/layout/README.md) |
| Existing libraries, measured (186 tests) | [../benchmarks/libs](../benchmarks/libs/README.md) |

Toolchain for every number: scarb / Cairo 2.19.4, snforge 0.61.0, metric = Sierra gas (`l2_gas`),
net of a per-group baseline. Observed weights: 1 step = 100, 1 range check = 70, 1 bitwise
builtin = 583. **The ranking of techniques depends on this cost table** — re-run the benchmarks
after every toolchain bump.

## 1. nalgebra and the physics stack

- nalgebra is ~49k lines: `base` (generic `Matrix<T, R, C, S>`), `geometry` (rotations,
  isometries, transforms), `linalg` (decompositions), plus satellites (glm, sparse, lapack).
- The generic dimension machinery (`Dim`/`Const`/`Dyn`, storage, allocators, views, SIMD,
  `ComplexField`, ref/val operator permutations) has no value in Cairo. What is worth keeping:
  concrete unrolled types, a thin scalar trait stack, `Unit<T>`, approximate comparisons.
- **The Rust physics stack has largely moved to glam**: parry is nalgebra-free since 0.26, and
  rapier ≥ 0.32 uses glam for every scalar path. nalgebra remains in rapier for the SIMD solver
  (irrelevant in Cairo), multibody joints / IK (`DMatrix`, `DVector`, LU, 6xN Jacobians, 6x6
  blocks), and soft bodies (small Cholesky, symmetric eigen).
- Consequence: glam-cairo serves the rapier-cairo hot path; **nalgebra-cairo's unique value is
  dimensions above 4 (`Vector6`, `Matrix6`), small decompositions, solvers, and dynamic
  algebra**, while still offering the full static 2/3/4 + geometry surface for users who want
  the nalgebra API. Both must share one scalar representation.

## 2. Existing Cairo libraries

| Library | Status on Cairo 2.19.4 | Verdict |
|---|---|---|
| alexandria | builds (registry 0.10.0) | Good workspace/packaging model. `linalg` is 90 lines of `Span` loops; gas tracking exists but never fails CI. No fixed-size linear algebra. |
| origami | `algebra` exports nothing (private modules), `Span`-backed, Laplace `det`/`inv` | `map` crate is a catalogue of loop-free idioms worth copying; `algebra` is to be avoided entirely. No gas tracking. |
| cubit | dormant (Cairo 2.7); tests no longer parse | Sign-magnitude: branchy add (4,050 vs 740), negative zero bug, Taylor `sin` 130k gas. Keep: `u128_sqrt` trick, polynomial coefficients, test vectors. |
| orion | 125 compile errors | Fixed point copied from cubit; dynamic tensors with per-element index arithmetic; no det/inverse/solve. Keep: the idea of generating test vectors from floats. |
| starknet-agentic | n/a (guides) | Optimization rules match ours (`DivRem` over `&`, BoundedInt limbs, `match` tables, no `pow()`), `AGENTS.md` conventions, SHA-pinned CI. |

Measured against a hand-written reference (net gas):

| op | library | reference |
|---|---:|---:|
| Q32.32 add | cubit 4,050 | 740 |
| dot3 | cubit 15,450 · origami 28,480 · orion 37,820 | **2,350** (fused BoundedInt) |
| 3x3 matmul | origami 359,330 · orion 349,360 | 60,840 (i128 path; ~26k with the final kernel) |
| 3x3 inverse | origami 1,076,420 | 98,490 |
| 3x3 determinant | origami 279,510 | 19,570 |
| integer sqrt u128 | alexandria `fast_sqrt` 740,770 | corelib `Sqrt` 1,180 |
| u64 `>> 16` | alexandria `BitShift` 18,510 | constant division 1,010 |

None of them can be a dependency of nalgebra-cairo.

## 3. Math vs bitwise vs loops

The working hypothesis — *simple math < bitwise < loops* — **mostly holds**:

| Claim | Verdict | Evidence |
|---|---|---|
| Loops are the most expensive | **Always true** | ~1,100 gas per iteration + ~1,200 fixed; lost by x8.5 (popcount) to x125 (sqrt) against the best loop-free variant |
| add/sub/mul beat bitwise | True up to 64 bits | u64 `+`/`*` 370 vs `&` 1,583 (isolated), 745 vs 895 (marginal) |
| divmod beats bitwise | **Tie** | isolated `% 2^k` 910 < `&` 1,683; marginal `&` 895 < `/` 1,152; one `DivRem` (910) replaces `/` + `%` (1,920) |
| math beats bitwise on wide types | **False** | u128 `*` 3,030; u256 `%` 5,850 vs `&` 2,566 |
| data-parallel bit tricks | bitwise wins | SWAR popcount 14,612 vs best math 43,240 |

Above all three: **corelib libfuncs and tables**. `Sqrt` 1,080 (x22 cheaper than the best
hand-written Newton), const-array lookup ~700, `match` table ~2,000, corelib `pow` ~14,800.
And below everything: **felt252 ops (100) and `BoundedInt` arithmetic** (no range checks until
the final downcast).

Other rules of thumb: a non-inlined call costs ~2,100-2,400 (≈ 300 + 100/felt); branches are
cheap (300-400) and "branchless" felt tricks are ~2x worse; widening conversions are free,
narrowing costs 270-540; signed `/` by a variable costs 4,220 — store reciprocals.

## 4. Scalar representation

| op (net gas) | sign-mag u64 (cubit) | native i64 | **i64 + BoundedInt** | **fused** | i128 Q64.64 | felt252 lazy | i32 Q16.16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| add | 1,620-2,840 | 640 | 640 | | 370 | 100 | 640 |
| mul | 2,250-2,350 | 2,910-2,990 | **1,850** | | 8,430-8,970 | 2,120 | 2,440-2,520 |
| div | 2,250 | 5,330 | 3,570 | | 13,500 | 4,850 | 4,860 |
| lt | 1,080-2,180 | 870 | 870 | | 870 | 1,790 | 870 |
| sqrt | 1,180 | 1,820 | 1,920 | | 7,360 | 1,720 | 1,820 |
| dot3 | 10,310 | 10,570 | 7,230 | **2,250** | 26,950 | 2,520 | 9,160 |
| cross3 | 19,960 | 20,640 | 14,120 | **6,650** | 53,850 | 7,160 | 17,820 |
| length3 | 10,830 | 12,330 | 9,250 | **2,220** | 34,430 | 8,580 | 10,920 |

- Winner: **Q32.32 in a native `i64`, multiplicative kernels on `core::internal::bounded_int`**
  (one felt mul, biased branch-free floor `div_rem` by 2^32, a single `downcast` as overflow check).
- **Fusing matters more than the representation**: accumulate unscaled products, rescale once per
  output scalar (dot3 x3.2, length3 x4.2, mat*mat 70k → 26k). Fused forms also tolerate
  intermediate overflow and round once (< 1 ulp).
- Q16.16 saves nothing (width is free up to 128 bits); Q64.64 costs 2.4-5.8x; felt252 is
  rejected (silent wrap-around is unacceptable in a proof, ordering costs a range check).
- Transcendentals, loop-free, typed Horner polynomials generated with interval bounds:
  sin/cos 11-12.5k (6.6e-9 / 4.7e-10), atan2 16-19k (6.3e-8 / 1.9e-9), acos 16.5k (1.2e-9),
  inv_sqrt 2,930, normalize3 9,750. LUT, CORDIC, Newton and Taylor all lose (12k-204k).

## 5. Layout and abstraction

- struct = tuple = `[T; N]` (identical Sierra); `Span` layouts cost +4k to +10k per op. Named-field
  structs win on ergonomics. Transpose, swizzles, construction are free.
- **Generic `Vec3<T>` with trait bounds is exactly zero-cost** (12 configurations), operators =
  methods = free functions, `@T` = `T` for `Copy` structs.
- The default inliner leaves small vector ops out of line: `#[inline(always)]` halves `add`
  (4,550 → 2,220). It is refused on functions with their own generics, accepted on methods of
  generic impls.
- Dynamic matrices: x3-x8 slower than static with a cheap scalar; affordable only as a secondary
  type (row-major `Span`, sequential traversal, felt252 lazy accumulation, dispatch to static
  kernels for small shapes: +2 %).
- Structured kernels pay off on the rigid-body path: `Sym3` inverse 241k vs 344k, `R diag(d) Rᵀ`
  291k vs 593k, cross 66k vs skew-matrix 101k (figures with the slow i128 scalar; ratios hold).

## 6. Shipped library figures (net gas, `fixed::Fixed` 0.3.0, since WP 7.1)

Measured on the merged code (snapshots in `gas/`), for the operations a physics engine calls most:

| operation | gas | note |
|---|---:|---|
| `Fixed` add / mul / div / sqrt | 640 / 1,580 / 3,300 / 1,820 | fixed-cairo's `fixed` 0.3.0 (div rounds to nearest) |
| `sin_cos` / `atan2` | 31,500 / 29,930 | `fixed::trig` (≈1.9× the former simba kernels) |
| `Vector3` dot / cross / norm / normalize | 1,980 / 5,540 / 2,220 / 11,830 | normalize: one prepared divisor (`div3`) |
| `Matrix3 * Matrix3` / `try_inverse` / `determinant` | 22,150 / 88,360 / 10,350 | fused rows |
| `SymMatrix3::quadform` (R·diag(d)·Rᵀ) / `try_inverse` | 30,430 / 67,000 | vs 45,000 / 88,960 generic |
| `Matrix6 * Matrix6` / `mul_vec` | 108,910 / 21,910 | one rescale per 6-term row |
| `q * q` / `uq.transform_vector` / `append_axisangle_linearized` | 11,550 / 22,070 / 35,770 | |
| `Isometry3::transform_point` / `inv_mul` / `*` | 23,270 / 38,640 / 35,320 | `inv_mul` on the fused `conj_mul` |
| `Isometry3::inverse_transform_point` | 26,190 | sign-folded conjugate |
| `Ldlt3::new` + `solve` / `Lu3::new` + `solve` | 45,660 / 65,080 | nearest division (was 36,990 / 56,820 on the floor scalar) |
| `SymmetricEigen3::new` / `Svd3::new` | 590,090 / 725,890 | 4 Jacobi sweeps |

Every figure has a `bench_*` test and, where a design choice was made, the losing variant next to it.

## 7. Measurement methodology (adopted project-wide)

- One snforge test per measurement, `bench_<group>__<variant>`, with a `bench_<group>__baseline`;
  `net = raw - baseline`. Sierra gas is deterministic: snapshots are compared for strict equality.
- Inputs through `black_box` (`#[inline(never)]` identity), results asserted.
- **Every benchmark test is `#[inline(never)]`**, otherwise snforge's wrapper inlining shifts the
  subtraction by ~1,020 gas at random.
- Report first-use *and* marginal cost when ranking primitives.
- `scripts/gas_report.py` produces the markdown report and the JSON snapshot; `--check` gates CI.
