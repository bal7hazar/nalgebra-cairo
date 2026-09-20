# Gas benchmarks of existing Cairo libraries

Empirical Sierra-gas measurements of **cubit**, **orion**, **alexandria** and **origami** on the
operations a fixed-point linear algebra library for a provable physics engine needs, each next to a
hand-written comparison point. Toolchain: scarb 2.19.4 / Cairo 2.19.4, snforge 0.61.0.

The full, generated report is [`GAS.md`](GAS.md) (50 groups, 131 variants plus their baselines).
This file explains how the numbers were obtained and what to conclude from them.

```sh
cd benchmarks/libs
snforge test 2>&1 | python3 ../scripts/gas_report.py --md GAS.md
```

This directory is a **standalone scarb package** (own `Scarb.lock` and `target/`), not a member of
the parent `benchmarks` workspace, so that third-party code never leaks into it.
`.cairofmtignore` keeps `scarb fmt` away from `vendor/`.

## What is measured, and how

| Library | Version measured | How it is consumed | Builds on 2.19.4 as a plain dependency? |
|---|---|---|---|
| alexandria (`alexandria_math`, `alexandria_linalg`) | 0.10.0 | **registry dependency** (scarbs.xyz) | yes |
| cubit (influenceth) | 1.4.0 @ `8007a30` | **vendored**, 36 patched lines, test code only | no: parser error in its `#[cfg(test)]` code, git and registry (1.3.1) alike |
| origami `algebra` (dojoengine) | 1.1.2 @ `1ddafcb` | **vendored**, visibility-only patch | compiles, but every item is private: unusable |
| orion (gizatechxyz) | 0.2.5 @ `bac0b42` | **minimal extraction** (see below) | no: 125 errors |

Each `vendor/<lib>/` holds the upstream `LICENSE` (all four are MIT) and a `NOTICE` with the
upstream URL, commit and the exact list of local changes. Summary of the patches:

- **cubit**: `assert(a <= b == false, ..)` became `assert((a <= b) == false, ..)` in 36 test
  assertions (`E1028 Consecutive comparison operators`, a parser error, so it fires even inside a
  dependency's test module). Dead duplicate directories and JS files dropped. No library line
  changed.
- **origami**: `pub` added on modules, structs, fields, traits and impls. Upstream declares
  `edition = "2024_07"` with `mod matrix;` in `lib.cairo`, hence `E2099 Item is not visible`.
- **orion**: timeboxed. Kept byte-identical: the `FixedTrait`, the whole `fp16x16` implementation
  (orion's own code), the `fp32x32` / `fp64x64` wrappers and `tensor/linalg/matmul.cairo`.
  Extracted with verbatim function bodies: `Tensor<T>`, a `TensorTrait` cut down from ~106 methods
  to `new / at / add / matmul`, the shape / broadcasting helpers they call, the broadcasting `add`,
  `MutMatrix` + `NullableVec`. Two alexandria helpers orion imports from a 2.5-era revision
  (`VecTrait`, `reverse`) are inlined. Orion has **no determinant / inverse / solve** operator:
  its "linalg" folder is `matmul`, `transpose`, `trilu`.
- Orion's `FP32x32` and `FP64x64` are not implementations: they are `use cubit::f64::Fixed as
  FP32x32` plus a delegating trait. The measured figures are identical to cubit's, as expected.

### Method

- One snforge test per variant, named `bench_<group>__<variant>`; Sierra gas is deterministic.
- Inputs go through `harness::black_box`, results are asserted.
- Every `bench_*` function (baselines included) is `#[inline(never)]`: snforge otherwise inlines
  some test bodies into its generated wrapper and not others, which shifts `raw - baseline` by up
  to ~1 000 gas at random.
- **Exact baselines.** Libraries use different representations (sign-magnitude struct, `i64`,
  spans, tensors), so "same inputs, same assertions" is enforced literally: in a group *every*
  test calls the same `#[inline(never)] inputs()` (which builds the inputs in **all**
  representations) and the same `#[inline(never)] check(..)` (which verifies one value per
  representation, passing the untouched input where a variant has nothing to report). The group
  baseline is therefore the exact fixed overhead of every variant; `net` is the operation alone.
- Containers (arrays, tensors, matrices) are built inside `inputs()`, i.e. **outside** the measured
  operation. The `build_vec3` group prices construction separately.
- Transcendental results are asserted against the true value with a 1e-3 tolerance; the accuracy
  really reached is printed by the `report_accuracy` tests and quoted below.

### The `reference` variants (`src/reference.cairo`)

The obvious hand-written form, **not** the proposed nalgebra.cairo design (the sibling packages
`primitives/`, `scalar/`, `layout/` explore that): two's complement native integers (`i32`, `i64`,
`i128`) widened with `WideMul` and divided back, `struct Vec3 { x, y, z: i64 }`, unrolled 3x3.
`*_fused` variants accumulate the wide products and rescale once per output instead of once per
product. `Fix64` is a newtype over `i64` with the operator traits, so that the generic containers
of origami / alexandria can be instantiated with the reference number: that separates the cost of
the *container* from the cost of the *number*.

`reference_bounded` (module `q32_bounded`, Q32.32 `mul` and fused `dot3`) is different in kind: it
is a copy of the `BoundedInt` biased-floor kernels of `benchmarks/primitives/src/bounded.cairo`,
i.e. the kernel nalgebra.cairo intends to ship, so that the library comparison includes it.

## Results (net gas)

### Scalars

| op | cubit f64 = orion FP32x32 (Q32.32) | reference `i64` (`i128` division) | orion FP16x16 | reference `i32` | cubit f128 = orion FP64x64 (Q64.64) | reference `i128` |
|---|---:|---:|---:|---:|---:|---:|
| add | 4 050 | **740** | 4 050 | **740** | 2 240 | **470** |
| sub | 4 560 - 4 650 | **740** | | | | |
| mul | **1 750** | 3 090, `reference_bounded`: **1 950** | **1 280** | 2 620 | **11 280** | 14 170 |
| div | **1 750** | 5 430 | **1 280** | 4 960 | **11 580** | 15 840 |
| sqrt | 5 110 | **1 920** | 2 350 | | 8 880 (32 useful fractional bits) | 9 360 (64 bits) |
| neg | 300 | 200 | | | | |
| `<` | 680 (different signs), 1 880 (same sign) | 870 | 680 | 870 | 680 | 870 |
| `!=` | 610 | 600 | | | | |

Two things stand out. Sign-magnitude makes **add/sub 5.5x to 6.3x more expensive** than a native
signed add (sign test, magnitude test, magnitude compare, then add or sub; Sierra charges the most
expensive branch whatever the inputs). Conversely the naive two's complement **mul/div is 1.8x / 3.1x
more expensive than cubit's** (re-checked after the `#[inline(never)]` fix: 3 090 vs 1 750), because
corelib's signed `i128` division is costly (split on the sign, unsigned divide, re-apply the sign);
cubit only ever divides unsigned. Neither is what nalgebra.cairo should ship.
`reference_bounded` is the `BoundedInt` biased-floor kernel copied from
`benchmarks/primitives/src/bounded.cairo` (range-tracked product, bias to non-negative, `div_rem`
by the constant 2^32, un-bias, one narrowing): **1 950**, i.e. within 11% of cubit's unsigned path
while keeping the 740 native add, a one-felt representation and floor (arithmetic shift)
semantics. Fused, the same kernel computes a whole 3D dot product for **2 350** (next table).

### Transcendentals (x = 0.75 rad)

| function | gas | absolute error |
|---|---:|---:|
| cubit f64 `sin` (8-term Taylor, recursive, 1 div per term) | 130 470 | 1 LSB (2e-10) |
| cubit f64 `sin_fast` (LUT, 256 entries over [0, pi/2] + lerp) | **30 320** | 2.3e-6 |
| reference `sin` (odd polynomial to x^11, Horner in x^2, 7 mul, 0 div) | 33 300 | 1 LSB (2e-10) |
| orion FP16x16 `sin` (silently = `sin_fast`) / its hidden Taylor version | 25 290 / 117 850 | < 1 LSB of Q16.16 |
| cubit f128 `sin` / `sin_fast` | 460 330 / 52 600 | 4e-17 / 2.3e-6 |
| alexandria `fast_sin` (degrees x 1e8, 10-entry table) | 23 170 | **4.9e-4** |
| cubit f64 `cos` / `cos_fast` | 135 220 / 35 070 | 1 LSB / 2.4e-6 |
| cubit f64 `tan` / `tan_fast` | 269 140 / 69 340 | 1 LSB / 7e-9 |
| cubit f64 `atan` / `atan_fast` | 83 740 / 57 290 | 7e-10 / 9.7e-7 |
| cubit f64 `acos` / `acos_fast` | 115 060 / 88 610 | 5e-10 / 1.3e-6 |
| cubit f64 `exp` / `ln` | 64 870 / 70 260 | 7e-10 / 2.2e-8 |

The reference polynomial is priced with the *slow* reference multiplication (3 090); with the
`BoundedInt` kernel (1 950) its 7 multiplications would bring it to about 25 000 for full Q32.32
accuracy, i.e. cheaper **and** 10 000x more accurate than the LUT.

### Vectors (Q32.32, 3D unless noted)

| op | reference bounded (fused) | reference fused (`i128`) | reference | cubit `Vec3` | origami `Vector<T>` (reference / cubit number) | alexandria `dot` (reference / cubit number) | orion `Tensor` |
|---|---:|---:|---:|---:|---:|---:|---:|
| dot | **2 350** | 6 360 | 12 680 | 15 450 | 19 950 / 28 480 | 20 450 / 28 880 | 37 820 |
| cross | | **13 210** | 22 490 | 28 830 | | | |
| norm | | **5 360** | 14 600 | 20 860 | | | |
| add | | | **4 550** | 12 150 | - / 26 010 | | |
| dot 2D | | **5 590** | | 7 850 (origami `Vec2<T>`: 7 850) | | | |
| build from 3 scalars | 0 | 0 | 0 | 0 | 1 200 | | 6 310 (+7 320 for one `at([0])`) |

### 3x3 matrices (Q32.32)

| op | reference | origami `Matrix<Fix64>` | origami `Matrix<cubit>` | orion `Tensor<FP32x32>` |
|---|---:|---:|---:|---:|
| matmul | **60 840** fused, 117 720 unfused | 302 140 | 359 010 | 349 040 |
| matmul, Q16.16 | **61 470** fused | | | 336 150 (`Tensor<FP16x16>`) |
| element-wise add | **13 950** | 79 850 | 110 330 | **794 210** |
| transpose | **300** | | 57 310 | |
| determinant | **19 570** | 257 700 | 279 720 | not provided |
| inverse | **98 490** | 1 041 580 | 1 076 100 | not provided |
| matrix x vector | **20 300** fused | | | 226 780 (`MutMatrix`, dictionary-backed) |

### alexandria primitives

| op | alexandria | corelib / literal | ratio |
|---|---:|---:|---:|
| `isqrt(u128)`: `fast_sqrt(x, 40)` vs `Sqrt::sqrt` | 740 770 | **1 180** | x628 |
| `x >> 16` (u64): `BitShift::shr` / `OptBitShift::shr` (LUT) vs `x / 0x10000` | 18 620 / 3 320 | **1 010** | x18 / x3.3 |
| `x << 16` (u64): `BitShift::shl` / `OptBitShift::shl` vs `x * 0x10000` | 20 063 / 4 863 | **470** | x43 / x10 |
| `2^n`: `pow` / `fast_power` / `const_pow::pow2` (LUT) vs `core Pow` | 17 410 / 91 120 / **1 540** | 13 950 | |
| `dot` of 3 `u64` | 11 800 | **2 150** unrolled | x5.5 |
| `kron` of two 3-vectors | 35 170 | **3 460** unrolled | x10 |
| L2 `norm` of 3 `u64` (10 Newton iterations) | 215 820 | **3 330** | x65 |
| `wad_mul` / `wad_div` (1e18 decimal on u256) | 22 700 / 29 730 | Q32.32 mul/div: 1 750 | x13 / x17 |

## Anti-patterns and their measured cost

| # | Anti-pattern | Where | Measured cost |
|---|---|---|---|
| 1 | Sign-magnitude number (`{ mag, sign: bool }`) | cubit, orion | add 4 050 vs 740 (x5.5), sub x6.3. A dot product is 3 mul + 2 add: the two adds (8 100) cost more than the three multiplications (5 250); cubit's dot3 is 15 450 where the fused `BoundedInt` kernel is 2 350 (**x6.6**). Also a second felt per number in calldata / `Serde`, and a negative zero: `0 * -x` yields `{ mag: 0, sign: true }`, which is `!=` zero (asserted by `finding_cubit_negative_zero`). |
| 2 | Rescaling after every product | everyone | dot 12 680 vs 6 360, cross 22 490 vs 13 210, matmul 117 720 vs 60 840: **x1.7 to x2**. `norm` rescales then re-widens for the sqrt: 14 600 vs 5 360 (x2.7), truncating the low 32 bits of the sum before the root for nothing. |
| 3 | `Span<T>` / `Array<T>` as the vector or matrix storage | origami, alexandria, orion | With the *same* number type: dot 19 950 vs 12 680 (+57%), matmul 302 140 vs 117 720 (x2.6), add 79 850 vs 13 950 (x5.7), transpose 57 310 vs 300 (x191: a struct shuffle is free, a span must be rebuilt element by element through bounds-checked `get` and `u8` div/mod index arithmetic). |
| 4 | Runtime shapes (`shape: Span<usize>`) | orion | Shape checks, `prepare_shape` / `adjust_output_shape`, strides: 3-vector dot 37 820 vs 15 450 with the same numbers (x2.4), construction 6 310 for a 3-vector, 7 320 to read one element through `at(indices)`. |
| 5 | Generic broadcasting on every element-wise op | orion | 3x3 + 3x3 = **794 210**, i.e. 88 000 per element (`unravel_index` + two `broadcast_index_mapping`, each allocating strides) against 1 550: **x57**. |
| 6 | Dictionary-backed matrix (`Felt252Dict<Nullable<T>>`) | orion `MutMatrix` | 3x3 matrix x vector 226 780 vs 20 300 (**x11**), construction excluded. |
| 7 | Recursive cofactor expansion with allocated minors | origami `det` / `inv` | det 257 700 vs 19 570 (**x13**); inv 1 041 580 vs 98 490 (**x10.6**): the inverse recomputes 9 minors as fresh arrays plus the full determinant, and performs 9 divisions instead of 1 reciprocal + 9 multiplications. |
| 8 | u256 arithmetic for the wide intermediate | cubit f128, alexandria wad | Q64.64 mul 11 280 vs Q32.32 1 750 (**x6.4**), div x6.6, `sin` 460 330 vs 130 470. Q64.64 is not affordable as the default scalar of a physics step. |
| 9 | Checked `u128 * u128` where a `u64` wide multiplication suffices | cubit `sqrt` | 5 110 vs 1 920 (x2.7) for the same `u128` square root. |
| 10 | Precision thrown away | cubit f128 `sqrt` | Computes `sqrt(mag) * 2^64 / 2^32`: a Q64.64 result whose low 32 bits are always zero (asserted in `bench_sqrt_q64__cubit`). Costs the same as the exact u256 square root (8 880 vs 9 360). |
| 11 | Taylor series evaluated term by term with a division per term | cubit / orion `sin`, `cos`, `tan` | sin 130 470 vs 33 300 for a Horner polynomial of the same accuracy (x3.9); `tan` = sin + cos + div = 269 140. |
| 12 | Misleading API | orion | `FixedTrait::sin / acos / asin / atan` silently call the `_fast` LUT versions (25 290), while the accurate versions are only reachable through the module path (117 850). |
| 13 | Newton iteration with a caller-supplied iteration count and `x/2` as the initial guess | alexandria `fast_sqrt`, `norm` | 40 iterations needed for a Q32.32 radicand (35 give a wrong answer, silently): 740 770 vs 1 180 (**x628**). `norm` of 3 small integers: x65. |
| 14 | Variable shifts computed through a generic recursive `pow(2, n)` | alexandria `BitShift` | x18 (shr) and x43 (shl) against a literal; even the LUT version is x3.3 / x10. Fixed-point scales are compile-time constants: never shift by a runtime amount. |
| 15 | Re-implementing core traits for a foreign type | orion `FP32x32` = cubit `Fixed` | Importing orion's `FP32x32Add/Mul/Div/PartialOrd` next to cubit's own impls is a hard error on 2.19.4 (`E2313 multiple implementations`); the benchmarks have to call them by path. |

## Assessment per library

### cubit

- **Reuse**
  - The constants: Q32.32 and Q64.64 `PI`, `HALF_PI`, `ln 2`, `log10 2`; the polynomial
    coefficient sets of `atan` (degree 10 on [0, 0.7] after the `sqrt(3)/3` range reduction), `exp2`
    (degree 8 on the fractional part) and `log2` (degree 8 on the mantissa); measured accuracy 7e-10
    (`atan`, `exp`) and 2e-8 (`ln`). They are directly portable to a two's complement Q32.32.
  - The decomposition ideas: `exp(x) = 2^(x * log2 e)` with an integer-part table, `log2` through the
    most significant bit, `atan` argument folding, `acos` / `asin` through `atan`.
  - The test vectors in `f64/math/*.cairo` and `f64/types/*.cairo` (values, tolerances, edge cases).
  - API surface worth mirroring: `FixedTrait` method names, `Vec2/3/4` with `dot`, `cross`, `norm`,
    component-wise operators, `StorePacking` of a fixed-point number into one felt.
- **Avoid**: sign-magnitude (#1), per-product rescale (#2), Taylor `sin` (#11), f128 as a default
  (#8), f128 `sqrt` (#10), the 256-entry if/else LUTs (`lut.cairo` is 1 331 lines for f64 and 1 550 for f128, 2.3e-6
  accuracy, not cheaper than a polynomial).
- **Status**: upstream `main` moved in 2026-07 but still targets "Cairo >= 2.7" with edition
  `2023_10`; it does not parse on 2.19.4 (test code only, 36 lines), the registry release 1.3.1
  neither. The ecosystem runs on personal forks (origami pins `bengineer42/cubit` branch
  `bump-cairo-gt-2.8`, orion pins an old rev). Depending on it means vendoring or forking.

### orion

- **Reuse**: almost nothing for this project. The FP16x16 code is a port of cubit (same algorithms,
  same sign-magnitude). The one good idea is the generic `FixedTrait<T, MAG>` / `NumberTrait<T, MAG>`
  pair that lets one algorithm run over several precisions, which is how a `Scalar` trait should be
  shaped (without the 55 mandatory methods of orion's `NumberTrait`). The ONNX-style operator catalogue is irrelevant to
  rigid-body physics; there is no determinant, inverse or solver.
- **Avoid**: everything structural: runtime shapes (#4), broadcasting (#5), dictionary matrices
  (#6), monolithic traits (one ~106-method `TensorTrait` that every element type must implement in
  full, which is also why a partial port is impossible without surgery), silent `_fast` dispatch
  (#12), core-trait impls on foreign types (#15).
- **Status**: last commit 2025-03, written for Cairo 2.5.3, not published on scarbs.xyz. 125
  compile errors on 2.19.4, plus two pinned dependencies that fail on their own. Effectively
  unmaintained.

### alexandria

- **Reuse**: `const_pow::pow2` / `pow10` (constant arrays: the cheapest runtime power of two
  measured, 1 540); the `OptBitShift` lookup-table idea if a runtime shift is ever unavoidable;
  `u512_arithmetics` and `i257` as references if a wider intermediate is needed. The package layout
  (one small crate per concern, registry releases, per-crate tests, a committed `gas_report.json`)
  is the maintenance model to copy.
- **Avoid**: `fast_sqrt` / `fast_cbrt` / `norm` (#13; corelib `Sqrt` is x628 cheaper and exact),
  `BitShift` (#14), `fast_power` (goes through u256: 91 120 for `2^16`), `linalg::dot` / `kron` on
  spans for fixed-size data (#3), `trigonometry` (decimal degrees x 1e8, 4.9e-4 error),
  `wad_ray_math` as a number format (#8).
- **Status**: healthy. 0.10.0 on scarbs.xyz (built for Cairo 2.16), compiles and runs unchanged on
  2.19.4. The only one of the four usable as a normal dependency. `alexandria_linalg` is three
  functions (`dot`, `kron`, `norm`): there is no matrix type to build on.

### origami (`algebra`)

- **Reuse**: the API shape of `Vec2<T>` (generic over the scalar, `splat`, `select(mask, a, b)`,
  swizzles `xx/xy/yx/yy`: all free at the Sierra level; `dot` costs exactly cubit's 7 850) and the
  fact that `Matrix<T>` / `Vector<T>` are generic over any `T` with the operator traits, which is
  what made the `Fix64` instantiation in this benchmark possible. The small test matrices of
  `matrix.cairo` (2x2 / 3x3 det and inverse) are reusable as test vectors.
- **Avoid**: span-backed matrices with `u8` dimensions and per-access bounds checks (#3), recursive
  cofactor `det` / `inv` (#7), `get(ref self, ..)` taking `ref` for a read.
- **Status**: compiles on 2.19.4 but **cannot be used from another crate** (all items private
  under edition 2024_07), not on the registry, depends on a personal fork branch of cubit. `Vec2`
  is the only fixed-size type; there is no `Vec3`, `Mat2`, `Mat3`, quaternion or transform.

## Conclusions for nalgebra.cairo

1. No existing library can be a dependency: three of four do not build or link on 2.19.4, and the
   fourth (alexandria) has no fixed-size linear algebra. What is reusable is data (coefficients,
   constants, test vectors) and API vocabulary, under MIT.
2. Fixed-size structs with unrolled arithmetic are **5x to 13x** cheaper than any span / tensor /
   dictionary container measured here, and struct shuffles (transpose, swizzles, construction) are
   free. Runtime-shaped containers have no place in the core.
3. Number representation: two's complement for add / sub / compare (x5.5 cheaper than
   sign-magnitude) but the rescale must not go through corelib's signed wide division (x1.8 to
   x3.1 slower than cubit's unsigned path). The `BoundedInt` biased-floor kernel gives both: mul
   1 950 (cubit 1 750, naive `i128` 3 090), add 740 (cubit 4 050). Q32.32 is the affordable
   precision; Q64.64 costs x6.4 per multiplication.
4. Fuse: accumulate wide products and rescale once per output (x1.7 to x2.7 on dot, cross, norm,
   matmul with `i128`), and take `norm` directly from the un-rescaled sum of squares. With
   `BoundedInt` the sums are range-check free and the gain compounds: dot3 = **2 350**, against
   6 360 (`i128` fused), 12 680 (unfused), 15 450 (cubit), 28 480 (origami), 37 820 (orion): the
   fused dot costs barely more than a single multiplication (1 950). The `*_fused` matrix figures
   above (matmul 60 840, det 19 570, inverse 98 490) still use the `i128` path and are upper bounds.
5. Transcendentals: Horner polynomials on a folded argument beat both of cubit's options (as
   accurate as the Taylor loop at a quarter of its price, 10 000x more accurate than the LUT at the
   same price).
