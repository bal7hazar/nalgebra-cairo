# Fixed-point and linear-algebra prior art in Cairo: Cubit and Orion

Benchmark study for `nalgebra.cairo` (scalar type design). All paths are absolute or relative to:

- `CUBIT` = `<scratchpad>/refs/cubit` (influenceth/cubit, `main` @ `8007a30`, v1.4.0)
- `ORION` = `<scratchpad>/refs/orion` (gizatechxyz/orion, `main` @ `bac0b42`, v0.2.5)
- `LOCAL_CUBIT` = the maintainer's fork `bal7hazar/cubit` (branch `cairo-2.9.2`, v1.3.0)
- `CORELIB` = `~/Library/Caches/com.swmansion.scarb/registry/std/v2.19.4/core/src`
- `BENCH` = `<scratchpad>/bench` and `<scratchpad>/bench_cubit` (throw-away snforge projects written
  for this report; scarb 2.19.4 + snforge 0.61.0)

Unlike what the brief allowed (static analysis only), a first round of micro-benchmarks **was run**;
numbers below tagged **[measured]** come from those runs. They are `l2_gas` deltas per call
(loop of N=1000 (or 100) calls to an `#[inline(never)]` function, minus a baseline loop). They are
indicative, not final: they must be re-run inside the real crate (see section 4.6).

---

## 0. TL;DR

1. Cubit and Orion share the same fixed-point core (Orion's FP types are a copy of Cubit's code, and
   FP32x32/FP64x64 literally re-export `cubit::f64`/`cubit::f128`). Both use **sign-magnitude
   structs** `{ mag: uN, sign: bool }`.
2. Sign-magnitude makes `add`/`sub`/comparison branchy (3.6k gas per add vs 2.1k for native `i64`),
   yields a **negative zero** that breaks `==`, and prevents fusing sums of products.
3. In Cairo, integer *width* is free up to 128 bits; what costs is the **number of range checks,
   branches, and wide (u256) ops**. Q64.64 mul costs ~3x Q32.32 mul because the product leaves the
   128-bit domain. Q16.16 is *not* cheaper than Q32.32.
4. Recommended scalar: **Q32.32 stored in a native `i64` (newtype `struct Fixed { raw: i64 }`)**, with
   arithmetic kernels written on `core::internal::bounded_int` so that a product (or a whole
   dot product / matrix row) is computed range-check-free in a wide `BoundedInt`, rescaled by **one**
   `div_rem` by `2^32` using a positive offset (branch-free floor), and range-checked **once**.
   Measured: mul 3.3k gas (cubit-style 3.75k, naive native i64 4.3k); fused dot3 **4.1k vs 20.5k**;
   fused norm3 **~5k vs ~25k**.
5. `sqrt` needs no Newton iteration: corelib `u128_sqrt` is a hint-verified libfunc (~4.5k gas
   including scaling). Trig should be unrolled polynomials (sin deg 9: ~3e-9 error for 6 muls),
   not Cubit's Taylor recursion (130k gas) nor its if-chain LUTs (30k-57k gas).
6. Orion's tensor/linalg layer contains nothing reusable for a static-dimension physics library:
   no det/inverse/solve exists, and matmul is a generic dynamic triple loop with `Array::append`.

---

## 1. Cubit

### 1.1 Layout, version, maintenance

- Two parallel implementations: `CUBIT/src/f64/**` (Q32.32, `mag: u64`) and `CUBIT/src/f128/**`
  (Q64.64, `mag: u128`). Each has `types/{fixed,vec2,vec3,vec4}.cairo`,
  `math/{ops,trig,hyp,comp,lut}.cairo`, `procgen/{rand,simplex3}.cairo`, `test/helpers.cairo`.
- Stale leftovers: `CUBIT/src/lib.cairo` still declares `mod math; mod procgen; mod test; mod types;`
  and top-level `src/math/{lut,trig}.cairo`, `src/types/vec2.cairo` exist as orphans from the
  pre-split layout.
- `CUBIT/Scarb.toml`: `version = "1.4.0"`, `cairo-version = ">=2.7.0"`, `edition = "2023_10"`,
  `cairo_test = "2.7.0"`. Only a `main` branch upstream in the shallow clone; last commit is the merge
  of PR #46 "cairo-2.7". MIT license (Unstoppable Games, 2023).
- Maintenance: effectively frozen. Upstream stopped at Cairo 2.7; `LOCAL_CUBIT` (branch
  `cairo-2.9.2`, last commit 2025-01-30) is the user's own bump. The diff upstream -> local is
  1360 lines but purely mechanical:
  - explicit generic args where inference changed:
    `WideMul::<u64, u64>::wide_mul(a.mag, b.mag)`, `Sqrt::<u128>::sqrt(...)`
    (`LOCAL_CUBIT/src/f64/math/ops.cairo:47,187,274`);
  - trait renames `FixedAddEq -> FixedAddAssign` etc. in test imports;
  - parenthesised comparisons in asserts (`assert((a <= b) == false, ...)`), explicit `: Fixed`
    annotations in tests, trailing commas (formatter).
  No algorithmic change. It still compiles as a path dependency under **scarb 2.19.4** (verified in
  `<scratchpad>/bench_cubit`), with deprecation warnings only.

### 1.2 Representation

`CUBIT/src/f64/types/fixed.cairo:18-29`:

```cairo
const ONE: u64 = 4294967296; // 2 ** 32
#[derive(Copy, Drop, Serde)]
struct Fixed { mag: u64, sign: bool }
```

`CUBIT/src/f128/types/fixed.cairo:17-30`: same with `mag: u128`, `ONE_u128 = 2**64`.

Consequences:

- 2 felts per scalar in memory, calldata and Serde (a `Vec3` is 6 felts, a 3x3 matrix 18 felts).
  Storage is packed to one felt via `StorePacking` (`fixed.cairo:513-527`), which uses a
  `U128DivRem` by `2^64` to unpack.
- Range is asymmetric vs. two's complement and *unchecked at the type level*: `mag` may use all 64
  bits (so the real range is +-2^32, not +-2^31), but `MAX`/`MIN` are not defined for f64.
- **Negative zero exists.** `mul` computes `sign: a.sign ^ b.sign` regardless of magnitude
  (`ops.cairo:185-190`) and `eq` compares both fields (`ops.cairo:54-56`), so
  `(-1 * 0) == 0` is **false**. `add` special-cases `a.mag == b.mag` to return `ZERO`, `neg` special
  cases zero, `sin` has `&& loop_res.mag != 0`, all to fight this; `mul`/`div`/`floor`/`round` do not.
  This is a correctness hazard for a physics engine (e.g. `if v.x == ZERO`).

### 1.3 Core operations (`CUBIT/src/f64/math/ops.cairo`)

```cairo
fn add(a: Fixed, b: Fixed) -> Fixed {                       // :15-29
    if a.sign == b.sign { return FixedTrait::new(a.mag + b.mag, a.sign); }
    if a.mag == b.mag { return FixedTrait::ZERO(); }
    if (a.mag > b.mag) { return FixedTrait::new(a.mag - b.mag, a.sign); }
    else { return FixedTrait::new(b.mag - a.mag, b.sign); }
}
fn sub(a: Fixed, b: Fixed) -> Fixed { return add(a, -b); } // :278 (neg has 2 more branches)
fn mul(a: Fixed, b: Fixed) -> Fixed {                       // :185-190
    let prod_u128: u128 = WideMul::wide_mul(a.mag, b.mag);
    return FixedTrait::new((prod_u128 / ONE.into()).try_into().unwrap(), a.sign ^ b.sign);
}
fn div(a: Fixed, b: Fixed) -> Fixed {                       // :46-52
    let a_u128: u128 = WideMul::wide_mul(a.mag, ONE);
    let res_u128 = a_u128 / b.mag.into();
    return FixedTrait::new(res_u128.try_into().unwrap(), a.sign ^ b.sign);
}
```

- **mul/div (f64)**: widen `u64 x u64 -> u128` (free, a felt mul), one `u128` divmod, one
  `u128 -> u64` downcast (range check), bool xor. This is actually near-optimal for sign-magnitude:
  3.75k gas **[measured]**. Rounding is truncation toward zero (symmetric), not floor.
- **mul/div (f128)** (`CUBIT/src/f128/math/ops.cairo:188-197, 51-60`): `u128_wide_mul -> u256`, then
  `u256_safe_div_rem` by `ONE_u256`, then `assert(high == 0)`. The u256 division is the most
  expensive integer op in corelib; dividing by a power of two this way is wasteful (a hi/lo
  recomposition `hi * 2^64 + lo / 2^64` does the same). 11.2k gas **[measured]**, vs 10.4k for the
  recomposition variant: `u128_wide_mul` itself dominates. Q64.64 mul is ~3x a Q32.32 mul.
- **add/sub**: 3 data-dependent branches + equality tests; 3.65k gas for mixed signs **[measured]**
  vs 2.07k for a native `i64 +`. `sub` = `neg` (2 branches) + `add`.
- **Comparison** (`ops.cairo:107-183`): `lt/le/gt/ge` each branch on sign, then up to two magnitude
  comparisons (`!=` and `<`) plus xor: 2.8k **[measured]** vs 1.6k for `i64 <` (one `i64_diff`).
- **Rounding**: `floor/ceil/round` (`ops.cairo:31-44, 95-105, 262-270`) use `u64_safe_divmod(mag, ONE)`
  then sign branches. `round` of `-0.4` returns `{0, true}` (negative zero again).
- **rem** (`:258`): `a - floor(a / b) * b`: a div, a floor (divmod), a mul, a sub: ~15k gas.
- **sqrt (f64)** (`:270-276`): `Sqrt::sqrt(a.mag.into() * ONE.into())` on u128 -> u64. This calls the
  `u128_sqrt` **libfunc** (`CORELIB/integer.cairo:173`), which is hint-computed and verified
  in-circuit, so there is no loop: ~4.5-4.8k gas **[measured]**. This is the one thing to keep.
- **sqrt (f128)** (`CUBIT/src/f128/math/ops.cairo:289-295`):
  `sqrt(mag) * ONE / sqrt(ONE)`, i.e. `u128_sqrt(mag) * 2^32`. The result has **only 32
  significant fractional bits** (low 32 bits are always zero) and costs an extra mul + div by a
  constant that is itself recomputed via `Sqrt::sqrt(ONE_u128)` on every call. Precision bug.
- **exp/exp2** (`:58-91`): `exp(x) = exp2(x * log2(e))`; integer part through `lut::exp2`
  (if-tree on the exponent), fractional part through an 8-term Horner polynomial (8 fixed muls),
  plus a full division `ONE / res` for negative inputs. ~65k gas **[measured]**.
- **ln/log2/log10** (`:133-176`): `lut::msb` (nested if-tree, `lut.cairo:2-106`), one division to
  normalise into [1,2), 8-term Horner polynomial; **recursive call** with `1/x` for `x < 1`
  (extra division). ~74k gas **[measured]**.
- **pow** (`:208-256`): integer exponent -> square-and-multiply `loop` with `u64_safe_divmod(n, 2)`;
  otherwise `exp(b * ln(a))` (~140k).
- **hyp** (`CUBIT/src/f64/math/hyp.cairo`): textbook formulas from `exp`/`ln`, e.g. `tanh` does one
  `exp` and three divisions. Not needed for physics.

### 1.4 Trigonometry (`CUBIT/src/f64/math/trig.cairo`)

- `sin` (`:148-157`): reduce with `mag % TWO_PI` then `divmod(PI)`, then
  `_sin_loop(a2, 7, ONE)` (`:188-198`): an **8-step recursion**, each step doing
  `a * a * acc / new_unscaled(div)`: 2 fixed muls + **1 fixed division** + 1 sub per step,
  and `a * a` is recomputed at every step. ~130k gas **[measured]**. Accuracy ~1e-9 (limited by
  accumulated truncation, not the series: Taylor-17 error is 4e-14).
- `cos(a) = sin(HALF_PI - a)` (`:138-140`): one extra sign-magnitude sub.
- `tan = sin / cos` (`:171-176`): two full evaluations + division (~265k).
- `sin_fast` (`:159-174`): fold to [0, pi/2], `lut::sin` returns `(start, low, high)` for a
  **256-slot** table (`CUBIT/src/f64/math/lut.cairo:212-1028`) implemented as a nested `if` tree
  (8 comparisons + `a / 26353589` division), then linear interpolation with a **division** by the
  slot width and a mul. ~30k gas **[measured]**; error bound h^2/8 = 4.7e-6.
- `atan` (`:61-101`): |x|>1 -> `1/x` (division); x>0.7 -> `(x - sqrt3/3)/(1 + x*sqrt3/3)` (mul +
  division); then a **degree-10 full polynomial** (10 muls, even and odd coefficients, not
  exploiting oddness). 84k gas on the cheap path **[measured]** (no inversion/shift), ~95k worst.
  Tests accept 1e-5 relative error.
- `atan_fast` (`:103-135`): same range reduction, then `lut::atan`, a **linear chain of 100
  `if slot == k`** (`lut.cairo:1029+`, average 50 comparisons), plus division-based interpolation.
  57k gas **[measured]**: the "fast" variant is barely faster.
- `asin(a) = atan(a / sqrt(1 - a*a))`, `acos(a) = asin(sqrt(1 - a*a))` with pi correction
  (`:16-58`): **two sqrt + one division + atan** for `acos`. 115k (`acos`), 89k (`acos_fast`)
  **[measured]**. `asin` special-cases `mag == ONE` to avoid division by zero, but values just
  below one lose all precision in `a / sqrt(1 - a^2)`.
- No `atan2`, no `sin_cos` combined evaluation, no `inv_sqrt`.

### 1.5 Conversions and API ergonomics

- `FixedTrait` (`fixed.cairo:33-78`) is a *non-generic* trait per width, exposing constructors
  (`new`, `new_unscaled`, `from_felt`, `from_unscaled_felt`) and all math as methods. Operators
  (`Add/Sub/Mul/Div/Rem/Neg`, `*Assign`, `PartialEq`, `PartialOrd`) and `Zero`/`One` are implemented
  (`fixed.cairo:414-552`). There is no scalar abstraction shared by f64 and f128, hence the whole
  tree is duplicated (`f64/` vs `f128/`, including the Vec types and the 1.3k-1.5k-line LUT files).
- `from_felt` (`fixed.cairo:93-96`) calls `utils::felt_sign`/`felt_abs` (`CUBIT/src/utils.cairo:9-22`),
  which converts to **u256 twice** to compare with `HALF_PRIME` and multiplies by `-1`/`1`. Used by
  every test helper (`assert_precise`), which inflates the test gas numbers.
- `Into`/`TryInto` for u8..u256 and i8..i128 (`fixed.cairo:270-412`); f64 <-> f128 conversions.
- Most wrappers are plain `fn` without `#[inline]`, e.g. `FixedAdd::add -> ops::add`
  (`fixed.cairo:428-432`): two call frames per `a + b` unless the compiler auto-inlines.

### 1.6 Vec2/Vec3/Vec4 (`CUBIT/src/f64/types/vec3.cairo`)

`struct Vec3 { x: Fixed, y: Fixed, z: Fixed }` with `new, splat, abs, cross, dot, floor, norm` and
component-wise/scalar `add/sub/mul/div/rem`. Implementations are the naive compositions:

```cairo
fn dot(a: Vec3, b: Vec3) -> Fixed { return (a.x * b.x) + (a.y * b.y) + (a.z * b.z); } // :144
fn norm(a: Vec3) -> Fixed { return dot(a, a).sqrt(); }                                 // :156
```

i.e. 3 rescaling divisions + 3 downcasts + 2 branchy adds per dot (20.5k gas **[measured]**), and
`norm` rescales down then `sqrt` rescales up again. No `normalize`, `length_squared`, `lerp`,
matrices, or quaternions. `div(Vec3, scalar)` does 3 divisions instead of 1 reciprocal + 3 muls.

### 1.7 Tests and gas annotations

- Inline `#[cfg(test)] mod tests` at the bottom of each file; ~140 tests for f64+f128.
- Assertions via `assert_precise(result, expected_felt, msg, Option<precision>)` and
  `assert_relative` (`CUBIT/src/f64/test/helpers.cairo`), default tolerance 430 raw = 1e-7; trig tests
  loosen to 1e-5.
- 144 `#[available_gas(N)]` annotations, but these are only **upper bounds** needed by the old
  cairo-test runner (e.g. `#[available_gas(3000000)]` for `test_acos`). There is **no gas
  tracking/regression**, no per-op benchmark, and no comparison between `x` and `x_fast` variants.
- Few edge cases: no overflow tests for `mul`, no negative-zero tests, sparse trig sample points.

### 1.8 Inventory of inefficiencies (Cubit)

| # | Where | Issue |
|---|-------|-------|
| 1 | `ops.cairo:15-29, 278` | sign-magnitude add/sub: 3-5 branches, `sub` = `neg` + `add` |
| 2 | `ops.cairo:107-183` | comparisons: sign branch + two magnitude compares |
| 3 | `ops.cairo:185-190` | negative zero from `mul`/`div`/`round`; `eq` then fails |
| 4 | `f128/math/ops.cairo:188-197, 51-60` | `u256` division by a power of two |
| 5 | `f128/math/ops.cairo:289-295` | sqrt loses 32 fractional bits, recomputes `sqrt(ONE)` |
| 6 | `trig.cairo:188-198` | Taylor recursion with a fixed **division** per term, `a*a` recomputed |
| 7 | `trig.cairo:61-101` | degree-10 dense polynomial for an odd function; 2 divisions in reduction |
| 8 | `lut.cairo:212+, 1029+` | LUTs as `if` trees / 100-long linear `if` chain; tuple of 3 u64 returned |
| 9 | `trig.cairo:159-174` | interpolation divides by the (constant) slot width instead of multiplying |
| 10 | `trig.cairo:16-58` | `acos` = 2 sqrt + div + atan |
| 11 | `vec3.cairo:144-158` | no fused dot/norm/cross; per-product rescale |
| 12 | `utils.cairo:9-22` | felt sign via two u256 conversions |
| 13 | tree | f64/f128 code duplicated instead of generic; 2 felts per scalar |
| 14 | tests | `available_gas` ceilings only, no gas regression tracking |

---

## 2. Orion

### 2.1 Version and status

`ORION/Scarb.toml`: v0.2.5, **`cairo-version = "2.5.3"`**, `.tool-versions` = scarb 2.6.4, depends on
`cubit` at rev `6275608` and three alexandria crates at a 2024 rev. Last commit 2025-03 is a README
patch; the project is archived in practice (Giza moved away). MIT. It will not compile on a current
toolchain without a port (old `integer::u32_wide_mul` paths, `AddEq` traits, `impl TCopy: Copy<T>`
style everywhere).

### 2.2 Numbers module

- `ORION/src/numbers/fixed_point/implementations/`: `fp8x23` (`mag: u32`, ONE=2^23), `fp16x16`
  (`mag: u32`, ONE=2^16), `fp8x23wide`/`fp16x16wide` (`mag: u64`, same ONE, more integer headroom
  for ML accumulations), `fp32x32`, `fp64x64`.
- **All are sign-magnitude** and the math is a line-for-line copy of Cubit:
  `fp16x16/math/core.cairo:15-29` (`add`), `:184-189` (`mul` = `u32_wide_mul` then `/ ONE`
  then `try_into`), `:45-51` (`div`), `:266-272` (`sqrt` = `u64_sqrt(mag * ONE)`), same
  `exp2`/`log2` Horner polynomials rescaled, same `_sin_loop`, same LUT if-trees
  (`fp16x16/math/lut.cairo`, 1930 lines, plus an `erf_lut`).
- `fp32x32/core.cairo:3-6` and `fp64x64/core.cairo:3-6` simply alias Cubit:
  `use cubit::f64::Fixed as FP32x32;` / `use cubit::f128::types::Fixed as FP64x64;`.
- One notable difference: in the FP8x23/FP16x16 `FixedTrait` impl, `sin/cos/acos/atan...` dispatch
  to the **`_fast` LUT variants** (`fp16x16/core.cairo:57-59, 81-83, 97-99, 152-154`), i.e. the
  series versions are dead code for users.
- `FixedTrait<T, MAG>` (`ORION/src/numbers/fixed_point/core.cairo`, 1147 lines, mostly docs) is a
  *generic* trait over the FP type and its magnitude type, which is the one design improvement over
  Cubit.
- **Signed integers**: `ORION/src/numbers.cairo:1433+` implements `NumberTrait<i8, i8>`,
  `<i16>`, `<i32>`, `<i64>`, `<i128>` on the *native* signed ints plus hand-written `Div` impls
  (`I8Div` at `:1692`, written before corelib had signed division). All transcendental methods are
  `panic(array!['not supported!'])` (`numbers.cairo:2107-2151`).
- `NumberTrait<T, MAG>` (`ORION/src/numbers.cairo:11-67`) is a 55-method "god trait" (math, trig,
  hyperbolic, `NaN`/`INF` sentinels, bitwise ops, `where`, `min/max`, `add/sub`...) implemented
  12 times in a **3495-line file**, each impl a wall of one-line forwarders. Operators (`Mul`,
  `AddEq`, `PartialOrd`...) are *not* part of it, so every generic function repeats 6-10 impl
  bounds (see `matmul` below). Static dispatch means no runtime cost per se, but the forwarders are
  not `#[inline]`, so each call can add a frame, and compile time/code size explode.

### 2.3 Tensor design (`ORION/src/operators/tensor/core.cairo`)

```cairo
#[derive(Copy, Drop)]
struct Tensor<T> { shape: Span<usize>, data: Span<T> }   // :12-16
```

Dynamic rank, dynamic shape, flat row-major data, immutable (every op allocates a new `Array`).
Index helpers are all loops that allocate:

- `ravel_index` (`:5891-5910`): loop over dims popping both spans;
- `unravel_index` (`:5913-5935`): loop with **a div and a mod per dimension** + `Array::append`,
  preceded by `len_from_shape` (another loop);
- `stride` (`:5938-5952`): loop + `append` + `reverse()`.

### 2.4 Linalg-ish operators

- `ORION/src/operators/tensor/linalg/` contains only `matmul.cairo`, `transpose.cairo`,
  `trilu.cairo`. **There is no determinant, inverse, solve, LU, Cholesky, QR or eigen anything**
  on `main` (grep for `determinant|inverse|solve|cholesky` returns only unrelated hits).
- `matmul` (`linalg/matmul.cairo:5-43`): 1D/2D only. `matrix_multiply` (`:100-139`) is the naive
  triple `while` loop, computing `i * n + k` and `k * p + j` (two u32 mul+add, each range-checked)
  and doing two bounds-checked span indexings per inner iteration, `sum += a * b` through the
  sign-magnitude ops, then `result_data.append(sum)`. 1D inputs go through
  `prepare_shape_for_matmul`, which **rebuilds the shape array with a loop** just to add a `1`.
- `gemm` (`ORION/src/operators/nn/functional/gemm.cairo:7-53`): `transpose` (allocating copy) ->
  `matmul` -> `mul_by_scalar` (another full copy, even when `alpha == 1`) -> broadcasting `+`.
- Element-wise ops with broadcasting (`tensor/math/arithmetic.cairo:6-29`): for **every element**,
  `unravel_index(n, shape)` (loop with div/mod + allocation), then two
  `broadcast_index_mapping` calls, each of which recomputes `stride(shape)` (loop + alloc +
  reverse) (`tensor/helpers.cairo:104-122`). Adding two 3-vectors therefore performs O(10) array
  allocations and ~6 div/mods. This is the root of Orion's reputation.
- `transpose` N-D (`linalg/transpose.cairo:8-46`): per element `unravel_index` + `find_axis` loop +
  `ravel_index`. `transpose2D` is a plain double loop.
- `MutMatrix<T>` (`ORION/src/operators/matrix.cairo:4-8`) wraps `NullableVec<T>`
  (`ORION/src/operators/vec.cairo:7-10`) = `Felt252Dict<Nullable<T>>`: every `get`/`set` is a dict
  access (~10k+ gas with squashing) plus a `Box`. Only `matrix_vector_product`, `argmax`,
  `softmax`, `sigmoid` exist.

### 2.5 Anti-patterns to avoid (Orion)

1. Dynamic shapes for objects whose size is known at compile time (Vec3, Mat3, Isometry).
2. `Array::append` + `.span()` as the universal output mechanism; no fixed-size structs/tuples.
3. Per-element index arithmetic with div/mod and re-derived strides.
4. Generic code with 6-10 trait bounds per function and a 55-method number trait.
5. Non-inlined forwarding layers (`TensorTrait -> math::x -> NumberTrait -> core_math::x`).
6. `Felt252Dict` for mutable dense data.
7. Identity shortcuts that cost more than they save (`if val == NumberTrait::zero()`) and
   full-copy scalings by one.
8. Tests: 2014 ONNX node tests generated by a Python `nodegen` (`ORION/tests/nodes/`), good for
   conformance, useless for performance; `#[available_gas(2000000000)]` everywhere.

### 2.6 What is worth reusing

- **Algorithms**: nothing beyond what Cubit already has (it is the same code).
- **LUT data**: Cubit/Orion sin/atan tables are easily regenerated from Python; the *format*
  (if-trees returning `(start, low, high)`) should not be reused.
- **Test vectors**: Cubit's numeric expectations in `ops.cairo`/`trig.cairo`/`hyp.cairo` tests
  (e.g. `exp(2) = 31735754293`, `log2(10) = 14267572527`, `acos(0.5) = 4497679235`) are handy as
  cross-checks for a Q32.32 implementation since they share our scale. Orion's `nodegen` idea
  (generate Cairo tests from a NumPy reference) **is** worth copying: generate expected values from
  Rust `nalgebra`/f64 for every function, with the tolerance stated per test.
- **Design idea**: a *small* generic scalar trait (Orion's `FixedTrait<T, MAG>` minus the bloat) so
  that vector/matrix code is written once. Keep it to what `nalgebra`'s `RealField`/`ComplexField`
  subset actually needs, and keep operators as supertrait-like bounds bundled in one place.

---

## 3. Comparison table

| Aspect | Cubit f64 / f128 | Orion FP8x23 / FP16x16 (+wide) / FP32x32 / FP64x64 |
|---|---|---|
| Representation | `{mag: u64\|u128, sign: bool}` | same struct; `u32` (or `u64` for wide); 32x32/64x64 = Cubit re-export |
| Felts per scalar | 2 | 2 |
| Range / resolution | +-2^32 / 2.3e-10; +-2^64 / 5.4e-20 | +-256 / 1.2e-7; +-32768 / 1.5e-5; wide: more int bits |
| Negative zero | yes (unhandled in mul/div/round) | yes |
| mul/div cost driver | f64: felt mul + u128 divmod + downcast; f128: `u128_wide_mul` + **u256 div** | u32 wide mul + u64 divmod + downcast (same gas class as f64: width is free) |
| add/sub/cmp cost driver | sign branches (3-5) | same |
| sqrt | `u128_sqrt` libfunc (f64 exact; f128 loses 32 bits) | `u64_sqrt` libfunc |
| exp/ln | LUT int part + 8-term Horner; recursion + division for x<1 | same (7-8 terms) |
| Trig strategy | Taylor recursion w/ division per term; `_fast` = 256-slot if-tree LUT + lerp; atan deg-10 poly or 100-slot if-chain | same code; trait methods route to `_fast` |
| atan2 / inv_sqrt / sin_cos | no / no / no | no / no / no |
| Vector/matrix types | Vec2/3/4 structs, naive dot/cross/norm | dynamic `Tensor<T>`, `MutMatrix` on dict; matmul/gemm/transpose only |
| API ergonomics | operators + method trait, per-width duplication | generic `FixedTrait<T,MAG>` + 55-method `NumberTrait`; heavy bounds |
| Tests | ~140 inline unit tests, tolerance helpers, gas ceilings only | same FP tests + 2000 generated ONNX node tests; gas ceilings only |
| Gas tracking | none | none |
| Cairo / edition | >=2.7.0 / 2023_10 (local fork 2.9.2; builds on 2.19.4 with warnings) | pinned 2.5.3 / 2023_10; does not build on current toolchains |
| License | MIT | MIT |
| Maintenance | dormant (version bumps only) | abandoned |

---

## 4. Design space for the `nalgebra.cairo` scalar

### 4.0 Cost model (what actually costs gas in Cairo)

- felt252 `+`, `-`, `*`: 1 step, no builtin. `BoundedInt` add/sub/mul compile to exactly these.
- Any *integer-typed* result (`u64`, `i64`, `u128`...) needs >= 1 range-check to be produced from a
  wider value; **types narrower than 128 bits need more work, not less** (two range checks to bound
  both sides). Hence Q16.16 in `i32` is not cheaper than Q32.32 in `i64`.
- divmod: hint + verification (`q*d + r == n`, `r < d`): a few range checks. Division by a constant
  costs the same as by a variable.
- Anything over 128 bits (`u128_wide_mul`, u256 ops) is 3-10x a 128-bit op.
- Branches cost little in steps but force both arms to be compiled, block fusing, and
  sign-dependent branches cannot be predicted away by the prover: every executed instruction is
  proven.
- Calls: a non-inlined call costs ~1-2k gas of frame/implicit shuffling, comparable to the op
  itself: the hot scalar ops must be `#[inline(always)]` or exposed as fused kernels.
- Loops: each iteration pays gas-withdrawal + counter + jump; fixed dimensions must be unrolled.

### 4.1 (a) Sign-magnitude struct `{mag: u64, sign: bool}` (Cubit-style)

Pros: `abs`, `neg`, sign tests are free; mul/div avoid any sign normalisation (div is the cheapest
of all candidates: 3.75k **[measured]**); battle-tested code exists.
Cons: 2 felts per scalar (doubles calldata/Serde/memory traffic of every vector and matrix);
add/sub/cmp are the slowest of all candidates (3.65k / 2.8k **[measured]**) and physics is
add/sub-heavy; negative zero; **sums of products cannot be fused** because every partial product
carries its own sign, so `dot`, `cross`, `mat*vec`, `mat*mat` pay one rescale + downcast + branchy
add per term (dot3 = 20.5k **[measured]**). Rejected.

### 4.2 (b) Native signed integers (`i64` Q32.32)

Verified in `CORELIB/integer.cairo` (2.19.4; identical structure since at least 2.8.2):

- `i64` add/sub: `i64_overflowing_add_impl` libfunc, checked, panics on overflow (`:2231-2249`):
  2.07k gas **[measured]**.
- `i64` compare: `i64_diff` libfunc (`:2271-2283`): 1.6k **[measured]**.
- `i64_wide_mul(i64, i64) -> i128` is a `nopanic`, implicit-free libfunc (`:2262`): a felt mul.
  (`i128 * i128` on the contrary goes through `abs_and_sign` + `u128_wide_mul` + branches
  (`:2356-2381`): **i128-based Q64.64 has no cheap widening mul**; there is no `i256`.)
- **Signed `DivRem` exists** for i8..i128 (`mod signed_div_rem`, `:2400-2556`), so `i128 / i128`
  works, but it is implemented as `constrain::<T, 0>` on both operands (4-way branch), negate,
  unsigned `bounded_int::div_rem`, negate back; it **truncates toward zero** and returns a
  remainder with the sign of the dividend.
- Naive Q32.32 mul `(wide_mul(a, b) / ONE_i128).try_into().unwrap()`: 4.3k **[measured]**, *slower*
  than sign-magnitude because of the signed division by a constant. Naive div: 6.8k **[measured]**.

So native `i64` is the right *storage* type (1 felt, native Serde/Store, free
`Into<felt252>`, cheap add/sub/cmp, overflow-checked) but corelib's signed `/` is the wrong
rescaling tool.

**(b') `i64` storage + `BoundedInt` kernels.** `core::internal::bounded_int` (unstable, behind
`#[feature("bounded-int-utils")]`, used by corelib itself and by garaga) accepts native ints as
operands. A product of two `i64` is a `BoundedInt<-(2^126 - 2^63), 2^126>` obtained with a single
felt mul and no range check. Adding the constant `2^126` makes it non-negative, one `div_rem` by
`UnitInt<2^32>` gives the floor quotient, subtracting `2^94` restores the sign, and one `downcast`
to `i64` is the overflow check. No branch at all (`BENCH/src/i64b.cairo`):

```cairo
pub fn mul(a: i64, b: i64) -> i64 {
    let p: Prod = bounded_int::mul(a, b);                       // felt mul
    let s: Shifted = bounded_int::add(p, 0x4000...0);           // + 2^126  (>= 0)
    let (q, _r) = bounded_int::div_rem(s, 0x100000000);         // floor(/ 2^32)
    let r: Res = bounded_int::sub(q, 0x400000000000000000000000); // - 2^94
    downcast(r).unwrap()                                        // single overflow check
}
```

Measured: **3.3k gas** (best checked mul). More importantly sums of products stay in the wide
domain: `dot3` = 3 felt muls + 2 felt adds + **one** rescale + **one** downcast = **4.1k gas vs
20.5k (sign-magnitude) and 19.3k (naive i64)** **[measured]**. `norm3` computed as
`u128_sqrt(x*x + y*y + z*z)` on the *raw* Q64.64 sum needs **no rescale at all** (sqrt of Q64.64 is
Q32.32) and cannot overflow: ~5k vs ~25k **[measured]**. The same trick applies to `cross`, `mat3 *
vec3` (3 rescales instead of 9), `mat3 * mat3` (9 instead of 27), quaternion products, and
`a*b + c` (mul-add).

Rounding semantics: floor (arithmetic shift), like every fixed-point C library, and unlike Cubit
(truncation). Floor has a consistent -0.5 LSB bias regardless of sign, which is preferable for
integration (no dead-zone around zero), and it is what the offset trick gives for free.

Division has no such trick (the divisor is variable and signed): `|a| * 2^32 / |b|` with two
`constrain` branches costs 5.1k **[measured]** (`BENCH/src/extra.cairo: i64b_div`) vs 3.75k for
sign-magnitude. Acceptable: divisions are rare in a solver if inverse masses/inertias are stored,
and `v / s` on vectors should be `inv = 1/s` once then fused muls.

Limits found empirically: `downcast` refuses source ranges >= 2^128
("Provided generic argument is unsupported", see `BENCH/w96.cairo.failed-experiment`), so the
value *after* rescale must span < 2^128. With ONE = 2^32 that allows sums of up to ~2^33 products
of i64 pairs: never a constraint in practice. Formats whose single product exceeds ~2^160 (e.g.
96-bit Q64.32) are not expressible this way.

Risk: `bounded_int` is an internal, feature-gated API; helper-trait impls must state exact bounds
(a wrong bound is a Sierra specialisation panic, not a soundness hole). It has been stable in shape
from 2.8 to 2.19. Mitigation: confine it to one `kernels` module with a pure-`i64`/`i128` fallback
behind a cfg/feature and identical tests.

### 4.3 (c) Raw `felt252` representation

Value `v` stored as `v mod P`, negatives as `P - |v|`. add/sub/neg/mul-by-integer are 1 step with
**zero range checks** (add measured at +0 gas over the call baseline). But:

- Every mul needs the rescale, which needs an integer: `(a*b + 2^126).try_into::<u128>()`,
  divmod, back to felt: 2.9k **[measured]**, only ~0.4k less than (b') because (b') does the very
  same thing minus the type erasure.
- Nothing bounds the value: after k unchecked adds/muls the "integer" may exceed the offset window
  and the next rescale silently wraps or panics far from the bug. Adding the missing range check
  per mul costs *more* than (b'): 4.5k with a u128-window check, 6.0k with `felt252 -> i64`
  **[measured]** (felt-to-small-int conversion is expensive).
- Comparison requires offset + u128 conversion: 3.5k **[measured]**, the worst of all candidates
  (2.2x `i64 <`). Collision detection, clamping, and `max/min` are comparison-heavy.
- `Serde`/storage accept any felt: no validation at the ABI boundary, which for a *provable* engine
  is a soundness issue (a malicious input of `2^200` is a valid `Fixed`).
- (b') already gets the "lazy felt arithmetic" benefit *inside* fused kernels, where the bounds are
  tracked statically by the type system instead of by convention.

Verdict: do not use felt252 as the public scalar. Use felt/BoundedInt accumulation *inside* kernels.

### 4.4 (d) Precision choice

Cost is flat for any format whose storage is <= 64 bits and product <= ~127 bits, so choose on
numerics alone:

| Format (storage) | Range | Resolution | Notes |
|---|---|---|---|
| Q16.16 (i32) | +-32768 | 1.5e-5 | `dt = 1/60` has 5e-4 relative error; `dt^2`, small inverse inertia, angular increments underflow; squared length overflows at |v| > 181. Not cheaper. Rejected. |
| Q48.16 (i64) | +-1.4e14 | 1.5e-5 | same precision problems as Q16.16 |
| **Q32.32 (i64)** | +-2.1e9 | 2.3e-10 | recommended |
| Q24.40 (i64) | +-8.4e6 | 9.1e-13 | squared length overflows at |v| > 2896; only if worlds are small |
| Q64.64 (i128/u128) | +-9.2e18 | 5.4e-20 | product is 256-bit: mul ~10-11k gas (3x), no fused kernels, no `i256`. Rejected as default. |

Q32.32 checks for game physics:

- `dt = 1/60` -> raw 71582788, relative representation error 1.2e-8. Velocity `v*dt` with
  v = 1e-3 m/s still has 5 significant digits. Gravity step `g*dt*dt` = 2.7e-3, fine.
- Floor bias: each mul loses on average 0.5 LSB = 1.2e-10. A body integrated for 1 hour at 60 Hz
  (216k steps, ~10 muls on the position path per step) drifts by ~2.5e-4 units worst case if all
  biases align: invisible in a game, and *identical on every machine*.
- Inverse mass/inertia: for m in [1e-3, 1e6] both `m` and `1/m` are representable with >= 3 digits;
  inertia of a 5 cm, 100 g sphere (1e-4) -> inverse 1e4, fine. Extremely light/small bodies need a
  documented lower bound (or unit scaling), as in any fixed-point engine.
- Dot/cross/squared length: the *result* must fit +-2.1e9, i.e. |v| <= 46340 units for
  `length_squared`. Inside fused kernels intermediates are ~128-bit so there is no *intermediate*
  overflow, and `norm`/`distance` use the raw-sum sqrt and are overflow-free for any representable
  vector. Provide `length_squared` as checked (panics on overflow) and document preferring
  `norm`/comparisons in the wide domain (`norm_squared_lt(v, r)` comparing raw sums).
- Angular quantities: angles in radians within +-pi, quaternions with components in [-1, 1] get
  the full 32 fractional bits: renormalisation error ~1e-9 per step, renormalise each step.
- Determinism: everything is exact integer arithmetic with specified floor rounding, no hints
  whose result is ambiguous (`divmod`/`sqrt` hints are fully constrained), so results are
  bit-identical across provers/sequencers. Two rules to enforce: (1) fused and unfused variants
  round differently (one floor vs three), so each public function must have **one** canonical
  implementation, and tests must pin exact raw outputs; (2) overflow must panic (never wrap),
  which `downcast(...).unwrap()` guarantees.

### 4.5 Recommendation

**Primary: `Fixed` = Q32.32, newtype over native `i64`.**

```cairo
#[derive(Copy, Drop, Serde, PartialEq, Debug, Hash, starknet::Store)]
pub struct Fixed { raw: i64 }          // 1 felt; struct wrapper is zero-cost in Sierra
pub const ONE: i64 = 0x100000000;
```

- add/sub/neg/cmp: native `i64` ops (checked), `#[inline(always)]`.
- mul, mul_add, dot2/3/4, cross, norm, mat*vec, mat*mat, quat*quat: `BoundedInt` wide kernels,
  single floor-rescale via offset, single `downcast`.
- div: abs/sign via `constrain`, unsigned `bounded_int::div_rem`; provide `recip()` and encourage
  multiplication by stored inverses.
- sqrt: `u128_sqrt(raw << 32)` (exact floor); norms via raw sum of squares.
- A slim `Scalar`/`RealField`-like trait (constants, `sqrt`, `sin_cos`, `atan2`, `acos`, `abs`,
  `min/max`, `floor/ceil/round`, `recip`, fused `mul_add`) + core operator traits, so Vector/Matrix
  code is generic and a second scalar can be plugged in.
- Validate at the boundary for free: `i64` Serde already rejects out-of-range felts.

**Fallback (if `bounded-int-utils` becomes unavailable or is judged too risky):** same `i64`
storage and public API, kernels implemented with `u128` arithmetic on offset values
(`(a_felt * b_felt + 2^126).try_into::<u128>()`, `u128` divmod, subtract `2^94`, convert to
`i64`), i.e. variant (c)'s mul *inside* a checked `i64` API. Expected ~4.5-5k per mul and ~6k per
dot3: still 3x better than Cubit on fused ops. Last resort: corelib `i64_wide_mul` + signed `i128`
division (4.3k/mul, truncation semantics: would change rounding, so decide before 1.0).

**Second scalar to keep possible, not to build now:** Q64.64 sign-less `i128` for a high-precision
mode, accepting ~3x mul cost.

### 4.6 Micro-benchmarks that must be (re)run with snforge to settle the design

Each as a gas-tracked unit test, N-iteration loop minus baseline, inputs with mixed signs, in the
real crate with final inlining attributes:

1. `add`, `sub`, `neg`, `lt`, `max` for: sign-magnitude, native `i64`, BoundedInt+downcast, felt.
2. `mul`: Cubit-style; `i64_wide_mul` + signed `/`; BoundedInt offset-floor; u128 offset-floor;
   felt unchecked / checked. Also with `#[inline(always)]` vs `#[inline(never)]` vs default, and
   with the `Fixed` newtype vs bare `i64` (confirm zero wrapper cost).
3. `div` and `recip`: sign-magnitude; signed corelib; BoundedInt with `constrain`; variant that
   branches once on `sign(a) ^ sign(b)`.
4. Fused vs unfused: `dot2/3/4`, `cross3`, `mul_add`, `mat2/3/4 * vec`, `mat3 * mat3`,
   `mat4 * mat4`, quaternion product, `norm`, `normalize` (1 sqrt + 1 recip + fused muls vs
   3 divisions).
5. `sqrt`: `u128_sqrt(raw << 32)` vs `u64_sqrt` + correction vs unrolled Newton (expected to lose).
   `inv_sqrt`: `sqrt` + `recip` vs unrolled Newton (expected 8k vs ~30k).
6. Table lookup styles for 64/256 entries: nested `if` tree (4.4k for 64 entries **[measured]**),
   const fixed-size array `.span()[i]` (2.9k), `match` on a felt/enum jump table (**1.8k**), and
   packing several entries per felt + divmod extraction.
7. `sin`, `cos`, `sin_cos`, `atan2`, `acos`, `asin`: polynomial degrees (5/7/9/11), LUT+lerp, and
   CORDIC-16/24 unrolled, each with max-error measurement against generated reference vectors.
8. Struct passing: `Vec3` by value vs snapshot, tuple returns vs struct returns, `Mat3` as 9 fields
   vs 3 `Vec3` columns (Sierra may shuffle more felts for nested structs).
9. Storage/Serde round-trip cost of `Fixed`, `Vec3`, `Mat3` (1 felt per scalar vs packed).
10. Cubit baselines for the same functions (already measured here: sin 130k, sin_fast 30k, atan
    84k, atan_fast 57k, acos 115k, exp 65k, ln 74k, sqrt 4.8k, mul 5.0k incl. input generation).

Summary of first-round measurements (l2_gas per op, scarb 2.19.4 / snforge 0.61.0):

| Op | sign-mag (Cubit-style) | native i64 | i64 + BoundedInt | felt252 |
|---|---|---|---|---|
| add | 3650 | 2070 | 2070 | ~0 (unchecked) |
| lt | 2795 | 1575 | (= native) | 3515 |
| mul | 3750 | 4320 | **3280** | 2910 unchecked / 4490-6020 checked |
| div | **3750** | 6760 | 5140 | n/a |
| dot3 | 20550 | 19300 | **4080 (fused)** | n/a |
| norm3 | ~25700 | n/a | **~5260 (fused, no rescale)** | n/a |
| Q64.64 mul | 11180 (u256 div) / 10380 (hi/lo) | n/a | not expressible | n/a |

---

## 5. Algorithm recommendations (loop-averse, add/mul/divmod first)

Notation: `M` = one fixed mul (~3.3k gas), `D` = one fixed div (~5k), `R` = range reduction
(divmod + a couple of branches, ~3-4k). Errors are absolute, from a near-minimax fit
(Lawson-weighted least squares) computed for this report; Q32.32 LSB = 2.3e-10. All polynomials
are evaluated by **unrolled Horner in `u = x*x`** inside one wide kernel where possible (keep
intermediate products unscaled when the bound allows, rescale once per Horner step at most).

### 5.1 sqrt

Use `core::num::traits::Sqrt` on `u128`: `sqrt(Fixed{raw}) = u128_sqrt((raw as u128) << 32)`.
Exact floor of the true root at full 32-bit fractional precision, ~4.5k gas, no loop (hint +
in-circuit verification). For vector norms use `u128_sqrt(sum of raw squares)` directly (no shift).
For sums exceeding u128 (|v| near the top of the range with 3+ components: 3 * 2^126 < 2^128 is
still fine; 4D needs care) fall back to `u256_sqrt` or pre-shift by 2 bits.
Newton/bit-by-bit methods are strictly worse here: do not implement them.

### 5.2 inverse sqrt / normalize

`inv_sqrt(x) = recip(sqrt(x))`: sqrt (4.5k) + one division (5k) ~ 9.5k, exact to 1 LSB.
Better precision for small x: compute `2^64 / u128_sqrt(raw << 32)` in one integer division (the
numerator constant is ONE^2), i.e. skip building the intermediate `Fixed`.
Unrolled Newton (`y <- y * (1.5 - 0.5 * x * y * y)`, 3M per iteration) from a linear initial guess
on the normalised mantissa needs 3 iterations for 3e-7 and 4 for ~1e-13 relative error (measured:
1.1e-1 -> 1.8e-2 -> 4.7e-4 -> 3.4e-7), plus an msb/normalisation step: >= 30k gas. Rejected; keep
only as a benchmark entry. `normalize(v)`: raw-sum norm, one reciprocal, one fused scale
(3 felt muls, 3 rescales).

### 5.3 sin / cos

1. Range reduction: `(k, r) = divmod(raw + offset, HALF_PI_raw)` on a non-negative shifted value
   (offset = multiple of 2*pi, keeps it branch-free), quadrant `k & 3` via a second tiny divmod by
   4 (divmod preferred over bitwise). Use symmetry so the polynomial argument is in [0, pi/2] (or
   [-pi/4, pi/4] with separate sin and cos polynomials, which halves the degree needed).
2. Polynomial, odd in x, near-minimax on [0, pi/2]:

   | degree | muls (incl. `x*x` and final `x*`) | max error |
   |---|---|---|
   | 5 | 4 | 6.8e-5 |
   | 7 | 5 | 5.9e-7 |
   | **9** | **6** | **3.3e-9** |
   | 11 | 7 | 1.3e-11 (below what Q32.32 rounding allows) |

   Recommended default: degree 9 (~R + 6M ~ 23k gas, vs Cubit 130k / 30k at 4.7e-6). Provide
   `sin_cos(x)` returning both from a single reduction (rotations always need both): ~R + 11M ~ 40k.
   With the [-pi/4, pi/4] split: sin deg 7 + cos deg 6 reach ~1e-9 with 4+4 muls.
3. LUT + linear interpolation is *not* competitive: 256 entries give only 4.7e-6 (64 entries:
   7.5e-5) and the lookup itself costs 1.8k (match) to 4.4k (if-tree) plus 2 muls; a cubic
   (Hermite, storing sin and using cos as derivative) on 64 entries reaches ~1e-9 but costs as much
   as the degree-9 polynomial. If a LUT is kept for benchmarking, implement it as a `match`
   jump table or const array span, never as an if-tree, and multiply by the precomputed inverse
   slot width instead of dividing.
4. CORDIC (24-32 unrolled iterations of shift-add) needs a divmod (shift) x2 and a branch per
   iteration: ~32 x 3 range-checked ops ~ 60-100k gas for 1e-9. Shifts are divmods in Cairo, which
   removes CORDIC's raison d'etre. Rejected; benchmark once for the record.

### 5.4 atan / atan2

`atan2(y, x)`:
1. Handle zero cases; work with `|x|`, `|y|` (2 `constrain`/abs), remember the quadrant.
2. `t = min/max` ratio in [0, 1]: **one division** (`D`), swap flag (octant) via one comparison.
3. Polynomial on [0, 1], odd, near-minimax: deg 9 -> 1.1e-5, deg 11 -> 1.7e-6, deg 13 -> 2.5e-7,
   deg 15 -> 3.8e-8 (9 muls). Better: one extra range split at `tan(pi/12)` using
   `t' = (t*sqrt3 - 1)/(t + sqrt3)` costs one more division but lets a **degree-7 polynomial
   reach 2.6e-9** (deg 5: 2.0e-7) on [0, tan(pi/12)]. Two candidates to benchmark:
   - A: `D` + deg-13 poly (8M): ~31k gas, 2.5e-7;
   - B: `D` + `D` + deg-7 poly (5M): ~27k gas, 2.6e-9.  <- expected winner
4. Octant/quadrant fix-up with constant adds/subs (`pi/2 - a`, `pi - a`, negate).
Compare with Cubit: 84-95k for 1e-5. `atan(x)` = `atan2(x, ONE)` sharing the kernel.

### 5.5 acos / asin

Avoid Cubit's `atan(x / sqrt(1 - x^2))` chain (2 sqrt + div + deg-10 poly). Use the
Abramowitz-Stegun form, accurate up to the endpoint singularity:

`acos(x) = sqrt(1 - x) * P(x)` for x in [0, 1], `acos(-x) = pi - acos(x)`, `asin = pi/2 - acos`.

| deg P | cost | max error |
|---|---|---|
| 3 | sqrt + 4M | 1.4e-4 |
| 5 | sqrt + 6M | 2.8e-6 |
| **7** | **sqrt + 8M ~ 31k gas** | **6.3e-8** |

Clamp the input to [-1, 1] (dot products of unit vectors routinely give 1 + 1 LSB) instead of
panicking like Cubit. If 1e-9 is required: for |x| <= 0.5 use an `asin` odd polynomial (deg 11),
otherwise `acos(x) = 2 * asin(sqrt((1 - x)/2))` (1 sqrt, same polynomial): ~35k.
For physics, prefer formulations that avoid `acos` entirely (`atan2(|a x b|, a . b)` for angles
between vectors: better conditioned near 0 and pi, and reuses the atan2 kernel).

### 5.6 exp / ln / pow (low priority for physics)

Keep Cubit's structure (`2^int * P(frac)`; `msb + P(mantissa)`) but: msb via a `match`/const table
on a divmod-normalised value instead of a 64-leaf if-tree; no recursion for x < 1 (normalise with
a negative exponent instead of dividing); fold `exp` for negative x as `2^-k * P(f)` (shift
= divmod by table constant) rather than `1 / exp(|x|)`.

### 5.7 Testing/gas-tracking conventions (derived from the gaps found)

- Every function ships with (1) exact-raw-value tests on hand-picked points (determinism pins),
  (2) reference-vector tests generated from f64 (`nodegen`-style script), asserting a documented
  max error, (3) edge cases (0, +-ONE, MIN, MAX, overflow panics, domain errors), and (4) a
  **gas snapshot test**: run via `snforge test` and compare `l2_gas` to a checked-in baseline
  (scarb script or CI diff), so alternative implementations (`sin_poly9` vs `sin_lut256`) can live
  side by side under a `bench` feature and be compared, which neither Cubit nor Orion can do.
- Benchmarks must feed **mixed-sign, non-constant inputs** through `#[inline(never)]` wrappers and
  subtract a baseline loop, as done in `BENCH/src/tests.cairo`.
