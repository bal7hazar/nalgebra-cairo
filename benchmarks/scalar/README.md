# Scalar representation benchmarks

Which fixed-point scalar should `nalgebra.cairo` (linear algebra for a provable game physics
engine) be built on? Cairo has no floats; this package implements the candidate representations
with identical semantics, measures every basic operation in Sierra gas, then benchmarks algorithm
alternatives (sqrt, inverse sqrt / normalize, sin/cos, atan2, acos) on the winner.

Toolchain: scarb 2.19.4, snforge 0.61.0. Full ranked tables: [`GAS.md`](GAS.md) (407 tests).

**TL;DR** — store a native `i64` in **Q32.32**, do the arithmetic with
`core::internal::bounded_int` (branch-free *floor* multiply, 1,850 gas), keep products **unscaled**
and rescale **once** per dot/cross/matrix row (`dot3`: 2,250 gas vs 10,570 for the plain `i64`
version and 10,310 for cubit-style sign-magnitude), use the corelib integer sqrt and generated
*typed* Horner polynomials for trigonometry (sin/cos: 12.5k gas at 4.7e-10 max error).

## Methodology

- One snforge test per measurement, named `bench_<group>__<variant>`; `scripts/gas_report.py`
  subtracts the group's `baseline` test. Sierra gas is deterministic: one call per test.
- Inputs **and expected values** go through `harness::black_box` (`#[inline(never)]` identity), so
  nothing is const-folded. Every benchmark asserts its result, so it is also a correctness test.
  The baseline has the same inputs and the same equality assertion on the same type
  (`assert!(e == e)`), only the operation is missing.
- Groups are `<op>_<storage>` (`mul_i64`, `dot3_felt`, ...): the fixed overhead of a test depends
  on the storage type (a `{u64, bool}` struct does not cost the same to move/compare as an `i64`),
  so each storage gets its own baseline. `summary.py` pivots the nets into the
  operation x representation table below.
- Every `bench_*` function is `#[inline(never)]`: snforge otherwise inlines some test bodies in
  its wrapper and not others, which shifts `raw - baseline` by ~1,000 gas.
- The first use of an operation in a function pays a little for the implicit/panic plumbing, so
  the table also reports a **marginal** cost: `(op chained 4 times - op once) / 3`.
- Operators of every representation are `#[inline(always)]`; compositions (`dot3`, `cross3`,
  `lerp`, `mul_add`) exist in a generic form (`src/generic.cairo`, written against `Add/Sub/Mul`
  like a generic library would) and, where the representation allows it, in a fused form.
- Signs: `pp`/`pn`/`nn` = (+,+) / (+,-) / (-,-) operands. Values are exactly representable in all
  formats (3.5, 2.25, ...) except for `div`, whose expected value follows the rounding mode.
- Tests are generated (`gen_tests.py`, `gen_algos.py`) so that all representations run the exact
  same battery. For the algorithms, expected values are the **bit-exact** outputs of Python integer
  models that mirror the Cairo code; the same models are swept over 4k-20k points against double
  precision math to get the max-error figures. A separate test checks the Cairo results against
  the *true* values with tolerances, and `src/tests/semantics.cairo` pins rounding modes,
  overflow panics and the range behaviour of fused forms.

```sh
cd benchmarks/scalar && python3 gen_tests.py && python3 gen_algos.py   # regenerate (+ accuracy table)
cd .. && scarb fmt -p scalar
snforge test -p scalar 2>&1 | tee /tmp/scalar.txt | python3 scripts/gas_report.py --md scalar/GAS.md
python3 scalar/summary.py < /tmp/scalar.txt                            # pivot table below
```

## Candidates

| | module | storage | format | mul | rounding | overflow |
|---|---|---|---|---|---|---|
| A | `signmag64` | `{ mag: u64, sign: bool }` | Q32.32 | `u64` wide mul -> `u128 / 2^32` (cubit `f64`) | trunc | panic |
| B | `i64q32` | `i64` | Q32.32 | `i64` wide mul -> corelib signed `i128 / 2^32` | trunc | panic |
| C | `i128q64` | `i128` | Q64.64 | magnitudes -> `u128` wide mul -> `u256`, recombined by math; also a BoundedInt limb version (`mul_bounded`) | trunc / floor | panic |
| D | `felt_fixed` | `felt252` (negatives as `P - x`) | Q32.32 | felt mul, `+2^127`, felt->`u128` range check, `div_rem 2^32` | floor | add/sub/neg **unchecked**, mul checked |
| E | `i32q16` | `i32` | Q16.16 | same as B one size down | trunc | panic |
| E' | `signmag32` | `{ mag: u32, sign: bool }` | Q16.16 | orion `FP16x16` style | trunc | panic |
| G | `bi64` | `i64` | Q32.32 | `bounded_int_mul` (1 felt mul, no range check) `+2^128`, `div_rem` by the constant `2^32`, `-2^96`, **one** `downcast` to `i64` | floor | panic |

F (biased unsigned, `u64` with bias 2^63) was not implemented: the bias only buys a cheaper
comparison (an `i64` `<` is already 870 gas), its add needs the same two-sided range check as the
`i64` add, and its mul needs exactly the BoundedInt machinery of G on top of removing two biases.
It cannot beat G.

Q64.64 **cannot** live in a `felt252` the way D does: products reach 2^254 > P ~ 2^251.

`core::internal::bounded_int` is usable in 2.19.4 behind `#[feature("bounded-int-utils")]`
(helper traits `AddHelper`/`SubHelper`/`MulHelper`/`DivRemHelper`/`ConstrainHelper` are public and
user-implementable; result types must be the *exact* interval, see the Python snippets in the
generators). Limits met: `downcast` refuses source ranges wider than 2^128, `div_rem` needs a
quotient < 2^128 and a non-negative dividend type (hence the constant positive offsets), and
`core::zeroable::IsZeroResult` is crate-private (the libfunc `bounded_int_is_zero` can be
re-declared with a local enum, see `algo/trig.cairo`).

## Results: representation x operation (net Sierra gas)

`G fused` / `D lazy` = products kept unscaled, a single rescale. `D soft` = mul without the final
"fits in i64" check, `D plain` = same algorithm without BoundedInt. `G (bi add)` = add/sub/neg/abs
done through BoundedInt instead of the native checked `i64` ops (no gain: the cost *is* the range
check). The `-200` of `abs` for sign-magnitude is an artefact (the compiler knows `sign == false`
in the comparison); read it as ~0.

| op | A SignMag64 | B I64Q32 | G BI64 | G (bi add) | G fused | C I128Q64 | D FeltQ32 | D lazy | D soft | D plain | E I32Q16 | E' SignMag32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `add` pp | 1800 | 640 | 640 | 640 |  | 370 | 100 |  |  |  | 640 | 1800 |
| `add` pn | 2440 | 640 | 640 | 640 |  | 370 | 100 |  |  |  | 640 | 2440 |
| `add` nn | 1620 | 640 | 640 | 640 |  | 370 | 100 |  |  |  | 640 | 1620 |
| `sub` pp | 2840 | 640 | 640 | 640 |  | 370 | 100 |  |  |  | 640 | 2840 |
| `sub` pn | 2200 | 640 | 640 | 640 |  | 370 | 100 |  |  |  | 640 | 2200 |
| `neg` p | 600 | 300 | 300 | 370 |  | 300 | 100 |  |  |  | 300 | 600 |
| `abs` p | -200 | 890 | 890 | 710 |  | 890 | 1910 |  |  |  | 890 | -200 |
| `abs` n | -200 | 1070 | 1070 | 1140 |  | 1070 | 1590 |  |  |  | 1070 | -200 |
| `mul` pp | 2250 | 2910 | 1850 |  |  | 8430 | 2120 |  | 1480 | 2590 | 2440 | 1780 |
| `mul` pn | 2260 | 2990 | 1850 |  |  | 8690 | 2120 |  | 1480 | 2590 | 2520 | 1790 |
| `mul` nn | 2350 | 2910 | 1850 |  |  | 8970 | 2120 |  | 1480 | 2590 | 2440 | 1880 |
| `div` pp | 2250 | 5330 | 3570 |  |  | 13500 | 4850 |  |  |  | 4860 | 1780 |
| `div` pn | 2260 | 5330 | 3640 |  |  | 13760 | 4920 |  |  |  | 4860 | 1790 |
| `lt` pp | 2180 | 870 | 870 |  |  | 870 | 1790 |  |  |  | 870 | 2180 |
| `lt` np | 1080 | 870 | 870 |  |  | 870 | 1790 |  |  |  | 870 | 1080 |
| `lt` nn | 2080 | 870 | 870 |  |  | 870 | 1790 |  |  |  | 870 | 2080 |
| `eq` p | 900 | 500 | 500 |  |  | 500 | 500 |  |  |  | 500 | 900 |
| `from_int` n | 2410 | 640 | 100 |  |  | 370 | 100 |  |  |  | 640 | 2410 |
| `to_int` p | 2280 | 3430 | 1110 |  |  | 3900 | 1650 |  |  |  | 3430 | 2280 |
| `to_int` n | 2650 | 3690 | 1110 |  |  | 4160 | 1650 |  |  |  | 3690 | 2650 |
| `sqrt` p | 1180 | 1820 | 1920 |  |  | 7360 | 1720 |  |  |  | 1820 | 1180 |
| `dot3` mix | 10310 | 10570 | 7230 |  | 2250 | 26950 | 6760 | 2520 |  |  | 9160 | 8900 |
| `cross3` mix | 19960 | 20640 | 14120 |  | 6650 | 53850 | 13520 | 7160 |  |  | 17820 | 17140 |
| `lerp` mix | 7140 | 4470 | 3330 |  | 2150 | 9630 | 2320 | 2420 |  |  | 4000 | 6670 |
| `mul_add` mix | 4500 | 3730 | 2590 |  | 2050 | 9160 | 2220 | 2320 |  |  | 3260 | 4030 |
| `length3` mix | 10830 | 12330 | 9250 |  | 2220 | 34430 | 8580 |  |  |  | 10920 | 9420 |
| `add4` pn | 7610 | 2860 | 2860 |  |  | 1780 | 400 |  |  |  | 2860 | 7610 |
| `mul4` pn | 8700 | 12100 | 7700 |  |  | 35620 | 8780 |  |  |  | 10220 | 6820 |
| `add` marginal (pn) | 1723 | 740 | 740 |  |  | 470 | 100 |  |  |  | 740 | 1723 |
| `mul` marginal (pn) | 2147 | 3037 | 1950 |  |  | 8977 | 2220 |  |  |  | 2567 | 1677 |

What the numbers say:

1. **`i64` storage + BoundedInt kernels (G) wins or ties everywhere except `add`/`sub`** (640 vs
   100 for raw felt adds) and `div`/`abs`/`sqrt` where sign-magnitude gets the sign for free.
   A multiply is +15 steps / +5 range checks for G versus +23 / +7 for B; it is sign-agnostic
   (same cost for `pp`, `pn`, `nn`), whereas A/B/C vary by up to 6%.
2. **Fusing is worth far more than the choice of representation**: `dot3` 2,250 (fused) vs 7,230
   (same representation, three rescales + two checked adds); `cross3` 6,650 vs 14,120; `length3`
   2,220 vs 9,250 - the unscaled sum of squares is exactly what the integer sqrt wants, so
   `length3` needs **no rescale at all**. Two extra products in a fused sum cost ~200 gas each.
   A Horner step `z * acc + c` fused is 2,050 vs 2,590.
3. **Sign-magnitude (cubit/orion style) is the most expensive design for linear algebra**: its
   add/sub is a 3-way branch (1,620-2,840 vs 640), it cannot accumulate unscaled products, and
   `dot3` ends up 4.6x more expensive than G fused. Its only wins are `div`, `abs` and `neg`.
4. **Native `i128` Q64.64 is 3-5x more expensive** on mul (8,430-8,970), div (13.5k), sqrt (7,360,
   `u256_sqrt`), dot3 (26,950). The BoundedInt limb multiply halves that (4,500; lazy dot3 13,160)
   but remains 2.4x / 5.8x the Q32.32 numbers. Its add is the cheapest checked add (370: one range
   check instead of two).
5. **`felt252` storage (D) is the gas runner-up**: add/sub/neg are 100 gas, the lazy dot3 is 2,520.
   But `lt` (1,790) and `abs` (1,590-1,910) need a range check to find the sign, `div` must convert
   both operands back to `i64`, and add/sub overflow is undetected.
6. **Q16.16 on `i32` saves nothing** (mul 2,440-2,520 native; a BoundedInt version would cost the
   same 1,850 as G since everything happens in a felt anyway) and its 1.5e-5 resolution is
   unusable for physics.
7. `from_int` (100) and floor `to_int` (1,110, branch-free, cannot fail) are nearly free with
   BoundedInt; the native versions cost 640 and 3,430-3,690.
8. Rounding: the branch-free **floor** multiply is 1,850; a **truncating** BoundedInt multiply needs
   a sign split of the product and costs 2,440-2,520 (`mul_i64` group in `GAS.md`). Checking the
   overflow before the division instead of after saves 100 gas (`bounded_precheck`, 1,750) but
   only works for a single product, so the generic `rescale` of a `Wide` accumulator was kept.

## Results: algorithms (on G, `BI64`)

| sqrt (`sqrt_algo`) | net gas | note |
|---|---:|---|
| corelib `u128_sqrt` on `raw << 32` (shift = free bounded mul) | **1,920** | exact floor, prover-guessed root verified with a few range checks |
| Newton, power-of-two seed from a compare tree, 6 unrolled iterations | 31,140 | exact floor |
| via multiplication-only inverse sqrt Newton (6 iterations) | 50,460 | error up to 2e-4 for large inputs |
| Newton loop until convergence (seed n/2+1) | 165k-204k | data dependent |

| inverse sqrt / normalize | net gas | note |
|---|---:|---|
| `inv_sqrt`: sqrt then `2^64 / s` as one unsigned bounded `div_rem` | **2,930** | |
| `inv_sqrt`: sqrt then generic signed division | 4,410 | |
| `inv_sqrt`: multiplication-only Newton | 48,510 | |
| `normalize3`: fused length, `2^96 / len` (64 fractional bits), 3 wide multiplies | **9,750** | ~1 ulp whatever the length |
| `normalize3`: fused length, Q32.32 inverse, 3 multiplies | 10,120 | loses precision for long vectors (|v| = 1e5 -> 4-5 significant digits) |
| `normalize3`: fused length, 3 floor divisions | 13,700 | 1 ulp |

| sin (`sin_algo`), any angle, range reduction by one `div_rem` by pi/2 | net gas | max abs error |
|---|---:|---:|
| degree 9, **typed Horner** (generated, BoundedInt end to end, 2^-64 internal precision) | **11,190** | 6.6e-9 |
| degree 11, typed Horner (same for `cos`: 12,600) | **12,500** | **4.7e-10** (2 ulp) |
| LUT 64 or 256 entries as const array + linear interpolation | 12,180 | 7.5e-5 / 4.7e-6 |
| LUT 64 entries as `match` + linear interpolation | 13,460 | 7.5e-5 |
| degree 7 / 9 / 11, Horner with fused `mul_add` on `BI64` | 15,080 / 17,130 / 19,180 | 1.2e-6 / 1.2e-8 / 3.1e-8 |
| degree 9, Horner with plain `*` and `+` | 20,090 | 1.2e-8 |
| cubit-style Taylor loop with integer divisions (7 terms) | 82,560 | 4.7e-10 |
| CORDIC, 20 unrolled iterations (shifts are signed divisions) | 138,640 | 1.9e-6 |

The typed Horner kernels (`src/algo/poly_typed.cairo`, generated by `gen_algos.py`) keep the
accumulator with 64 fractional bits in the BoundedInt domain. Interval arithmetic done by the
generator proves every bound at compile time, so a Horner step is `mul, add const, div_rem,
add const` (~1.3k gas) with **no overflow check and no sign branch**, and the result is *more*
accurate than the Q32.32 Horner (whose tiny high-order coefficients are quantized: degree 11 is
worse than degree 9 there). Table size does not change the cost of an array LUT, but a LUT is both
slower and 4 orders of magnitude less accurate than the degree 9 polynomial.

| atan2 / acos | net gas | max abs error |
|---|---:|---:|
| `atan2`: octant reduction, unsigned bounded division, typed odd degree 15 on [0, 1] | 16,320 | 6.3e-8 |
| `atan2`: same, typed degree 19 | **18,840** | **1.9e-9** |
| `atan2`: degree 15 on `BI64` with fused `mul_add` | 30,540 | 6.4e-8 |
| `atan2`: second reduction at tan(pi/8) + degree 9 (one more division) | 31,420 | 7.0e-9 |
| `acos`: `sqrt(1 - x) * P9(x)` on [0, 1] mirrored, typed | **16,510** | **1.2e-9** |
| `acos`: `sqrt(1 - x) * P7(x)` on `BI64` (Abramowitz-Stegun 4.4.46 form, refitted) | 23,370 | 2.9e-8 |
| `acos`: `atan2(sqrt(1 - x^2), x)` | 36,190 | 7.3e-9 |

Coefficients are Chebyshev (near-minimax) fits computed with mpmath. Angles beyond a few turns
inherit the quantization of pi/2 (2^-33 per quarter turn), like any fixed-point implementation.

## Range and accuracy for game physics

- **Q32.32 on `i64`**: range +-2.147e9, resolution 2.33e-10. World coordinates of +-1e6 use 21 of
  the 31 integer bits. `dt = 1/60` is stored with a relative error of 4e-9; `g * dt^2 = 2.7e-3`
  is 1.2e7 ulp; an angular increment of 1e-4 rad/s * dt = 1.7e-6 rad is still 7,000 ulp (4
  significant digits). Q16.16 (1.5e-5) would round that increment to zero: slow rotations and
  small impulses simply vanish, which rules out E/E' regardless of gas.
- **Products are the problem, not storage**: a squared length overflows Q32.32 as soon as
  |v| > 46,340, and any product of two world coordinates (1e12) does. With the plain
  representations (A, B) this forces `i128` intermediates by hand everywhere. With G the
  intermediates are *always* wide: every product lives unscaled in a `Wide =
  BoundedInt<-2^128, 2^128>` accumulator that holds any sum of 4 full-range products, and only the
  final result must fit. Consequences, pinned by `src/tests/semantics.cairo`:
  `dot3((6e4, 6e4, 0), (6e4, -6e4, 0)) = 0` works fused and panics unfused;
  `length3(1e6, -1e6, 1e6) = 1,732,050.80...` is exact to 1 ulp although the squares are 1e12.
  What still must fit is the *result*: |dot| < 2.1e9, i.e. keep dot/cross products on relative
  vectors (lever arms, contact offsets, velocities), not on absolute world positions.
- The fractional/integer split is gas-neutral on this kernel (only the constant `2^32` changes):
  Q40.24 (range 5.5e11, resolution 6e-8) or Q24.40 cost exactly the same. Q32.32 is the balanced
  choice and makes `from_int`/`to_int` align with `i32`.
- **Floor vs truncation**: floor is branch-free and is a true arithmetic shift, so it commutes with
  translation (`floor(x + n) = floor(x) + n`), which avoids the "dead zone" asymmetry of
  truncation around zero. Its bias is -0.5 ulp (1.2e-10) per rescale - irrelevant next to the
  integration error of a 60 Hz step - and fused forms pay it once instead of 3-4 times.
- Storage: an `i64` packs 3 per storage felt, an `i128` only 1. For an on-chain engine that alone
  is a 3x difference in state cost, on top of the 2.4-5.8x compute difference.

## Recommendation

**Representation: `BI64`-style - a newtype over a native `i64`, Q32.32, with all multiplicative
arithmetic done through `core::internal::bounded_int`** (`src/bi64.cairo`).

- **Precision**: Q32.32 (32 fractional bits). The split is a constant and can be revisited for
  free.
- **Rounding**: floor (toward -inf) for `mul`, `div`, fused forms and `to_int`. Branch-free, sign
  agnostic: 1,850 gas vs 2,440-2,520 for a truncating BoundedInt multiply and 2,910-2,990 for the
  native `i128` path.
- **Overflow policy**: panic, checked **once per kernel** by the final `downcast` to `i64`.
  Intermediates cannot wrap (they are typed `BoundedInt`s, max 2^129 << P). add/sub/neg keep the
  native checked `i64` operators (640 gas; going through BoundedInt costs the same because the
  cost is the two-sided range check itself).
- **API shape - this matters more than the representation** (3.2x on `dot3`, 4.2x on `length3`):
  expose an unscaled product `wmul(a, b)` and a `Wide` accumulator with a single `rescale`, and
  write every vector/matrix kernel on top of it (dot, cross, matrix rows, `mul_add`, `lerp`,
  quaternion products: up to 4 products per rescale). A generic `T: Add + Mul` code path should
  exist only as a fallback. Measured: `dot3` 2,250, `cross3` 6,650, `lerp` 2,150, `mul_add` 2,050.
- **sqrt**: corelib `u128_sqrt` on the widened value (1,920 gas, exact floor) - every Newton
  variant is 16-100x more expensive. **Vector length**: integer sqrt of the *unscaled* sum of
  squares (2,220 gas, no rescale, no intermediate overflow). **Inverse length / normalize**: one
  unsigned bounded `div_rem` with a constant numerator, inverse kept with 64 fractional bits
  (`normalize3`: 9,750 gas, 1 ulp).
- **Trigonometry**: range reduction by one `div_rem` by pi/2, then generated *typed* Horner
  kernels: `sin`/`cos` degree 11 (12,500 / 12,600 gas, 4.7e-10), `atan2` degree 19 on [0, 1] after
  octant reduction (18,840 gas, 1.9e-9), `acos` as `sqrt(1 - |x|) * P9(|x|)` (16,510 gas, 1.2e-9).
  No LUT, no CORDIC, no Taylor loop. If 6e-9 / 6e-8 accuracy is enough, degree 9 sin (11,190)
  and degree 15 atan2 (16,320) save ~10%.
- Owner's rule of thumb confirmed and sharpened: math (`div_rem` by constants) beats everything,
  and *type-level bounds beat math*: the cheapest operation is the range check you can prove you
  do not need.

**Runner-up: `FeltQ32` (D), Q32.32 in a `felt252` with lazy rescaling** - 100-gas add/sub/neg,
lazy `dot3` 2,520 (+12%), `cross3` 7,160 (+8%), `mul` 2,120 (+15%). Rejected as the default
because add/sub overflow is silent, comparisons and `abs` need a range check to recover the sign
(1,790 vs 870), division needs a round trip through `i64`, and values must be converted and
range-checked anyway to be stored compactly. It only wins on add-heavy code, and fused kernels
already absorb most additions.

Fallbacks: if `bounded-int-utils` ever becomes unavailable, the same `i64` storage works with the
plain corelib kernels of `I64Q32` (B: mul 2,910, dot3 10,570) without touching user code or
stored data. If more range is required, `I128Q64` with the BoundedInt limb multiply (mul 4,500,
lazy dot3 13,160, sqrt 7,360) is the option - at 2.4-5.8x the compute and 3x the storage.
