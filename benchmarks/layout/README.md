# `layout` — data layout & abstraction style benchmarks

Empirical Sierra-gas micro-benchmarks used to choose the data layout and the abstraction style of
`nalgebra.cairo` (fixed-point linear algebra for a provable physics engine).

Toolchain: scarb 2.19.4 / Cairo 2.19.4 / snforge 0.61.0. Full numbers: [`GAS.md`](./GAS.md)
(106 groups, 648 tests). All figures below are **net Sierra gas** (`l2_gas`, raw minus the group
baseline); 100 gas = 1 Cairo step, 1 range check = 70 gas.

```sh
cd benchmarks
snforge test -p layout 2>&1 | python3 scripts/gas_report.py --md layout/GAS.md
# regenerate the test matrix (Cairo has no test macros):
cd layout && python3 gen_tests.py && scarb fmt -p layout
```

## Methodology

* One snforge `#[test]` per variant, named `bench_<group>__<variant>`; every group has a
  `baseline` with the same opaque inputs and the same number of scalar assertions but without the
  operation. Sierra gas is deterministic, so a single call per test is enough.
* Inputs are **scalars** passed one by one through `harness::black_box`, and results are asserted
  **scalar by scalar**, so the baseline is identical for every layout of a group. A variant
  therefore pays `construct from scalars + operation + read every component back`. For static
  layouts the first and last parts are free (see `*_construct_*` groups: net 0); for `Span`
  layouts they are not (2 240 gas to build a 3-vector, 2 570 to build + unbox a 3x3), and that is
  part of the real price of those layouts.
* Every entry of every result is asserted. This is not only for correctness: unused `felt252`
  results are dead-code-eliminated, which silently turns a 27-mul product into a 9-mul one.
* Scalar: `Fixed { raw: i64 }`, Q32.32, `mul = ((a as i128) * (b as i128) / 2^32) as i64` using
  corelib `i128` `Mul`/`Div`. Key experiments are repeated with plain `i64` and `felt252`
  (`*_i64`, `*_felt` groups) to separate the layout cost from the scalar cost: with `felt252` the
  arithmetic is ~100 gas/op, so what remains is the pure layout/abstraction overhead.
* Every `bench_*` test function and every generic runner is `#[inline(never)]`: snforge inlines
  some test bodies into its generated wrapper and not others, which otherwise shifts
  `raw - baseline` by ~1 000 gas at random. Residual noise after the fix is about +-200 gas
  (branch alignment of the assertion), so differences below ~300 gas are not significant.
* Layout x scalar x op matrices are monomorphisations of one generic body (`vec3::runners`,
  `mat3::runners`), instantiated by the generated `*_gen.cairo` files. Hand-written groups
  (`call_*`) give the same numbers as the generic runners, so the runner itself is neutral.

| file | content |
|---|---|
| `src/fixed.cairo` | `Fixed` Q32.32, `Scalar` glue for `Fixed` / `i64` / `felt252` |
| `src/vec3.cairo` | Q1: 6 `Vec3` layouts behind one trait + generic benchmark bodies |
| `src/vecn.cairo` | Q1: `Vec2` / `Vec4`, component access |
| `src/mat3.cairo`, `src/mat6.cairo`, `src/static_mats_gen.cairo` | Q2: 6 `Mat3` layouts, `Mat2`/`Mat4`/block `Mat6`, dynamic NxN |
| `src/abstraction.cairo`, `src/vec3_variants_gen.cairo`, `src/scalar_variants_gen.cairo` | Q3: generic vs concrete, call style, inline policies |
| `src/passing.cairo` | Q3: by value vs `@snapshot` vs `Box` |
| `src/composite.cairo` | Q4: quaternion, isometry, normalize, skew, symmetric 3x3 |
| `src/lazy.cairo` | Q5: lazy rescaling |
| `gen_tests.py` | generator of every `*_gen.cairo` file |

Approximate unit costs worth keeping in mind (derived from differences between the tables below):

| item | gas |
|---|---:|
| `felt252` add / mul | 100 |
| `i64` add / sub (overflow-checked) | ~750 |
| `i64` mul | ~900 |
| `Fixed` mul as specified for this study (generic `i128 * i128`, `i128 /`, narrowing) | **~11 000** |
| same with the `i64_wide_mul` libfunc for the product (§5) | ~3 700 |
| same with the `BoundedInt` biased-floor kernel of the `primitives` package (§5) | ~2 600 |
| `Fixed` div | ~13 500 |
| `Fixed` sqrt (`u128_sqrt` on the widened value) | ~8 000 |
| non-inlined call of a small panicking function (6 felts in, 3 out) | ~2 300 |
| each extra felt of argument across a non-inlined call | ~100 |
| `span[i]` (bounds-checked read) | ~500–800 |
| `array.append` | ~400–500 |
| one loop iteration (gas withdrawal + recursion), body excluded | ~2 000–3 000 |

## 1. Vector layout

`Vec3`, net gas. `struct` = `{x, y, z}`, `array` = `[T; 3]` newtype (destructured), `tuple` =
`(T, T, T)` newtype, `span_box` = `Span<T>` read with `multi_pop_front::<3>()`, `span_index` =
`Span<T>` read with `span[i]`, `span_loop` = `Span<T>` of unknown length, loops.

| op (scalar) | struct = array = tuple | span_box | span_index | span_loop |
|---|---:|---:|---:|---:|
| construct + read back (any) | **0** | 2 310 | 2 240 | 2 240 |
| add (`Fixed`) | **4 550** | 8 300 | 11 170 | 14 340 |
| add (`felt252`) | **300** | 5 350 | 8 820 | 12 120 |
| scale (`Fixed`) | **33 150** | 35 830 | 37 430 | 40 980 |
| neg (`Fixed`) | **900** | 4 580 | 5 380 | 10 420 |
| dot (`Fixed`) | **34 730** | 36 270 | 38 810 | 42 330 |
| dot (`i64`) | **5 930** | 8 070 | 10 610 | 13 910 |
| dot (`felt252`) | **500** | 3 240 | 6 480 | 8 210 |
| cross (`Fixed`) | **66 590** | 70 640 | 73 510 | 93 150 |
| cross (`felt252`) | **900** | 5 950 | 9 420 | 30 710 |
| equals (`Fixed`) | **600** | 3 340 | 7 180 | 9 580 |
| chain `(a + b*k - b) x a . b` (`Fixed`) | **143 370** | 150 640 | 162 100 | 196 570 |
| chain (`felt252`) | **2 200** | 13 870 | 28 630 | 65 990 |

* **Struct, fixed-size array and tuple compile to exactly the same Sierra** (identical gas on
  every op, every scalar): `struct_construct` / `struct_deconstruct` are free and all three are
  "N felts on the stack". The choice is purely ergonomic, and the struct wins: named fields,
  `v.x` access, `#[derive(PartialEq, Debug, Serde)]`. A `[T; 3]` can only be read for free by
  destructuring; `arr.span()[i]` costs 1 670 gas (`access_y`), more than indexing a real span
  (1 170).
* `Span` layouts pay an allocation per result (~2 200), a bounds check per read, and loop
  overhead. The *absolute* overhead (+4k to +10k per op, +25k for a looped cross product) is
  masked on mul-heavy ops by today's 11k `Fixed` mul (+4 % to +40 %), but it is already x2–x3
  on add / sub / neg, and x6 to x40 everywhere once the scalar is cheap (`felt252` rows, and
  see §5 which does make the scalar cheap).
* Vec2 / Vec4 behave the same (`vec2_*`, `vec4_*`): struct = array; `span_loop` dot is
  +5 600 (Vec2) / +9 600 (Vec4) gas, i.e. ~+2.4k per element.
* Runtime index into a static vector: `match i { 0 => x, 1 => y, 2 => z }` costs 710, cheaper
  than a span index (1 270) — an `Index` impl on the struct is affordable when needed.

## 2. Matrix layout

`Mat3`, net gas. `fields` = 9 named fields, `cols` = 3 column `Vec3` with ops written through
vector ops, `array9` = `[T; 9]`, `array3x3` = `[[T; 3]; 3]`, `dyn_index` = `Span<T>` + dims with
`data[i * cols + j]` loops (orion / alexandria style), `dyn_seq` = same storage, sequential
traversal (`slice` + `pop_front`, rhs transposed once) and `multi_pop_front::<9>()` + static
kernel for the 3x3-only ops (det / inverse).

| op (`Fixed`) | fields = array9 = array3x3 | cols | dyn_seq | dyn_index |
|---|---:|---:|---:|---:|
| construct + read back | **0** | 0 | 2 570 | 9 560 |
| mat * vec | **100 430** | 108 550 | 152 670 | 181 190 |
| mat * mat | **297 030** | 329 350 | 451 880 | 498 530 |
| transpose | **0** | 0 | 50 560 | 62 880 |
| determinant | **99 290** | 101 320 | 100 960 | 122 490 |
| inverse (adjugate) | **344 830** | 354 050 | 349 070 | 574 210 |
| add | **11 190** | 14 050 | 32 530 | 39 720 |
| scale | **96 390** | 99 450 | 112 410 | 119 600 |

| op (`felt252`: pure layout overhead) | static (all four) | dyn_seq | dyn_index |
|---|---:|---:|---:|
| mat * vec | **1 500** | 56 950 | 85 830 |
| mat * mat | **4 700** | 165 120 | 212 610 |
| transpose | **0** | 49 860 | 62 180 |
| determinant | **1 400** | 3 970 | 27 700 |
| inverse | **11 000** | 15 240 | 181 460 |

* The four static layouts are again the same Sierra. With `felt252` even `cols` is identical
  (4 700): **the layout is free, transpose is free (a renaming)**. The +3 % to +11 % of `cols`
  with `Fixed` / `i64` is *not* a layout cost: it is the call overhead of the non-inlined
  `Vec3::scale/add/cross` helpers it is composed of (see §3) — composing matrix kernels out of
  vector ops is fine if and only if those ops are `#[inline(always)]`.
* Dynamic matrices: the loop machinery costs ~3.5k–6k gas per inner-product term with
  sequential traversal and ~6.5k–7.6k with index arithmetic; a dynamic transpose costs ~5.5k
  per element (it is free for static types).

N x N product, unrolled static type vs dynamic loops (`matmul_n*`; static N=6 is a 2x2 block
matrix of `Mat3`):

| N | scalar | static | dyn_seq | dyn_index | dyn_seq / static |
|---|---|---:|---:|---:|---:|
| 2 | `Fixed` | 88 210 | 161 030 | 158 320 | x1.83 |
| 3 | `Fixed` | 296 730 | 447 700 | 491 280 | x1.51 |
| 4 | `Fixed` | 703 810 | 976 690 | 1 127 420 | x1.39 |
| 6 | `Fixed` | 2 427 900 | 3 069 170 | 3 708 400 | x1.26 |
| 2 | `i64` | 11 410 | 85 510 | 83 160 | x7.5 |
| 3 | `i64` | 37 530 | 192 620 | 236 820 | x5.1 |
| 4 | `i64` | 89 410 | 372 110 | 523 940 | x4.2 |
| 6 | `i64` | 354 300 | 1 028 230 | 1 670 240 | x2.9 |
| 3 | `felt252` | 4 600 | 161 020 | 205 220 | x35 |
| 4 | `felt252` | 11 400 | 293 830 | 445 660 | x26 |
| 6 | `felt252` | 47 200 | 752 990 | 1 395 000 | x16 |

"Dynamic storage, static kernel" (`dyn_static_kernel`, N=3: check dims, two
`multi_pop_front::<9>()`, unrolled product, re-pack into an array) costs 302 640 with `Fixed`
(**+2 %** over static) and 13 010 with `felt252` (vs 161 020 for the loop): a dynamic type is
affordable *if* it dispatches to unrolled kernels for the small shapes.

## 3. Abstraction cost

Generic vs concrete, inline policy (`abs_vec3_*`, `Fixed`):

| op | generic = concrete, default | generic = concrete, `inline(always)` | `inline(never)` |
|---|---:|---:|---:|
| add | 4 550 | **2 220** | 4 550 |
| dot | 34 730 | **32 720** | 34 730 |
| cross | 66 590 | **64 540** | 66 590 |
| chain (5 ops) | 143 370 | **132 580** | 143 370 |

* **Genericity is exactly zero-cost**: `Vec3<T>` with `+Add<T> +Sub<T> +Mul<T> +Copy<T> +Drop<T>`
  bounds and a hand-monomorphised `Vec3Fixed` give identical gas in every configuration. So do an
  associated-type based API (`WideScalar::Wide`, §5) and generic free functions.
* **Trait method = free function = operator trait** (`call_add`: 4 450 each; 2 120 each when
  inlined). The call style is a pure ergonomics decision.
* **The default inliner does not inline vector ops**: default == `inline(never)` for
  `add/dot/cross`. Each non-inlined call costs ~2 000–2 300 gas (arguments + `PanicResult`
  wrapping/matching), i.e. more than the body of a `Vec3` add. On the other hand the default
  *does* inline the small `Fixed::mul` wrapper (default == always); forcing it out of line costs
  +1 800 gas per multiplication (`inline_*`: mat*mat 297 030 -> 346 440, +17 %).
  Bodies made only of non-panicking `felt252` arithmetic do get inlined by default, anything
  with checked integer arithmetic does not: the heuristic is opaque, so be explicit.
* Compiler restriction found on the way: `#[inline(always)]` is **rejected on functions that
  have their own impl generic parameters** (`fn f<T, +Add<T>>`, error E2143) but accepted on
  methods of a generic *impl* (`impl VecOps<T, +Add<T>> of ...`). Small generic helpers must
  therefore live in generic impls, not be generic free functions.
* Operators: corelib binary operator traits are homogeneous (`Mul<T>: (T, T) -> T`). `v + v`,
  `v - v`, `-v`, `m * m`, `q * q` can be operators; `m * v`, `v * k`, `iso * point` cannot and
  need named methods (`mul_vec`, `scale`, `transform_point`). Only the `*Assign<Lhs, Rhs>` family
  is heterogeneous (`v *= k` is possible).
* Impl lookup: operator impls must be defined in the module of the type (or imported by the
  caller) — put them next to the struct.

By value vs snapshot vs `Box` (`pass_*`, callee `#[inline(never)]`, `felt252` payload so the
call dominates; `x3` = same argument passed to three calls):

| payload | value x1 | snapshot x1 | box x1 | value x3 | snapshot x3 | box x3 |
|---|---:|---:|---:|---:|---:|---:|
| `Vec3` (3 felts) | 600 | 600 | 1 200 | 1 800 | 1 800 | 2 400 |
| isometry (7 felts) | 1 000 | 1 000 | 1 600 | 3 000 | 3 000 | 2 800 |
| `Mat3` (9 felts) | 1 200 | 1 200 | 1 800 | 3 600 | 3 600 | 3 000 |
| `Mat4` (16 felts) | 1 900 | 1 900 | 2 500 | 5 700 | 5 700 | 3 700 |
| `Mat3<Fixed> * Vec3<Fixed>` (real body) | 100 330 | 100 330 | – | – | – | – |

* **`@T` of a `Copy` struct is the same Sierra as `T`**: identical gas everywhere, the snapshot
  only adds `*` noise in the code. `dup` is free; what costs is the `store_temp` of every felt of
  every argument at each *non-inlined* call: a by-value call costs **~300 + 100 per felt**
  (600 / 1 000 / 1 200 / 1 900 for 3 / 7 / 9 / 16 felts), and nothing at all once the callee is
  inlined (`value_default_inline_x3`: 300 for three calls). This confirms on composite types
  (isometry, `Mat3`, `Mat4`) what the `primitives` package measured on flat structs.
* `Box<T>` makes each call flat (600 gas whatever the size) after a one-time ~600 boxing; it
  only wins from the third non-inlined call of the same >= 7-felt value, and the gain
  (2 000 gas for a `Mat4` passed 3 times) is less than one fixed-point mul. Returning a `Mat4`
  boxed is slower than by value (2 400 vs 1 800).

Code size (Sierra statements of one workload: 2 mat*mat, inverse, det, 3 cross, 3 dot...):
everything `inline(never)` 5 670; vector ops inlined + scalar mul out of line 6 534; defaults
8 992; everything `inline(always)` 10 715. Inlining the *scalar mul* is what bloats the program
(27 copies per mat*mat), inlining vector ops is nearly free in size. If contract size ever
matters, the knob to turn is the scalar mul (-37 % statements for +17 % gas), not the vector ops.
The 636-test crate compiles in ~5 s; no compile-time issue observed.

## 4. Composite patterns (`Fixed`)

Cost is essentially `11k x number of Fixed multiplications`, so the algorithmic choice is what
matters:

| pattern | variant | gas | muls |
|---|---|---:|---:|
| quaternion product | Hamilton | 175 950 | 16 |
| rotate 1 vector | expanded `v + 2w(u x v) + 2u x (u x v)` | **173 190** | 15 |
| | quat -> `Mat3` (106 290) then mat * vec | 207 520 | 9 + 9 |
| | `q * v * q^-1` with two Hamilton products | 354 400 | 32 |
| rotate 4 vectors | expanded, 4 times | 693 060 | 60 |
| | convert once, 4 mat * vec | **508 810** | 45 |
| isometry * point | quaternion rotation | 178 340 | 15 |
| | matrix rotation (already stored) | **107 460** | 9 |
| isometry * isometry | quaternion (7 scalars) | **355 890** | 31 |
| | matrix (12 scalars) | 407 170 | 36 |
| normalize | norm only (dot + sqrt) | 42 450 | 3 |
| | 3 divisions | **82 920** | 3 + 3 div |
| | 1 division + 3 muls | 84 430 | 6 + 1 div |
| `[a]x * b` | `a.cross(b)` | **66 490** | 6 |
| | build skew matrix, mat * vec | 100 930 | 9 |
| `[a]x * M` | cross per column | **199 670** | 18 |
| | skew matrix, mat * mat | 297 030 | 27 |
| `M * M^T` | 6 symmetric dots -> `Sym3` | **210 380** | 18 |
| | transpose (free) + mat * mat | 296 530 | 27 |
| `R diag(d) R^T` (world inertia) | scale columns + 6 symmetric dots -> `Sym3` | **291 190** | 27 |
| | two mat * mat | 592 760 | 54 |
| symmetric * vec | `Sym3` (6 fields) | 100 030 | 9 |
| | full `Mat3` | 100 330 | 9 |
| symmetric inverse | `Sym3` | **240 830** | 15 + 6 div |
| | full `Mat3` | 344 330 | 21 + 9 div |

* Rotating with the expanded quaternion formula beats going through a matrix for one or two
  vectors; from 3 vectors on, convert once (break-even: 106k conversion vs 66k saved per vector).
  A body whose frame transforms many points per step should cache its `Mat3`.
* `Sym3` does not make mat * vec cheaper, but it cuts the inverse by 30 % and the products that
  *produce* symmetric matrices by 30–50 %. It is worth having as a first-class type.
* The multiplication-by-zero entries of a skew matrix are paid in full: never materialise `[v]x`.
* Division ~ multiplication (13.5k vs 11k): `x / n` three times beats `inv = 1 / n; x * inv`
  and is more accurate (single truncation).

## 5. Lazy rescaling

A Q32.32 product is `(a * b) >> 32`. In a dot product the shift (and the narrowing to `i64`) can
be done once on the accumulated wide sum. Two independent axes are measured: *eager* (rescale
per product) vs *lazy* (rescale per output scalar), and the kernel used for product / sum /
rescale:

| kernel | product | sum | rescale |
|---|---|---|---|
| `i128` (the `Fixed` of this study) | generic `i128 * i128` | checked `i128 +` | signed `i128 /`, trunc |
| `wide_mul` | `i64_wide_mul` libfunc | checked `i128 +` | signed `i128 /`, trunc |
| `felt` | `felt252 *` (1 step) | `felt252 +` (1 step, cannot wrap) | `felt252 -> i128` range check + signed `/`, trunc |
| `felt_floor` | idem | idem | bias by 2^127, unsigned `u128 /`, floor |
| `bounded_floor` | `bounded_int::mul` (no range check) | `bounded_int::add` (no range check) | bias, `bounded_int::div_rem` by 2^32, one `downcast`, floor (kernel from `benchmarks/primitives`) |

| op | eager `i128` | eager `wide_mul` | eager `bounded_floor` | lazy `i128` | lazy `wide_mul` | lazy `felt` | lazy `felt_floor` | lazy `bounded_floor` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dot | 34 630 | 12 580 | 9 460 | 28 310 | 6 260 | 6 090 | 2 990 | **2 250** |
| cross | 66 490 | – | – | 57 210 | – | 13 110 | 11 740 | – |
| mat * vec | 100 330 | 34 180 | 24 820 | 81 570 | 15 220 | 14 310 | 12 940 | **10 080** |
| mat * mat | 296 930 | 98 480 | 70 400 | 240 050 | 41 600 | 38 470 | 33 080 | **25 780** |
| mat * mat, dynamic `Span` storage | 452 080 | – | – | – | – | 207 030 | – | – |

(In steps: eager `i128` dot = 267 steps + 61 range checks, lazy `felt` = 55 + 8, lazy
`felt_floor` = 35 + 7. The `lazy_bounded_floor` dot, 2 250, reproduces the figure measured
independently by the `primitives` package.)

* **The scalar kernel matters more than anything else in this report**: the straightforward
  `i128 * i128` product costs ~7k of the 11k of a `Fixed` mul (sign split, `u128_wide_mul`,
  overflow checks). `i64_wide_mul` brings an eager mul to ~3.7k, the `BoundedInt` kernel to
  ~2.6k (eager mat*mat: 297k -> 98k -> 70k).
* **Lazy rescaling is a further x2.5 to x4 on top of the best eager kernel** (dot 9 460 -> 2 250,
  mat*vec 24 820 -> 10 080, mat*mat 70 400 -> 25 780), and x9–x15 against the naive scalar. With
  a cheap product, a row of n terms costs `~2k + 0.1k*n` instead of `2.6k*n`.
* It composes with genericity at zero cost (`lazy_felt_generic_api`, a `WideScalar` trait with
  an associated `Wide` type: 6 090 = the hand-written `lazy_felt`).
* **Overflow.** `felt252`: each product is < 2^126 in magnitude and the prime is ~2^251, so no
  realistic number of terms can wrap: *no intermediate overflow at all*; only the final value is
  range-checked (it must fit `i128`, else its rescale could not fit `i64` anyway, then `i64`).
  `BoundedInt`: the bounds of the sum are tracked in the type (3 terms: +-3 * 2^126), same
  property, but one helper-impl set per row length (dot2 / dot3 / dot4...), whereas the felt
  accumulator handles any length, including dynamic ones. Both panic exactly when the true
  result is out of range, whereas eager can also panic on an intermediate term. A checked `i128`
  accumulator can overflow on an intermediate sum only from 4 terms up and only at the extreme
  edge of the type (tests `lazy_felt_survives_intermediate_overflow`,
  `lazy_i128_intermediate_overflow_panics`).
* **Rounding.** Eager rounds every product (error up to n ulp, biased); lazy rounds once
  (error < 1 ulp): strictly more accurate, but results differ from a naive term-by-term
  implementation in the last bit (test `lazy_rounding_is_tighter`: for an exact value of
  -1.5 ulp, eager-trunc gives 0, eager-floor -3, lazy-trunc -1, lazy-floor -2). Any reference
  implementation used for differential testing (Rust, client-side prediction) must use the same
  summation rule. Floor is cheaper than truncation (unsigned division, no sign branch) and is
  what an arithmetic shift does in every other fixed-point library; pick one for the whole
  library (scalar-level decision).
* **Limits.** Only one level of products can be deferred: `(a*b)*c` needs a rescale in between.
  Kernels must be written as "sum of products of *inputs*" (dot, cross, mat*vec, mat*mat,
  `q_mul`, quaternion -> matrix, `Sym3` products) to benefit; chained formulas (expanded
  quaternion rotation) get one rescale per product level.
* **Consequence for the rest of this report.** All `Fixed` tables above use the 11k scalar, which
  hides overheads. With a 2–3k scalar mul and lazy rows, a non-inlined call (2.3k) costs as
  much as a whole dot product, and the `Span`-based layouts are x3–x8 slower instead of
  +5–60 %: read the `i64` / `felt252` rows of §1–§2 as the realistic ones.

## 6. Compile-time / code-size side effects

See the code-size paragraph of §3. Summary: forced inlining of vector/matrix ops is cheap in
size; forced inlining of the scalar multiplication is what grows the Sierra program (+60 %
statements between all-never and defaults). Nothing here came close to a compile-time problem.

## Design decisions

1. **Vectors: plain structs with named fields** (`Vec2 {x,y}`, `Vec3 {x,y,z}`, `Vec4`,
   `Quat {x,y,z,w}`). Struct, `[T; N]` and tuple are the same Sierra (0 gas difference on all
   ops and scalars); structs are the most ergonomic and derive-friendly. No array-backed storage:
   the only free access to `[T; N]` is destructuring, `span()[i]` costs 1 670.
2. **Matrices: static structs; column-major columns of vectors like nalgebra/glam is fine**
   (`Mat3 { c1, c2, c3 }` or 9 named fields — both are the same 9 felts; transpose and
   conversions between them are free). Write the hot kernels (mat*vec, mat*mat, det, inverse)
   as explicit scalar formulas rather than compositions of vector ops, or make sure every vector
   op involved is `#[inline(always)]`; otherwise `cols` pays +3–11 % (`Fixed`) to +47 % (`i64`)
   in calls.
3. **Provide `Sym3` (6 fields) as a first-class type** and structured kernels
   (`M*M^T -> Sym3`, `R diag(d) R^T -> Sym3`, `[v]x * M` by columns, `Sym3::inverse`): -30 % to
   -50 % on the rigid-body inertia path. Never materialise skew or diagonal matrices. Cache a
   `Mat3` next to the quaternion for bodies that transform >= 3 points per step.
4. **Generic `Vec3<T>` / `Mat3<T>` with trait bounds: yes.** Monomorphisation is exactly
   zero-cost (identical gas to hand-written concrete types in all 12 compared configurations),
   including associated types. Keep a `Scalar`-style trait; no need for a concrete
   `Vec3Fixed` fork.
5. **By value (`Copy`) everywhere; no `@snapshot` parameters, no `Box`.** `@T` of a `Copy`
   struct is the same Sierra as `T` (identical gas for 3, 7, 9 and 16 felts); a non-inlined call
   costs ~300 + 100 gas per felt either way, and `Box` only wins ~2k gas in contrived cases.
   Derive `Copy, Drop` on every math type, including `Mat4` and isometries.
6. **Inlining policy:** `#[inline(always)]` on every small op of vectors / quaternions
   (constructors, accessors, add, sub, neg, scale, dot, cross, component-wise ops): the default
   inliner leaves them out of line and each call costs 2 000–2 300 gas, i.e. -50 % on `add`,
   -8 % on a typical expression with the 11k scalar, and as much as the op itself once the
   scalar kernel is cheap (§5). Leave big kernels (mat*mat, inverse, isometry composition) at the
   default. Leave the scalar `mul` at the default (the compiler inlines the wrapper) unless
   program size becomes a constraint — then `inline(never)` on the scalar mul trades +17 % gas
   for -37 % Sierra statements. Because `#[inline(always)]` is refused on functions with their
   own impl generics, implement ops as methods of generic impls, not as generic free functions.
7. **Operators and methods:** cost-neutral (operator = trait method = free function). Implement
   `Add/Sub/Neg` for vectors and matrices, `Mul` for `Mat*Mat` and `Quat*Quat`, `PartialEq`
   by derive; corelib operators being homogeneous, `mat.mul_vec(v)`, `v.scale(k)`,
   `iso.transform_point(p)` must be named methods (plus `MulAssign<V, T>` for `v *= k`).
   Define operator impls in the module of the type so callers need no import.
8. **Dynamic `DVector` / `DMatrix`: affordable only as a secondary type.** With the 11k scalar a
   good dynamic product is +26 % (N=6) to +83 % (N=2); with a realistic cheap scalar the gap is
   x3–x8 (`i64` rows; measured 207k vs 38k for a lazy-felt 3x3 product) because the loop
   machinery costs 3.5k–7k gas per inner-product term and ~2.4k per element for element-wise
   ops. Use static types for every dimension known at compile time (2, 3, 4, and 6 as 2x2 blocks
   of `Mat3`). When a dynamic type is needed (constraint Jacobians, N-body solvers):
   `struct DMatrix<T> { data: Span<T>, rows: usize, cols: usize }`, **row-major, immutable
   `Span`**; no index arithmetic in inner loops (walk rows with `slice` / `pop_front`, transpose
   the rhs once: neutral at N=2, up to -17 % (`Fixed`) / -46 % (`felt252`) at N=6 vs
   `data[i*cols+j]`); felt252 lazy accumulation in the inner product (the only lazy kernel that
   works for a runtime length); read fixed-size chunks with `multi_pop_front::<N>()` (one bounds
   check instead of N: reading back a 3x3 costs ~300 gas instead of ~7 000); and **dispatch to
   the unrolled static kernels for small shapes** (3x3 through the static kernel: +2 % over the
   static type with `Fixed`, +16 % with `i64`, vs x5 for the loop).
9. **Lazy rescaling: make it the default for every sum-of-products kernel** (dot, cross,
   mat*vec, mat*mat, quaternion product, quaternion -> matrix, `Sym3` products): accumulate the
   unrescaled products, rescale once per output scalar. x2.5–x4 cheaper than term-by-term
   multiplication with the best known eager kernel (mat*mat 70k -> 26k, dot 9.5k -> 2.3k), x9–x15
   against the naive `i128` scalar (297k -> 26k), no intermediate overflow, tighter rounding
   (< 1 ulp instead of n ulp). Use the `BoundedInt` biased-floor row kernels for the static
   sizes (2 / 3 / 4 terms; cheapest, 2 250 per dot3) and the `felt252` accumulator for dynamic
   lengths and as the generic fallback (2 990–6 090 per dot3). Expose it generically through the
   scalar trait (`type Wide; fn wide_mul(a, b) -> Wide; fn rescale(Wide) -> T;`, `Wide = T` for
   plain integers) — measured zero-cost. Whatever the accumulator, never build the scalar on
   the generic `i128 * i128` (7k of its 11k are avoidable). Document the rounding rule (one
   floor per output scalar) as part of the numeric spec since it is observable in the last bit,
   and settle trunc vs floor once at the scalar level (floor is cheaper and branchless).
