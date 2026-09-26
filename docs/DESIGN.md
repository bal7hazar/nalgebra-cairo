# Design decisions

Each decision is justified by a measurement in [BENCHMARK.md](BENCHMARK.md). A decision may only
be overturned by a new measurement committed under `benchmarks/`.

## D1 — Workspace

```
Scarb.toml            virtual workspace, shared versions, edition 2024_07, no `starknet` dependency;
                      `simba = "0.2.0"` (registry, bal7hazar/simba-cairo) is the scalar layer
crates/nalgebra/      package `nalgebra`: `base`, `geometry`, `linalg` modules (mirrors upstream)
benchmarks/           standalone workspaces: design-time micro-benchmarks (never a dependency)
scripts/              gas_report.py (report + snapshot + CI gate), check.sh
tools/oracle/         Rust generator of golden vectors from upstream nalgebra (f64) — outputs are committed
docs/                 research, benchmark synthesis, design, roadmap
```

Upstream nalgebra depends on `simba` for its scalar abstraction, and simba is a trait layer over the
primitive `f32` / `f64` shared by glam, parry and rapier. We mirror that exactly (WP 7.1): the primitive
is fixed-cairo's `fixed` package (registry dependency, pinned version), `simba` only implements the
traits for it, and nalgebra is generic over them.

## D2 — Scalar: fixed-cairo's `fixed::Fixed` (Q32.32 on `i64`)

- There is ONE Q32.32 implementation in the stack: `fixed` (fixed-cairo), `Fixed { raw: i64 }`,
  range ±2.1e9, resolution 2.3e-10. nalgebra-cairo has none of its own (the former `simba::fixed`
  was deleted in WP 7.1: two implementations with independent rounding choices would break the
  bit-exact determinism rapier-cairo relies on when it mixes glam and nalgebra values).
- Pinned by version (`fixed = "0.4.0"`): fixed-cairo bumps MINOR on any numeric change, so the
  version pins every golden file here.
- Numeric semantics are `fixed`'s: `+ - neg` checked `i64`; `*`, fused kernels (`wide::dot*`,
  `mul_add`, `mul_sub`, `Acc`) rescale ONCE with **floor**; `sqrt` / `norm*` floor of the exact root;
  **`/`, `recip`, `from_ratio` round to nearest, ties to even, like `f64 /`** (owner's ruling: the
  Rust reference); `%` is the exact truncated remainder, like Rust's float `%`; constants are
  rounded to nearest; `fixed::trig` rounds its final rescale to nearest.
- **Overflow: panic** (`'Fixed: overflow'`, `'Fixed: division by zero'`, ...). Never wrap.
- Anything `fixed` lacks (a kernel, a rounding mode) is an escalation to the fixed-cairo
  orchestrator, never a local reimplementation (WP 7.1 obtained `wide::Acc` in 0.2.0 and the
  nearest division in 0.3.0 that way).

## D3 — Scalar abstraction

nalgebra types are generic over `T: Real`, as nalgebra-rs is over `T: RealField` (measured zero-cost
after monomorphisation: `bench_real_dot3__generic` equals the direct `fixed` call). `simba`
implements the traits for `fixed::Fixed` by delegation only (`#[inline(always)]` methods of impls,
no arithmetic of its own). The trait exposes **fused kernels**, not just operators:

```cairo
pub trait Real<T> {
    // constants, conversions, abs / signum / min / max / clamp / floor / ceil / round / recip
    fn sum_prod2(a0: T, b0: T, a1: T, b1: T) -> T;            // a0*b0 + a1*b1, one rescale
    fn sum_prod3(..) -> T;  fn sum_prod4(..) -> T;
    fn diff_prod(a: T, b: T, c: T, d: T) -> T;                 // a*b - c*d (cross, det2)
    fn mul_add(a: T, b: T, c: T) -> T;  fn lerp(a: T, b: T, t: T) -> T;
    fn norm2(x: T, y: T) -> T;  fn norm3(..) -> T;  fn norm4(..) -> T;   // sqrt of raw sum of squares, no rescale
    fn sqrt(self: T) -> T;  fn div(a: T, b: T) -> T;  fn div3(x0: T, x1: T, x2: T, d: T) -> (T, T, T); // .. div16
    fn sin_cos(self: T) -> (T, T);  fn atan2(y: T, x: T) -> T;  fn acos(self: T) -> T;  // ...
}
```

Longer sums (6-term rows of `Matrix6`, solves, Frobenius norms) use the `Wide` accumulator, `fixed::wide::Acc`
(`wide_add_prod` / `wide_sub_prod` / `wide_rescale` / `wide_sqrt`; +200 gas per extra product). `wide_mul_scalar(w, s)`
(`Acc::mul_narrow`) is the exact triple product `(a·b − c·d)·e` with ONE rounding. Every division goes through
`Real::div` (nearest), and one divisor serving ≥ 3 quotients goes through `Real::div3..div16` (`fixed::wide::RecipNearest`,
bit-identical to per-element division, cheaper from 3 quotients). Every sum-of-products in the library goes through a fused kernel:
**never `a * b + c * d` with two rescales.**

Approximate equality (`abs_diff_eq`, `relative_eq`) is defined on raw units (ulp), since
nalgebra's float epsilons are meaningless in fixed point.

**The one exception to "neither more nor less"** (owner ruling, 2026-09-24): the fused kernels of
`Real` (`sum_prod*`, `diff_prod`, `norm*`, `norm_squared*`, `lerp`, `wide_*`, `div3..div16`) and the
constants simba-rs has no name for (`TWO`, `HALF`, …) have no counterpart in simba-rs's `RealField`.
They stay, confined to the scalar trait layer (nalgebra's public API is strictly upstream's):
expressing nalgebra with upstream-named scalar methods only was measured at +94 % to +326 % gas per
sum of products, up to 5 ulp more error, and norms overflowing above |x| > 46,341 where `f64` does
not (WP 8.0 report). Everything with a simba-rs name uses it (`is_sign_negative`, `T::pi()`, …).

## D4 — Types: static, unrolled, by value

- Named-field `Copy` structs: `Vector2/3/4/6<T>`, `Matrix2/3/4/6<T>` (`m11..m33`, `Matrix3::new`
  takes row-major arguments like upstream; Serde order is column-major like upstream), `Point2/3`,
  `Unit<V>`, `Quaternion`, `UnitQuaternion`, `UnitComplex`, `Rotation2/3`, `Translation2/3`,
  `Isometry2/3`, `Similarity2/3`.
- `Matrix6` is 2x2 blocks of `Matrix3`; `Vector6` is two `Vector3` (spatial algebra layout).
- Structured kernels stay crate-internal since WP 8.0 (strict public API): `SymMatrix2/3` (Jacobi
  state, SVD Gram matrix, LDLᵀ input), `quadform`, `mul_transpose`, `cross_matrix_mul`, `adjugate`;
  public signatures take and return upstream's `MatrixN`.
- No loops, no `Array`, no `Span` in static types. Hot kernels are explicit scalar formulas over
  fused scalar kernels, not compositions of vector ops.
- By value everywhere (no `@T`, no `Box`): identical Sierra, simpler API.
- Operators: `Add/Sub/Neg` on vectors and matrices, `Mul` for `M*M` and `Q*Q`. Heterogeneous
  products are named methods as corelib operators are homogeneous: `m.mul_vec(v)`, `v.scale(k)`,
  `iso.transform_point(p)`. Upstream names are kept wherever an operation exists upstream.
- `#[inline(always)]` on small ops (constructors, accessors, add, sub, neg, scale, dot, cross);
  default inlining for large kernels (mat*mat, inverse, isometry composition). Implement ops as
  methods of generic impls (the attribute is refused on generic free functions).

## D5 — Dynamic types: `DMatrix`, `DVector`, `RowDVector` (WP 8.5-P13)

Behind the Scarb feature `dynamic` (in `default`, D9), in `base/dynamic*.cairo`. Measured in
`crates/tests_dynamic/src/layout.cairo` (net gas, `Fixed`, the losing layouts kept there):

| kernel | column-major `Span`, indexed | column-major `Span`, contiguous runs | row-major `Span` | `Felt252Dict<Nullable>` |
|---|---:|---:|---:|---:|
| one component `m[(i, j)]` | **2 180** | — | 2 180 | 302 060 (incl. build) |
| sum of a column, 16 rows | — | **28 410** | 56 400 (strided) | — |
| product 6x6 | 1 295 550 | **584 500** | 939 160 | 2 506 740 |
| product 16x16 | 23 161 150 | **7 560 380** | 14 441 560 | 35 822 140 |
| resize 6x6 → 7x8 | — | **146 290** | — | 623 860 |

- **Storage: column-major like upstream** (`VecStorage`): `DMatrix<T> { data: Span<T>, nrows,
  ncols }`, `DVector<T> { data: Span<T> }`, `RowDVector<T> { data: Span<T> }` (three structs, like
  the static `Vector3` / `RowVector3` / `Matrix3x1`: their constructors differ in arity upstream,
  `DVector::zeros(n)` vs `DMatrix::zeros(r, c)`). `from_vec(nrows, ncols, data)`, `as_slice`,
  `Serde` (`data, nrows, ncols`: `VecStorage`'s field order) and linear indexing are
  column-major. The former D5 (row-major, dispatch to static kernels) is superseded: row-major
  costs +61 % on the 6x6 product and +91 % on 16x16, and makes every upstream column-major
  contract a transposition.
- **Containers: `Span<T>`** (a view of a write-once `Array<T>`): the types are `Copy` (upstream
  `Clone`, like every Cairo value type here, `RENAMES`), constructors from an `Array` take it
  without copy. `Felt252Dict` is 2 to 5 times dearer on every kernel (every read goes through the
  dictionary, squashed on drop); its only advantage, in-place writes, does not pay for the reads.
- **In-place forms** (`resize_mut`, `resize_vertically_mut`...): `ref self`, rebuilt then
  reassigned (upstream does the same: `*self = self.clone().resize(..)`). Element writes build a
  new span (Cairo memory is write-once).
- **Loops** are allowed (runtime sizes); inner loops walk contiguous runs with
  `multi_pop_front::<4>` (dot product of 16: 21 430 gas, vs 35 370 with one `pop_front` per term
  and 58 370 indexed; 8-chunks save 7 % at 16 but lose 19 % at 6); dot products of up to 6 terms
  are straight-line (6 850 vs 14 240 for 6). The product transposes the left factor once (strided
  reads, 4 per iteration: -12 %) so that every output is ONE fused sum of products of two
  contiguous runs, accumulated in `Real::Wide` and floored once.
- **Dispatch to the static kernels**: the square products up to 6x6 (`DMatrix * DMatrix`,
  `DMatrix * DVector`) convert to `MatrixN` / `VectorN`, multiply and convert back, bit-identical
  (every output is the exact sum floored once either way): 3x3 25 150 gas instead of 139 680, 6x6
  121 210 instead of 542 520, 6x6 by a vector 30 310 instead of 215 470. Only the squares: the 216
  rectangular pairs would compile 216 static products into every dependent that multiplies
  dynamic matrices.
- **Partially dynamic aliases**: `MatrixXx1` IS `DVector` and `Matrix1xX` IS `RowDVector` (upstream
  names the same types); `Matrix2xX..Matrix6xX` and `MatrixXx2..MatrixXx6` are aliases of `DMatrix`
  whose fixed dimension is an invariant of the value, not of the type (Cairo has no const
  generics; ten more structs with the whole dynamic surface would cost their compile memory to
  every dependent for no behaviour, `tools/shapegen/DESIGN.md` §1.1). The static edition forms
  return them with the right dimension (`Matrix3x4::insert_columns` is a `Matrix3xX` of 3 rows);
  their constructors take both dimensions (`DMatrixTrait::zeros(3, n)` for upstream's
  `Matrix3xX::zeros(n)`).
- **Static shapes** (generated by `tools/shapegen/dynamic.py`, gated with the rest): the
  conversions `Into<MatrixRxC, DMatrix>`, `Into<VectorN, DVector>`, `Into<RowVectorN, RowDVector>`
  (one array literal each, in the target's module: Cairo finds an impl in the module of its trait
  or of one of its types); per shape `<S>DynamicTrait` with upstream's edition forms whose result
  is dynamic (`insert_columns` on a `Matrix3x4` is a `Matrix3xX`, on a `RowVector3` a
  `RowDVector`: the shape converted, then the dynamic kernel), `from_vec` / `from_iterator` /
  `from_row_iterator` (an `Array` / `Span`), `len`, `is_empty`, `compress_*`; and upstream's
  `insert_fixed_columns::<D>` .. `remove_fixed_rows::<D>` whose result is another static shape,
  as `InsertFixedColumns<M, Out, T>`... (like `FixedRows`: the output type selects `D`, one impl
  per pair so an impossible size is a compile error), run on the `Matrix6` canvas of
  `FixedResize`: 1 100 gas for `Matrix3x4 -> Matrix3x6`, against 5 200 for a direct per-pair
  literal (`bench_matrix3x4_insert_fixed_columns__alt_direct`), and 80 canvas literals instead of
  720.
- **Cost of the feature** (cold `scarb build -p nalgebra`, this machine): default features 7.00 GB
  / 110 s CPU before, 7.55 GB / 121 s after; `--no-default-features` 5.89 GB before, 5.92 GB after
  (the gated modules are only parsed). The test package `nalgebra_tests_dynamic` (`dynamic` +
  `closures`) peaks at 7.9 GB (`scarb build --test`).
- Panics keep upstream's meaning with the existing `felt252` messages: `nalgebra: dimension
  mismatch` (operand shapes), `nalgebra: index out of bounds` (component, insertion / removal
  index), `nalgebra: wrong slice length` (`from_vec` / `from_iterator` length), `nalgebra: diagonal
  too long` (`from_partial_diagonal`).

## D6 — Algorithms

| Function | Algorithm | Cost / accuracy (benchmarks/scalar) |
|---|---|---|
| scalar (`fixed`) | fixed-cairo's kernels, see its `docs/DESIGN.md`; nalgebra-cairo measures them through `Real` | add 640, mul 1,580, div 3,300, recip 2,820, sqrt 1,820, sin_cos 31,500, atan2 29,930 |
| norm | `fixed::wide::norm*` / `Acc::sqrt` of the *unscaled* sum of squares | 2,220 (norm3) |
| Jacobi `c = 1/√(1+t²)` | `recip(sqrt(1 + t²))` (more accurate SVD3 than the 96-bit-reciprocal `inv_norm2`, kept as the loser) | 6,750 |
| det / inverse ≤ 4 | closed forms (cofactors; 4x4 determinant from 2x2 minors) on fused kernels, integer pre-scaled inverse; `try_inverse` returns `Option` on an exactly zero determinant | det3 10,350 / inv3 88,360 |
| Cholesky / LDLᵀ ≤ 6 | unrolled; LDLᵀ is the crate-internal kernel (no sqrt, indefinite accepted), exposed as upstream's `UDU` (run on the reversed matrix `J·A·J`) since WP 8.0 | new+solve 3x3: LDLᵀ 45,660 vs Cholesky 66,090; 6x6: 161,000 vs 200,690 |
| LU ≤ 6 | unrolled with partial pivoting (unpivoted variant fails on a permuted identity) | new 3x3 37,840, solve 27,240; 6x6 new 252,920, solve 72,240 |
| symmetric eigen 2x2 / 3x3 | closed form 2x2; 4-sweep cyclic Jacobi 3x3 (the fixed point: a 5th sweep changes nothing), residual ≤ 31 ulp·max(1, max\|m\|) | 27,330 / 590,090 |
| SVD 2x2 / 3x3, polar | symmetric eigen of `MᵀM`, renormalised eigenvectors, `σ = \|M·v\|`, `U` orthonormal by construction | 72,360 / 725,890 |
| QR 2/3/4 | modified Gram-Schmidt (Householder is 2.7× dearer and further from upstream's factors) | new 32,380 / 77,890 / 151,680 |
| rotations | `UnitComplex` / `UnitQuaternion` as raw pairs/quads (no `Unit` wrapper), Hamilton product on the `Wide` accumulator, algebraic `rotation_between`, quaternion transform for 1 vector and matrix for ≥ 2 | `q*q` 11,550, `uq.transform_vector` 22,070, `rotation_between` 62,870 |
| isometries | `rotate_translate` fused kernel (translation folded into the accumulator), direct `inv_mul` on the crate-internal fused `conj_mul` (conjugate signs folded into the accumulation, no negation), quaternion internally, `lerp_nlerp` (trig-free) | `Isometry3::transform_point` 23,270, `inv_mul` 38,640, `*` 35,320 |

Figures are net gas on `fixed` 0.4.0 (snapshots in `gas/`). When the best implementation is ambiguous, ship the variants behind one trait
(`one trait, one impl per algorithm`), benchmark them side by side, export the cheapest.

## D7 — Tests and gas tracking (definition of done)

Every public function ships with:
1. `test_<fn>_<scenario>` unit tests: exact values for integer-valued cases, golden vectors from the
   oracle (`tools/oracle`: Rust nalgebra in f64 on inputs quantised to Q32.32) with a tolerance in
   ulp, identities (`M·M⁻¹ ≈ I`, `q → R → q`), and `#[should_panic]` for overflow / singular cases;
2. `bench_<group>__<variant>` gas tests (`#[inline(never)]`, `black_box` inputs, group baseline),
   one variant per alternative implementation when several exist;
3. an updated `gas_report.json` snapshot — CI fails on any difference, so every gas change is
   visible in the PR diff.

Scalar kernels additionally have a bit-exact Python integer model used to generate expectations.

## D9 — Families are Scarb features, all in `default` (owner, 2026-09-26)

Compiling `nalgebra` costs every consumer and every test package the whole library (6.1 GB peak,
92 s CPU cold after WP 8.3-P06; heaviest test packages ~10.4 GB against 16 GB CI runners), and
~350 parity items remain. Measured in spike WP 8.1d (`tools/shapegen/DESIGN.md` §2.11, raw data in
`tools/shapegen/budget-results.jsonl`): Scarb 2.19.4 features resolve per compilation unit, and
gating `statistics`, `blas` and the closure methods alone cuts the library by 19 % (memory) and
39 % (CPU), about 1 GB per test package that opts out.

- Every **leaf family** (nothing ungated in the crate uses it) is a Scarb feature, and **every
  feature is in `default`**: `nalgebra = "x.y"` exposes the full nalgebra-rs surface (parity holds
  by default). A feature may be added to `default`, never removed from it.
- New families are gated from day one: dynamic matrices, sparse, glam conversions, macros; the
  completion / views / geometry surfaces follow once their internal uses are cut (spike plan §6).
- Test packages depend on `nalgebra` with `default-features = false` plus the features they test
  (helper crates such as `tests_utils` too: features are unified per compilation unit).
- `scarb build -p nalgebra --no-default-features` must pass: part of the local checks of any WP
  that touches a gated family.
- Structural deviation from nalgebra-rs (which has no such features), documented here and in the
  README: a consumer may write `nalgebra = { version = "x.y", default-features = false, features
  = [...] }` for a lighter build.

## D10 — Experimental Cairo features (owner, 2026-09-26)

Two experimental features of the Cairo compiler, both enabled by corelib itself, are enabled in
`crates/nalgebra/Scarb.toml` (`experimental-features`):

- `associated_item_constraints`: the `Sum` / `Product` impls of upstream (`Iterator` item
  constraints) and simpler generic bounds (e.g. the generic QR methods);
- `user_defined_inline_macros`: the construction macros `matrix!`, `vector!`, `point!`,
  `dmatrix!`, `dvector!` (and the other macros of nalgebra-rs) as declarative Cairo macros,
  without a Rust procedural-macro plugin (feature `macros`, in `default`). Measured by WP P21: a
  consumer needs NO experimental feature to call the macros or `iter.sum()` / `iter.product()`.
  Cairo 2.19 facts: a macro body resolves prelude names only through `$defsite::`, which also
  reaches private items of the defining crate.

A toolchain bump that changes either feature is handled in its dedicated PR (`benchmarks/`
re-run, as every toolchain bump).

## D11 — glam conversions in a separate package `nalgebra_glam` (owner, 2026-09-26)

nalgebra-rs puts its glam conversions in the crate behind a `convert-glam0XX` feature. Scarb 2.19.4
has no optional dependencies (`optional = true` is rejected), and depending on `glam = "0.4.0"`
costs every consumer +0.46 GB / +4 s CPU of cold compile (measured on an empty package). The
conversions therefore live in the package `nalgebra_glam` (`crates/nalgebra_glam`), published
together with `nalgebra` 0.1.0 and counted in `docs/API_PARITY.md`: a project that wants them
adds `nalgebra_glam` next to `nalgebra` and `glam`, the Cairo counterpart of enabling the feature,
and brings the impls into scope with `use nalgebra_glam::prelude::*;` (Cairo finds an `Into` impl
only when it is in scope). Publication order: `nalgebra` before `nalgebra_glam`.

## D8 — Interop with glam-cairo / rapier-cairo

One scalar across the three repositories: fixed-cairo's `fixed::Fixed`, a registry dependency pinned
by version (rapier-cairo must pin the same `fixed` / `glam` / `glamx` version, or two incompatible
`Fixed` types meet). nalgebra's results are bit-identical to glam's wherever both call the same
`fixed` kernel. Conversions (WP 6.2) follow `nalgebra/src/third_party/glam`: mind `Matrix3::new`
(row-major) vs `from_cols`, and `Quaternion::new(w, i, j, k)` vs `from_xyzw`. A module importing both
`fixed::FixedTrait` (or `TrigTrait` / `ExpTrait`) and `simba::prelude::*` gets E2046 on method syntax
(`x.abs()`): call `Real::abs(x)` or import one of them.
