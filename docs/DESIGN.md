# Design decisions

Each decision is justified by a measurement in [BENCHMARK.md](BENCHMARK.md). A decision may only
be overturned by a new measurement committed under `benchmarks/`.

## D1 — Workspace

```
Scarb.toml            virtual workspace, shared versions, edition 2024_07, no `starknet` dependency
crates/simba/         package `simba`: scalar traits + `Fixed` (Q32.32) + fused kernels + transcendentals
crates/nalgebra/      package `nalgebra`: `base`, `geometry`, `linalg` modules (mirrors upstream)
benchmarks/           standalone workspaces: design-time micro-benchmarks (never a dependency)
scripts/              gas_report.py (report + snapshot + CI gate), check.sh
tools/                offline generators (polynomial coefficients, test vectors) — outputs are committed
docs/                 research, benchmark synthesis, design, roadmap
```

Upstream nalgebra depends on `simba` for its scalar abstraction; we keep the same split so the
scalar can be shared with glam.cairo / rapier.cairo without pulling linear algebra.

## D2 — Scalar: Q32.32 on native `i64`, BoundedInt kernels

- `#[derive(Copy, Drop, PartialEq, Serde, Default, Hash)] struct Fixed { raw: i64 }` — one felt,
  unique zero, range ±2.1e9, resolution 2.3e-10.
- add / sub / neg / comparisons: native checked `i64` ops.
- mul and every sum of products: `core::internal::bounded_int` (feature `bounded-int-utils`):
  unscaled products in a typed wide accumulator, one biased floor `div_rem` by 2^32, one
  range check (the only overflow check). 1,750 gas per mul, 2,150 per dot3 (as shipped in `simba`).
- div: floor division, biased numerator, one branch on the divisor sign (2,740-2,820); `rem` is the
  floored modulo (sign of the divisor, unlike Rust's truncated `%`). Division is the expensive op: APIs take and store
  reciprocals where upstream divides repeatedly (inverse mass, inverse inertia, `recip()`).
- **Rounding: floor, once per output scalar.** Part of the numeric spec (observable in the last bit).
- **Overflow: panic.** Never wrap: a wrapped value would still be a valid proof.
- Fallback if `bounded-int-utils` ever breaks: same storage and API on corelib `i64_wide_mul` +
  unsigned `DivRem` (≈ +60 % on mul). The internal feature is confined to `simba::fixed::kernels`.

## D3 — Scalar abstraction

nalgebra types are generic over `T` with a `Real`-style trait (measured zero-cost after
monomorphisation), so an alternative scalar (glam.cairo's, a future Q64.64) plugs in by
implementing the trait. The trait exposes **fused kernels**, not just operators:

```cairo
pub trait Real<T> {
    // constants, conversions, abs / signum / min / max / clamp / floor / ceil / round / recip
    fn sum_prod2(a0: T, b0: T, a1: T, b1: T) -> T;            // a0*b0 + a1*b1, one rescale
    fn sum_prod3(..) -> T;  fn sum_prod4(..) -> T;
    fn diff_prod(a: T, b: T, c: T, d: T) -> T;                 // a*b - c*d (cross, det2)
    fn mul_add(a: T, b: T, c: T) -> T;  fn lerp(a: T, b: T, t: T) -> T;
    fn norm2(x: T, y: T) -> T;  fn norm3(..) -> T;  fn norm4(..) -> T;   // sqrt of raw sum of squares, no rescale
    fn sqrt(self: T) -> T;  fn inv_sqrt(self: T) -> T;
    fn sin_cos(self: T) -> (T, T);  fn atan2(y: T, x: T) -> T;  fn acos(self: T) -> T;  // ...
}
```

Longer sums (6-term rows of `Matrix6`, dynamic dot products) use the explicit `Wide` accumulator
(`wide_add_prod` / `wide_sub_prod` / `wide_rescale`; +200 gas per extra product). Every sum-of-products in the library goes through a fused kernel:
**never `a * b + c * d` with two rescales.**

Approximate equality (`abs_diff_eq`, `relative_eq`) is defined on raw units (ulp), since
nalgebra's float epsilons are meaningless in fixed point.

## D4 — Types: static, unrolled, by value

- Named-field `Copy` structs: `Vector2/3/4/6<T>`, `Matrix2/3/4/6<T>` (`m11..m33`, `Matrix3::new`
  takes row-major arguments like upstream; Serde order is column-major like upstream), `Point2/3`,
  `Unit<V>`, `Quaternion`, `UnitQuaternion`, `UnitComplex`, `Rotation2/3`, `Translation2/3`,
  `Isometry2/3`, `Similarity2/3`.
- `Matrix6` is 2x2 blocks of `Matrix3`; `Vector6` is two `Vector3` (spatial algebra layout).
- First-class structured types: `SymMatrix3` (6 fields, rapier's `SdpMatrix3`), `SymMatrix2`.
  Never materialise skew-symmetric or diagonal matrices: provide `cross_matrix_mul`, `gcross`,
  `quadform` (`R diag(d) Rᵀ → Sym3`), `mul_transpose → Sym3`.
- No loops, no `Array`, no `Span` in static types. Hot kernels are explicit scalar formulas over
  fused scalar kernels, not compositions of vector ops.
- By value everywhere (no `@T`, no `Box`): identical Sierra, simpler API.
- Operators: `Add/Sub/Neg` on vectors and matrices, `Mul` for `M*M` and `Q*Q`. Heterogeneous
  products are named methods as corelib operators are homogeneous: `m.mul_vec(v)`, `v.scale(k)`,
  `iso.transform_point(p)`. Upstream names are kept wherever an operation exists upstream.
- `#[inline(always)]` on small ops (constructors, accessors, add, sub, neg, scale, dot, cross);
  default inlining for large kernels (mat*mat, inverse, isometry composition). Implement ops as
  methods of generic impls (the attribute is refused on generic free functions).

## D5 — Dynamic types (secondary)

`DVector<T>` / `DMatrix<T>`: `{ data: Span<T>, rows, cols }`, row-major, immutable; sequential
traversal (`pop_front` / `multi_pop_front::<N>`), no index arithmetic in inner loops, lazy wide
accumulation in inner products, dispatch to the static kernels when the shape is ≤ 4 or 6.
Built only once the static surface is complete, and scoped by what multibody dynamics needs.

## D6 — Algorithms

| Function | Algorithm | Cost / accuracy (benchmarks/scalar) |
|---|---|---|
| sqrt | corelib `u128_sqrt` on the widened value (exact floor) | 1,820 |
| norm | `u128_sqrt` of the *unscaled* sum of squares (no rescale, overflow tolerant) | 2,220 |
| inv_sqrt | `isqrt(floor(2^96 / raw))`: exact floor, one rounding | 2,830 |
| sin, cos, sin_cos | range reduction by one divmod + typed Horner, degree 11 | 12,500 / 4.7e-10 |
| atan2 | one ratio, typed Horner degree 19 | 18,840 / 1.9e-9 |
| acos, asin | `sqrt(1-\|x\|) · P9(\|x\|)` | 16,510 / 1.2e-9 |
| det / inverse ≤ 4 | closed forms (cofactors) on fused kernels; `try_inverse` returns `Option` | — |
| Cholesky / LDLᵀ ≤ 6 | unrolled; LDLᵀ preferred (no sqrt) | — |
| LU ≤ 6 | unrolled with partial pivoting | — |
| symmetric eigen 2x2 / 3x3 | closed form 2x2; fixed-sweep cyclic Jacobi 3x3 (no convergence loop) | — |
| SVD 2x2 / 3x3, polar | via symmetric eigen of `MᵀM` | — |

Polynomial coefficients and interval-typed Horner code are **generated** (`tools/`), with bounds
proven by the generator, so no overflow checks are needed inside the evaluation.
When the best implementation is ambiguous, ship the variants behind one trait
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

## D8 — Interop with glam.cairo / rapier.cairo

One scalar representation across the three repositories (Q32.32 `i64`, floor). Conversions follow
`nalgebra/src/third_party/glam`: mind `Matrix3::new` (row-major) vs `from_cols`, and
`Quaternion::new(w, i, j, k)` vs `from_xyzw`. Where the scalar type finally lives (here in `simba`
or in glam.cairo) is an open coordination point; nalgebra being generic over `Real` keeps both
options open.
