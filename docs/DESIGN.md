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
(`wide_add_prod` / `wide_sub_prod` / `wide_rescale`; +200 gas per extra product). `wide_mul_scalar(w, s)` is the terminal
`Wide × T` op, `floor(w·s / 2^64)`: exact triple products `(a·b − c·d)·e` with ONE rounding (1,580 net, cheaper than a
plain `mul`; reserved for the 4x4 / 6x6 cofactor determinants). Every sum-of-products in the library goes through a fused kernel:
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
| sin, cos, sin_cos | octant reduction at scale 2^61 + generated interval-typed Horner (degree 11/10), fits of the form `1 + z·G(z)` so key values are exact | 12,800 / 16,820 (pair) — 0.53 ulp |
| tan | sin/cos at 2^-48 then one division | 19,500 |
| atan, atan2 | division-free fast path for \|x\| ≤ 1, generated Horner | 13,600 / 15,400 — 1.12 ulp |
| asin, acos | generated Horner; `acos = 2·asin(sqrt((1-x)/2))` above 1/2 | 11,420 / 11,430 — ~1 ulp |
| exp, ln | reduction to a power of two by a generated comparison tree, then Horner | 14,290 / 12,580 — 0.5 ulp |
| det / inverse ≤ 4 | closed forms (cofactors; 4x4 determinant from 2x2 minors) on fused kernels, integer pre-scaled inverse; `try_inverse` returns `Option` on an exactly zero determinant | det3 10,660 / inv3 88,940 |
| Cholesky / LDLᵀ ≤ 6 | unrolled; LDLᵀ preferred (no sqrt, indefinite accepted) | new+solve 3x3: LDLᵀ 36,990 vs Cholesky 53,800; 6x6: 132,590 vs 165,330 |
| LU ≤ 6 | unrolled with partial pivoting (unpivoted variant fails on a permuted identity) | new 3x3 33,260, solve 23,560; 6x6 new 239,410, solve 65,020 |
| symmetric eigen 2x2 / 3x3 | closed form 2x2; 4-sweep cyclic Jacobi 3x3 (the fixed point: a 5th sweep changes nothing), residual ≤ 31 ulp·max(1, max\|m\|) | 24,530 / 550,770 (eigenvalues only 303,470) |
| SVD 2x2 / 3x3, polar | symmetric eigen of `MᵀM`, renormalised eigenvectors, `σ = \|M·v\|`, `U` orthonormal by construction | 64,740 / 675,400 |
| QR 2/3/4 | modified Gram-Schmidt (Householder is 2.7× dearer and further from upstream's factors) | new 26,710 / 66,580 / 134,840 |
| rotations | `UnitComplex` / `UnitQuaternion` as raw pairs/quads (no `Unit` wrapper), Hamilton product on the `Wide` accumulator, algebraic `rotation_between`, quaternion transform for 1 vector and matrix for ≥ 2 | `q*q` 11,860, `uq.transform_vector` 23,230, `rotation_between` 49,440 |
| isometries | `rotate_translate` fused kernel (translation folded into the accumulator), direct `inv_mul` on the fused `conj_mul` (conjugate signs folded into the accumulation, no negation), quaternion internally, `lerp_nlerp` (trig-free) | `Isometry3::transform_point` 24,430, `inv_mul` 40,110, `*` 36,790 |

Polynomial coefficients and interval-typed Horner code are **generated** (`tools/polygen`), with
bounds proven by the generator, so no overflow checks are needed inside the evaluation; the Python
model replays the generated programs, so it cannot diverge from the Cairo (`polygen.py --check`).
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
`Quaternion::new(w, i, j, k)` vs `from_xyzw`. Settled: glam.cairo owns the shared `fixed::Fixed`
(rapier.cairo consumes it), nalgebra stays generic over `Real`, and `simba_fixed` bridges by
implementing `Real` / `Transcendental` for `fixed::Fixed` through simba's own kernels (zero
overhead, bit-identical wherever nalgebra goes through `Real`). Known gap: nalgebra still reaches
`/` through the corelib operator, and `fixed::Fixed` truncates where simba floors — a
`Real::div` / `Real::rem` routing would close it (WP 4.5).
