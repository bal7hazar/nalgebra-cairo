# Design decisions

Each decision is justified by a measurement in [BENCHMARK.md](BENCHMARK.md). A decision may only
be overturned by a new measurement committed under `benchmarks/`.

## D1 — Workspace

```
Scarb.toml            virtual workspace, shared versions, edition 2024_07, no `starknet` dependency
crates/simba/         package `simba`: scalar traits (`Real`, `Transcendental`) implemented for glam.cairo's `fixed::Fixed`
crates/nalgebra/      package `nalgebra`: `base`, `geometry`, `linalg` modules (mirrors upstream)
benchmarks/           standalone workspaces: design-time micro-benchmarks (never a dependency)
scripts/              gas_report.py (report + snapshot + CI gate), check.sh
tools/oracle/         Rust generator of golden vectors from upstream nalgebra (f64) — outputs are committed
docs/                 research, benchmark synthesis, design, roadmap
```

Upstream nalgebra depends on `simba` for its scalar abstraction, and simba is a trait layer over the
primitive `f32` / `f64` shared by glam, parry and rapier. We mirror that exactly (WP 7.1): the primitive
is glam.cairo's `fixed` package (registry dependency, pinned version), `simba` only implements the
traits for it, and nalgebra is generic over them.

## D2 — Scalar: glam.cairo's `fixed::Fixed` (Q32.32 on `i64`)

- There is ONE Q32.32 implementation in the stack: `fixed` (glam.cairo), `Fixed { raw: i64 }`,
  range ±2.1e9, resolution 2.3e-10. nalgebra.cairo has none of its own (the former `simba::fixed`
  was deleted in WP 7.1: two implementations with independent rounding choices would break the
  bit-exact determinism rapier.cairo relies on when it mixes glam and nalgebra values).
- Pinned by version (`fixed = "0.3.0"`): glam.cairo bumps MINOR on any numeric change, so the
  version pins every golden file here.
- Numeric semantics are `fixed`'s: `+ - neg` checked `i64`; `*`, fused kernels (`wide::dot*`,
  `mul_add`, `mul_sub`, `Acc`) rescale ONCE with **floor**; `sqrt` / `norm*` floor of the exact root;
  **`/`, `recip`, `from_ratio` round to nearest, ties to even, like `f64 /`** (owner's ruling: the
  Rust reference); `%` is the exact truncated remainder, like Rust's float `%`; constants are
  rounded to nearest; `fixed::trig` rounds its final rescale to nearest.
- **Overflow: panic** (`'Fixed: overflow'`, `'Fixed: division by zero'`, ...). Never wrap.
- Anything `fixed` lacks (a kernel, a rounding mode) is an escalation to the glam.cairo
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
| scalar (`fixed`) | glam.cairo's kernels, see its `docs/DESIGN.md`; nalgebra.cairo measures them through `Real` | add 640, mul 1,580, div 3,300, recip 2,820, sqrt 1,820, sin_cos 31,500, atan2 29,930 |
| norm | `fixed::wide::norm*` / `Acc::sqrt` of the *unscaled* sum of squares | 2,220 (norm3) |
| Jacobi `c = 1/√(1+t²)` | `recip(sqrt(1 + t²))` (more accurate SVD3 than the 96-bit-reciprocal `inv_norm2`, kept as the loser) | 6,750 |
| det / inverse ≤ 4 | closed forms (cofactors; 4x4 determinant from 2x2 minors) on fused kernels, integer pre-scaled inverse; `try_inverse` returns `Option` on an exactly zero determinant | det3 10,350 / inv3 88,360 |
| Cholesky / LDLᵀ ≤ 6 | unrolled; LDLᵀ preferred (no sqrt, indefinite accepted) | new+solve 3x3: LDLᵀ 45,660 vs Cholesky 66,090; 6x6: 161,000 vs 200,690 |
| LU ≤ 6 | unrolled with partial pivoting (unpivoted variant fails on a permuted identity) | new 3x3 37,840, solve 27,240; 6x6 new 252,920, solve 72,240 |
| symmetric eigen 2x2 / 3x3 | closed form 2x2; 4-sweep cyclic Jacobi 3x3 (the fixed point: a 5th sweep changes nothing), residual ≤ 31 ulp·max(1, max\|m\|) | 27,330 / 590,090 |
| SVD 2x2 / 3x3, polar | symmetric eigen of `MᵀM`, renormalised eigenvectors, `σ = \|M·v\|`, `U` orthonormal by construction | 72,360 / 725,890 |
| QR 2/3/4 | modified Gram-Schmidt (Householder is 2.7× dearer and further from upstream's factors) | new 32,380 / 77,890 / 151,680 |
| rotations | `UnitComplex` / `UnitQuaternion` as raw pairs/quads (no `Unit` wrapper), Hamilton product on the `Wide` accumulator, algebraic `rotation_between`, quaternion transform for 1 vector and matrix for ≥ 2 | `q*q` 11,550, `uq.transform_vector` 22,070, `rotation_between` 62,870 |
| isometries | `rotate_translate` fused kernel (translation folded into the accumulator), direct `inv_mul` on the fused `conj_mul` (conjugate signs folded into the accumulation, no negation), quaternion internally, `lerp_nlerp` (trig-free) | `Isometry3::transform_point` 23,270, `inv_mul` 38,640, `*` 35,320 |

Figures are net gas on `fixed` 0.3.0 (snapshots in `gas/`). When the best implementation is ambiguous, ship the variants behind one trait
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

One scalar across the three repositories: glam.cairo's `fixed::Fixed`, a registry dependency pinned
by version (rapier.cairo must pin the same `fixed` / `glam` / `glamx` version, or two incompatible
`Fixed` types meet). nalgebra's results are bit-identical to glam's wherever both call the same
`fixed` kernel. Conversions (WP 6.2) follow `nalgebra/src/third_party/glam`: mind `Matrix3::new`
(row-major) vs `from_cols`, and `Quaternion::new(w, i, j, k)` vs `from_xyzw`. A module importing both
`fixed::FixedTrait` (or `TrigTrait` / `ExpTrait`) and `simba::prelude::*` gets E2046 on method syntax
(`x.abs()`): call `Real::abs(x)` or import one of them.
