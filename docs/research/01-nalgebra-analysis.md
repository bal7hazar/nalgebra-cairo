# nalgebra — source-grounded analysis for a Cairo port (`nalgebra.cairo`)

Reference checkouts analysed (all paths below are relative to
`scratchpad/refs/` unless absolute):

| Repo | Version | HEAD |
|---|---|---|
| `nalgebra/` | 0.35.0 (edition 2024, MSRV 1.89) | `3320ecc` — 2026-06-30 "fix: Cholesky::new returns None non-positive-definite complex matrices" |
| `rapier/` | 0.35.3 | `28d0ba9` — 2026-09-19 "feat: add support for soft-bodies" |
| `parry/` | 0.31.1 | `3383f51` — 2026-09-18 "Release v0.31.1" |
| `glamx` 0.3.0, `simba` 0.10.0 | (read from `~/.cargo/registry/src/*/`) | — |

Size of the thing being ported (Rust LoC, `wc -l`):

| Area | LoC |
|---|---|
| `nalgebra/src/base` | 19 009 |
| `nalgebra/src/geometry` | 21 630 |
| `nalgebra/src/linalg` | 8 180 |
| `nalgebra/src/sparse` (legacy) | 1 683 |
| `nalgebra-glm` + `nalgebra-sparse` + `nalgebra-lapack` + `nalgebra-macros` | 20 921 |
| `nalgebra/tests` | 10 116 (+ ~425 doctest blocks in `src/`) |

> **Headline finding (details in §3).** The physics stack has *already left nalgebra* for its
> public/scalar math. `parry` 0.26+ has **zero** dependency on nalgebra (it uses `glamx`, a thin
> extension crate over `glam`). `rapier` 0.32+ uses glam/glamx for every scalar (`Real`) code path
> and keeps nalgebra for exactly three things: (a) the AoSoA-SIMD instantiation of the constraint
> solver (`na::Vector3<SimdReal>` etc.), (b) multibody joints / IK (`DMatrix`, `DVector`,
> `Matrix6xX`, `LU<Dyn,Dyn>`, BLAS-like `gemm/gemv/quadform`, `pseudo_inverse`), and (c) a few
> soft-body helpers (`SMatrix` Cholesky, `Matrix3::symmetric_eigen`). In Cairo there is no SIMD, so
> (a) collapses onto the scalar path. **Consequently `glam.cairo` owns the hot path of
> `rapier.cairo`; `nalgebra.cairo` is needed for small-N / dynamic-N linear algebra
> (multibody, soft body, IK) and as the generic-dimension fallback.**

---

## 1. Crate / module architecture

### 1.1 Workspace layout

```
nalgebra/
├── Cargo.toml            workspace root + the `nalgebra` crate
├── src/
│   ├── lib.rs            re-exports base::*, geometry::*, linalg::*; free fns (clamp, wrap, zero, one, convert, distance, center…)
│   ├── base/             Matrix<T,R,C,S>, storage, dims, allocator, ops, BLAS-like kernels, norms, Unit<T>
│   ├── geometry/         Point, Rotation, Quaternion, UnitComplex, DualQuaternion, Translation, Isometry,
│   │                     Similarity, Transform (Affine/Projective), Scale, Reflection, Perspective3, Orthographic3
│   ├── linalg/           decompositions + solve/inverse/determinant/exp/pow/convolution
│   ├── sparse/           LEGACY CsMatrix (feature "sparse"); superseded by nalgebra-sparse
│   ├── io/               Matrix-Market parser (pest grammar, feature "io")
│   ├── debug/            RandomOrthogonal, RandomSDP test helpers (feature "debug")
│   ├── proptest/         proptest strategies for matrices/vectors (feature "proptest-support")
│   └── third_party/      conversions: glam 0.30–0.33, mint, alga, encase
├── nalgebra-macros/      proc-macros: matrix!, vector!, point!, dmatrix!, dvector!, stack!
├── nalgebra-glm/         GLM-flavoured free-function façade (TVec3, TMat4, Qua; glm::rotate, glm::look_at…)
├── nalgebra-sparse/      COO / CSR / CSC, sparsity patterns, spmm/spadd/spsolve, CscCholesky, MatrixMarket IO
├── nalgebra-lapack/      LAPACK-backed Cholesky/LU/QR/ColPivQR/SVD/Eigen/SymmetricEigen/Schur/Hessenberg/QZ
├── tests/                integration tests (core/, geometry/, linalg/, sparse/, macros/, proptest/)
└── benches/              criterion benches
```

Dependencies that shape the design (`nalgebra/Cargo.toml`): `simba` (scalar abstraction),
`num-traits`, `num-complex`, `num-rational`, `approx` (tolerance comparisons), `typenum`
(type-level integers, legacy bridge to const generics), `matrixmultiply` (optimised f32/f64 gemm on
`std`). Everything else is optional (serde, rkyv, rand, bytemuck, mint, glam, rayon, …).

### 1.2 `src/base` — the core type design

| File | Content | Cairo relevance |
|---|---|---|
| `base/matrix.rs` (2381 L) | `pub struct Matrix<T, R, C, S> { pub data: S, _phantoms }`. Generic methods: `transpose`, `adjoint`, `map`, `zip_map`, `fold`, `apply`, `cast`, `diagonal`, `trace`, `symmetric_part`, `to_homogeneous`/`from_homogeneous`, `push`, `cross`, `perp`, `cross_matrix`, `angle`, `scale`/`unscale`, iterators, `Eq`/`PartialOrd`/approx impls. | Methods: yes. The 4-parameter generic struct: no (see 1.3). |
| `base/dimension.rs` | `Dim` trait, `Dyn(pub usize)` (runtime), `Const<const R: usize>` (compile-time), `DimName`, type-level arithmetic `DimAdd/DimSub/DimMul/DimMin/DimMax/DimDiff/DimSum…`, `U1..U127` aliases, `ToTypenum`/`ToConst` bridges. | Meaningless in Cairo: no const-generic arithmetic, no type-level `DimSum`. Replace by concrete types. |
| `base/storage.rs`, `array_storage.rs`, `vec_storage.rs`, `matrix_view.rs` | `RawStorage`/`RawStorageMut`/`Storage`/`StorageMut`/`IsContiguous`/`ReshapableStorage`. `ArrayStorage<T,R,C>([[T; R]; C])` — **column-major**, stack. `VecStorage<T,R,C>` — heap `Vec<T>` + runtime dims. `ViewStorage(Mut)` — pointer + shape + (row,col) strides, borrowed sub-matrix. | Only the *semantics* matter: column-major ordering (affects `from_column_slice`, iteration order, indexing `m[i]`). Views are meaningless: Cairo has no mutable aliasing/pointers; `Span<T>` is read-only, `Array<T>` is append-only. |
| `base/allocator.rs`, `default_allocator.rs` | `Allocator<R,C>` trait with associated `Buffer<T>`; `DefaultAllocator` picks `ArrayStorage` for `Const×Const`, `VecStorage` if any dim is `Dyn`. `Reallocator` for resize. Source of the infamous `where DefaultAllocator: Allocator<D,D>` bounds. | Meaningless. It exists purely to let one generic impl serve static and dynamic sizes. |
| `base/constraint.rs` | `ShapeConstraint` + `DimEq`, `SameNumberOfRows/Columns`, `AreMultipliable` — compile-time shape checking for mixed static/dynamic operands. | Meaningless once each shape is its own type (the type system checks shapes for free). For dynamic matrices: runtime `assert`. |
| `base/scalar.rs` | `pub trait Scalar: 'static + Clone + PartialEq + Debug {}` — *no algebra*. Algebra comes from simba (`ClosedAdd…`, `ComplexField`, `RealField`, `SimdRealField`). | Meaningful: keep a small trait stack (see 1.4). |
| `base/alias.rs`, `alias_view.rs`, `alias_slice.rs` | `Matrix1..6`, `MatrixRxC` for all R,C ≤ 6, `MatrixXxN`, `MatrixNxX`, `DMatrix`, `SMatrix<T,R,C>`, `OMatrix`; `Vector1..6`, `RowVector1..6`, `DVector`, `SVector`, `OVector`, `UnitVector1..6`; view aliases. | The alias list *is* the API surface to port: each alias becomes a concrete Cairo struct. |
| `base/coordinates.rs` | `Deref` of `Matrix<…,Const<N>,…>` to `#[repr(C)]` structs `X, XY, XYZ, XYZW, XYZWA, XYZWAB`, `M2x2…M6x6` → gives `v.x`, `m.m12`. | In Cairo just make them real struct fields (`x, y, z` / `m11…m33`). This is the natural representation. |
| `base/construction.rs` | `zeros`, `identity`, `from_element`, `repeat`, `from_fn`, `from_iterator`, `from_row_slice`, `from_column_slice`, `from_rows`, `from_columns`, `from_diagonal`, `from_diagonal_element`, `from_partial_diagonal`, `from_vec`, `new(...)` per shape, `x()/y()/z()/ith()`, `x_axis()…` (as `Unit`), `new_random`, Bounded/Zero/One impls. | Yes (minus random). |
| `base/ops.rs` | `Index/IndexMut` (linear & `(i,j)`), `Neg`, `Add/Sub` (+ all ref combos, + `Assign`), scalar `Mul/Div`, matrix `Mul` (via `gemm`), `tr_mul`, `ad_mul`, `kronecker`, `Sum`/`Product`, `mul_to`. | Yes — one impl per concrete type, fully unrolled. No ref/val permutations needed (Cairo `Copy` structs). |
| `base/blas.rs`, `blas_uninit.rs` | BLAS-1/2/3-style kernels: `dot`, `dotc`, `tr_dot`, `axpy`, `axcpy`, `gemv`, `gemv_tr`, `gemv_ad`, `sygemv`/`hegemv`, `ger`, `gerc`, `syger`/`hegerc`, `gemm`, `gemm_tr`, `gemm_ad`, `quadform`, `quadform_tr` (+ `_with_workspace`). Uses `matrixmultiply` for large f32/f64. | For static types these are just operators. For Tier-4 dynamic matrices they are the right primitives (rapier multibody calls exactly these). |
| `base/norm.rs` | `Norm` trait + `EuclideanNorm`, `LpNorm(i32)`, `UniformNorm`; `norm`, `norm_squared`, `magnitude`, `lp_norm`, `apply_norm`, `metric_distance`, `normalize(_mut)`, `try_normalize(_mut)`, `set_magnitude`, `cap_magnitude`, `simd_try_normalize`, `orthonormalize`, `orthonormal_subspace_basis`. | Yes (L2, L1, L∞). `LpNorm` needs `powf` → skip. |
| `base/unit.rs` | `Unit<T> { value: T }` + `Normed` trait; `new_normalize`, `try_new(eps)`, `new_unchecked`, `new_and_get`, `renormalize`, **`renormalize_fast`** (`v *= 0.5*(3 - |v|²)`, no sqrt), `into_inner`. Newtype that statically tags "norm == 1". `UnitVectorN`, `UnitQuaternion`, `UnitComplex`, `UnitDualQuaternion` are all `Unit<…>`. | Meaningful and cheap in Cairo (a wrapper struct). `renormalize_fast` is a gift for fixed point. |
| `base/componentwise.rs` | `abs`, `component_mul/div` (+`_assign`, `cmpy`, `cdpy`), `inf`, `sup`, `inf_sup`, `add_scalar`. | Yes. |
| `base/min_max.rs` | `amax/amin` (abs), `camax/camin`, `max/min`, `iamax/iamin/argmax/argmin/imax/imin`, `iamax_full`, `icamax`. | Yes (pivot search in LU uses `icamax`). |
| `base/statistics.rs` | `sum`, `product`, `mean`, `variance`, row/column variants, `compress_rows/columns`. | Low value. |
| `base/interpolation.rs` | `Vector::lerp`, `Vector::slerp`, `Unit<Vector>::slerp/try_slerp` (uses `acos`, `sin`, `sqrt`). | lerp yes; slerp optional. |
| `base/cg.rs` | Homogeneous-matrix computer-graphics helpers on `Matrix3/4`: `new_scaling`, `new_translation`, `new_rotation`, `from_scaled_axis`, `from_euler_angles`, `from_axis_angle`, `new_orthographic`, `new_perspective`, `look_at_rh/lh`, `face_towards`, `append/prepend_scaling/translation…`, `transform_point/vector`. | Rendering only → glam.cairo's territory, or out of scope. |
| `base/edition.rs` | Structural edits: `fill*`, `set_row/column/diagonal`, `swap_rows/columns`, `remove_row(s)/column(s)`, `insert_row(s)/column(s)`, `resize*`, `fixed_resize`, `reshape_generic`, `select_rows/columns`, `upper/lower_triangle`. | Tier 4 only (dynamic). |
| `base/indexing.rs`, `iter.rs`, `par_iter.rs` | `get/get_mut/index` with ranges; iterators; rayon column iterators. | Minimal getters only. |
| `base/conversion.rs` | `From` arrays/tuples/nested arrays/slices/views; `SubsetOf` (simba) numeric casts; static⇄dynamic. | `from_array`/`to_array` and tuple conversions only. |
| `base/swizzle.rs` | `xy()`, `yx()`, `xyz()`, `zyx()`, … | Optional sugar (glam.cairo more likely). |
| `base/properties.rs` | `is_square`, `is_identity(eps)`, `is_orthogonal(eps)`, `is_special_orthogonal(eps)`, `is_invertible`. | Yes, cheap and useful for tests. |
| `base/matrix_simba.rs`, `uninit.rs`, `helper.rs`, `rkyv_wrappers.rs` | `SimdValue` impl for matrices (AoSoA lanes), `MaybeUninit` plumbing, serialization. | Meaningless. |

### 1.3 The `Matrix<T, R, C, S>` design: what survives in Cairo

`Matrix<T, R, C, S>` is a single struct parameterised by scalar `T`, row dim `R`, col dim `C`
(each either `Const<N>` or `Dyn`), and storage `S`. *Every* vector, matrix, row-vector, view and
dynamic matrix in nalgebra is this one type (`Vector3<T> = Matrix<T, Const<3>, Const<1>,
ArrayStorage<T,3,1>>`). Ninety percent of the trait machinery (`Allocator`, `ShapeConstraint`,
`DimName`, `Storage`, `SameShapeAllocator`, `DimMin<D, Output = D>` …) exists only so that *one*
generic `impl` block covers static, dynamic, owned and borrowed variants with compile-time shape
checks where possible.

| Abstraction | Rust purpose | Verdict for Cairo |
|---|---|---|
| `Matrix<T,R,C,S>` single type | One impl for all shapes/storages | **Drop.** Use concrete structs `Vector2<T>…Vector6<T>`, `Matrix2<T>…Matrix4<T>` (+ `Matrix6<T>` later) with named fields. Cairo loops cost steps (each iteration = function-call-like recursion + bounds-checked array access); unrolled struct arithmetic compiles to straight-line felt ops. |
| Generic over `T` | f32/f64/complex/SIMD/fixed/rational/dual numbers | **Keep, but thin.** Generic `T` with a `Real`-like trait lets you swap Q-formats (Q32.32 vs Q64.64) and lets tests run with exact integer scalars. Cairo monomorphises generics, so there is no runtime cost, but compile time and Sierra size grow; limit to 1–2 instantiations. |
| `Dim`, `Const<N>`, `Dyn`, typenum | Type-level shapes | **Drop.** Cairo has `[T; N]` but no arithmetic on `N`, no `impl<const N>` over expressions like `N+1` (needed by `to_homogeneous`, `Transform<D>` = `(D+1)×(D+1)`). |
| `Storage`/`ArrayStorage`/`VecStorage`/views | Owned vs borrowed, static vs heap, strides | **Drop** for static types. For Tier 4: one `DMatrix { data: Array<T>/Span<T>, nrows, ncols }`, column-major, immutable-functional (rebuild arrays), no views — or `Felt252Dict`-backed for in-place algorithms (LU, Cholesky). |
| `Allocator`/`DefaultAllocator`/`Reallocator` | Pick storage type from dims | **Drop.** |
| `ShapeConstraint` | Compile-time shape compat | **Drop** (free with concrete types; `assert!` for dynamic). |
| `Scalar` / simba `RealField`/`ComplexField` | Numeric tower | **Keep, reduced** (see 1.4). Complex numbers meaningless for physics → drop `ComplexField`, `conjugate`, `adjoint`, `dotc`, `hegemv`, `gerc`… (on reals they alias `transpose`/`dot`/`gemv`/`ger`). |
| `Unit<T>` | "norm is 1" newtype | **Keep.** Zero-cost in Cairo too; avoids re-normalising, documents invariants (`UnitQuaternion`, `UnitVector3`). |
| SIMD AoSoA (`SimdValue`, `SimdRealField`, `simd_*` methods, `if_else`, `select`, lanes) | `Vector3<f32x4>` = 4 vectors packed | **Meaningless.** The STARK VM has no data parallelism. Note the consequence: nalgebra geometry code is written *branch-free* with `simd_gt(...).if_else3(...)` (e.g. `UnitQuaternion::from_rotation_matrix`, `geometry/quaternion_construction.rs:339`). In Cairo use ordinary `if` — cheaper, since only the taken branch is proven. |
| `approx` (`AbsDiffEq`, `RelativeEq`, `UlpsEq`) impls (43 impl blocks) | Tolerant equality | **Keep `abs_diff_eq` + `relative_eq`** with fixed-point epsilons. `UlpsEq` is float-specific → drop. |
| Ref/val operator permutations (`&a + &b`, `a + &b`, …; generated by `op_macros.rs`) | Avoid moves of big matrices | **Drop.** All static types are `Copy + Drop` in Cairo. |
| serde / rkyv / bytemuck / mint / rand / rayon | Ecosystem glue | **Drop**; `Serde` + `starknet::Store` derives replace serialization. |

### 1.4 The scalar tower (simba) and what to retain

nalgebra's own `Scalar` has no algebra. The algebra lives in `simba`:

* `simba::scalar::ComplexField` — ~60 methods: `real, imaginary, modulus, norm1, scale, unscale,
  signum, floor, ceil, round, trunc, fract, mul_add, abs, hypot, recip, conjugate, sin, cos,
  sin_cos, tan, asin, acos, atan, sinh…atanh, sinc, sinhc, cosc, log*, ln, ln_1p, sqrt, try_sqrt,
  exp, exp2, exp_m1, powi, powf, powc, cbrt, is_finite`.
* `simba::scalar::RealField: ComplexField + PartialOrd` — adds `atan2, clamp, copysign, min, max,
  is_sign_positive/negative, min_value/max_value`, constants `pi, two_pi, frac_pi_2…, e, ln_2…`.
* `SimdRealField`/`SimdComplexField` — lane-wise versions (`simd_sqrt`, `simd_sin_cos`, …);
  for plain scalars they forward to the above.

**Important precedent:** simba already ships a fixed-point backend
(`simba-0.10.0/src/scalar/fixed_impl.rs`, feature `partial_fixed_point_support`, built on the
`fixed` + `cordic` crates). It implements *only* `sqrt, exp, sin, cos, tan, sin_cos, asin, acos,
atan, atan2` (+ arithmetic, `abs`, `floor/ceil/round`, `signum`, comparisons); everything else
(`ln`, `powf`, `sinh`, `cbrt`, `hypot`, …) is `unimplemented!()`. parry even has a feature
`improved_fixed_point_support` (`parry/src/query/gjk/gjk.rs:828`,
`query/point/point_triangle.rs:149`, `transformation/convex_hull3/initial_mesh.rs:47`) that tweaks
epsilon-sensitive branches for fixed point. This is an existence proof that **the 10-function set
{sqrt, sin, cos, tan, asin, acos, atan, atan2, exp, sin_cos} is sufficient for the
dimforge physics stack**, and a pointer to the three spots in parry where fixed-point needed
algorithmic changes.

Proposed Cairo trait stack: `Scalar` (Copy+Drop+PartialEq+Debug) → `Ring` (Add/Sub/Mul/Neg/Zero/One)
→ `Real` (Div, PartialOrd, abs, signum, min, max, clamp, floor/ceil/round, sqrt, recip, constants,
epsilon) → `RealTrig` (sin, cos, sin_cos, tan, asin, acos, atan2) → optional `RealExp` (exp, ln,
powi). Decompositions then bound on exactly what they need (LU: `Real`; Cholesky: `Real`+sqrt;
rotations: `RealTrig`).

### 1.5 `src/geometry`

Every type is split across `X.rs` (struct + inherent methods), `X_construction.rs`,
`X_ops.rs` (operator matrix), `X_conversion.rs` (`SubsetOf`, `From`), `X_alias.rs`,
`X_simba.rs` (`SimdValue`), sometimes `X_coordinates.rs` (Deref to `.x/.y`), `X_interpolation.rs`.

| Type (file) | Definition |
|---|---|
| `OPoint<T, D>` / `Point<T, const D>` (`point.rs:60`) | `{ coords: OVector<T, D> }` — affine point; differs from a vector only in typing (`Point - Point = Vector`, `Point + Vector = Point`, translation affects points not vectors, homogeneous w = 1). Aliases `Point1..6`. |
| `Rotation<T, const D>` (`rotation.rs:72`) | `{ matrix: SMatrix<T, D, D> }` — special-orthogonal matrix. `Rotation2`, `Rotation3`. 2D/3D specifics in `rotation_specialization.rs`. |
| `Quaternion<T>` (`quaternion.rs:45`) | `{ coords: Vector4<T> }` stored `[i, j, k, w]`, constructor order `new(w, i, j, k)`. `UnitQuaternion<T> = Unit<Quaternion<T>>` (`:1020`). |
| `UnitComplex<T>` (`unit_complex.rs:32`) | `Unit<num_complex::Complex<T>>` — 2D rotation as (cos, sin). |
| `DualQuaternion<T>` (`dual_quaternion.rs:61`) | `{ real: Quaternion, dual: Quaternion }`; `UnitDualQuaternion` = rigid motion, `sclerp`. |
| `Translation<T, const D>` (`translation.rs:40`) | `{ vector: SVector<T, D> }`. |
| `Scale<T, const D>` (`scale.rs:40`) | `{ vector: SVector<T, D> }` non-uniform scale. |
| `Isometry<T, R, const D>` (`isometry.rs:92`) | `{ rotation: R, translation: Translation<T, D> }`, generic over `R: AbstractRotation<T, D>` (`abstract_rotation.rs:7`). Aliases: `Isometry2 = Isometry<T, UnitComplex<T>, 2>`, `Isometry3 = Isometry<T, UnitQuaternion<T>, 3>`, `IsometryMatrix2/3` (rotation-matrix flavour). |
| `Similarity<T, R, const D>` (`similarity.rs:56`) | `{ isometry, scaling: T }` (uniform scale, non-zero). |
| `Transform<T, C: TCategory, const D>` (`transform.rs:160`) | `{ matrix: OMatrix<T, D+1, D+1> }`, phantom category `TGeneral`/`TProjective`/`TAffine` with type-level category multiplication (`TCategoryMul`). Aliases `Transform2/3`, `Projective2/3`, `Affine2/3`. |
| `Reflection<T, D, S>` (`reflection.rs:10`) | `{ axis: Vector, bias: T }` — Householder reflector; **used internally by QR/Hessenberg/Bidiagonal** (`linalg/householder.rs`). |
| `Perspective3<T>` (`perspective.rs:44`), `Orthographic3<T>` (`orthographic.rs:42`) | `{ matrix: Matrix4<T> }` projection matrices with getters/setters, `project_point`, `unproject_point`. |

`AbstractRotation` is the only geometry abstraction worth mirroring (a Cairo trait with
`identity, inverse, transform_vector, transform_point, inverse_transform_*`), and only if
`Isometry` is to be shared between quaternion- and matrix-backed rotations; otherwise write
`Isometry2` and `Isometry3` as two concrete structs.

### 1.6 `src/linalg`

Factorisations as structs owning the packed factors, plus `decomposition.rs` which hangs
convenience methods on `Matrix` (`m.lu()`, `m.cholesky()`, `m.svd(true,true)`,
`m.symmetric_eigen()`, `m.polar()`…). Building blocks: `householder.rs`, `givens.rs`,
`permutation_sequence.rs`, `solve.rs` (triangular solves), `balancing.rs`. Full table in §2.3.

### 1.7 Satellite crates

| Crate | What it is | Cairo verdict |
|---|---|---|
| `nalgebra-macros` | `matrix![1,2;3,4]`, `vector![]`, `point![]`, `dmatrix![]`, `dvector![]`, `stack![]` (block matrices). rapier re-exports `vector!`/`point!` in its prelude. | Cairo has no proc-macros usable like this; `Vector3Trait::new(x,y,z)` suffices. Out of scope. |
| `nalgebra-glm` | GLSL-style free functions (`glm::dot`, `glm::rotate`, `glm::perspective`, `glm::quat_slerp`…) over `TVec<T,N>`, `TMat<T,R,C>`, `Qua<T>`. No new maths. | Out of scope (pure façade). |
| `nalgebra-sparse` | `CooMatrix`, `CsrMatrix`, `CscMatrix`, `SparsityPattern`, ops (`spmm_*`, `spadd_*`, `spsolve_csc_lower_triangular`), `CscCholesky` (symbolic+numeric), conversions, MatrixMarket IO. | Out of scope. (rapier's FEM soft body wrote its *own* `SkylineCholesky`, `rapier/src/dynamics/solver/soft_fem/soft_fem_skyline.rs`, rather than use this.) |
| `nalgebra-lapack` | FFI to LAPACK/OpenBLAS/MKL/Accelerate. | Meaningless. |
| `src/sparse` | Older `CsMatrix`, `CsCholesky`, feature-gated. | Out of scope. |
| `src/io` | Matrix Market text parser. | Out of scope. |
| `src/third_party/glam` | `From` conversions nalgebra⇄glam (`glam_isometry.rs`, `glam_matrix.rs`, `glam_point.rs`, `glam_quaternion.rs`, `glam_rotation.rs`, `glam_similarity.rs`, `glam_translation.rs`, `glam_unit_complex.rs`) for glam 0.30–0.33. | **Meaningful as a spec**: it defines the exact type correspondences `nalgebra.cairo ⇄ glam.cairo` should offer (Vec3⇄Vector3, Quat⇄UnitQuaternion, Mat3⇄Matrix3/Rotation3, (Vec3,Quat)⇄Isometry3, Affine3A⇄Similarity…). rapier relies on it (`glamx` feature `nalgebra`; `rapier/src/utils/mod.rs:107-130` `vect_to_na`, `mat_to_na`). |

---

## 2. Feature inventory

Legend for "Num" column: `+−×` ring ops only · `÷` division · `√` sqrt · `trig` sin/cos ·
`atrig` asin/acos/atan2 · `exp` exp/ln/powf · `iter` data-dependent iteration (convergence loop).

### 2.1 Base: vectors and matrices

#### Vectors — `Vector1..6`, `RowVector1..6`, `SVector<T,N>`, `DVector`, `UnitVector1..6`

| Group | Operations (source) | Num |
|---|---|---|
| Constructors | `new(x,…)`, `zeros`, `repeat`/`from_element`, `from_fn`, `from_iterator`, `from_row_slice`/`from_column_slice`, `from_vec`, `x()/y()/z()/w()/a()/b()`, `ith(i, v)`, `x_axis()…` → `Unit`, `ith_axis`, `identity` (`base/construction.rs`) | — |
| Access | fields `.x .y .z .w .a .b` via Deref (`base/coordinates.rs`), `v[i]`, `get(i)`, `iter`, `as_slice`, swizzles `xy() xyz() yzx()…` (`base/swizzle.rs`), `fixed_rows::<N>(i)`, `rows(i,n)` views, `push(e)` (N→N+1), `remove_row`, `insert_row` | — |
| Ops traits | `Neg`, `Add/Sub` (+Assign), `Mul<T>/Div<T>` (+Assign), `T * v` for primitive T, `Index/IndexMut`, `PartialEq`, `PartialOrd` (component-wise partial order!), `Sum`, `Zero`, `Bounded`, `Default`, `Hash`, `Display` (`base/ops.rs`, `base/matrix.rs`) | +−×÷ |
| Products | `dot`, `dotc`, `tr_dot`, `cross` (3D; also works for 2D returning 1×1 via `perp`), `perp` (2D scalar cross), `cross_matrix()` (3D skew-symmetric), `kronecker`, outer product `u * v.transpose()`, `component_mul/div` (`base/blas.rs`, `base/matrix.rs`, `base/componentwise.rs`) | +−×(÷) |
| Norms | `norm`, `norm_squared`, `magnitude(_squared)`, `lp_norm(p)`, `apply_norm(&N)`, `metric_distance`, `normalize(_mut)`, `try_normalize(_mut)(eps)`, `set_magnitude`, `try_set_magnitude`, `cap_magnitude(max)`, `Unit::new_normalize`, `Unit::try_new`, `Unit::new_and_get`, `renormalize`, `renormalize_fast` (`base/norm.rs`, `base/unit.rs`) | √ ÷ (`lp_norm`: exp) |
| Angles / interp | `angle(&other)` (acos of clamped normalised dot; `base/matrix.rs`), `lerp`, `slerp`, `Unit::slerp/try_slerp` (`base/interpolation.rs`) | atrig, trig, √ |
| Component-wise | `abs`, `inf`, `sup`, `inf_sup`, `add_scalar`, `map`, `zip_map`, `zip_zip_map`, `apply`, `fold`, `cast` | — |
| Reductions | `min/max`, `amin/amax`, `camax`, `argmin/argmax`, `imin/imax`, `iamax/iamin`, `sum`, `product`, `mean`, `variance` (`base/min_max.rs`, `base/statistics.rs`) | ÷ for mean |
| Basis | `orthonormalize(&mut [v])` (Gram-Schmidt), `orthonormal_subspace_basis(vs, f)` (`base/norm.rs`) | √ ÷ |
| Homogeneous | `to_homogeneous()` (appends 0 for vectors), `from_homogeneous` (divides by last) | ÷ |
| Conversions | `From<[T;N]>`, `Into<[T;N]>`, tuples, `From<[Vector<Simd::Element>; LANES]>`, mint, glam, `cast::<U>()`, `convert` via `SubsetOf` (`base/conversion.rs`) | — |
| Approx | `abs_diff_eq`, `relative_eq`, `ulps_eq` | — |

#### Matrices — `Matrix1..6`, `MatrixRxC` (R,C ≤ 6), `SMatrix<T,R,C>`, `DMatrix`, `MatrixXxN`, `MatrixNxX`

| Group | Operations | Num |
|---|---|---|
| Constructors | `new(m11, m12, …)` **row-major argument order, column-major storage**; `zeros`, `identity`, `from_element`, `from_diagonal(&v)`, `from_diagonal_element`, `from_partial_diagonal`, `from_rows(&[row])`, `from_columns(&[col])`, `from_row_slice`, `from_column_slice`, `from_fn(|i,j|)`, `from_vec` | — |
| Access | `.m11…m66`, `m[(i,j)]`, `m[k]` (column-major linear), `row(i)`, `column(j)`, `fixed_view::<R,C>(i,j)`, `view((i,j),(r,c))`, `rows/columns`, `diagonal()`, `upper_triangle/lower_triangle`, `row_iter/column_iter` | — |
| Ops traits | `Neg`, `Add/Sub`, scalar `Mul/Div`, `Mat*Mat`, `Mat*Vec`, `RowVec*Mat`, `MulAssign`, `One` (identity), `Product` | +−× |
| Structure | `transpose(_mut/_to)`, `adjoint`, `conjugate`, `tr_mul` (Aᵀ·B without forming Aᵀ), `ad_mul`, `trace`, `symmetric_part`, `hermitian_part`, `map_diagonal`, `set_diagonal`, `set_row/column`, `swap_rows/columns`, `fill*`, `fill_upper_triangle_with_lower_triangle`… (`base/edition.rs`) | — |
| BLAS-like | `gemm`, `gemm_tr`, `gemv`, `gemv_tr`, `sygemv`, `ger`, `syger`, `axpy`, `axcpy`, `quadform` (α·M·Q·Mᵀ… precisely `self = α·lhs·mid·lhsᵀ + β·self`), `quadform_tr`, `cmpy`/`cdpy` | +−× |
| Properties | `is_square`, `is_identity(eps)`, `is_orthogonal(eps)`, `is_special_orthogonal(eps)`, `is_invertible` (`base/properties.rs`) | (LU) |
| Linalg hooks | `determinant`, `try_inverse(_mut)`, `pseudo_inverse(eps)`, `rank(eps)`, `singular_values`, `eigenvalues`, `complex_eigenvalues`, `symmetric_eigenvalues`, `solve_lower/upper_triangular*`, `pow(n)`, `exp()`, `lu() full_piv_lu() qr() col_piv_qr() cholesky() udu() lblt() svd() svd_unordered() polar() schur() hessenberg() bidiagonalize() symmetric_eigen() symmetric_tridiagonalize()` (`linalg/decomposition.rs`) | see 2.3 |
| CG helpers (3×3 / 4×4 homogeneous) | `new_scaling`, `new_nonuniform_scaling(_wrt_point)`, `new_translation`, `new_rotation(_wrt_point)`, `from_scaled_axis`, `from_euler_angles`, `from_axis_angle`, `new_perspective`, `new_orthographic`, `look_at_rh/lh`, `face_towards`, `new_observer_frame`, `append/prepend_(nonuniform_)scaling/translation(_mut)`, `transform_point`, `transform_vector` (`base/cg.rs`) | trig, √, ÷ |
| Norms | Frobenius `norm`, `norm_squared`, `apply_norm` (L1/L∞ as vector norms), `normalize` | √ |
| Dynamic-only | `resize(_mut)`, `resize_vertically/horizontally(_mut)`, `insert/remove_row(s)/column(s)`, `reshape_generic`, `select_rows/columns`, `zeros(r,c)`, `from_*` with runtime dims | — |

### 2.2 Geometry

#### `Point<T, D>` (`geometry/point*.rs`)

| Group | Operations |
|---|---|
| Constructors | `new(x,…)` (Point1..6), `origin()`, `from(coords)`, `from_slice`, `from_homogeneous(v) -> Option` (÷ by w), `from_coordinates` |
| Ops (`point_ops.rs`) | `Point − Point → Vector`, `Point ± Vector → Point`, `−Point`, `Point × T`, `Point ÷ T`, `Matrix × Point`, `Index`, `.x/.y/.z` via Deref (`point_coordinates.rs`) |
| Methods | `coords` field, `to_homogeneous()` (w = 1), `lerp(&rhs, t)`, `map`, `apply`, `cast`, `inf/sup/inf_sup`, `iter`, `len`; free fns `na::distance`, `na::distance_squared`, `na::center` (`lib.rs`) |
| Num | +−×÷, √ for distance |

#### `Unit<T>` — see `base/unit.rs` above. `Deref<Target = T>`, `Neg`, `AsRef`, `PartialEq`, approx.

#### `Rotation2` / `Rotation3` (`rotation.rs`, `rotation_specialization.rs`, `rotation_construction.rs`, `rotation_ops.rs`, `rotation_interpolation.rs`)

| Group | Rotation2 | Rotation3 | Num |
|---|---|---|---|
| Constructors | `identity`, `new(angle)`, `from_scaled_axis(Vector1)`, `from_matrix_unchecked`, `from_basis_unchecked([x,y])`, `from_matrix(m)` / `from_matrix_eps(m, eps, max_iter, guess)` (iterative projection onto SO(n)), `rotation_between(a,b)`, `scaled_rotation_between(a,b,s)` | + `new(axisangle)`, `from_axis_angle(&Unit, angle)`, `from_scaled_axis`, `from_euler_angles(r,p,y)`, `face_towards(dir, up)`, `look_at_rh/lh`, `new_observer_frames` | trig; atrig for `rotation_between`; iter for `from_matrix` |
| Queries | `angle()`, `angle_to(&other)`, `scaled_axis()`, `matrix()`, `into_inner()`, `to_homogeneous()` | + `axis()`, `axis_angle()`, `euler_angles()`, `euler_angles_ordered(seq, extrinsic)` | atrig (`atan2`, `acos`), √ |
| Algebra | `inverse()` = `transpose()` (+`_mut`), `R*R`, `R*Vector`, `R*Point`, `R*Unit<Vector>`, `R / R`, `R*Matrix`, `powf(n)`, `rotation_to(&other)`, `renormalize()` | same | `powf`: atrig+trig |
| Transform | `transform_point`, `transform_vector`, `inverse_transform_point`, `inverse_transform_vector`, `inverse_transform_unit_vector` | same | +−× |
| Interp | `slerp(&other, t)` | `slerp`, `try_slerp` (via quaternion) | trig, atrig |

#### `UnitComplex` (`unit_complex*.rs`) — 2D rotation, 2 scalars

`identity`, `new(angle)`, `from_angle`, `from_cos_sin_unchecked(c,s)`, `from_complex(_and_get)`,
`from_rotation_matrix`, `from_basis_unchecked`, `from_matrix(_eps)`, `from_scaled_axis`,
`rotation_between(_axis)`, `scaled_rotation_between(_axis)`; `angle()` (atan2), `sin_angle()`,
`cos_angle()`, `scaled_axis`, `axis_angle`, `angle_to`, `rotation_to`, `complex()`, `conjugate`,
`inverse(_mut)`, `powf`, `slerp`, `to_rotation_matrix`, `to_homogeneous`,
`transform_point/vector`, `inverse_transform_point/vector/unit_vector`; ops `UC*UC`, `UC/UC`,
`UC*Vector2`, `UC*Point2`, `UC*Rotation2`, `UC*Isometry/Similarity/Translation`.
Num: `new` needs sin+cos; `angle` needs atan2; everything else is ring ops.

#### `Quaternion` / `UnitQuaternion` (`quaternion*.rs` — largest geometry surface)

| Group | `Quaternion<T>` (general) | `UnitQuaternion<T>` (rotation) | Num |
|---|---|---|---|
| Constructors | `new(w,i,j,k)`, `from_parts(scalar, vector)`, `from_real`, `from_imag`, `identity`, `from_vector(Vector4)`, `from_polar_decomposition(scale, theta, axis)` | `identity`, `new(axisangle)` / `from_scaled_axis` (implemented as `exp(axisangle/2)`: `quaternion_construction.rs:706`), `new_eps`, `from_axis_angle` (`sin_cos(angle/2)`, `:265`), `from_quaternion` (normalises), `from_euler_angles`, `from_rotation_matrix` (branchy √-based, `:339`), `from_matrix(_eps)` (iterative), `from_basis_unchecked`, `rotation_between(_axis)`, `scaled_rotation_between(_axis)`, `face_towards`, `look_at_rh/lh`, `new_observer_frames`, `mean_of(iter)` (needs symmetric eigen 4×4) | √, trig; `rotation_between`: √ + normalise; `mean_of`: iter |
| Parts | `scalar()`/`w`, `vector()`/`imag()`, `i j k w` fields via Deref, `as_vector(_mut)`, `coords` | `quaternion()`, `into_inner()`, same accessors | — |
| Algebra | `+ − ×(Hamilton) × T ÷ T`, `conjugate(_mut)`, `try_inverse(_mut)`, `simd_try_inverse`, `dot`, `inner`, `outer`, `project`, `reject`, `left_div`, `right_div`, `lerp`, `half`, `squared`, `normalize(_mut)`, `norm(_squared)`, `magnitude(_squared)` | `q1*q2`, `q1/q2`, `conjugate`, `inverse(_mut)` (= conjugate), `rotation_to`, `angle_to`, `dot`, `lerp` (non-unit result), `nlerp`, `slerp`, `try_slerp(eps)` (flips sign for shortest path, `quaternion.rs:1237`), `powf`, `append_axisangle_linearized(&v)` (`q + ½(v,0)q`, renormalise — **trig-free integrator**, `:1630`), `renormalize`, `renormalize_fast` | ÷ √; slerp: atrig+trig; `powf`: atrig+trig |
| Transcendentals | `exp`, `exp_eps`, `ln`, `powf`, `sqrt`, `polar_decomposition`, `sin cos tan asin acos atan sinh cosh tanh asinh acosh atanh`, `is_pure`, `pure` | `ln`, `powf`, `angle()` (`2·atan2(‖v‖, |w|)`, `:1072`), `axis()`, `axis_angle()`, `scaled_axis()`, `euler_angles()` / `to_euler_angles()` | exp, atrig, trig |
| Transform | — | `q * Vector3` = `t = 2 v×r; r + w t + v×t` (`quaternion_ops.rs:377`, 2 cross products, **no trig, no matrix**), `q * Point3`, `q * Unit<Vector3>`, `transform_point/vector`, `inverse_transform_point/vector/unit_vector` | +−× |
| Conversions | `From<Vector4>`, `[T;4]`, glam `Quat` | `to_rotation_matrix()` (products only), `to_homogeneous()`, `From<Rotation3>`, `Into<Matrix3/4>`, glam, mint | +−× |

#### `DualQuaternion` / `UnitDualQuaternion` (`dual_quaternion*.rs`)

`from_real_and_dual`, `from_real`, `identity`, `normalize(_mut)`, `conjugate(_mut)`,
`try_inverse(_mut)`, `lerp`; unit: `from_parts(translation, rotation)`, `from_isometry`,
`from_rotation`, `to_isometry`, `rotation()`, `translation()`, `inverse(_mut)`, `isometry_to`,
`lerp`, `nlerp`, `sclerp`, `try_sclerp` (screw interpolation: acos, sin, cos, √), `to_homogeneous`,
`transform_point/vector`, `inverse_transform_*`; ops with quaternions, translations, isometries,
points, vectors. Not used by rapier/parry.

#### `Translation<T, D>` (`translation*.rs`)

`new(x,…)`, `identity`, `from(vector)`, `.vector`, `.x/.y/.z`, `inverse(_mut)`, `to_homogeneous`,
`transform_point`, `inverse_transform_point`, `cast`; ops: `T*T`, `T/T`, `T*Point`,
`T*Rotation → Isometry`, `T*UnitQuaternion → Isometry`, `T*Isometry`, `T*Similarity`. Ring ops only.

#### `Isometry2` / `Isometry3` / `IsometryMatrix2/3` (`isometry*.rs`)

| Group | Operations | Num |
|---|---|---|
| Constructors | `identity`, `from_parts(translation, rotation)`, `new(translation_vec, axisangle)` (2D: angle), `translation(x,y,z)`, `rotation(axisangle)`, `rotation_wrt_point(r, p)`, `face_towards(eye, target, up)`, `look_at_rh/lh`, `new_observer_frame`, `From<Translation>/<Rotation>/<[Isometry<Simd::Element>; N]>` | trig for `new`; √ for look_at |
| Algebra | `inverse(_mut)`, **`inv_mul(&rhs)`** (`self⁻¹ * rhs` without forming the inverse — the single most used composite in collision detection), `Iso*Iso`, `Iso/Iso`, `Iso*Rotation`, `Iso*Translation`, `Iso*Point`, `Iso*Vector` (rotation only), `Iso*Unit<Vector>`, `*=` variants | +−× |
| Mutators | `append_translation_mut`, `append_rotation_mut`, `append_rotation_wrt_point_mut`, `append_rotation_wrt_center_mut` | +−× |
| Transform | `transform_point`, `transform_vector`, `inverse_transform_point`, `inverse_transform_vector`, `inverse_transform_unit_vector` | +−× |
| Interp | `lerp_slerp(&other, t)`, `try_lerp_slerp(&other, t, eps)` (`isometry_interpolation.rs`) | atrig, trig |
| Conversions | `to_homogeneous()`, `to_matrix()` → (D+1)², `From/Into` Similarity/Transform/DualQuaternion, glam `(Vec3, Quat)`/`Affine` | +−× |

#### `Similarity2/3`, `SimilarityMatrix2/3` (`similarity*.rs`)

`identity`, `new(translation, axisangle, scaling)`, `from_parts`, `from_isometry`, `from_scaling`,
`scaling()`, `set_scaling`, `inverse(_mut)` (÷ scaling), `prepend/append_scaling(_mut)`,
`append_translation/rotation(_wrt_point/_wrt_center)_mut`, `transform_point/vector`,
`inverse_transform_point/vector`, `to_homogeneous`, `face_towards`, `look_at_*`; full operator matrix
with Isometry/Rotation/Translation/Point/Vector. Not used by rapier/parry.

#### `Transform2/3`, `Affine2/3`, `Projective2/3` (`transform*.rs`)

`identity`, `from_matrix_unchecked`, `matrix()`, `into_inner`, `to_homogeneous`, `set_category`,
`try_inverse(_mut)` (General), `inverse(_mut)` (Affine/Projective; uses the 3×3/4×4 closed-form
inverse), `transform_point` (projective divide for `TGeneral`/`TProjective`), `transform_vector`,
`inverse_transform_point/vector`; ops with every other transform type, the result category being
computed at type level by `TCategoryMul`. Not used by rapier/parry.

#### `Scale1..6` (`scale*.rs`)

`new`, `identity`, `try_inverse(_mut)`, `inverse_unchecked`, `pseudo_inverse`, `to_homogeneous`,
`transform_point`, `try_inverse_transform_point`; ops `S*S`, `S*Point`, `S*Vector`, `S*T`. ÷ only.

#### `Reflection1..6` (`reflection.rs`)

`new(axis: Unit, bias)`, `new_containing_point(axis, &pt)`, `axis()`, `bias()`, `reflect(&mut m)`,
`reflect_with_sign`, `reflect_rows`, `reflect_rows_with_sign`. Internal workhorse of Householder
QR/Hessenberg/bidiagonalisation.

#### `Perspective3`, `Orthographic3` (`perspective.rs`, `orthographic.rs`)

`new(aspect, fovy, znear, zfar)` / `new(l,r,b,t,n,f)`, `from_fov`, `from_matrix_unchecked`,
`inverse()` (closed form), `as_matrix`, `as_projective`, `to_projective`, `to_homogeneous`, getters
`aspect fovy znear zfar left right bottom top`, setters, `project_point`, `unproject_point`,
`project_vector`. Needs `tan`/`atan`. Rendering only.

### 2.3 `linalg` — decompositions and solvers

All are generic over `Dim` (static or dynamic) and `ComplexField` unless noted; "loops" counts
`for/while/loop` statements in the file as a proxy for unroll difficulty.

| Feature (file, LoC) | API | Algorithm | Num requirements | Notes for Cairo |
|---|---|---|---|---|
| **determinant** (`determinant.rs`, 61) | `m.determinant()` | closed form for n ≤ 3 (cofactor expansion), LU otherwise | +−× (n≤3) | 3×3 = triple products → at Q-format scale S the raw product is S³; do in wide ints / felt and rescale once. |
| **try_inverse** (`inverse.rs`, 290) | `try_inverse()`, `try_inverse_mut()` | closed form n=1,2,3 (adjugate ÷ det), **n=4 unrolled MESA `do_inverse4`**, LU otherwise; fails iff `det.is_zero()` (exact zero test!) | +−×, 1 ÷ (can use one reciprocal then ×) | Exact-zero singularity test is meaningless in fixed point → use `abs(det) <= eps`. |
| **solve (triangular)** (`solve.rs`, 780) | `solve_lower/upper_triangular(_mut)`, `tr_solve_*`, `ad_solve_*`, `*_unchecked`, `*_with_diag` | forward/back substitution | ÷ per row | Trivial to unroll for n ≤ 4. |
| **LU** partial pivoting (`lu.rs`, 389) | `LU::new`, `l()`, `u()`, `p()`, `unpack`, `solve(_mut)`, `try_inverse(_to)`, `determinant`, `is_invertible`; helpers `gauss_step(_swap)` | Doolittle with row pivot via `icamax` | ÷, abs, compare | **Used by rapier multibody** as `LU<Real, Dyn, Dyn>` (`multibody.rs:113`). No sqrt, no iteration → most fixed-point-friendly general solver. |
| **FullPivLU** (`full_piv_lu.rs`, 268) | same + `q()` | row+column pivoting | ÷, abs | More robust for ill-conditioned; O(n³) pivot search. |
| **Cholesky** (`cholesky.rs`, 450) | `Cholesky::new`, `new_with_substitute`, `new_unchecked`, `pack_dirty`, `l()`, `l_dirty()`, `unpack(_dirty)`, `solve(_mut)`, `inverse`, `determinant`, `ln_determinant`, `rank_one_update`, `insert_column`, `remove_column` | column Cholesky–Crout: `axpy` updates then `sqrt(diag)`, ÷ (`:221-260`); returns `None` if pivot ≤ 0 | √ and ÷ per column; `ln` only for `ln_determinant` | **Used by rapier soft bodies** on `SMatrix<3,3>`/`SMatrix<6,6>` (`soft_elastic_constraint.rs:275`). For fixed point prefer an LDLᵀ variant (no √) — nalgebra's `UDU` is that. |
| **UDU** (`udu.rs`, 99) | `UDU::new`, `u`, `d`, `d_matrix()` | UDUᵀ for symmetric matrices, no pivoting | ÷ only | √-free SPD factorisation. Good fixed-point candidate. RealField only. |
| **LBLT** Bunch–Kaufman (`lblt.rs`, 503; new in 0.35) | `new`, `l_permuted`, `d`, `solve(_mut)`, `determinant` | symmetric-indefinite with 1×1/2×2 pivots | ÷, abs | Complex control flow (36 loops). Low priority. |
| **QR** Householder (`qr.rs`, 308) | `QR::new`, `q()`, `r()`, `unpack`, `unpack_r`, `q_tr_mul`, `solve(_mut)`, `try_inverse`, `is_invertible`, `determinant` | Householder reflections (`householder.rs`: `reflection_axis_mut` uses norm → √, signum, ÷) | √, ÷, signum | Least squares. Not used by physics stack (except inside nalgebra's 3×3 SVD). |
| **ColPivQR** (`col_piv_qr.rs`, 338) | same + `p()` | QR with column pivoting (rank revealing) | √, ÷ | — |
| **Hessenberg** (`hessenberg.rs`, 155) | `new`, `h()`, `q()`, `unpack(_h)` | Householder similarity | √, ÷ | Only a stepping stone to Schur. |
| **Bidiagonal** (`bidiagonal.rs`, 381) | `new`, `u()`, `d()`, `v_t()`, `diagonal`, `off_diagonal`, `unpack` | Golub–Kahan bidiagonalisation | √, ÷, signum | Stepping stone to SVD. |
| **SymmetricTridiagonal** (`symmetric_tridiagonal.rs`, 171) | `new`, `q()`, `diagonal`, `off_diagonal`, `unpack(_tridiagonal)`, `recompose` | Householder | √, ÷ | Stepping stone to SymmetricEigen. |
| **SymmetricEigen** (`symmetric_eigen.rs`, 469) | `new`, `try_new(m, eps, max_niter)`, `recompose`, `m.symmetric_eigenvalues()`; pub `wilkinson_shift` | tridiagonalise → implicit symmetric QR with Wilkinson shifts and Givens rotations (`givens.rs`), subproblem deflation; special-cases the trailing 2×2 block in closed form | √, ÷, abs, signum, **iter** (convergence test `|off| ≤ eps(|d_m|+|d_n|)`, `:196`) | **Used by rapier** (`Matrix3::symmetric_eigen`, `soft_body_cluster.rs:782`). The physics stack otherwise uses glamx's *closed-form* solvers: 2×2 quadratic (`glamx/src/eigen2.rs`, √ only) and 3×3 Eberly/trigonometric (`glamx/src/eigen3.rs`: √, `acos`, `cos`). For Cairo: closed-form or fixed-sweep Jacobi, not the iterative QR. |
| **SVD** (`svd.rs`, 903; `svd2.rs`, `svd3.rs`) | `SVD::new(m, u?, v?)`, `try_new(…, eps, max_niter)`, `new_unordered`, `singular_values`, `rank(eps)`, `pseudo_inverse(eps)`, `solve(b, eps)`, `recompose`, `to_polar`, `sort_by_singular_values`; `m.polar()`, `m.try_polar()` | general: bidiagonalise + implicit-shift QR (Golub–Kahan) with Givens, `hypot`; **2×2 closed form** (`svd2.rs`: 2 √, 2 `atan2`, 2 `sin_cos`); **3×3**: eigen of AᵀA + QR (`svd3.rs`, McAdams-style) | √, ÷, hypot, **iter**; 2×2: atrig+trig | **Used by rapier IK only**: `SMatrix<6,6>::pseudo_inverse(1e-5)` (`multibody_ik.rs:76`). Replaceable by damped-least-squares via Cholesky/LDLᵀ since `JJᵀ + λ²I` is SPD. |
| **Schur** (`schur.rs`, 628) | `Schur::new`, `try_new`, `unpack`, `eigenvalues`, `complex_eigenvalues` | Hessenberg + Francis double-shift QR | √, ÷, **iter**, complex output | General (non-symmetric) eigenvalues. Not needed. |
| **Eigen** (`eigen.rs`, 113) | (crate-private/unfinished) | — | — | Skip. |
| **exp** (`exp.rs`, 556) | `m.exp()` | Padé-13 scaling & squaring (Higham 2005/2009) with 1-norm estimator | `powf`, `exp`, abs, solve | Not needed. |
| **pow** (`pow.rs`, 71) | `m.pow(n: u32)`, `pow_mut` | exponentiation by squaring | +−× | Trivial. |
| **convolution** (`convolution.rs`, 150) | `convolve_full/valid/same` | direct | +−× | Not needed. |
| **balancing** (`balancing.rs`, 84) | `balance_parlett_reinsch`, `unbalance` | Parlett–Reinsch | √, ÷, iter | Not needed. |
| **givens** (`givens.rs`, 165) | `GivensRotation::new/try_new/identity/cancel_x/cancel_y/c/s/inverse/rotate/rotate_rows` | — | √ (norm of 2-vector), ÷ | Needed iff Jacobi/QR eigen is ported. |
| **householder** (`householder.rs`, 150) | `reflection_axis_mut`, `clear_column_unchecked`, `clear_row_unchecked`, `assemble_q` | — | √, ÷, signum | Needed iff QR family is ported. |
| **PermutationSequence** (`permutation_sequence.rs`, 165) | `identity`, `append_permutation`, `permute_rows/columns`, `inv_permute_*`, `determinant` (±1) | — | — | Needed by LU. |

---

## 3. CRITICAL — what the physics stack actually uses

### 3.1 Current dependency state (precise)

**parry 0.31.1 — nalgebra-free.**

* `parry/Cargo.toml` `[workspace.dependencies]`: `glamx = "0.3"`, `simba = "0.10.1"`, `approx`,
  `num-traits` — **no `nalgebra` entry at all**. `crates/parry3d/Cargo.toml` depends on `glamx`
  (features `i32`), `simba` (feature `wide`, for the SIMD lane types and the `RealField`
  trait re-export), `approx`.
* `grep -rnE 'nalgebra|\bna::' parry/src` → 7 hits, all comments/stale doctest text
  (`shape/trimesh.rs:823`, `shape/heightfield3.rs:2`, `transformation/convex_hull_utils.rs:24…`).
* Migration happened in **parry 0.26.0** (`parry/CHANGELOG.md:241-310`): "migrates parry from
  nalgebra to glam (via the glamx crate) for future compatibility with rust-gpu".
* `parry/src/math/mod.rs` is the authoritative alias table:

| parry alias | 3D | 2D |
|---|---|---|
| `Real` | `f32` / `f64` | same |
| `Vector` | `glam::Vec3` / `DVec3` | `Vec2` / `DVec2` |
| `IVector` | `IVec3` / `I64Vec3` | `IVec2` / `I64Vec2` |
| `AngVector` | `Vec3` | `Real` |
| `Matrix` | `Mat3` | `Mat2` |
| `Pose` | `glamx::Pose3` (Quat + Vec3) | `glamx::Pose2` (Rot2 + Vec2) |
| `Rotation` | `glamx::Rot3` = `glam::Quat` | `glamx::Rot2` (unit complex) |
| `Orientation` | `Vec3` | `Real` |
| `SpatialVector` | `[Real; 6]` | `Vec3` |
| `AngularInertia` / `SdpMatrix` | `parry::utils::SdpMatrix3<Real>` (own type, 6 scalars) | `Real` / `SdpMatrix2` |
| `CrossMatrix` | `Mat3` | `Vec2` |
| `SymmetricEigen` | `glamx::SymmetricEigen3` | `glamx::SymmetricEigen2` |
| (no `Point`) | positions are plain `Vector` — "parry no longer distinguishes Point from Vector" | |

  From simba parry only re-exports the *traits* `ComplexField`, `RealField` (for `.sqrt()`, `.abs()`…
  on `Real`) and the SIMD scalar `SimdReal`.

**rapier 0.35.3 — hybrid.**

* `rapier/Cargo.toml:68-70`: `nalgebra = "0.35"` (default-features off, `macros`),
  `glamx = "0.3.1"`, `simba = "0.10.2"`. `crates/rapier3d/Cargo.toml`: nalgebra is a
  `[target.'cfg(not(target_arch = "spirv"))'.dependencies]` entry together with
  `glamx/nalgebra` (conversion feature) — i.e. **rapier already compiles a subset of itself with
  no nalgebra at all** (the SPIR-V/GPU build).
* Migration: **rapier 0.32.0, 9 Jan 2026** (`rapier/CHANGELOG.md:320-360`): "nalgebra is still
  used internally for SIMD code and multibody Jacobians (via `SimdVector<N>`, `SimdPose<N>`,
  `DVector`, `DMatrix`)".
* `rapier/src/lib.rs:48` `pub extern crate nalgebra as na;` and `lib.rs:157-315` `pub mod math`:
  `pub use parry::math::*` (glam aliases) **plus** nalgebra aliases:

| rapier alias (`src/lib.rs`) | nalgebra type | Used for |
|---|---|---|
| `SimdVector<N>` | `na::Vector2<N>` / `na::Vector3<N>` | SoA solver |
| `SimdAngVector<N>` | `N` / `na::Vector3<N>` | SoA solver |
| `SimdPoint<N>` | `na::Point2/3<N>` | SoA solver |
| `SimdPose<N>` | `na::Isometry2/3<N>` | SoA solver |
| `SimdRotation<N>` | `na::UnitComplex<N>` / `na::UnitQuaternion<N>` | SoA solver |
| `SimdMatrix<N>` | `na::Matrix2/3<N>` | SoA solver |
| `SimdAngularInertia<N>` | `N` / `parry::utils::SdpMatrix3<N>` (not nalgebra) | SoA solver |
| `Dim`, `AngDim` | `na::U2/U3`, `na::U1/U3` | multibody view sizes |
| `DVector`, `DMatrix` | `na::DVector<Real>`, `na::DMatrix<Real>` | multibody, generic joints, solver velocity buffer |
| `Jacobian<N>` (+`View`, `ViewMut`) | `na::Matrix3xX` / `na::Matrix6xX` | multibody |
| `TangentImpulse<N>` | `na::Vector1<N>` / `na::Vector2<N>` | contact friction impulse storage |

* File-level census of `nalgebra|na::` mentions in `rapier/src` (34 files). Grouped by role:

| Role | Files | nalgebra items |
|---|---|---|
| **(a) SIMD/generic solver instantiation** | `utils/scalar_type.rs:120-138` (`impl ScalarType for SimdReal`), `utils/pos_ops.rs`, `utils/rotation_ops.rs`, `utils/angular_inertia_ops.rs`, `utils/cross_product.rs`, `utils/cross_product_matrix.rs`, `utils/dot_product.rs`, `utils/component_mul.rs`, `utils/copysign.rs`, `utils/orthonormal_basis.rs`, `utils/matrix_column.rs`, `utils/simd_select.rs`, `dynamics/solver/contact_constraint/*.rs`, `dynamics/solver/joint_constraint/joint_velocity_constraint.rs`, `joint_constraint_builder.rs`, `geometry/contact_pair.rs` | `Vector1/2/3<SimdReal>`, `Matrix2/3<SimdReal>`, `UnitQuaternion<SimdReal>`, `UnitComplex<SimdReal>`, `Isometry2/3<SimdReal>`, `Translation2/3`, `na::zero()`, `na::vector!`, traits `SimdValue`, `SimdPartialOrd`, `SimdRealField`, `Scalar` |
| **(b) Multibody joints & IK** | `dynamics/joint/multibody_joint/multibody.rs` (imports `DMatrix, DVectorView, DVectorViewMut, Dyn, LU, OMatrix, SMatrix, SVector, StorageMut`), `multibody_joint.rs`, `multibody_ik.rs`, `dynamics/solver/joint_constraint/generic_joint_constraint*.rs`, `dynamics/solver/solver_body.rs` (`DVectorView::from_slice`), `velocity_solver.rs` (`generic_solver_vels: DVector`), `joint_constraints_set.rs`, `generic_contact_constraint_element.rs`, `dynamics/rigid_body_components.rs:698-725` (`as_vector() -> &na::Vector3/6`) | `DMatrix`, `DVector`, `RowDVector`, views, `Matrix3xX/6xX`, `SMatrix<6,6>`, `SVector<6>`, `LU<Real,Dyn,Dyn>::new/solve_mut`, `gemm`, `gemm_tr`, `gemv_tr`, `quadform`, `tr_mul_to`, `axpy`, `cmpy`, `dot`, `norm`, `norm_squared`, `rows/rows_mut/fixed_rows(_mut)/fixed_view_mut`, `copy_from(_slice)`, `fill`, `resize_vertically_mut`, `pseudo_inverse(1e-5)`, `na::clamp` |
| **(c) Soft bodies** | `dynamics/solver/soft_constraint/soft_element_constraint/soft_elastic_constraint.rs:18-23,275` (`StrainVector = SVector<Real, 3|6>`, `StrainMatrix = SMatrix<Real, 3|6, 3|6>`, `StrainJacobian = SMatrix<Real, DIM, 3|6>`, `.cholesky().inverse()`), `dynamics/soft_body/soft_body_cluster.rs:771-790` (`na::Matrix3::symmetric_eigen()` → pseudo-inverse of inertia) | `SMatrix`, `SVector`, `Cholesky`, `SymmetricEigen`, `set_column`, `row_mut/column_mut().fill`, `u * u.transpose() / λ` |
| **(d) Trait imports only** | `pipeline/physics_hooks.rs` (`ComplexField`), `control/character_controller.rs` (`RealField`, `Vector2`), `dynamics/joint/revolute_joint.rs` (`RealField`), `geometry/collider.rs:14` (`na::Unit`, halfspace normal), `soft_body_builder_shapes.rs` | re-exported simba traits, `Unit` |
| **(e) Prelude sugar** | `lib.rs:332-344` | `vector!`, `point!` macros + `pub extern crate nalgebra` |

* **The subtle but decisive point about (a).** The velocity solver is written *once*, generically
  over `N: ScalarType` (`utils/scalar_type.rs:20`). `impl ScalarType for Real` maps the associated
  types to **glam** (`Pose`, `Vector`, `Matrix`, `Rotation`); `impl ScalarType for SimdReal` maps
  them to **nalgebra**. In the shipped native build the contact constraints are *only* instantiated
  with `SimdReal` (`contact_constraints_set.rs:46-53`:
  `Vec<ContactWithCoulombFriction<SimdReal>>`), while joint constraints exist in both flavours
  (`JointConstraint<Real, 1>` on glam and `JointConstraint<SimdReal, SIMD_WIDTH>` on nalgebra,
  `joint_velocity_constraint.rs:146,391`) — so today nalgebra *is* on rapier's hot path on
  CPUs. But the code never calls nalgebra API directly: it goes through rapier's own tiny ops traits
  (`gdot` 122×, `gcross` 29×, `component_mul` 42×, `transform_vector` 35×, `column` 31×,
  `select` 29×, `gcross_matrix`, `orthonormal_basis`, `copy_sign_to`, `imag`, `to_mat`, `inverse`,
  `inverse_transform_point`, `simd_max/clamp/gt`…). A Cairo port instantiates `ScalarType` for the
  fixed-point `Real` only, so **(a) disappears entirely** and the trait bounds in
  `scalar_type.rs:26-111` become the *exact* operation contract the chosen vector/matrix/rotation
  types must satisfy — whichever Cairo library supplies them.

### 3.2 What glamx adds on top of glam (the part nalgebra used to supply)

`glamx-0.3.0/src/` (3 102 LoC): `rot2.rs` (`Rot2`: unit complex; `new(angle)`, `from_cos_sin`,
`angle`, `inverse`, `slerp`, `to_mat`, `transform_vector`…), `rot3.rs` (15 L: `Rot3 = Quat`),
`pose2.rs`/`pose3.rs` (`Pose{rotation, translation}`: `identity`, `from_parts`, `from_translation`,
`from_rotation`, `inverse`, `inv_mul`, `transform_point/vector`, `inverse_transform_point/vector`,
`prepend/append_translation/rotation`, `lerp`, `to_mat`…), `matrix_ext.rs` (`MatExt`: `abs`,
`try_inverse`, `swap_cols`, `swap_rows`, `symmetric_eigen`, `symmetric_eigenvalues`, `svd`),
`eigen2.rs`/`eigen3.rs` (closed-form symmetric eigen; 3×3 = Eberly's robust trigonometric method
using `sqrt`, `acos`, `cos`), `svd2.rs`/`svd3.rs`. **This is exactly the nalgebra feature subset
the physics stack could not live without** — and it is ~3k lines, versus nalgebra's ~49k.

### 3.3 Operation frequency in the physics stack (method-call histogram, `src/` only)

`grep -rhoE '\.<name>\(' --include='*.rs' src | wc -l` — names are glam-flavoured; nalgebra
equivalents in parentheses.

| Operation (nalgebra name) | parry | rapier |
|---|---|---|
| `dot` | 397 | 126 (+122 `gdot`) |
| `abs` | 137 | 101 |
| `inverse` (rot/pose/mat) | 116 | 41 |
| `length` (`norm`) | 114 | 99 |
| `normalize` / `try_normalize` / `normalize_or_zero` | 100 / 41 / 22 | 7 / 17 / 3 |
| `max` / `min` | 96 / 66 | 173 / 90 |
| `length_squared` (`norm_squared`) | 77 | 51 |
| `cross` / `perp_dot` (`perp`) / `gcross` | 75 / 34 / 3 | 59 / 9 / 66 |
| `transform_point` / `inverse_transform_point` | 49 / 39 | 14 / 18 |
| `transform_vector` / `inverse_transform_vector` | 1 / 6 | 44 / 0 |
| `inv_mul` | 31 | 10 |
| `clamp` | 20 | 40 |
| `signum` / `copysign` | 20 / 1 | 5 / 1 |
| `abs_diff_eq` (approx) | 20 | — |
| `distance` / `distance_squared` | 22 / 4 | — |
| `sqrt` (scalar) | 11 | 22 |
| `sin` / `cos` / `sin_cos` | 14 / 13 / 4 | 8 / 3 / 2 |
| `acos` / `asin` / `atan2` | 2 / 0 / 0 | 2 / 2 / 1 |
| `floor` / `ceil` / `round` | 11 / 17 / 1 | 0 / 1 / 0 |
| `lerp` | 7 | 6 |
| `transpose` / `determinant` / `try_inverse` | 4 / 3 / 1 | 22 / 12 / 1 |
| `col` / `column` | 1 | 32 (+31) |
| `cross_matrix` / `gcross_matrix` | — | 3 / 10 |
| `orthonormal_basis` | 7 | 14 |
| `symmetric_eigen` | 3 | 1 (+ own `symmetric_eigen` for soft bodies) |
| `to_mat` (rotation→matrix) | 1 | 9 |
| `angle` / `to_scaled_axis` / `to_axis_angle` | 4 / 1 / 0 | 12 / 4 / 1 |
| `from_scaled_axis` (rotation integration) | — | 5 call sites |
| `exp` / `powi` / `powf` | 0 / 4 / 0 | 2 / 1 / 3 |
| `is_finite` / `is_nan` (float-only) | 5 / 6 | 69 / 1 |
| `kronecker` (outer product) | 3 | 1 |
| `pseudo_inverse` | — | 1 |
| `slerp` / `nlerp` / `lerp_slerp` | 0 | 0 |

Observations: (1) the workload is overwhelmingly `dot`/`cross`/`norm`/`min`/`max`/`abs` on 2–3
component vectors plus pose composition; (2) **inverse trig is nearly absent** (7 call sites in the
whole stack) and forward trig is rare (≈45 sites, mostly shape tessellation and joint-angle
handling); (3) `slerp` is *never* used; (4) `is_finite` (69× in rapier) is float-specific and
becomes a no-op / overflow check in fixed point.

### 3.4 Ranked list — nalgebra features by importance for a game physics engine

Rank reflects (frequency in rapier+parry) × (is it on the per-step hot path) × (can the engine
work without it). "Owner" is the recommended Cairo home.

| # | Feature | Where it is needed | Owner |
|---|---|---|---|
| 1 | `Vector2/3`: `+ − ×s ÷s`, `neg`, `dot`, `cross`/`perp`, `norm(_squared)`, `normalize`/`try_normalize`, `component_mul`, `abs`, `inf/sup` (min/max), indexing | everything | glam.cairo (hot path); nalgebra.cairo mirrors |
| 2 | 3D rotation as unit quaternion: `q*q`, `inverse`=conjugate, `q*v`, `to_rotation_matrix`, `from_scaled_axis`/`new(axisangle)`, `append_axisangle_linearized`, `renormalize(_fast)`, `imag`, `angle`/`scaled_axis`; 2D rotation as `UnitComplex`: `new(angle)`, `*`, `inverse`, `angle`, `to_rotation_matrix` | body orientation, integration, joints, every narrow-phase query | glam.cairo (`Quat`, `Rot2`) |
| 3 | `Isometry2/3` (Pose): `from_parts`, `*`, `inverse`, **`inv_mul`**, `transform_point/vector`, `inverse_transform_point/vector`, `append/prepend_translation/rotation`, `translation`/`rotation` accessors | every shape-vs-shape query, colliders' local frames, joints' anchors | glam.cairo (`Pose2/3` per glamx) |
| 4 | `Matrix2/3`: `M*v`, `M*M`, `transpose`, `column`, `from_columns`, `from_diagonal`, `determinant`, `try_inverse`, `cross_matrix`, outer product, symmetric 3×3 (`SdpMatrix3`: `inverse_unchecked`, `quadform`, `mul_vec`, `add_diagonal`) | inertia tensors, effective-mass blocks, joint Jacobian blocks, 2×2 block solver | glam.cairo (`Mat2/3`) + `SdpMatrix` in parry.cairo |
| 5 | Scalar helpers: `clamp`, `min/max`, `abs`, `signum`/`copysign`, `wrap` (angle wrap), constants `PI`…, `abs_diff_eq`/`relative_eq` | everywhere | shared scalar crate (Tier 0) |
| 6 | `orthonormal_basis` / `orthonormal_subspace_basis` (tangent frame from a normal) | contact friction directions, joint frames, GJK/EPA | glam.cairo/rapier.cairo util |
| 7 | Symmetric eigendecomposition 2×2 / 3×3 | principal inertia axes (`parry/src/mass_properties/mass_properties.rs:215` — once per body creation), convex-hull init, VHACD, soft-body inertia pseudo-inverse | **nalgebra.cairo** Tier 3 (or glam.cairo per glamx) |
| 8 | `Unit<Vector>` (`UnitVector2/3`) | half-space normals, joint axes, ray directions; type-level invariant | either; cheap |
| 9 | `Vector1/Vector2` impulse accumulators, `SVector<6>`/`Vector6`, `SMatrix<6,6>`/`Matrix6` (spatial algebra), `Matrix3x6`-style blocks | contact tangent impulses; multibody; soft-body strain | **nalgebra.cairo** Tier 1/3 |
| 10 | Small SPD solve: `Cholesky` (or LDLᵀ/`UDU`) `+ inverse/solve` on 3×3 and 6×6 | soft-body elastic block inversion; damped-least-squares IK; 2-contact block solver | **nalgebra.cairo** Tier 3 |
| 11 | `DVector`/`DMatrix` + BLAS-like kernels (`gemm`, `gemv_tr`, `quadform`, `axpy`, `cmpy`, `tr_mul_to`, `dot`, `norm_squared`), row/col block copy, `resize_vertically`, and **`LU<Dyn,Dyn>::new` + `solve_mut`** | multibody (reduced-coordinate) joints: mass-matrix assembly `Jᵀ M J` and solve each step | **nalgebra.cairo** Tier 4 |
| 12 | `SVD`/`pseudo_inverse` (6×6) | multibody inverse kinematics only | Tier 4 / replace with #10 |
| 13 | `Point` as a distinct type | none any more (parry dropped it) — API nicety only | optional |
| 14 | `lerp`, `slerp`, `lerp_slerp`, `nlerp` | not used by the engine; useful to *clients* for render interpolation | glam.cairo |
| 15 | `Rotation2/3` (matrix form), Euler angles, `look_at`, `face_towards`, `rotation_between` | user-facing constructors only | low |
| 16 | `Similarity`, `Transform/Affine/Projective`, `Scale`, `Perspective3`, `Orthographic3`, `DualQuaternion`, `Reflection` (public), `base/cg.rs` | **not used** by rapier/parry | out of physics scope |
| 17 | QR, ColPivQR, FullPivLU, Schur, Hessenberg, Bidiagonal, LBLT, `exp`, convolution, statistics | **not used** | out of physics scope |

### 3.5 Implication for the nalgebra.cairo ⇄ glam.cairo split

1. **glam.cairo (+ a glamx-like layer: `Rot2`, `Pose2/3`, `MatExt`, `SymmetricEigen2/3`) is the
   library `parry.cairo` and the scalar path of `rapier.cairo` should be written against.** It
   mirrors upstream 1:1 (`parry/src/math/mod.rs`), which keeps porting mechanical.
2. **nalgebra.cairo's unique value** is what glam does not have: N > 4 static vectors/matrices
   (`Vector6`, `Matrix6`, `Matrix3x6`/`Matrix6xN`), generic small-N decompositions (Cholesky / LDLᵀ /
   LU / symmetric eigen / SVD 2×2–3×3), and the dynamic `DVector`/`DMatrix` + LU stack for
   multibody joints. If `rapier.cairo` v1 excludes multibody joints, IK and soft bodies (reasonable
   for a provable game engine: impulse joints cover most gameplay), then **rapier.cairo v1 needs
   nothing from nalgebra.cairo** — exactly like upstream's SPIR-V build.
3. To avoid duplicating Tier 0–2 work, nalgebra.cairo should either (i) *re-use* glam.cairo's
   `Vec2/3/4`, `Mat2/3/4`, `Quat` as the backing representation of `Vector2/3/4`, `Matrix2/3/4`,
   `UnitQuaternion` (newtype or plain alias + extension traits with nalgebra naming), or (ii) share
   one scalar crate and provide `From/Into` conversions patterned on
   `nalgebra/src/third_party/glam/common/*.rs`. Note the convention differences to pin down in the
   conversion layer: nalgebra `Matrix3::new` takes **row-major** arguments / glam `Mat3::from_cols`
   is column-major (see `rapier/src/utils/mod.rs:124-128` for the hand-written shuffle);
   nalgebra `Quaternion::new(w, i, j, k)` vs glam `Quat::from_xyzw(x, y, z, w)` (both *store* xyzw).

---

## 4. Numeric primitives required from the scalar type

Counts are `.f(`/`.simd_f(` call sites in `nalgebra/src/{base,geometry,linalg}`; the last column is
demand from the physics stack (§3.3).

| Primitive | Where nalgebra needs it (files) | Physics-stack need | Fixed-point implementation note |
|---|---|---|---|
| `+ − ×` | everything | everything | `×` = wide multiply then shift; in Cairo do products in `felt252`/`u256`/`i128`, **defer the rescale**: a dot product of N terms needs *one* shift, not N. |
| `÷`, `recip` | `normalize`, `inverse`, all solves/decompositions, `from_homogeneous`, `Similarity::inverse`, projections | high | Most expensive primitive (u256/u128 divmod + sign handling). Compute one reciprocal and multiply (nalgebra's 3×3 inverse does 9 divisions by `det` — use 1). |
| `sqrt` | `base/norm.rs` (3), `base/unit.rs`, `base/interpolation.rs`, `geometry/quaternion*.rs` (12), `rotation_specialization.rs` (4), `unit_complex*.rs` (2), `dual_quaternion.rs`, `similarity_conversion.rs`; `linalg/cholesky.rs`, `householder.rs` (3), `givens.rs` (3), `symmetric_eigen.rs`, `schur.rs`, `svd2.rs` (2), `balancing.rs` (2) | high (`norm`, `normalize`: >400 sites) | Cairo corelib has integer `u128_sqrt`/`u256_sqrt` (hint + verify: cheap). Scale input by 2^frac first. |
| inverse sqrt | not a primitive in nalgebra (`normalize` = `÷ norm`); `Unit::renormalize_fast` is a 1-step Newton rsqrt around 1 (`base/unit.rs:176`) | — | Optional optimisation: `normalize` via one `sqrt` + one `recip` + N `×`. Use `renormalize_fast` for quaternions drifting near 1. |
| `abs`, `signum`, `copysign` | `componentwise.rs`, `min_max.rs` (10), pivoting in `lu.rs`/`full_piv_lu.rs`/`lblt.rs`, `householder.rs`, `qr.rs`, `bidiagonal.rs`, `symmetric_eigen.rs`, `rotation_specialization.rs` (6) | high | trivial with sign-magnitude or two's-complement representation. |
| `min`, `max`, `clamp`, comparisons | `lib.rs` (`clamp`, `wrap`, `partial_*`), `matrix.rs::angle` (clamp before `acos`), `cap_magnitude`, `inf/sup` | very high (rapier: `max` 173, `min` 90, `clamp` 40) | trivial. |
| `sin`, `cos`, `sin_cos` | `quaternion_construction.rs` (`from_axis_angle`, `from_euler_angles`), `quaternion.rs` (`exp`, `powf`, trig fns), `rotation_specialization.rs` (`Rotation2::new`, `Rotation3::from_axis_angle`, euler; 7 `sin_cos`), `unit_complex_construction.rs`, `interpolation.rs` (slerp), `dual_quaternion.rs` (sclerp), `svd2.rs`, `base/cg.rs` | low-moderate (≈45 sites; rotation-from-angle, shape tessellation, motors) | LUT + interpolation or polynomial (cubit `f64/math/trig.cairo`, `lut.cairo` are prior art). Always expose `sin_cos` together. |
| `tan` | `perspective.rs`, `orthographic.rs::from_fov`, `quaternion.rs::tan` | none | skip or `sin/cos`. |
| `acos`, `asin` | `matrix.rs::angle`, `interpolation.rs::slerp`, `quaternion.rs` (2+1), `rotation_specialization.rs` (3+1: `angle`, euler), `dual_quaternion.rs` | very low (4 sites) | via `atan2(sqrt(1−x²), x)`; clamp input to [−1, 1] (fixed-point rounding *will* produce 1+ε). |
| `atan2`, `atan` | `rotation_specialization.rs` (12: `Rotation2::angle`, euler extraction), `unit_complex.rs::angle`, `quaternion.rs::angle` (`2·atan2(‖v‖,|w|)`), `svd2.rs` (2), `perspective.rs::fovy` | low (rapier: joint angle limits via `angle()`; 1 explicit `atan2`) | CORDIC or rational approximation; needed for `UnitComplex::angle` / `UnitQuaternion::angle/scaled_axis` (joint limits, motors). |
| `exp`, `ln` | `quaternion.rs` (`exp` 8, `ln` 11 — mostly *scalar* exp/ln inside quaternion `exp/ln/powf`), `cholesky.rs::ln_determinant`, `linalg/exp.rs` | ~none (rapier: 2 `exp` for damping-style decay) | Note `UnitQuaternion::new(axisangle)` is *named* `exp` but only needs `sin`, `cos`, `sqrt`, `÷` (`exp` of a pure quaternion). Provide scalar `exp` only if a consumer asks. |
| `powi` | `norm.rs` (Lp), `udu.rs` | 5 sites, exponent 2–3 | repeated multiply. |
| `powf`, `cbrt`, `hypot`, hyperbolic, `sinc/sinhc/cosc` | `norm.rs::lp_norm`, `quaternion.rs::powf`, `rotation*.rs::powf`, `linalg/exp.rs` (14), `svd.rs` (`hypot` 2) | 3 `powf` sites in rapier | out of scope; `hypot` = `sqrt(a²+b²)` with pre-scaling to avoid overflow. |
| `floor`, `ceil`, `round`, `trunc`, `fract` | not used by nalgebra algorithms (only via `ComplexField` passthrough / `map`) | parry: `floor` 11, `ceil` 17 (voxel/heightfield grids, broad-phase cell indices) | bit-mask on the fractional bits; mind negative numbers. |
| `rem_euclid` / `wrap` | `lib.rs::wrap` | parry 4× (angles, grid wrap) | integer modulo. |
| Constants `pi`, `two_pi`, `frac_pi_2`, `e`… | trig call sites | yes | compile-time `const`. |
| `epsilon` + `abs_diff_eq`/`relative_eq` | `try_normalize(eps)`, `Unit::try_new`, `is_identity/is_orthogonal(eps)`, `try_slerp(eps)`, `from_matrix_eps`, every iterative decomposition's convergence test (`symmetric_eigen.rs:196`, `svd.rs`, `schur.rs`), `pseudo_inverse(eps)`, `rank(eps)`; tests (770 `relative_eq!` uses) | parry: `DEFAULT_EPSILON = Real::EPSILON`, 20 `abs_diff_eq`, GJK/EPA tolerances | **The most delicate item.** f32 ε = 1.19e-7, f64 ε = 2.2e-16; a Q32.32 ulp is 2.3e-10, Q16.16 is 1.5e-5. Every hard-coded tolerance must be re-derived relative to the chosen Q-format; nalgebra's *relative* convergence tests `|off| ≤ eps·(|a|+|b|)` underflow to `0 ≤ 0` in fixed point — use absolute tolerances of a few ulps and fixed iteration caps. |
| `is_zero()` exact test | `inverse.rs` (`determinant.is_zero()`), `cholesky.rs` (pivot ≤ 0), `lu.rs` (`diag.is_zero()`) | — | Replace by `abs(x) <= tol`; exact-zero tests pass garbage-but-nonzero pivots of 1 ulp that then blow up the division. |
| `is_finite`, `is_nan` | `ComplexField::is_finite` | rapier 69× | no NaN/Inf in fixed point → becomes overflow policy: decide **panic vs saturate** once (a panic makes the proof fail = DoS vector for an on-chain game; saturation hides bugs). |

Minimal scalar contract for Tiers 1–3: `+ − × ÷ neg abs signum min max clamp sqrt floor ceil`
+ `sin_cos`, `atan2`, `acos` + constants + epsilon comparison. That is a superset of simba's
fixed-point backend minus `exp`/`tan`.

---

## 5. Testing approach in nalgebra, and what is reusable

### 5.1 Layout (`nalgebra/tests/`, 10 116 LoC; entry `tests/lib.rs`)

| Dir | Files | Style |
|---|---|---|
| `tests/core/` | `matrix.rs` (1367 L: construction, indexing, ops, iterators, `push`, `cross`, swizzles, `apply`, `zip_map`, parallel iterators…), `edition.rs` (702: insert/remove/resize/swap/fill), `matrix_view.rs` (337), `conversion.rs` (356), `blas.rs` (120: `gemm/gemv/ger/quadform` vs naive), `cg.rs`, `empty.rs` (0-sized matrices), `reshape.rs`, `variance.rs`, `serde.rs`, `rkyv.rs`, `mint.rs`, `macros.rs`, `matrixcompare.rs`, `helper.rs` (`RandScalar`, `RandComplex`) | Mostly **exact example-based** tests with small integer-valued matrices → *directly portable*, scalar-agnostic. |
| `tests/geometry/` | `rotation.rs` (359), `quaternion.rs` (266), `dual_quaternion.rs` (302), `isometry.rs` (276), `similarity.rs` (280), `unit_complex.rs` (161), `point.rs` (102), `projection.rs` (64) | **Property-based** (`proptest!`): round-trips (`from_euler_angles ∘ euler_angles = id`, `q → R → q`), group laws (`inverse * self = identity`, composition order, all operator permutations give equal results), agreement between representations (`UnitQuaternion` vs `Rotation3` vs `Isometry` vs homogeneous `Matrix4` applied to the same point), `rotation_between(a,b) * a ∥ b`, slerp endpoints. |
| `tests/linalg/` | one file per decomposition: `lu`, `full_piv_lu` (472), `qr`, `col_piv_qr`, `cholesky` (218), `svd` (515), `eigen` (282), `schur`, `hessenberg`, `bidiagonal`, `tridiagonal`, `udu`, `lblt`, `inverse`, `solve`, `exp`, `pow`, `balancing`, `convolution` | Property-based **reconstruction identities**: `L·Lᵀ ≈ M`, `P·L·U ≈ M`, `Q·R ≈ M`, `QᵀQ ≈ I`, `U·Σ·Vᵀ ≈ M`, `M·solve(M,b) ≈ b`, `M·M⁻¹ ≈ I`, each generated by macro for `Dyn` sizes 1..=20 *and* for static 2×2…6×6 (`gen_tests!(f64, …)`, also complex). Plus hand-picked regression matrices (e.g. `svd.rs` singular/ill-conditioned cases, `symmetric_eigen.rs` Wilkinson-shift edge cases in-module `:405-470`). |
| `tests/proptest/mod.rs` (370) | strategies: `PROPTEST_MATRIX_DIM = 1..=20`, `PROPTEST_F64 = -100.0..=100.0`, `vector2()…`, `matrix3()…`, `dmatrix()`, `point3()`, `rotation3()`, `unit_quaternion()`, `isometry3()`, `similarity3()`, `unit_dual_quaternion()`… | 27 test files use `proptest!`. Public generators also live in `src/proptest/mod.rs`. |
| `tests/sparse/`, `tests/macros/` (+`trybuild`) | legacy sparse; macro compile-fail tests | not relevant. |
| `src/**` doctests | ≈425 fenced examples, one per public method, with concrete numbers (`assert_relative_eq!(rot * Vector3::x(), Vector3::y(), epsilon = 1.0e-6)`) | **excellent seed corpus of example vectors.** |
| `src/debug/` | `RandomOrthogonal` (product of Householder reflections), `RandomSDP` (`Q·diag(|λ|+1)·Qᵀ`) | reusable *idea* for generating well-conditioned inputs. |

### 5.2 `approx` usage

`relative_eq!`/`assert_relative_eq!` dominate (397 + 373 uses); no `ulps_eq`. Tolerances: 
`epsilon = 1.0e-7` (332×), `1.0e-5` (30×), `1.0e-6` (19×), a handful of `1e-9/1e-10` and a few loose
ones (`0.001`, `0.6`). `relative_eq` semantics: pass if `|a−b| ≤ epsilon` **or**
`|a−b| ≤ max_relative · max(|a|,|b|)`. nalgebra implements `AbsDiffEq/RelativeEq/UlpsEq` for
`Matrix`, `Unit`, `Point`, every geometry type (component-wise; 43 impl blocks).

For Cairo: implement `abs_diff_eq(a, b, eps)` and `relative_eq(a, b, eps, max_rel)` on the scalar
and derive them field-wise for each struct. Tolerances must scale with the Q-format and with
the operation count (errors accumulate ≈ 1 ulp per multiply): e.g. with Q32.32 use ~2⁻²⁴ for
single ops, ~2⁻¹⁶ for decompositions with inputs in ±100.

### 5.3 Reuse strategy — Rust nalgebra as the oracle

1. **Golden-vector generator (recommended core).** A small Rust binary depending on
   `nalgebra = "0.35"` (f64) that, for each ported function, draws inputs from the same ranges
   as `tests/proptest/mod.rs` (seeded RNG → deterministic), *quantises the inputs to the Q-format
   first* (so Rust and Cairo see bit-identical inputs), evaluates in f64, and emits Cairo test
   files (`#[test] fn …() { let got = f(FixedTrait::from_raw(..)); assert_close(got, expected_raw, tol) }`)
   or a JSON consumed by a Cairo test-generating script. Tolerance per function = a bound chosen
   from the fixed-point error analysis, not from nalgebra's `1e-7`.
2. **Port the property tests as properties** where they are scalar-agnostic and cheap:
   `inverse * self ≈ identity`, `q → R → q`, `transform ∘ inverse_transform ≈ id`,
   `L·Lᵀ ≈ M`, `M·x ≈ b` after `solve`. Cairo has no proptest; emulate with a fixed table of
   N generated cases per property (generated by the Rust tool), or a tiny LCG inside the test. Keep
   N small — each Cairo test runs in the VM.
3. **Port `tests/core/*.rs` example tests almost verbatim** — they use integer-valued entries,
   so results are exact in fixed point (no tolerance needed) and they pin down conventions
   (column-major order, `new` argument order, `cross` handedness, homogeneous coordinates).
4. **Doctest mining.** Extract the ≈425 doctest blocks mechanically; most are 3–6 lines with literal
   inputs and an `assert_relative_eq!`. They document intended semantics of each method
   (e.g. `Isometry::inv_mul`, `UnitQuaternion::rotation_between`, `append_axisangle_linearized`).
5. **Differential test against the fixed-point reference in Rust.** Because simba already has
   `fixed` + `cordic` scalars, one can instantiate *nalgebra itself* with e.g.
   `simba::scalar::FixedI64<U32>` and compare the Cairo Q32.32 results against it — this gives an
   oracle for the *expected fixed-point error envelope* (not bit-exactness: rounding modes and trig
   algorithms differ), and quickly reveals which nalgebra algorithms misbehave in fixed point before
   any Cairo is written.
6. **Cross-library conformance with glam.cairo**: for the overlapping types, assert
   `nalgebra.cairo` and `glam.cairo` agree bit-for-bit (they should share the scalar crate), using
   the conversion table of `nalgebra/src/third_party/glam/common/`.
7. **Step-count benchmarks as tests.** nalgebra's `benches/` (criterion) measures time; the Cairo
   analogue is asserting `steps`/gas budgets per op (e.g. `Matrix3 * Matrix3`, `Isometry3::inv_mul`,
   `Cholesky6::new`) to catch regressions — the project is, after all, a *benchmark*.

---

## 6. Proposed porting scope, in tiers

Risk scale — **L** low, **M** medium, **H** high — judged for a fixed-point, no-float,
loop-expensive VM. "Phys" = needed by the rapier/parry feature set (✔ core, ○ optional
subsystem, ✘ unused).

### Tier 0 — scalar foundation

| Item | Phys | Risk | Notes |
|---|---|---|---|
| Fixed-point type (suggest signed Q32.32 in i64-ish magnitude or Q64.64 for headroom) + `Add/Sub/Mul/Div/Neg/PartialOrd` | ✔ | **M** | Central design decision shared with glam.cairo/rapier.cairo. Evaluate cubit (`refs/cubit/src/f64`, `f128`), orion (`refs/orion/src/numbers/fixed_point/implementations/fp32x32.cairo`, `fp64x64.cairo`…), alexandria, origami. Overflow policy (panic vs saturate vs wrap) must be decided here. Products of 3 scalars (det 3×3, triple products, quadform) overflow narrow formats. |
| `abs signum copysign min max clamp wrap floor ceil round` | ✔ | L | |
| `sqrt` | ✔ | L | integer sqrt hint + range check. |
| `recip`, wide-mul + deferred rescale helpers (`mul_add`, `dot2/3/4` primitives) | ✔ | **M** | Biggest performance lever; design the API so vector code can accumulate unscaled products. |
| `sin cos sin_cos` | ✔ | M | LUT/polynomial; accuracy target ~1e-6; angle reduction via `wrap`. |
| `atan2 acos asin` | ✔ (few sites) | M | CORDIC loop is step-expensive (≈32 iterations); prefer rational/polynomial approximations. Domain clamping mandatory. |
| `exp ln powi` | ○ | M | defer. |
| `tan powf cbrt hypot` hyperbolics | ✘ | — | skip. |
| Epsilon model + `abs_diff_eq`/`relative_eq` traits | ✔ | **M** | Conceptual risk (see §4). |
| Trait stack `Scalar → Ring → Real → RealTrig` | ✔ | L | Keeps Tier 1+ generic so integer scalars can be used in exact tests. |

### Tier 1 — static vectors, matrices, points (all unrolled structs)

| Item | Phys | Risk | Notes |
|---|---|---|---|
| `Vector2/3/4`: ctor, consts (`zeros`, `x()`…, `repeat`), `+ − neg ×s ÷s`, `dot`, `cross`/`perp`, `component_mul/div`, `abs`, `inf/sup`, `min/max/amax/imax…`, `norm_squared`, `norm`, `normalize`, `try_normalize`, `cap_magnitude`, `lerp`, `push`/`xy()/xyz()`, `to_homogeneous`, arrays/tuples conv | ✔ | L | Overlaps glam.cairo 100 % → share or alias. `norm` of large vectors: `x²+y²+z²` needs 2·bits+2 headroom. `normalize` of tiny vectors loses all precision → `try_normalize(min_norm)` semantics matter. |
| `Vector1`, `Vector5`, `Vector6` | ○ | L | `Vector6` = spatial velocity/force (multibody, solver body `as_vector`). Pure boilerplate; consider code-generating. |
| `Matrix2/3/4`: `new` (row-major args), `identity`, `zeros`, `from_diagonal(_element)`, `from_columns/rows`, `column/row`, `transpose`, `M±M`, `M×s`, `M×M`, `M×v`, `tr_mul`, `trace`, `diagonal`, `cross_matrix`, outer product, `kronecker` (2×2 only), `swap_rows/columns`, `abs`, `is_identity/is_orthogonal(eps)` | ✔ | L | `Matrix3×Matrix3` = 27 mul: use deferred rescale (9 shifts not 27). `Matrix4` is graphics-only. |
| Rectangular `Matrix2x3`, `Matrix3x2`, `Matrix3x6`, `Matrix6x3`, `Matrix6` | ○ | M (volume) | Needed only with multibody/soft body. `Matrix6` products = 216 mul each → proof-cost hotspot; exploit block structure (3×3 blocks) instead of naive unrolling. |
| `Point2/3` (`coords`, `origin`, `P−P`, `P±V`, `lerp`, `center`, `distance(_squared)`, homogeneous) | ○ | L | parry dropped it; keep as a thin wrapper for API fidelity or skip. |
| `Unit<T>` + `UnitVector2/3`: `new_normalize`, `try_new`, `new_unchecked`, `new_and_get`, `into_inner`, `renormalize(_fast)`, `x_axis()`… | ✔ | L | |
| `orthonormal_subspace_basis`, `orthonormalize` | ✔ | L–M | needs robust branch on the largest component; uses normalize. |
| Generic `Matrix<T,R,C,S>`, views, storage, allocator, `Dyn` | ✘ | — | **Do not port.** |

### Tier 2 — geometry

| Item | Phys | Risk | Notes |
|---|---|---|---|
| `UnitComplex` / `Rotation2`: `new(angle)`, `identity`, `from_cos_sin_unchecked`, `*`, `inverse`, `angle` (atan2), `to_rotation_matrix`, `transform_*`, `inverse_transform_*`, `rotation_to`, `angle_to`, `renormalize` | ✔ (2D) | L | 2 scalars; composition is 4 mul. Drift → renormalise periodically. |
| `Quaternion` core: `new(w,i,j,k)`, `from_parts`, Hamilton product, `conjugate`, `dot`, `norm`, `normalize`, `try_inverse`, `lerp` | ✔ | L | |
| `UnitQuaternion`: `identity`, `from_axis_angle`, `from_scaled_axis`/`new`, `q*q`, `inverse`, `q*v` (2-cross form), `to_rotation_matrix`, `from_rotation_matrix`, `angle`, `axis`, `scaled_axis`, `axis_angle`, `rotation_to`, `angle_to`, `append_axisangle_linearized`, `nlerp`, `renormalize(_fast)` | ✔ (3D) | **M** | Precision drift of the unit constraint under repeated fixed-point products is the main risk: renormalise every integration step (`renormalize_fast` costs 4 mul + 4 mul, no sqrt). `from_scaled_axis` near zero angle: `sin(θ/2)/θ` → use series branch (nalgebra's `exp_eps`). `from_rotation_matrix`: 4-way branch on trace, each with `sqrt` and ÷ by a possibly small denom → precision loss near 180°. |
| `slerp`/`try_slerp`, `powf`, `ln`/`exp` (general), `rotation_between`, `scaled_rotation_between`, euler angles (`from_euler_angles`, `euler_angles`), `look_at_*`, `face_towards`, `mean_of` | ○/✘ | M–H | `slerp` needs `acos` + 2 `sin` + ÷ by `sin θ` (ill-conditioned near 0 and π → nlerp fallback). `euler_angles` has gimbal-lock branches with `atan2`. Unused by the engine → defer. |
| `Rotation3` (matrix): ctor from axis-angle/quaternion, `*`, `inverse`=transpose, `transform_*`, `from_matrix_unchecked`, `renormalize` | ○ | L–M | `from_matrix_eps` (iterative projection) → **H**, skip. |
| `Translation2/3` | ○ | L | glamx folds it into `Pose`; a plain `Vector` suffices. |
| `Isometry2/3`: `identity`, `from_parts`, `new`, `translation`, `rotation`, `*`, `inverse`, **`inv_mul`**, `transform_point/vector`, `inverse_transform_point/vector/unit_vector`, `append_translation/rotation(_wrt_point)_mut`, `to_homogeneous`, `lerp_slerp` | ✔ | L–M | Pure composition of Tier-1/2 pieces. `inv_mul` and `inverse_transform_point` avoid forming inverses (fewer mul → fewer rounding steps). `lerp_slerp` inherits slerp risk. |
| `Similarity2/3`, `Scale` | ✘ | L | one ÷ in inverse. Defer. |
| `Transform/Affine/Projective` | ✘ | M | 4×4 inverse + type-level categories (no Cairo analogue → 3 separate types). Defer/skip. |
| `Perspective3`, `Orthographic3`, `base/cg.rs` | ✘ | M | rendering; belongs (if anywhere) in glam.cairo. Skip. |
| `DualQuaternion` | ✘ | M–H | `sclerp` trig-heavy. Skip. |
| `Reflection` | ✘ (internal to QR) | L | only if Householder QR is ported. |

### Tier 3 — small static linear algebra (unrolled per size)

| Item | Phys | Risk | Notes |
|---|---|---|---|
| `determinant` 2×2, 3×3 (closed form), 4×4 | ✔ | **M** | Triple products: magnitude up to range³ and down to ulp³. Compute in wide integers at scale S³ (or S²·S) and rescale once; otherwise small-determinant matrices read as singular. |
| `try_inverse` 2×2, 3×3 (adjugate × 1/det), 4×4 (MESA unrolled, `linalg/inverse.rs:143`) | ✔ | **M–H** | Division by small det amplifies quantisation error by cond(M). Use `abs(det) > tol` guard; return `Option`. For SPD matrices prefer LDLᵀ. For inertia tensors prefer the symmetric 6-scalar form (`SdpMatrix3::inverse_unchecked`). |
| Triangular solves 2×2…6×6 | ○ | L | |
| `Cholesky` 2×2, 3×3, (6×6): `new → Option`, `solve`, `inverse`, `determinant`, `l()` | ○ (soft body, block solver) | **M** | n sqrt + n ÷. Positive-definiteness test `pivot > 0` must become `pivot > tol`; `new_with_substitute` semantics (clamp pivot) is a good fixed-point default. |
| **LDLᵀ / `UDU`** 3×3, 6×6 | ○ | L–M | √-free; n divisions. Recommended over Cholesky for fixed point whenever only `solve`/`inverse` is needed. |
| `LU` (partial pivot) 2×2…4×4 (6×6): `new`, `solve`, `try_inverse`, `determinant` | ○ | M | No sqrt, no iteration. Pivot by max-abs improves fixed-point conditioning. Unrolling 6×6 by hand is large → use bounded loops over `[T; 36]`/struct with a generated body. |
| `SymmetricEigen` 2×2 (closed form: `wilkinson_shift`-style quadratic; √ only) | ✔ (2D inertia is scalar, so mostly geometry utils) | L–M | Catastrophic cancellation when eigenvalues are close: use the numerically stable form (`glamx/src/eigen2.rs` is *not* the stable one; nalgebra's `wilkinson_shift`, `symmetric_eigen.rs:323`, is). |
| `SymmetricEigen` 3×3 | ✔ (principal inertia, once per body; convex hull; soft-body) | **H** | Options: (a) nalgebra's tridiagonal-QR — iterative, relative-eps convergence test breaks in fixed point, Givens rotations need √+÷ per sweep; (b) glamx/Eberly closed form — needs `acos` + `cos` and a normalisation by `p = sqrt(p2/6)` that is ill-conditioned for near-isotropic tensors (spheres, cubes — *common in games*!), must special-case `p2 ≈ 0`; (c) **cyclic Jacobi with a fixed number of sweeps (4–6)** — only √, ÷, ×; unconditionally stable, deterministic step count → best fit for a provable VM. Eigenvectors must be re-orthonormalised; degenerate eigenvalues give arbitrary (but valid) bases — tests must check `V·Λ·Vᵀ ≈ M`, not compare vectors with the oracle. |
| `SVD` 2×2 (closed form: 2√ + 2 `atan2` + 2 `sin_cos`, `svd2.rs`), `polar` 2×2/3×3, `SVD` 3×3 (`svd3.rs` = eigen(AᵀA) + QR) | ○ (soft-body corotational/FEM) | **H** | AᵀA squares the condition number — in fixed point that halves the usable bits. 2×2 can be done trig-free via half-angle identities. Defer until soft bodies are in scope. |
| `pseudo_inverse`, `rank` | ○ (IK) | H | depends on SVD; for IK replace with damped least squares + LDLᵀ. |
| `pow(u32)` | ✘ | L | trivial. |
| `QR`, `ColPivQR`, `FullPivLU`, `LBLT` static | ✘ | M | not needed by physics; skip. |

### Tier 4 — dynamic `DVector`/`DMatrix` and general decompositions

| Item | Phys | Risk | Notes |
|---|---|---|---|
| `DVector`, `DMatrix` containers (column-major `Array<T>` + dims), `zeros`, `from_fn`, `get/set`, `rows`/`columns` copy-out, `copy_from`, `fill`, `resize_vertically` | ○ (multibody) | **H** | Cairo `Array` is append-only and immutable once written; "in-place" algorithms need `Felt252Dict<T>` (each access = dict squash cost) or rebuilding arrays per update (O(n²) copies). No borrowed views → `fixed_rows_mut`, `rows_mut` patterns used ~40× in `multibody.rs` have no cheap analogue. |
| BLAS-like: `dot`, `axpy`, `cmpy`, `gemv`, `gemv_tr`, `gemm`, `gemm_tr`, `quadform`, `tr_mul_to`, `norm_squared` | ○ | **H** (cost) | Loops with indexed access; `gemm` is O(n³) array reads, each a bounds-checked `Span::at`. A 20-dof multibody → 8 000 mul + ~24 000 reads per `quadform`. Proving cost likely prohibitive per physics step. |
| `LU<Dyn,Dyn>` + `solve_mut` | ○ | H | as above; pivoting requires row swaps → dict-backed storage. |
| `Cholesky<Dyn>` (+ rank-one update, insert/remove column) | ✘/○ | H | multibody mass matrix is SPD → LDLᵀ would do instead of LU. |
| General `SymmetricEigen`, `SVD`, `QR`, `Schur`, `Hessenberg`, `Bidiagonal` for `Dyn` | ✘ | **H** | data-dependent iteration counts (unbounded loops are a proving-cost and DoS hazard), relative-epsilon convergence tests, heavy Householder/Givens traffic. |
| `Matrix3xX`/`Matrix6xX` Jacobians, `DVectorView(Mut)` slices into a shared velocity buffer | ○ | H | Entire design relies on aliasing mutable views → needs re-architecture (explicit offsets + functional updates), not a port. |
| `edition.rs` (insert/remove rows/cols), `statistics.rs`, `convolution.rs` | ✘ | M | skip. |

Recommendation: **treat Tier 4 as a separate, later project gated on a decision that
`rapier.cairo` supports multibody (reduced-coordinate) joints.** Upstream already treats that
subsystem as optional (`alloc`-gated; absent from the SPIR-V build). If it is wanted, prefer a
fixed-capacity design (`Matrix6xN` with N ≤ small const) and articulated-body (Featherstone O(n))
algorithms that only need 6×6 blocks (Tier 3) rather than porting `DMatrix` + dense LU.

### Tier 5 — out of scope

| Item | Why |
|---|---|
| `nalgebra-sparse`, `src/sparse`, `src/io` (Matrix Market) | No consumer; sparse formats rely on mutable index arrays; parsing text on-chain is pointless. |
| `nalgebra-lapack` | FFI. |
| SIMD / AoSoA (`SimdValue`, `simd_*`, `wide`, lanes, `select`/`if_else`) | No data parallelism in the VM. Port the *scalar* instantiation of rapier's `ScalarType`. |
| Complex numbers (`ComplexField`, `Complex<T>`, `adjoint`, `dotc`, `hegemv`, Hermitian variants, complex Schur/eigenvalues) | No physics use; doubles every kernel. (`UnitComplex` is ported as a plain 2-scalar struct, not via a complex type.) |
| `nalgebra-glm` | Façade only. |
| `nalgebra-macros` (`matrix!`, `vector!`, `stack!`) | No equivalent macro system needed; constructors suffice. |
| Generic dimension machinery (`Dim`, `Const`, `Dyn`, typenum, `Allocator`, `ShapeConstraint`, storage traits, views/slices, `uninit`) | Inexpressible and pointless in Cairo. |
| serde / rkyv / bytemuck / mint / alga / encase / rand / rayon / defmt / quickcheck integrations | Replaced by Cairo `Serde`/`Store`/`Debug` derives. |
| Matrix `exp`, `Schur`, `Hessenberg`, `balancing`, general `Eigen`, `convolution`, `statistics`, `LpNorm`, quaternion transcendental zoo (`sinh`, `acosh`, …) | Scientific-computing features with `powf`/`exp`/unbounded iteration; zero physics use. |
| `Perspective3`/`Orthographic3`/`cg.rs`, `DualQuaternion`, `Transform` categories | Rendering/animation; if ever needed they belong to glam.cairo. |

### 6.1 Suggested order of work and sizing

1. **Tier 0** shared with glam.cairo (one scalar crate; this is where the benchmark effort —
   steps per mul/div/sqrt/sin_cos/atan2 — pays off for all three projects).
2. **Tier 1 + the ✔ rows of Tier 2** (`Vector2/3`, `Matrix2/3`, `Unit`, `UnitComplex`,
   `UnitQuaternion`, `Isometry2/3`) — ≈ the glamx-sized subset (~3–5k LoC of Cairo), backed by
   golden vectors from Rust nalgebra f64. Decide alias-vs-duplicate w.r.t. glam.cairo up front.
3. **Tier 3 core**: det/inverse 2×2–3×3(–4×4), LDLᵀ + Cholesky 2×2/3×3, LU ≤ 4, symmetric eigen
   2×2 (closed form) and 3×3 (fixed-sweep Jacobi). This is where nalgebra.cairo is genuinely
   differentiated and what a `rapier.cairo` with mass-property computation needs.
4. **Tier 1 extensions** (`Vector6`, `Matrix6`, rectangular blocks) + 6×6 LDLᵀ/LU **only when** a
   consumer (soft bodies / articulated bodies) is scheduled.
5. **Tier 4** only after an explicit go/no-go on multibody joints, preferably replaced by a
   fixed-capacity / Featherstone design.

### 6.2 Cross-cutting fixed-point risks (apply to every tier)

* **Range vs precision.** World coordinates ±10⁴ with Q32.32 are fine; *squared* norms reach 10⁸·3
  (OK in 31 integer bits only barely) and inertia-tensor products / determinants overflow. Use
  wide intermediates everywhere a product of ≥ 2 scaled values is summed.
* **Normalisation of small vectors** (contact normals from tiny penetration vectors, GJK
  directions): fixed point has *absolute* not relative precision → a vector of length 10⁻⁶ has
  ~12 significant bits in Q32.32. Pre-scale by a power of two before `norm`/`normalize`
  (`parry`'s `improved_fixed_point_support` branches show where upstream hit this).
* **Exact-zero and relative-epsilon tests** in nalgebra (`is_zero()`, `eps·(|a|+|b|)`) must be
  replaced by absolute ulp-based tolerances.
* **Data-dependent loops** (QR iterations, `from_matrix_eps`, SVD) → fixed iteration counts for
  deterministic, bounded proving cost.
* **Determinism is free** (integer arithmetic) — a genuine advantage over upstream, which needs
  `enhanced-determinism`/`libm` gymnastics; but it also means *bit-exact agreement with glam.cairo*
  is achievable and should be a tested invariant.
* **Panics are proof failures.** Every `unwrap`, overflow, division by zero and out-of-bounds in
  the math layer becomes a liveness risk for an on-chain game loop; prefer `Option`-returning
  variants (`try_normalize`, `try_inverse`, `Cholesky::new → Option`) as the *primary* API and keep
  panicking versions as thin wrappers.
