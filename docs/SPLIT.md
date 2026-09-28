# Package split of nalgebra (WP 9-NS1: inventory and cut plan)

> Research, no code moved. Status: plan for the programme session; nothing here is decided until
> it is sent back. Every number below was measured on throwaway builds (Scarb 2.19.4, this
> machine, 2026-09-28) with the helpers of `tools/split/` (§10). Rule: `docs/PLAN.md` M9 and
> `pm/decisions/2026-09-28-package-granularity-rule.md`; gate definitions as the programme session
> made them precise (§0).

## 0. Summary

- **The cut**: 32 sub-crates `nalgebra_*` of 3,700 to 35,500 physical library lines (largest:
  `nalgebra_types6`, 35,462), an acyclic dependency graph, and the facade `nalgebra` (11,700 lines:
  root functions, macros, re-exports). Upstream's module paths are kept INSIDE every sub-crate
  (`base/matrix3.cairo` stays `base::matrix3` in whichever crate hosts its items) and the facade
  rebuilds the 0.1.0 tree module by module.
- **Zero-break, proved**: a prototype of the whole cut compiles, and a consumer naming all 4,284
  public paths of 0.1.0 plus methods, operators, products, norms, solves, LU and the macros with
  0.1.0 imports compiles against the facade prototype (§3.4). No method changes trait: every
  inherent trait (`Matrix3Trait`…) moves whole. That is the glam-cairo pattern the programme
  session asked for (§1, point 2).
- **Gate 2 (marginal cost ≤ 5 s / 1 GB)**: every sub-crate passes. The largest measured
  marginals are `nalgebra_dynamic` 4.6 s and `nalgebra_dim6b` 2.7 s (median of 2–3 cold builds;
  noise on this VM is ±1.5–3 s, §6.3). Memory is ≤ 0.65 GB for every crate.
- **Gate 3 (declared closures ≤ 15 s / 3 GB)**: `nalgebra_glam`'s closure costs **10.5 s /
  1.85 GB** (today 37 s / 5.5 GB); static 1–4 + geometry 12.4–14.6 s / 2.39 GB; static 1–4 +
  LU/Cholesky/QR 12.3–15.0 s / 2.3 GB; static 1–3 + SVD/eigen 8.6–9.6 s / 1.6 GB; `core` + the
  pivoting decompositions 4.8 s / 0.94 GB. Two closures sit close to 15 s: they need the GitHub
  runner measurement before the cut is fixed (§6.3).
- **The facade fails gate 3 by construction** (it depends on every sub-crate): 58.8 s / 10.0 GB
  cold, against 96.7 s / 10.4 GB for 0.1.0 today. Scarb has no optional dependency, so
  `default-features = false` on the facade no longer saves anything (§4). A user who wants a
  cheap build depends on the sub-crates.
- **Zero steps**: the split moves code, it does not rewrite it, except for three kinds of small
  internal change listed in §3.3. Every move PR proves it with `tools/split/gas_compare.py`
  (§5).
- **Release**: a non-breaking patch release, nalgebra 0.1.1 and every sub-crate 0.1.1 in
  dependency order, then nalgebra_glam 0.1.1 on `nalgebra_core` + `nalgebra_dim3v` +
  `nalgebra_dim3` (§9).

## 1. Cairo coherence and impl placement (measured)

Throwaway workspace `tools/split/probes/` (six packages; its README lists every probe and its
expected outcome), Scarb 2.19.4 / Cairo 2.19.4.

| question | answer (probe) |
|---|---|
| Orphan rule? May crate C implement a trait of crate A for a type of crate B? | **No orphan rule**: it compiles (`pc`: `Tr<B>`, `Add<A>`, `MulM<B, A>`). |
| Where does a caller find an impl without importing it? | Only in the **module of the trait or of one of the trait's generic-argument types** (`m1`: `MulM<A, B>` in `B`'s module, another crate than `A`'s and the trait's, is found; `q2`: `Into<A, B>` in `B`'s module is found). **Nowhere else**: not in a third crate (`m2`, `m3`, `m6`, `m7`), and not even in another module of the type's own crate (`q1`: `Mul<A>` in `pa::shapes::inner`). |
| Core traits (`Add`, `Mul`, `Into`, `PartialEq`, `Serde`…) for a type of another crate | Allowed, same lookup rule: the impl must sit in the module of one of its argument types (for `Add<Matrix3>`, only `Matrix3`'s module). |
| `#[generate_trait]` outside the type's crate | Allowed (`m5`). Its methods need the trait in scope, as today. |
| Impl lookup through a facade | The rule above holds through re-exports (`e1`). An impl sitting in a third crate becomes callable only if imported, and a facade glob (`use facade::*`) imports it (`e2b`). Importing an **impl** also puts its trait's methods in scope, which can make method calls ambiguous: seen in the prototype (`Normed::dot` against `Vector3Trait::dot`). |
| What does `pub use sub::*` re-export? | Every public item and module of `sub`, including module paths (`e4`: `facade::shapes::A`). Two globs exporting the same name are **ambiguous at use** (`e10`: E2087); an explicit `pub use` (or an item of the facade's own, such as its `pub mod`) wins over the globs. |
| Macros across crates | A declarative macro of another crate can be called by path or imported (`m8`), re-exported (`e5`), and a macro body must reach its crate through `$defsite::` (`e6`). nalgebra's macros resolve `$defsite::super::Matrix3Trait`, i.e. the root of the crate that defines them, so they stay in the facade. |
| Supertraits | None in Cairo 2.19: moving a method to another trait changes what a caller must import. The plan moves no method (§3). |
| Features across crates | `dep/feature` forwarding works (`e11`: `extra = ["pa/extra"]`). Features **unify per compilation unit**: if any crate of the unit enables a dependency's default features, they are on for all (so the sub-crates depend on each other with default features off whenever they have features at all). |

The programme session's two points from glam-cairo's cut (`PK-G-glam-cut-plan.md` §9):
1. "A core-trait impl is found without import only in the type's own module": **confirmed and
   made precise**. It is the module of ANY type among the trait's generic arguments (`Into<A, B>`
   in `B`'s module works), and not another module of the same crate. Every core-trait impl with a
   single nalgebra type (`Add<Matrix3>`…) therefore stays with its struct, and the plan keeps
   struct and core-trait impls in the same crate and module (the solver checks it: §3.2).
2. "No supertraits; keep the types and the methods that return them together, move only the
   rest": **confirmed** (probes `e8`, `e9`). The plan goes further than glam's (no method changes
   trait at all): every inherent trait moves whole, into a crate ABOVE every type its methods
   build (§2.3). No public path changes (proved in §3.4).

## 2. Inventory

### 2.1 Library lines (0.1.0, `scripts/consumer_cost.py` rules, 499,786 in total)

| family | lines |
|---|---:|
| linalg | 184,400 |
| base: the 36 static shapes, max dimension 6 | 93,469 |
| base: shapes, max dimension 5 | 53,495 |
| geometry | 33,638 |
| base: shapes, max dimension 4 | 28,808 |
| base: shapes, max dimension 1–3 | 28,175 |
| base: statistics | 18,357 |
| base: dynamic | 18,136 |
| base: solve | 13,907 |
| base: blas | 12,103 |
| macros | 4,653 |
| sparse + io | 3,482 |
| base: cg, point swizzles, transpose, unit, points 2/3, views (declarations), errors, kernels | 6,065 |
| root + lib | 1,098 |

The 36 shape files (194k lines) split into: inherent traits (`MatrixRxCTrait`, `…AngleTrait`,
`…InternalTrait`, `…EditTrait`) 108.8k; views (`FixedView`, `FixedRows`, `FixedColumns`,
`PadTo6`, `CropFrom6`) 35.9k; products (`MatrixMul`, `MatrixTrMul`) 20.1k; `MatrixKronecker`
5.1k; indexing 3.7k; norms 3.2k; core-trait impls (`Add`, `Sub`, `Neg`, `Mul`, assignments,
`PartialOrd`, `Bounded`, `One`, `Sum`, `Product`, `IndexView`, array conversions) 13.5k; structs
0.5k.

Linalg per family and dimension band (lines):

| family | shared | 1–3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|
| svd (+ svd2/svd3) | 5,507 | 3,956 | 4,074 | 7,221 | 11,570 |
| col_piv_qr | 78 | 2,663 | 3,653 | 7,525 | 14,389 |
| bidiagonal | 78 | 2,104 | 2,893 | 5,835 | 11,186 |
| full_piv_lu | 78 | 1,957 | 2,466 | 4,701 | 8,391 |
| schur | 19 | 1,277 | 2,104 | 4,323 | 8,148 |
| qr | 2,602 | 1,650 | 1,430 | 3,067 | 5,543 |
| permutation_sequence | 10,224 | | | | |
| lu (+ steps, inverse, `Perm1..6`) | 6,623 | 993 | 948 | | 2,281 |
| symmetric_eigen | | 918 | 1,000 | 1,821 | 2,940 |
| lblt | 22 | 1,087 | 846 | 1,262 | 1,847 |
| symmetric_tridiagonal / hessenberg / eigen | 67 | 1,468 | 1,165 | 2,337 | 3,994 |
| cholesky, ldlt, udu, cholesky_update | 4,817 | | | | |
| householder, givens, balancing, exp, pow | 6,946 | | | | |

### 2.2 Method

`tools/split/edges.py` cuts every library file into its top-level items (test-only code
excluded, `consumer_cost.py`'s rules) and records per item the names it references (types,
traits, constants, functions; resolved through the file's own `use` statements, homonyms
included), its method calls (`.m(`, resolved against the traits in scope: an
over-approximation), the generic arguments of every impl and its visibility. 6,600 items,
edges between items. `tools/split/plan.py` places the items: rule-based homes, lifting to a
fixpoint (every dependency ends in the same or a lower crate, hence an acyclic crate graph), and a
check of §1's lookup rule for every impl of a public trait with type arguments ("anchor").
`tools/split/prototype.py` then emits a real workspace of the plan and the compiler proves it (an
edge the analysis missed or over-approximated shows up as a build error).

### 2.3 Cross-family edges that decide the cut

| edge | what | consequence |
|---|---|---|
| inherent trait of dimension k → struct of dimension k + 1 | `insert_column` / `insert_row` (every shape), `push`, `to_homogeneous`, `from_homogeneous`, the `Vector2::xxx` swizzles | Within base the 36 shapes form ONE strongly connected component if inherent traits sit with their structs (120k lines). Resolved by placing inherent traits ABOVE the types of the next band (`nalgebra_dim4` above `nalgebra_types5`). |
| shape → geometry | `MatrixRxC::div_rotation` (12 shapes with 2 or 3 columns) names `Rotation2` / `Rotation3`; `Rotation3`'s own `Div` impl calls `Rotation3Trait`, which calls `Matrix3Trait` | The dimension 1–3 methods and the 2D/3D rotations, quaternions, isometries and similarities are one knot: they share `nalgebra_dim3`. |
| shape → geometry | `ApproxEqTrait` (crate-private helper of `relative_eq` / `ulps_eq`, in `geometry/quaternion.cairo`) used by every shape | Moves to `nalgebra_core` (not a public path). |
| base → linalg | `Matrix6Trait::is_invertible` / `is_special_orthogonal` run `Lu6` | `Lu6`, `Perm6` and their permutation impls sit with `Matrix6` in `nalgebra_dim6` (DESIGN D9 already notes it). |
| product → larger shape | `MatrixMul<A, B>` output ≤ the larger operand; `MatrixTrMul<Matrix6x2, Matrix6x3>` is `mul_mat` of the `Matrix2x6` literal | Products stay in the module of their LARGER operand (moved there when it is the right one). The dimension-6 types cannot be split: outer products (`Vector6 * RowVector6 → Matrix6`) and transposed products tie the 6×c and r×6 families together in both directions (brute force over the 2¹¹ partitions). |
| Kronecker | `Matrix2 ⊗ Matrix3 → Matrix6` (output larger than both operands) | `MatrixKronecker` and its 196 impls go to their own crate, in the trait's module. |
| views | `FixedView<Matrix6, Matrix3>`… 441 + 252 impls, used by no other item of the library | Own crates, impls in the trait's module (`nalgebra_views`, `nalgebra_edition`). |
| norms | `Norm<EuclideanNorm, Matrix3, T>` calls `Matrix3Trait::norm`, and `apply_norm` (inherent) is generic over `Norm` | Impls move to the module of the MARKER struct (`EuclideanNorm`…), which moves to `nalgebra_norm` above every inherent trait; the `Norm` trait stays in `core`. |
| solve | `MatrixSolve` is ONE blanket impl whose `impl K: SolveKernel<M, B>` is resolved at the caller | `SolveKernel` impls must stay anchored: in the shape modules of each type band. |
| permutations, Givens | public `PermuteRows<P, M>`, `GivensRotate<T, M>` | Declarations and `Perm1..5` in `core`, impls in the shape modules of each band; `Perm6` impls in `nalgebra_dim6`. |
| `nalgebra_glam` → 1- and 4-dimensional point / translation types | conversions of `Point4`, `Translation4` | These TYPES (and their core-trait impls) move to `nalgebra_core`; their methods stay above. |

## 3. The cut

### 3.1 Crates

Ordered lowest first (a crate depends only on crates above it in this table). "Lines": physical
lines of the prototype crate (`consumer_cost.py --lines-only`, imports included; the real crate
will be a little smaller). Marginal cost: cold build of an empty consumer of the crate minus an
empty consumer of its direct dependencies together (§6).

| crate | lines | content (upstream module paths kept) | direct deps (besides `simba`) | marginal s / GB |
|---|---:|---|---|---:|
| `nalgebra_core` | 20,571 | the 36 structs and their core-trait impls; products, transposed products, indexing, solve kernels, permutations and Givens impls of shapes up to 4x4; the declarations of every generic trait (`MatrixMul`, `MatrixTrMul`, `MatrixIndex`, `Norm`, views, `MatrixSolve`, `PermuteRows`, `GivensRotate`, `LuSteps`); `Point1/2/3/4`, `Translation1/4`, `Unit`, `errors`, kernels | – | 1.6 / 0.45 |
| `nalgebra_dim3v` | 5,474 | methods of `Matrix1`, `Vector2`, `Vector3` | core | 1.4 / 0.10 |
| `nalgebra_dim3` | 23,957 | methods of the other shapes up to 3x3 and the 2D/3D geometry: quaternions, unit complex, rotations, translations 2/3, isometries, similarities, dual quaternions | core, dim3v | 2.4 / 0.54 |
| `nalgebra_types5` | 22,815 | types of dimension 5 (struct, core-trait impls, products, indexing, solve / permutation / Givens impls, edit kernels); `Point4Trait`, `Translation4Trait` | core, dim3 | 0.8 / 0.40 |
| `nalgebra_dim4` | 16,312 | methods of the dimension-4 shapes | core, dim3, types5 | 0.0 / 0.30 |
| `nalgebra_geometry` | 13,511 | projective / affine / transform, perspective, orthographic, scales and reflections 1–4, `Point1`, `Translation1` methods, `cg` | core, dim3, dim4, types5 | 1.3 / 0.30 |
| `nalgebra_types6` | 35,462 | types of dimension 6 (as `types5`) | core, dim3, geometry, types5 | 3.6 / 0.65 |
| `nalgebra_dim5` | 16,512 | methods of `Matrix5`, `Matrix5xC`, `Vector5` | core, dim3, dim4, types5, types6 | 0.6 / 0.30 |
| `nalgebra_dim5a` | 10,756 | methods of `Matrix2x5`, `Matrix3x5`, `Matrix4x5`, `RowVector5` | core, dim5, types5, types6 | 0.8 / 0.21 |
| `nalgebra_dim6a` | 24,652 | methods of `Matrix6xC`, `Vector6`; dimension-6 edit kernels | core, dim3, dim4, dim5, types5, types6 | 2.5 / 0.39 |
| `nalgebra_dim6` | 15,523 | methods of `Matrix6`; `Lu6`, `Perm6` and its permutation impls | core, dim6a, types6 | 1.3 / 0.19 |
| `nalgebra_dim6b` | 17,273 | methods of `MatrixRx6`, `RowVector6` | core, dim6, dim6a, types5, types6 | 2.7 / 0.31 |
| `nalgebra_edition` | 15,838 | `FixedRows`, `FixedColumns`, `FixedResize`, `PadTo6`, `CropFrom6` and their impls | core, dim6a, types5, types6 | 1.5 / 0.22 |
| `nalgebra_views` | 23,485 | `FixedView` and its 441 impls | core, edition, types5, types6 | 2.3 / 0.31 |
| `nalgebra_kronecker` | 5,964 | `MatrixKronecker` and its impls | core, types5, types6 | 2.3 / 0.11 |
| `nalgebra_norm` | 4,044 | the norm markers and every `Norm` impl | every `dim*`, types5, types6 | 0.5 / 0.12 |
| `nalgebra_geometry_nd` | 6,043 | points, translations, scales, reflections of dimension 5 and 6 | core, types5, types6 | 0.9 / 0.17 |
| `nalgebra_statistics` | 18,576 | `base::statistics` | core, types5, types6 | 2.0 / 0.23 |
| `nalgebra_blas` | 12,322 | `base::blas` | core, types5, types6 | 1.4 / 0.36 |
| `nalgebra_linalg` | 10,722 | LU, Cholesky, LDLᵀ / UDU, Cholesky update, QR up to 4x4; Householder / LU steps | core, dim3, dim4 | 1.1 / 0.24 |
| `nalgebra_linalg_svd` | 11,152 | symmetric eigen and SVD up to 4x4 (the per-band part of the SVD kernels) | core, dim3, dim3v | 1.4 / 0.25 |
| `nalgebra_linalg_pivot` | 12,879 | column-pivoting QR, full-pivoting LU, LBLᵀ up to 4x4 | core | 0.0 / 0.25 |
| `nalgebra_linalg_spectral` | 13,095 | bidiagonal, Schur, eigen, Hessenberg, tridiagonal, balancing, exp / pow up to 4x4 | core, dim3, dim3v, dim4, linalg | 2.1 / 0.32 |
| `nalgebra_linalg5` | 15,075 | LU / Cholesky / QR / eigen / SVD of dimension 5 | core, linalg_svd, types5 | 1.7 / 0.28 |
| `nalgebra_linalg5_pivot` | 13,712 | pivoting decompositions of dimension 5 | core, types5 | −1.3 / 0.24 |
| `nalgebra_linalg5_spectral` | 14,793 | spectral decompositions of dimension 5 | core, dim3, dim4, dim5, linalg, linalg_spectral, types5 | 1.6 / 0.33 |
| `nalgebra_linalg6` | 26,231 | QR / Cholesky / eigen / SVD of dimension 6 | core, dim6, linalg5, linalg_svd, types5, types6 | 1.8 / 0.53 |
| `nalgebra_linalg6_pivot` | 24,893 | pivoting decompositions of dimension 6 | core, dim6, types5, types6 | 2.5 / 0.36 |
| `nalgebra_linalg6_bidiagonal` | 11,474 | bidiagonal of dimension 6 | core, linalg, types5, types6 | 0.9 / 0.21 |
| `nalgebra_linalg6_spectral` | 14,460 | Schur, eigen, Hessenberg, tridiagonal of dimension 6 | core, dim3, dim4, dim5, dim6, linalg, linalg_spectral, types5, types6 | 1.9 / 0.36 |
| `nalgebra_dynamic` | 18,360 | `DMatrix`, `DVector`, `RowDVector`, the dynamic forms of the static shapes | core, every `dim*`, edition, types5, types6 | 4.6 / 0.51 |
| `nalgebra_sparse` | 3,654 | `sparse` (legacy `CsMatrix`, `CsCholesky`) and `io` (Matrix Market) | core, dynamic, dim6, types5, types6 | 0.5 / 0.10 |
| **`nalgebra`** (facade) | 11,699 | root functions (`nalgebra::distance`…), the macros (`matrix!`…), the 0.1.0 module tree re-exporting every sub-crate | every sub-crate | gate 3 only: 58.8 s / 10.0 GB |

Negative marginals are measurement noise (§6.3); the CPU-time column of §6 is steadier.

Dependency graph (transitive edges omitted):

```mermaid
graph BT
  dim3v --> core
  dim3 --> dim3v
  types5 --> dim3
  dim4 --> types5
  geometry --> dim4
  types6 --> geometry
  dim5 --> types6
  dim5 --> dim4
  dim5a --> dim5
  dim6a --> dim5
  dim6 --> dim6a
  dim6b --> dim6
  edition --> dim6a
  views --> edition
  kronecker --> types6
  geometry_nd --> types6
  statistics --> types6
  blas --> types6
  norm --> dim6b
  norm --> dim5a
  linalg --> dim4
  linalg_svd --> dim3
  linalg_pivot --> core
  linalg_spectral --> linalg
  linalg5 --> linalg_svd
  linalg5 --> types5
  linalg5_pivot --> types5
  linalg5_spectral --> linalg_spectral
  linalg5_spectral --> dim5
  linalg6 --> linalg5
  linalg6 --> dim6
  linalg6_pivot --> dim6
  linalg6_bidiagonal --> linalg
  linalg6_bidiagonal --> types6
  linalg6_spectral --> linalg_spectral
  linalg6_spectral --> dim6
  dynamic --> dim6b
  dynamic --> dim5a
  dynamic --> edition
  sparse --> dynamic
  nalgebra --> views
  nalgebra --> norm
  nalgebra --> sparse
  nalgebra --> kronecker
  nalgebra --> geometry_nd
  nalgebra --> statistics
  nalgebra --> blas
  nalgebra --> linalg6_spectral
  nalgebra --> linalg6_bidiagonal
  nalgebra --> linalg6_pivot
  nalgebra --> linalg6
  nalgebra --> linalg5_pivot
  nalgebra --> linalg_pivot
  nalgebra_glam --> dim3
```

Scopes: the crates cannot follow upstream's module tree one to one (upstream's `base` is 272k
lines, one module per shape would give a knot, §2.3): they follow it "as far as the dimension
split allows". The type / method separation is the price of zero-break (§2.3), and every file of
the upstream tree keeps its module path inside its crate.

### 3.2 How each impl is placed (coherence)

Machine-checked by `plan.py` (0 anchor violations in the final layout) and by the prototype build:

| impl | placement |
|---|---|
| core trait of ONE nalgebra type (`Add<Matrix3>`, `Serde`, `PartialOrd`, `Index`, array `Into`) | the struct's module, hence the struct's crate. |
| `MatrixMul<A, B>`, `MatrixTrMul<A, B>`, `MatrixIndex` | the module of the larger operand (1,374 impls change module in all, most to their trait's module: next rows); `Matrix5x3 * Matrix3x2` stays in `matrix5x3` (`nalgebra_types5`); `Matrix2x5 * Matrix5x6` moves to `matrix5x6` (`nalgebra_types6`). |
| `FixedView`, `FixedRows`, `FixedColumns`, `FixedResize`, `PadTo6`, `CropFrom6`, `MatrixKronecker` | the trait's module in the family crate (801 + 196 impls move out of the shape files). |
| `Norm<Marker, M, T>` | the marker's module (`nalgebra_norm`, 144 impls). |
| `MatrixInfSup` (root) | the trait's module in the facade (36 impls). |
| `SolveKernel` (crate-private, but a blanket impl resolves it at the caller), `PermuteRows`, `GivensRotate` | the shape modules of each type band; `Perm6`'s in `perm` of `nalgebra_dim6`. |
| impls of crate-private traits with no blanket caller (`LuInvert`, `BlasTranspose`, `LuSteps`) | the crate of their callers, imported there (not a public path). |
| cross-crate products (`Matrix5x3 * Matrix3x2`, `Matrix2x5 * Matrix5x6`) | as above: the operand of the higher band holds them, so every product is found with 0.1.0's imports. |

### 3.3 The only code changes (everything else moves verbatim)

1. Impls change MODULE (never body) as listed in §3.2. An impl is rarely named by users; the
   public ones keep their 0.1.0 path through an explicit re-export in the facade (the prototype
   does it for `Point2Index` & co.).
2. `SvdRightTrait` (crate-private, `right2`..`right6` in one generated trait) is split into one
   trait per band (`SvdRightImpl`, `SvdRightImpl5`, `SvdRightImpl6`): same bodies, callers renamed.
3. Three `Normed` impls (`Matrix1`, `Vector5`, `Vector6`) call the inherent `norm`; they get a
   type-level kernel, called by both, through an `#[inline(always)]` wrapper so no step changes.
   The prototype stubs their bodies (no cost difference).
4. `ApproxEqTrait` (crate-private) moves from `geometry::quaternion` to `nalgebra_core`.
5. Every `pub(crate)` item used across crates becomes `pub` (no new public path in the facade:
   the facade only re-exports what 0.1.0 exports; the sub-crates do expose these items, a
   deviation to document in their READMEs as "internal, no stability promise").

### 3.4 The facade keeps every 0.1.0 path (proof)

`tools/split/facade.py` writes the facade of the prototype: for every public module of 0.1.0,
`pub use nalgebra_<crate>::<same path>::*` from each sub-crate that hosts that module, the
module's own `pub use` lists verbatim (root re-exports, `base::{Matrix3, …}`), explicit
re-exports for the impls that changed module, and the facade's own `root` and `macros`. It
then writes `path_proof`, a consumer with one `use nalgebra::<path>;` for each of the **4,284
public paths** of 0.1.0 (every public item of every public module, every name of every `pub use`)
and a `usage` module exercising, with the imports a 0.1.0 user writes:
`m.transpose()`, `m * t + m`, `p.mul_mat(v)`, `m.tr_mul(m)`, `s.apply_norm(EuclideanNorm {})`,
`m.solve_lower_triangular(w)`, `m.lu()`, `m.determinant()`, `v.norm()`,
`Matrix5x3.mul_mat(Matrix3x5)` (a cross-crate product), `FixedView::fixed_view(Matrix6)`,
`Matrix6::trace`, `UnitQuaternion::transform_vector`, `Isometry3::transform_vector`, and the
macros `matrix!`, `vector!`, `point!`, `dmatrix!`. `scarb build -p path_proof` succeeds; a
deliberately wrong path or method in the same file fails (checked). Two facade details found
by the proof: the inline `errors` modules need one explicit re-export each (else ambiguous
globs), and an impl re-exported by 0.1.0 at `nalgebra::geometry::Point2Index` needs an explicit
re-export at its old module after it moves to `Point2`'s module.

## 4. Features

Scarb has no optional dependency (DESIGN D11), so the facade depends on every sub-crate
unconditionally, and a feature cannot remove a sub-crate from a facade user's build.

| 0.1.0 feature | after the split |
|---|---|
| `statistics`, `blas`, `dynamic`, `sparse`, `io`, `eigen`, `svd`, `qr`, `cholesky_update`, `full_piv_lu`, `col_piv_qr`, `lblt`, `hessenberg`, `bidiagonal`, `schur`, `exp` | become crates (§3.1). In the facade they stay DECLARED, all in `default`, as **no-ops** (a manifest that names them keeps building: no break). |
| `closures` | gates methods inside the inherent traits (`map`, `fold`, `zip_*`…). Recommended: drop the gating (compiled always: the measured marginals include them and pass) and keep the name as a no-op. Keeping it gated inside the `dim*` crates (forwarded by the facade, `closures = ["nalgebra_dim3/closures", …]`, measured to work) buys little once the crates are split. |
| `macros` | stays a real feature of the facade (the macros live there). |
| `default-features = false` on the facade | still valid, **saves nothing** apart from the macros: 58.8 s / 10.0 GB cold, against 28.7 s / 5.2 GB for `nalgebra` 0.1.0 with `default-features = false` today (NS0). This is a **cost regression for that configuration** (no API change). Those users should depend on the sub-crates they use, e.g. `nalgebra_core` + `nalgebra_dim3` + `nalgebra_linalg`. |

Plainly: **a facade user pays the whole library** (58.8 s / 10.0 GB cold, down from 96.7 s /
10.4 GB with 0.1.0's default features); the sub-crates are the way to pay less. Sub-crates depend
on each other with default features (none have features once `closures` is dropped); if some keep
features, they depend on each other with `default-features = false` (features unify per unit,
§1).

## 5. Tests and gas

- **Test packages** (`crates/tests_*`, `crates/shapes_tests_*`: 60 packages, already outside
  `src/`) keep their names, so their gas keys (`nalgebra_tests_linalg::…`) do not change. Each
  one depends on the sub-crates it tests instead of `nalgebra` with features; its
  `use nalgebra::X` lines are rewritten to `use nalgebra_<crate>::<module>::X` (the rewrite of
  `tools/split/glam_proto.py`, generalised). Depending on the facade instead would keep the test
  code byte-identical, but every test package would compile the whole library (10 GB).
- **In-crate tests and benches** (174 benches and 249 tests in 27 library files, testing
  crate-private items) move with their module; their keys change package prefix only
  (`nalgebra::base::matrix2::tests` → `nalgebra_core::base::matrix2::tests`). The alexandria
  model puts tests outside `src/`; these few test crate-private items and stay inline, as AGENTS
  allows today (a deviation to confirm).
- **Zero step change, proved per move**: `tools/split/gas_compare.py OLD_GAS_DIR NEW_GAS_DIR`
  drops the `nalgebra` / `nalgebra_<crate>` package segment of every module path, compares every
  benchmark, and fails if one changes value, disappears or appears (checked: identical
  directories and a renamed package prefix both give "3,758 benchmarks, 0 changed"). The gas
  snapshot FILES are per CI shard (`gas/<shard>.json`); `gas_report.py` names shards after the
  package, so the move PR regenerates the files of the moved modules and runs `gas_compare.py`
  against `main`.
- Why no step change is expected: the whole dependency graph is compiled as one unit and
  inlining crosses crates (R7 §4); the moved code is byte-identical except §3.3, whose three items
  keep the same function bodies.

## 6. Measurements

### 6.1 Protocol

`tools/split/measure.py` builds a trivial consumer cold (`SCARB_INCREMENTAL=false`, fresh
`target/`, `scarb fetch` first), 2 or 3 times, median, under the shared build lock
(`flock ~/orchestrator/heavy-build.lock`, so lock waits are excluded): wall time, CPU time
(user + sys) and peak RSS. Marginal = consumer(crate) − consumer(its direct dependencies);
closure = consumer(crates) − consumer(no dependency). Baseline: 1.4–1.9 s wall, 0.54 GB. Machine:
shared 8-vCPU VM, `RAYON_NUM_THREADS=4`, other projects building at times.

### 6.2 Results (final layout)

Marginal costs: table of §3.1 (wall s / GB). CPU seconds of the same measurements, the steadier
figure: core 4.3, dim3v 3.1, dim3 6.7, types5 2.7, dim4 2.9, geometry 4.4, types6 4.8, dim5 3.3,
dim5a 1.3, dim6a 5.8, dim6 1.6, dim6b 7.3, edition 2.9, views 6.6, kronecker 6.2, norm 2.1,
geometry_nd 2.2, statistics 4.4, blas 5.6, linalg 2.4, linalg_svd 3.8, linalg_pivot 1.1,
linalg_spectral 2.2, linalg5 2.8, linalg5_pivot −0.3, linalg5_spectral 3.5, linalg6 7.2,
linalg6_pivot 5.2, linalg6_bidiagonal 3.0, linalg6_spectral 9.3, dynamic 10.7, sparse −0.2.

Declared closures (gate 3, over the no-dependency baseline):

| closure | crates (with their dependencies) | wall s | CPU s | GB | verdict |
|---|---|---:|---:|---:|---|
| `nalgebra_glam` | `nalgebra_glam` prototype → core, dim3v, dim3, glam 0.4.0, fixed | 10.0–10.5 | 25.2 | 1.85 | pass |
| static 1–3 + SVD / eigen | core, dim3v, dim3, linalg_svd | 8.6–9.6 | 22.3 | 1.59 | pass |
| `core` + pivoting decompositions ≤ 4 | core, linalg_pivot | 4.8 | 11.3 | 0.94 | pass |
| static 1–4 + geometry (3D game physics and rendering) | core, dim3v, dim3, types5, dim4, geometry | 12.4–14.6 | 32.6–35.4 | 2.39 | pass, **close** |
| static 1–4 + LU / Cholesky / QR | core, dim3v, dim3, types5, dim4, linalg | 12.3–15.0 | 31.8–36.4 | 2.31 | pass, **close** |
| the facade `nalgebra` | everything | 58.8 | 150.9 | 10.04 | **fails by construction** |
| (for reference) `glam` 0.4.0 alone | | 3.0 | 7.4 | 0.70 | |

`types5` is in every closure with `dim4`: `Matrix4Trait::insert_column` builds a `Matrix4x5`
(zero-break costs about 1–2.5 s there).

### 6.3 Close decisions (to confirm on a GitHub runner)

Noise on this VM: the same closure measured twice in one session gave 12.9 and 14.3 s; the
marginal of the former 27k-line `dim3` read 4.3, 4.6 and 5.6 s in three sessions (it was split
for that reason). Close calls, to be measured on a GitHub runner with interleaved cold builds
(glam used 15 per variant) before the moves start:
- the two "static 1–4" closures (12.3–15.0 s against 15 s);
- `nalgebra_dynamic` (4.6 s marginal), `nalgebra_dim6b` / `nalgebra_dim6a` / `nalgebra_types6`
  (2.5–3.6 s, but 5.8–7.3 s CPU);
- `nalgebra_types6` lines: 35,462 against 40,000 (the tightest line margin; it cannot be split
  without rewriting the transposed products, §2.3).
A runner measurement needs a workflow change (orchestrator files): escalated (REPORT).

### 6.4 Rejected cuts (measured)

| layout | why rejected |
|---|---|
| one crate per dimension band, inherent traits with their structs | 120k-line strongly connected component (§2.3). |
| all 36 structs + all products at the bottom (`v3`) | static 1–4 + geometry closure 16.5 s / 2.86 GB (fail): the products of dimension 6 were in every closure. |
| one `base5` (26.5k lines of methods) | marginal 6.9 s (fail). |
| one `dim6` (Matrix6 + r×6 methods + LU6, 32k) | marginal 4.4–6.1 s. |
| one `linalg5_ext` / `linalg6_spectral` of 24–27k lines | marginals 6.1 s and 5.8 s. |
| one `linalg` ≤ 4 (19.5k) | static 1–4 + linalg closure 14.8–16.4 s. |
| `solve` / `perm` as one crate over all dimensions | every small decomposition's closure pulled `types6` (19.6 s). |

## 7. Generators and tooling

Built in WP 9-NS2 (usage: `docs/ORCHESTRATOR.md`, "Repository tooling" and the move-PR checklist).

- **Crate map** `tools/split/crates.toml`, read by `tools/split/cratemap.py`: `[crates]` lists the
  planned crates (the names of §3.1; NS1b renames them there only), lowest first, each with the
  Cairo package that hosts it TODAY; `facade` names the crate of what no rule places (root,
  macros); `[[rule]]`s place every top-level item, first match wins, by module file (globs), item
  class (`struct`, `inherent`, `impl`...), name, implemented trait, shape, band (of the file, of
  the impl's type arguments or of the item name), with a `lift` of every impl above its type
  arguments; the module of an impl follows §3.2 (the first anchor in its package). Committed in
  single-crate mode (every crate hosted by `nalgebra`). With every crate hosted by its own
  package (`cratemap.py --split-map`) it reproduces NS1's `final` plan item by item (5,706 /
  5,706 items, crate and module: `cratemap.py --compare-plan`).
- **Generators** `tools/shapegen`, `tools/linalggen`: their library outputs go through
  `cratemap.route_outputs`: each item into `crates/<package dir>/src/<same module file>`, a file
  whose items all go to one package moved whole, the facade keeping the module doc and
  `pub use nalgebra_<crate>::<path>::*` plus one explicit `pub use` per public impl moved to an
  anchor module (its 0.1.0 path), the sub-crates their `pub mod` roots. `use` / `pub mod` lines
  are added when missing (`scarb fmt` sorts them), moved impls sit in marked blocks per generator,
  so the two generators are idempotent after each other. Single-crate mode: byte-identical output.
  Split mode on a throwaway checkout: the same 356 files and 4,571 generated items as
  `prototype.py` (`cratemap.py --compare-tree`).
- `scripts/api_parity.py`: scans every package of the crate map plus `nalgebra_glam`; an item
  is reported at its facade path (a moved impl at the module of its explicit facade re-export), so
  the report is byte-identical in both modes.
- **Path proof** `tools/split/public_paths.py` (CI job `Path proof`): the public surface is every
  public module (`internal` excluded), every public item (impls included) and every `pub use`
  name, globs followed across the packages of the map. 0.1.0 has **9,289** public paths (the
  4,284 of §3.4, 4,622 public impls, the modules), frozen in `public_paths_0.1.0.txt`;
  `--check` proves the facade exports exactly that set plus `public_paths_added.txt` and builds a
  consumer naming every path with the `usage` checks of §3.4 (4 min, 11.3 GB on this machine).
- `tools/split/gas_compare.py --base <git ref or dir> --head gas/`: the zero-step proof of §5.
- `tools/split/rewrite_imports.py`: the test-package rewrite of §5 (dry run by default).
- `consumer_cost.toml`: gates on every `nalgebra_*` sub-crate (lines + MARGINAL cost, the
  column the orchestrator adds); `[closures]`: `nalgebra_glam`, `static3_svd`, `core_pivot`,
  `static4_geometry`, `static4_factor` (§6.2); the facade reported, not gated.
- CI: one test shard per test package as today; the `Path proof` job; the gas job unchanged plus
  `gas_compare.py` in move PRs (PR template checkbox).
- `tools/split/`: the helpers of this study (§11), reusable by rapier / glam.

## 8. `nalgebra_glam`

Depends on `nalgebra_core`, `nalgebra_dim3v`, `nalgebra_dim3` (measured: the prototype rewritten
from `use nalgebra::X` to the defining sub-crates compiles against exactly these), plus `glam`
and `fixed`. Its closure: **10.0–10.5 s, 1.85 GB** (target 15 s / 3 GB; today 37 s / 5.5 GB). Its
public API does not change: its conversion impls take the same types, which the facade
re-exports.

## 9. Order of the moves and of the releases

The moves go bottom-up, so every PR creates crates below what remains in `nalgebra`, and
`nalgebra` (the future facade) re-exports them at once: every PR keeps every public path and
proves zero step change (`gas_compare.py`) and unchanged paths (`path_proof`).

| WP | content | size |
|---|---|---|
| NS2 | tooling only: shapegen / linalggen emit into per-class output crates (byte-identical output while everything maps to `nalgebra`), `gas_compare.py`, import rewriter for test packages, `path_proof` generator, `api_parity.py` over several crates | tools |
| NS3 | `nalgebra_core` (types ≤ 4, trait declarations, `ApproxEqTrait`, points / translations 1–4, solve / perm / Givens declarations and ≤ 4 impls) | ~20k |
| NS4 | `nalgebra_dim3v`, `nalgebra_dim3` (with the 2D/3D geometry), then `nalgebra_glam` on them | ~29k |
| NS5 | `nalgebra_types5`, `nalgebra_dim4`, `nalgebra_geometry` | ~50k (generator-driven) |
| NS6 | `nalgebra_types6`, `nalgebra_geometry_nd`, `nalgebra_kronecker` | ~47k (generator-driven) |
| NS7 | `nalgebra_dim5`, `_dim5a`, `_dim6a`, `_dim6` (with `Lu6`), `_dim6b` | ~85k (generator-driven) |
| NS8 | `nalgebra_edition`, `_views`, `_norm`, `_statistics`, `_blas` | ~74k |
| NS9 | `nalgebra_linalg`, `_linalg_svd`, `_linalg_pivot`, `_linalg_spectral` (with the `SvdRightTrait` split) | ~48k |
| NS10 | `nalgebra_linalg5*`, `nalgebra_linalg6*` | ~110k (linalggen-driven) |
| NS11 | `nalgebra_dynamic`, `nalgebra_sparse`; `nalgebra` becomes the pure facade (root, macros, re-exports, features as no-ops); one README per package; release script; CI gates | ~27k |

Release (no publication without the programme session's written go): one shared version,
**0.1.1** (a non-breaking patch: paths, API and numeric results unchanged; the only visible
change is §4's cost of `default-features = false` on the facade), published in this dependency
order: `nalgebra_core`, `_dim3v`, `_dim3`, `_types5`, `_dim4`, `_geometry`, `_types6`, `_dim5`,
`_dim5a`, `_dim6a`, `_dim6`, `_dim6b`, `_edition`, `_views`, `_kronecker`, `_norm`,
`_geometry_nd`, `_statistics`, `_blas`, `_linalg`, `_linalg_svd`, `_linalg_pivot`,
`_linalg_spectral`, `_linalg5`, `_linalg5_pivot`, `_linalg5_spectral`, `_linalg6`,
`_linalg6_pivot`, `_linalg6_bidiagonal`, `_linalg6_spectral`, `_dynamic`, `_sparse`,
`nalgebra`, `nalgebra_glam` (34 packages per release).

## 10. Risks

- **Crate count**: 32 sub-crates + facade + `nalgebra_glam` per release. The release script
  must publish in order and verify each against the registry (rapier's does).
- **Technical scopes**: `types5` / `dim5a` are not upstream module names. Upstream's paths
  survive inside each crate and through the facade, but a sub-crate user sees the dimension
  split. Alternative names are cosmetic.
- **Noise**: §6.3; the plan should be re-measured on a GitHub runner before NS3.
- **Over-approximation**: the item graph over-approximates method calls (every trait in scope
  with a method of that name). The prototype build is the proof, and it compiles, but the
  generator changes of each move must reproduce the prototype's placement.
- **Facade cost for `default-features = false` users** (§4), to be announced in the CHANGELOG.
- **Sub-crates expose former `pub(crate)` items** (§3.3.5).
- **Growth**: `nalgebra_types6` has 11 % of line margin; a new family of dimension-6 impls goes
  to a family crate (in its trait's module), not to `types6`.

## 11. Helpers (`tools/split/`)

| file | role |
|---|---|
| `crates.toml`, `cratemap.py` | the crate map and its engine (placement, routing of the generators; `--show`, `--place`, `--split-map`, `--compare-plan`, `--compare-tree`) (NS2) |
| `public_paths.py`, `public_paths_0.1.0.txt` | the public surface and the path proof (NS2) |
| `rewrite_imports.py` | `use nalgebra::X` of test packages -> the sub-crates (NS2) |
| `gas_compare.py` | zero-step proof of a move (`--base REF --head gas/`) |
| `files.py` | library lines per file (`consumer_cost.py`'s rules) |
| `edges.py` | the item graph (JSON) |
| `plan.py`, `layouts.py` | placement solver; `layouts.py` keeps every candidate (`naive`, `z`, `v1`..`v6`, `final`) |
| `prototype.py` | emits the throwaway workspace of a plan |
| `facade.py` | facade prototype + `path_proof` (the 4,284 item paths) |
| `glam_proto.py` | `nalgebra_glam` rewritten on the sub-crates |
| `measure.py` | marginal and closure costs (cold, under the build lock) |
| `probes/` | the coherence probes of §1 |

Reproduce NS1: `python3 tools/split/edges.py crates/nalgebra --json /tmp/e.json`;
`python3 tools/split/plan.py /tmp/e.json final --json /tmp/p.json`;
`python3 tools/split/prototype.py /tmp/e.json /tmp/p.json /tmp/proto`;
`python3 tools/split/facade.py /tmp/e.json /tmp/proto`;
`python3 tools/split/glam_proto.py /tmp/e.json /tmp/p.json /tmp/proto`; then in `/tmp/proto`
`scarb build -p path_proof`, and `flock ~/orchestrator/heavy-build.lock python3
tools/split/measure.py /tmp/proto --cache /tmp/m.json --crate core …`.

Reproduce NS2's check of the crate map: `python3 tools/split/cratemap.py --split-map /tmp/split.toml`;
`python3 tools/split/cratemap.py --map /tmp/split.toml --compare-plan /tmp/e.json /tmp/p.json`;
on a throwaway copy (`git archive HEAD | tar -x -C /tmp/co`), `NALGEBRA_CRATE_MAP=/tmp/split.toml`
`python3 tools/shapegen/shapegen.py` and `python3 tools/linalggen/generate.py` there, then
`python3 tools/split/cratemap.py --map /tmp/split.toml --compare-tree /tmp/co /tmp/proto`.

## 12. Decisions of the programme session (2026-09-28, plan approved)

1. **Cost regression of `default-features = false` on the facade** (28.7 s / 5.2 GB → 58.8 s /
   10.0 GB): accepted. The facade is for parity with nalgebra-rs paths; a light build depends on
   sub-crates. Conditions: a CHANGELOG note with the figures; the facade's README opens with a table
   "what you need → which crates to depend on → measured cost" for the declared closures; a
   facade feature that no longer does anything is **removed** in the release that ships the split
   (and the removal is announced), not kept as a no-op.
2. **Release load** (34 packages, one version 0.1.1, fixed order, each verified against the
   registry): accepted. The release script is **resumable** (a failure at package k restarts at
   k) and **refuses to start** unless the `Consumer cost` job is green at the release commit. The
   smallest crates merge with their natural neighbour when no declared closure worsens (fewer
   packages at equal cost; the orchestrator's call).
3. **Names and internals**: names are public and permanent; band names are acceptable only if they
   say what is inside (dimension and content, no bare `a` / `b` suffix). The final list (name,
   one-line description, lines, direct dependencies) is approved by the programme session and shown
   to the owner **before NS3**. Former `pub(crate)` items that become `pub` live under a module
   path that says it (`internal::`), are documented "internal, no stability promise", are never
   re-exported by the facade, and the path-proof tool proves that the public surface of 0.1.0 and
   of the facade are equal (nothing more, nothing less). Inline tests of crate-private items stay
   in `src/`.
4. **Gates**: the two close closures (static 1–4 + geometry, static 1–4 + LU / Cholesky / QR) must
   pass on a GitHub runner with interleaved repeats (median < 15 s) **before NS3**; if one fails,
   the closure's heaviest crate is cut further rather than the gate relaxed.
5. `nalgebra_glam` on `glam_core` + `glam_int` (glam 0.4.1): agreed, measured in NS4.

Order: NS2 (tooling, crate names read from a configuration) → NS1b (final names, small-crate
merges, GitHub-runner measurement of the closures) → NS3..NS11.
