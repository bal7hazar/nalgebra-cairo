# nalgebra-cairo

Linear algebra for **provable game physics**: a Cairo port of the Rust
[nalgebra](https://nalgebra.rs) crate on fixed-cairo's Q32.32 fixed point, designed gas-first.

Part of a stack porting reputable Rust crates to Cairo, repository by repository like the Rust
ecosystem: [fixed-cairo](https://github.com/bal7hazar/fixed-cairo) (the Q32.32 scalar, the `f64` of
the stack), [simba-cairo](https://github.com/bal7hazar/simba-cairo) (the scalar traits),
nalgebra-cairo (this repository), [glam-cairo](https://github.com/bal7hazar/glam-cairo),
[glamx-cairo](https://github.com/bal7hazar/glamx-cairo) and
[rapier-cairo](https://github.com/bal7hazar/rapier-cairo), towards games whose whole physics is
provable.

> Status: coverage of nalgebra-rs 0.35.0 at 99.9 % (1,885 of 1,886 items,
> [API_PARITY.md](https://github.com/bal7hazar/nalgebra-cairo/blob/main/docs/API_PARITY.md)): the
> 36 static shapes and 54 aliases, dynamic and sparse matrices, geometry (rotations, quaternions,
> dual quaternions, isometries, similarities, scale, reflection, projections) and the
> decompositions (LU, QR, Cholesky, SVD, eigen, Schur, ...). The one item left is upstream's
> `io::cs_matrix_from_matrix_market(path)` (a Cairo program has no file system; the `_str` form is
> there). Release 0.1.1 (the package split, no API change) is being prepared ([CHANGELOG](https://github.com/bal7hazar/nalgebra-cairo/blob/main/CHANGELOG.md)).

## Installation

```toml
[dependencies]
nalgebra = "0.1.0"
fixed = "0.4.0"          # the scalar: `Vector3<Fixed>`, `Matrix4<Fixed>`...
simba = "0.2.0"          # `use simba::prelude::*;` for the scalar traits (`Real`, `Transcendental`)
```

`nalgebra = "0.1.0"` has every feature on (`default`), the whole nalgebra-rs surface. A dependent
that needs only a part of it compiles much less by depending on the packages it uses instead of the
facade (0.1.1, see [Packages](#packages)); the features are the [feature table](#features):

```toml
nalgebra = { version = "0.1.0", default-features = false, features = ["qr"] }
```

The glam-cairo conversions are a separate package (see [glam conversions](#glam-conversions)):

```toml
nalgebra_glam = "0.1.0"
glam = "0.4.0"
```

Versioning (pre-1.0): **numeric results are part of the API**. Any change of a result, down to the
last bit of a `fixed` value, is a MINOR bump (`0.1.x` → `0.2.0`); a patch release never changes
a result. A breaking bump of `fixed` or `simba` (their types are in our signatures) is a MINOR bump too.
See the [CHANGELOG](https://github.com/bal7hazar/nalgebra-cairo/blob/main/CHANGELOG.md).

## Packages

The library is cut into 54 packages behind the `nalgebra` facade (docs/SPLIT.md §18): a number in a
crate name is exactly that dimension (the dimension of a rectangular shape is max(rows, columns),
dimension 1 is folded into `nalgebra_core` and `nalgebra_static_core`), and dimension k builds on every
dimension below it. `nalgebra` re-exports every package at the 0.1.0 paths (9,289 paths, proved in CI),
so a 0.1.0 consumer changes nothing; a consumer that needs part of the library depends on the packages
it uses and pays only for them. Every package is at most 40,000 library lines and adds at most
5 s / 1 GB to a cold build over its own dependencies (CI job `Consumer cost`).

The costs below are cold builds over an empty consumer on a GitHub runner (medians of interleaved
rounds, docs/SPLIT.md §18.2); a package pulls its dependencies, so the cost of a row is the cost of the whole set, not a sum. The facade and `dynamic` figures are those of the last CI run (5 rounds, one runner per shard, so absolute figures move by several tens of percent from one run to another); the closure figures are the 15-round measurements of docs/SPLIT.md §18.2 (views / blocks: after the move of the fixed-size edition to `nalgebra_blocks`), and [`docs/PACKAGES.md`](https://github.com/bal7hazar/nalgebra-cairo/blob/main/docs/PACKAGES.md) holds the last CI run.

| What you need | Depend on | Measured cost |
|---|---|---:|
| the upstream API at upstream's paths | [`nalgebra`](https://github.com/bal7hazar/nalgebra-cairo/tree/main/crates/nalgebra) | 29.8 s / 9.96 GB (0.1.0: 96.7 s / 10.4 GB; [CHANGELOG](https://github.com/bal7hazar/nalgebra-cairo/blob/main/CHANGELOG.md)) |
| the types only (vectors, matrices, points, operators, products, indexing) | `nalgebra_core`, `nalgebra_types2` .. `nalgebra_types6` | marginal: `types2` 0.3 s, `types3` 0.4 s, `types4` 0.9 s, `types5` 1.0 s, `types6` 2.5 s |
| the named methods of dimensions 2-4 (`norm()`, `normalize()`, `dot`, `cross`, `transpose`, `inverse`, `insert_*`...) | `nalgebra_static2`, `nalgebra_static3`, `nalgebra_static4` (`static4` pulls the two others and the types) | marginal of `static4`: 1.0 s / 0.33 GB |
| 2D / 3D geometry (rotations, quaternions, isometries, similarities) | `nalgebra_static3`, `nalgebra_geometry2`, `nalgebra_geometry3` | 6.3 s / 1.43 GB (`static3_geometry`) |
| geometry 2-4 with the transforms | `nalgebra_static4`, `nalgebra_geometry2` .. `nalgebra_geometry4`, `nalgebra_transform2`, `nalgebra_transform3` | 11.2 s / 2.44 GB (`static4_geometry`) |
| conversions to glam-cairo | `nalgebra_glam` (needs `glam` >= 0.4.1) | 7.4 s / 1.77 GB (`nalgebra_glam`; `glam_core` alone 1.3 s / 0.56 GB) |
| LU, Cholesky, LDL / UDU, QR, inverse of dimensions 2-4 | `nalgebra_static4`, `nalgebra_linalg2`, `nalgebra_linalg3`, `nalgebra_linalg4` | 8.9 s / 2.06 GB (`static4_factor`) |
| SVD and symmetric eigen of dimensions 2-3 | `nalgebra_static3`, `nalgebra_linalg_svd_eigen2`, `nalgebra_linalg_svd_eigen3` | 5.8 s / 1.23 GB (`static3_svd`) |
| SVD and symmetric eigen of dimensions 2-4 | `nalgebra_static4`, `nalgebra_linalg_svd_eigen2` .. `nalgebra_linalg_svd_eigen4` | 6.9 s / 2.11 GB (`static4_svd`) |
| column-pivoting QR, full-pivoting LU, LBLT of dimensions 2-4 | `nalgebra_linalg_pivot2` .. `nalgebra_linalg_pivot4` (types only) | 3.5 s / 1.04 GB (`core_pivot`) |
| bidiagonal, Schur, eigen, Hessenberg, `exp` / `pow` of dimensions 2-4 | `nalgebra_linalg_spectral2` .. `nalgebra_linalg_spectral4` | with the methods of dimension 4: see the packages' marginals in `docs/PACKAGES.md` |
| statistics (`mean`, `variance`...) of dimensions 2-4 | `nalgebra_static4`, `nalgebra_statistics2` .. `nalgebra_statistics4` | 8.6 s / 1.91 GB (`static4_statistics`); the family whole, 2-6: 11.9 s / 2.79 GB (`static4_statistics_all`) |
| BLAS (`gemm`, `gemv`, `axpy`, `ger`...) | `nalgebra_static4`, `nalgebra_blas` | 12.4 s / 2.88 GB (`static4_blas`) |
| dynamic matrices (`DMatrix`, `DVector`) and sparse matrices | `nalgebra_dynamic`, `nalgebra_sparse` | marginal: `dynamic` 2.6 s / 0.32 GB, `sparse` 0.6 s / 0.10 GB (they pull every method crate) |
| dimension 5 or 6 | see [Dimensions 5 and 6](#dimensions-5-and-6) | 9.4 - 21.5 s, 3.05 - 4.71 GB |
| blocks, views, norms as generic traits | see [Blocks, views and norms](#blocks-views-and-norms) | 17.4 - 18.7 s, 3.87 - 4.18 GB with static 2-4 |

Every package has a README (what it holds, its dependencies, when to depend on it). By family:
- shared: [`nalgebra_core`](https://github.com/bal7hazar/nalgebra-cairo/tree/main/crates/core) (trait declarations, errors, dimension-1 types), `nalgebra_static_core` (dimension-1 methods), `nalgebra_linalg_core` (shared linalg kernels);
- types: `nalgebra_types2` .. `nalgebra_types6`; methods: `nalgebra_static2` .. `nalgebra_static5`, `nalgebra_static6_tall`, `nalgebra_static6_wide` (the two halves of dimension 6, cut to stay under the line gate);
- geometry: `nalgebra_geometry2` .. `nalgebra_geometry6`, `nalgebra_transform2`, `nalgebra_transform3`;
- decompositions: `nalgebra_linalg2` .. `6` (LU, Cholesky, QR), `nalgebra_linalg_svd_eigen2` .. `6`, `nalgebra_linalg_pivot2` .. `6`, `nalgebra_linalg_spectral2` .. `6`;
- families: `nalgebra_statistics2` .. `nalgebra_statistics6`, `nalgebra_blas`, `nalgebra_blocks`, `nalgebra_views`, `nalgebra_norm`, `nalgebra_dynamic`, `nalgebra_sparse`;
- the facade `nalgebra` and the glam conversions `nalgebra_glam`.

Its scalar layer is the registry package `simba = "0.2.0"`
([simba-cairo](https://github.com/bal7hazar/simba-cairo): `Real` / `Transcendental` implemented for
fixed-cairo's Q32.32 [`fixed::Fixed`](https://github.com/bal7hazar/fixed-cairo) 0.4.0, the scalar
shared by the whole stack), like nalgebra-rs depends on simba-rs.

A dependent that wants the light build names the packages instead of the facade:

```toml
[dependencies]
nalgebra_static4 = "0.1.1"
nalgebra_linalg_svd_eigen4 = "0.1.1"
nalgebra_linalg_svd_eigen3 = "0.1.1"
nalgebra_linalg_svd_eigen2 = "0.1.1"
```

### Dimensions 5 and 6

Dimension k builds on every dimension below it: a shape of dimension 5 names the impls of dimensions
2-4, and the types, the methods and the decompositions of dimension 6 pull everything under them. So
a dimension-5 or 6 closure costs more than the whole of dimensions 2-4, and the library budgets it
apart (docs/SPLIT.md §19): **15 s / 3 GB** for a declared closure of dimensions up to 4,
**20 s / 4.5 GB** for one that includes dimension 5 or 6 (gates 1 and 2, 40,000 lines and 5 s / 1 GB,
apply to every package). Cold builds over an empty consumer on a GitHub runner, median of 15
interleaved rounds:

| Closure | Packages (besides their types) | Time / memory | Budget |
|---|---|---:|---|
| `static5` | `static5` | 12.9 s / 3.05 GB | 20 s / 4.5 GB |
| `static5_factor` | `static5`, `linalg5` | 10.9 s / 3.13 GB | 20 s / 4.5 GB |
| `static5_svd` | `static5`, `linalg_svd_eigen5` | 10.4 s / 3.53 GB | 20 s / 4.5 GB |
| `static5_pivot` | `static5`, `linalg_pivot5` | 9.4 s / 3.26 GB | 20 s / 4.5 GB |
| `static5_spectral` | `static5`, `linalg_spectral5` | 14.8 s / 3.35 GB | 20 s / 4.5 GB |
| `static5_geometry` | `static5`, `geometry5` | 13.3 s / 3.14 GB | 20 s / 4.5 GB |
| `static6` | `static6_wide` (pulls `static6_tall`) | 17.3 s / 3.93 GB | 20 s / 4.5 GB |
| `static6_factor` | `static6_wide`, `linalg6` | 17.7 s / 4.11 GB | 20 s / 4.5 GB |
| `static6_pivot` | `static6_wide`, `linalg_pivot6` | 18.5 s / 4.30 GB | 20 s / 4.5 GB |
| `static6_geometry` | `static6_wide`, `geometry6` | 18.0 s / 4.01 GB | 20 s / 4.5 GB |
| `svd_eigen6` | `linalg_svd_eigen6` with its types, without the dimension-6 method crates | 12.1 s / 3.00 GB | 20 s / 4.5 GB |
| `pivot6` | `linalg_pivot6` (pulls `static6_wide` for `Perm6`) | 12.8 s / 4.30 GB | 20 s / 4.5 GB |
| `static6_svd` | `static6_wide`, `linalg_svd_eigen6` | 21.5 s / 4.70 GB | documented, not gated |
| `static6_spectral` | `static6_wide`, `linalg_spectral6` | 21.5 s / 4.64 GB | documented, not gated |

The two combined closures, **static 6 + SVD / eigen 6 (21.5 s / 4.70 GB) and static 6 + spectral 6
(21.5 s / 4.64 GB)**, are a little over the 20 s / 4.5 GB budget in the 15-round measurement (11.6 s / 4.38 GB and
15.2 s / 4.13 GB in the last CI run, whose runners were faster); they are documented
here and not gated (owner decision, docs/SPLIT.md §18.7). The decomposition itself does not need the
dimension-6 methods (`Matrix6::svd()` and the products live in `nalgebra_types6` and
`nalgebra_linalg_*6`): SVD / eigen 6 alone costs 12.1 s / 3.00 GB, and the overrun is the methods of
every dimension-6 shape (`static6_tall` and `static6_wide`, 55,932 lines) on top.

For comparison, 0.1.0 was one package: a cold build of `nalgebra` with the default features cost
**97 s / 10.4 GB** whatever you used of it.

### Blocks, views and norms

`nalgebra_blocks`, `nalgebra_views` and `nalgebra_norm` are advanced use. They are the generic forms
(`FixedView::fixed_view(m, i, j)` over any shape, the `Norm` markers with their generic impls, the
block traits, the fixed-size edition `insert_fixed_rows`... and the Kronecker products); a generic trait's dimension 5-6 impls can only sit in its own
crate, so they cannot be cut per dimension and each one pulls dimensions 5-6 (`blocks` pulls
`static6_tall`, `norm` every method crate). "Static 2-4 + family" on the runner, documented and not
gated (docs/SPLIT.md §18.4, §18.7):

| Closure | Packages | Time / memory |
|---|---|---:|
| `static4_blocks` | `static4`, `blocks` | 17.4 s / 3.87 GB |
| `static4_views` | `static4`, `views` | 17.9 s / 4.18 GB |
| `static4_norm` | `static4`, `norm` | 18.7 s / 4.04 GB |

**The everyday methods do not need them**: `norm()`, `normalize()`, `apply_norm`, `lp_norm`, `dot`,
`cross`, `transpose`, the products, `fixed_rows` / `fixed_columns` / `fixed_view` as methods,
`insert_*` / `remove_*` and `kronecker` are methods of the inherent traits (`Vector3Trait`...) of the
`nalgebra_static*` packages. `nalgebra_blas` (12.4 s / 2.88 GB with static 2-4) and the per-dimension
`nalgebra_statistics*` do fit the budget of dimensions up to 4.

### glam conversions

nalgebra-rs converts to and from glam behind its `convert-glam0XX` features. Scarb has no optional
dependencies and `glam` costs every dependent +0.46 GB of cold compile, so the Cairo counterpart is
a separate package: add `nalgebra_glam` next to `nalgebra` and `glam` and `use
nalgebra_glam::prelude::*;` (upstream's `From` is `Into`, `TryFrom` is `TryInto` returning an
`Option`; details in the [package README](https://github.com/bal7hazar/nalgebra-cairo/tree/main/crates/nalgebra_glam)).
`nalgebra_glam` 0.1.1 depends on the sub-crates it converts and on `glam_core`, so it needs
**`glam` >= 0.4.1** (which re-exports `glam_core`).

## Features

`nalgebra = "x.y"` exposes the full nalgebra-rs surface: every Scarb feature is in `default`. A
structural deviation from nalgebra-rs ([DESIGN D9](https://github.com/bal7hazar/nalgebra-cairo/blob/main/docs/DESIGN.md)):
since the split the facade depends on every package (Scarb has no optional dependency), so
`default-features = false` no longer removes a family from the build, except where a feature still
gates code **inside** a package: `closures` and the linalg families.

```toml
nalgebra = { version = "x.y", default-features = false, features = ["qr"] }
```

**Features that still save compile work** (they gate code in the packages and the facade's
re-exports; the measured cost of the facade with `default-features = false` is in the
[CHANGELOG](https://github.com/bal7hazar/nalgebra-cairo/blob/main/CHANGELOG.md)):

| Feature | Gates |
|---|---|
| `closures` | the methods that take a closure: `map`, `fold`, `apply`, `zip_map`, `fill_with`... (forwarded to every `nalgebra_static*` package and `nalgebra_dynamic`) |
| `eigen` | `SymmetricEigen1..6`, `symmetric_eigen`, `symmetric_eigenvalues`, `wilkinson_shift` (`linalg::symmetric_eigen*`) |
| `svd` | `Svd1..Svd6x5`, `svd`, `singular_values`, `rank`, `pseudo_inverse`, `polar`, `svd_ordered2/3` (`linalg::svd*`; enables `eigen`) |
| `qr` | `Qr1..Qr6x5`, `qr` (`linalg::qr`) |
| `cholesky_update` | `rank_one_update`, `insert_column`, `remove_column` of the Cholesky factors (`linalg::cholesky_update`) |
| `full_piv_lu` | `FullPivLu1..FullPivLu6x5`, `full_piv_lu` (`linalg::full_piv_lu`) |
| `col_piv_qr` | `ColPivQr1..ColPivQr6x5`, `col_piv_qr` (`linalg::col_piv_qr`) |
| `lblt` | `Lblt1..Lblt6`, `lblt` (`linalg::lblt`, Bunch-Kaufman) |
| `macros` | `matrix!`, `vector!`, `point!`, `stack!`, and with `dynamic` `dmatrix!` / `dvector!` (`nalgebra::macros`, upstream's feature of the same name) |
| `hessenberg` | `Hessenberg1..6`, `hessenberg`; `SymmetricTridiagonal1..6`, `symmetric_tridiagonalize`; `balance_parlett_reinsch`, `unbalance`; `clear_column_unchecked`, `clear_row_unchecked`, `assemble_q` (`linalg::hessenberg`, `linalg::symmetric_tridiagonal`, `linalg::balancing`, `linalg::householder_steps`) |
| `bidiagonal` | `Bidiagonal1..Bidiagonal6x5`, `bidiagonalize` (`linalg::bidiagonal`) |
| `schur` | `Schur1..6`, `schur`, `try_schur`, `eigenvalues`, `complex_eigenvalues`; `Eigen1..6` (`linalg::schur`, `linalg::eigen`; enables `hessenberg`) |
| `exp` | `exp` (Padé approximant with scaling and squaring) and `pow` / `pow_mut` on `Matrix1..6` (`linalg::exp`, `linalg::pow`) |

**No-op features** (deprecated in 0.1.1, **removed in 0.2.0**): kept so that a manifest naming them
keeps building; they save nothing since the split, because the code they used to leave out lives in
packages the facade always depends on (they still gate the facade's re-exports, as in 0.1.0). To leave
this code out, depend on the packages that hold it instead of the facade:

| Feature | Gates | Depend on instead |
|---|---|---|
| `statistics` | `mean`, `variance`, `column_mean`... (`base::statistics`) | `nalgebra_statistics2` .. `nalgebra_statistics6` |
| `blas` | `gemm`, `gemv`, `axpy`, `ger`, `quadform`... (`base::blas`) | `nalgebra_blas` |
| `dynamic` | `DMatrix`, `DVector`, `RowDVector`, `Matrix3xX`...; the static `insert_columns`, `remove_fixed_rows`, `from_vec`...; `convolve_full` / `convolve_same` / `convolve_valid` (`base::dynamic`, [DESIGN D5](https://github.com/bal7hazar/nalgebra-cairo/blob/main/docs/DESIGN.md)) | `nalgebra_dynamic` |
| `sparse` | `CsMatrix`, `CsVector`, `CsCholesky`, the sparse triangular solves, `axpy_cs`, `cumsum` (`nalgebra::sparse`; enables `dynamic`) | `nalgebra_sparse` |
| `io` | `cs_matrix_from_matrix_market_str` (`nalgebra::io`, Matrix Market; enables `sparse`) | `nalgebra_sparse` |

### Macros, crate-root functions, `Sum` / `Product`

The construction macros are Cairo declarative macros with upstream's MATLAB-like syntax (`,`
between the columns, `;` between the rows), and they compile to the constructor they stand for
(same gas):

```cairo
use nalgebra::{matrix, point, stack, vector};

let m = matrix![1, 2, 3; 4, 5, 6];       // Matrix2x3 (row-major, like Matrix2x3Trait::new)
let v = vector![x, y, z];                // Vector3
let p = point![x, y];                    // Point2
let b = stack![m, m; m, m];              // Matrix4x6 from static blocks
let d = nalgebra::dmatrix![1, 2; 3, 4];  // DMatrix (feature `dynamic`)
```

nalgebra-cairo enables the experimental Cairo features `user_defined_inline_macros` (the macros)
and `associated_item_constraints` (the `Sum` / `Product` impls) in its own manifest ([DESIGN
D10](https://github.com/bal7hazar/nalgebra-cairo/blob/main/docs/DESIGN.md)). **A dependent needs neither**: calling the macros, `iter.sum()` or
`iter.product()` compiles without any `experimental-features` entry in the dependent's
`Scarb.toml` (measured by the test package `crates/tests_root`, which enables none). Cairo forms:
`stack!` takes static blocks only (write a zero block out, `Matrix2x3Trait::zeros()`, instead of
upstream's `0`); `dmatrix!` takes up to 16 rows and panics on rows of different lengths (a compile
error upstream); there are no 0-sized forms and no `const` use.

The free functions of upstream's crate root are there too (`nalgebra::distance(p, q)`, `center`,
`convert`, `try_convert`, `partial_cmp`, `wrap`, `clamp`...), each as cheap as the method or kernel
it wraps. `Sum` / `Product` are corelib's `core::iter` traits: `array.into_iter().sum()` for owned
matrices, `*span.into_iter().sum()` for snapshots (upstream's `Sum<&Matrix>`).

## Why it is fast

Measured on Cairo 2.19.4 ([full synthesis](https://github.com/bal7hazar/nalgebra-cairo/blob/main/docs/BENCHMARK.md)):

| | existing Cairo libraries | nalgebra-cairo kernels |
|---|---:|---:|
| fixed-point add | 4,050 gas | 640 |
| dot3 | 15,450 - 37,820 | 1,980 |
| 3x3 inverse | 1,076,420 | 88,360 |
| sin_cos (pair) | 30,320 - 130,360 (`sin` alone) | 31,500 (`fixed::trig`) |

No loops, no dynamic containers in static types, products accumulated unscaled and rescaled once,
range checks deferred to a single downcast per kernel. Every function ships with gas benchmarks,
and CI fails on any unreviewed gas change.

## Documentation

- [Benchmark synthesis](https://github.com/bal7hazar/nalgebra-cairo/blob/main/docs/BENCHMARK.md) — what was measured and learned
- [Design decisions](https://github.com/bal7hazar/nalgebra-cairo/blob/main/docs/DESIGN.md)
- [Orchestration strategy](https://github.com/bal7hazar/nalgebra-cairo/blob/main/docs/ORCHESTRATOR.md) — how work is split between sub-agents
- [Execution plan](https://github.com/bal7hazar/nalgebra-cairo/blob/main/docs/PLAN.md)
- [Research reports](https://github.com/bal7hazar/nalgebra-cairo/tree/main/docs/research) — nalgebra, alexandria, starknet-agentic, origami, cubit, orion
- [Design-time benchmarks](https://github.com/bal7hazar/nalgebra-cairo/tree/main/benchmarks) — ~2,100 reproducible gas measurements

## Development

```bash
asdf install            # scarb 2.19.4, starknet-foundry 0.61.0
./scripts/check.sh      # fmt, lint, build, tests, gas snapshot
```

See [AGENTS.md](https://github.com/bal7hazar/nalgebra-cairo/blob/main/AGENTS.md) for the contribution rules (they apply to humans too).

## License

MIT
