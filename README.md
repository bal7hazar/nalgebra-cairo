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

> Status: coverage of nalgebra-rs 0.35.0 at 64 % ([API_PARITY.md](docs/API_PARITY.md)): the 36
> static shapes and 54 aliases, geometry (rotations, quaternions, dual quaternions, isometries,
> similarities, scale, reflection) and small decompositions; ~5,000 tests. Release 0.1.0 when the
> coverage is complete (see the [plan](docs/PLAN.md)).

## Packages

| Package | Content |
|---|---|
| [`nalgebra`](crates/nalgebra) | `base` (vectors, matrices), `geometry` (rotations, isometries), `linalg` (decompositions), generic over `simba::scalar::Real` |
| [`nalgebra_glam`](crates/nalgebra_glam) | conversions between nalgebra-cairo and [glam-cairo](https://github.com/bal7hazar/glam-cairo) types (`Vec3` <-> `Vector3`, `Mat4` <-> `Isometry3`...), the Cairo counterpart of nalgebra-rs's `convert-glam` features ([DESIGN D11](docs/DESIGN.md)) |

Its scalar layer is the registry package `simba = "0.2.0"`
([simba-cairo](https://github.com/bal7hazar/simba-cairo): `Real` / `Transcendental` implemented for
fixed-cairo's Q32.32 [`fixed::Fixed`](https://github.com/bal7hazar/fixed-cairo) 0.4.0, the scalar
shared by the whole stack), like nalgebra-rs depends on simba-rs.

### glam conversions

nalgebra-rs converts to and from glam behind its `convert-glam0XX` features. Scarb has no optional
dependencies and `glam` costs every dependent +0.46 GB of cold compile, so the Cairo counterpart is
a separate package: add `nalgebra_glam` next to `nalgebra` and `glam` and `use
nalgebra_glam::prelude::*;` (upstream's `From` is `Into`, `TryFrom` is `TryInto` returning an
`Option`; details in the [package README](crates/nalgebra_glam)).

## Features

`nalgebra = "x.y"` exposes the full nalgebra-rs surface: every Scarb feature is in `default`. The
families that cost compile time and memory to every dependent, and that nothing else in the crate
uses, can be turned off (a structural deviation from nalgebra-rs, [DESIGN D9](docs/DESIGN.md)):

```toml
nalgebra = { version = "x.y", default-features = false, features = ["statistics"] }
```

| Feature | Gates |
|---|---|
| `statistics` | `mean`, `variance`, `column_mean`... (`base::statistics`) |
| `blas` | `gemm`, `gemv`, `axpy`, `ger`, `quadform`... (`base::blas`) |
| `closures` | the methods that take a closure: `map`, `fold`, `apply`, `zip_map`, `fill_with`... |
| `dynamic` | `DMatrix`, `DVector`, `RowDVector`, `Matrix3xX`...; the static `insert_columns`, `remove_fixed_rows`, `from_vec`...; `convolve_full` / `convolve_same` / `convolve_valid` (`base::dynamic`, [DESIGN D5](docs/DESIGN.md)) |
| `sparse` | `CsMatrix`, `CsVector`, `CsCholesky`, the sparse triangular solves, `axpy_cs`, `cumsum` (`nalgebra::sparse`; enables `dynamic`) |
| `io` | `cs_matrix_from_matrix_market_str` (`nalgebra::io`, Matrix Market; enables `sparse`) |
| `eigen` | `SymmetricEigen1..6`, `symmetric_eigen`, `symmetric_eigenvalues`, `wilkinson_shift` (`linalg::symmetric_eigen*`) |
| `svd` | `Svd1..Svd6x5`, `svd`, `singular_values`, `rank`, `pseudo_inverse`, `polar`, `svd_ordered2/3` (`linalg::svd*`; enables `eigen`) |
| `qr` | `Qr1..Qr6x5`, `qr` (`linalg::qr`) |
| `cholesky_update` | `rank_one_update`, `insert_column`, `remove_column` of the Cholesky factors (`linalg::cholesky_update`) |
| `full_piv_lu` | `FullPivLu1..FullPivLu6x5`, `full_piv_lu` (`linalg::full_piv_lu`) |
| `col_piv_qr` | `ColPivQr1..ColPivQr6x5`, `col_piv_qr` (`linalg::col_piv_qr`) |
| `lblt` | `Lblt1..Lblt6`, `lblt` (`linalg::lblt`, Bunch-Kaufman) |
| `macros` | `matrix!`, `vector!`, `point!`, `stack!`, and with `dynamic` `dmatrix!` / `dvector!` (`nalgebra::macros`, upstream's feature of the same name) |
| `hessenberg` | `Hessenberg1..6`, `hessenberg`; `SymmetricTridiagonal1..6`, `symmetric_tridiagonalize`; `balance_parlett_reinsch`, `unbalance`; `clear_column_unchecked`, `clear_row_unchecked`, `assemble_q` (`linalg::hessenberg`, `linalg::symmetric_tridiagonal`, `linalg::balancing`, `linalg::householder_steps`; 0.19 GB) |
| `bidiagonal` | `Bidiagonal1..Bidiagonal6x5`, `bidiagonalize` (`linalg::bidiagonal`; 0.37 GB) |
| `schur` | `Schur1..6`, `schur`, `try_schur`, `eigenvalues`, `complex_eigenvalues`; `Eigen1..6` (`linalg::schur`, `linalg::eigen`; enables `hessenberg`; 0.40 GB more) |

Turning the first three off cuts a cold build of the library by about 16 % of the memory and 32 % of the
CPU time ([measurements](tools/shapegen/DESIGN.md)); the test packages of this repository do it. `dynamic` adds about 0.55 GB to a cold build,
`sparse` and `io` about 0.14 GB more, and nothing when they are off. The linalg families (WP 8.5-P15)
add, to a cold build without the default features: `svd` + `eigen` 0.63 GB, `col_piv_qr` 0.43 GB,
`full_piv_lu` 0.29 GB, `qr` 0.23 GB, `eigen` alone 0.11 GB, `lblt` 0.09 GB, `cholesky_update` 0.02 GB.

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
D10](docs/DESIGN.md)). **A dependent needs neither**: calling the macros, `iter.sum()` or
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

Measured on Cairo 2.19.4 ([full synthesis](docs/BENCHMARK.md)):

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

- [Benchmark synthesis](docs/BENCHMARK.md) — what was measured and learned
- [Design decisions](docs/DESIGN.md)
- [Orchestration strategy](docs/ORCHESTRATOR.md) — how work is split between sub-agents
- [Execution plan](docs/PLAN.md)
- [Research reports](docs/research) — nalgebra, alexandria, starknet-agentic, origami, cubit, orion
- [Design-time benchmarks](benchmarks) — ~2,100 reproducible gas measurements

## Development

```bash
asdf install            # scarb 2.19.4, starknet-foundry 0.61.0
./scripts/check.sh      # fmt, lint, build, tests, gas snapshot
```

See [AGENTS.md](AGENTS.md) for the contribution rules (they apply to humans too).

## License

MIT
