//! The decompositions of `nalgebra_linalg6` (upstream `nalgebra::linalg`) of dimension 6:
//! `cholesky` (`Cholesky6`), `cholesky_update` (its updates), `udu` (`Udu6`, on the `LDLᵀ` kernel
//! under `internal`), `inverse` (`Matrix6InverseTrait`), `qr` and `svd` (every shape with 6 rows or
//! columns) and `symmetric_eigen6`; `Lu6` is in `nalgebra_static6_wide`, the rest of the module in
//! the other `nalgebra_linalg*` crates and in the facade `nalgebra`, which re-exports every item at
//! its 0.1.0 path. Features (docs/SPLIT.md §4): `qr`, `cholesky_update`, `eigen`, `svd` (on
//! `eigen`), as in `nalgebra`.

pub mod cholesky;
#[cfg(feature: 'cholesky_update')]
pub mod cholesky_update;
pub mod inverse;
#[cfg(feature: 'qr')]
pub mod qr;
#[cfg(feature: 'svd')]
pub mod svd;
#[cfg(feature: 'eigen')]
pub mod symmetric_eigen6;
pub mod udu;
