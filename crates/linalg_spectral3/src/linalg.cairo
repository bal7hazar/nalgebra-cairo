//! The decompositions of `nalgebra_linalg_spectral3` (upstream `nalgebra::linalg`) of dimension 3:
//! `bidiagonal`, `eigen`, `exp`, `hessenberg`, `pow`, `schur`, `symmetric_tridiagonal`; the rest of
//! the module is in the other `nalgebra_linalg*` crates, `nalgebra_types*`, `nalgebra_static6_wide`
//! (`Lu6`) and the facade `nalgebra`, which re-exports every item at its 0.1.0 path. Features
//! (docs/SPLIT.md §4): `hessenberg` (`linalg::hessenberg`, `linalg::symmetric_tridiagonal`,
//! `linalg::balancing`, `linalg::householder_steps`); `bidiagonal` (`linalg::bidiagonal`); `schur`
//! (`linalg::schur`, `linalg::eigen` (on `hessenberg`)); `exp` (`linalg::exp`, `linalg::pow`), as
//! in `nalgebra`.

#[cfg(feature: 'bidiagonal')]
pub mod bidiagonal;
#[cfg(feature: 'schur')]
pub mod eigen;
#[cfg(feature: 'exp')]
pub mod exp;
#[cfg(feature: 'hessenberg')]
pub mod hessenberg;
#[cfg(feature: 'exp')]
pub mod pow;
#[cfg(feature: 'schur')]
pub mod schur;
#[cfg(feature: 'hessenberg')]
pub mod symmetric_tridiagonal;
