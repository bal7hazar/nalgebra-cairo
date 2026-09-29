//! The spectral decompositions of `nalgebra_linalg_spectral6` (upstream `nalgebra::linalg`) of
//! dimension 6: `bidiagonal`, `hessenberg`, `symmetric_tridiagonal`, `schur`, `eigen`, `exp` and
//! `pow` (the balancing and Householder building blocks of dimension 6 sit with their types in
//! `nalgebra_shapes6`); the rest of the module is in the other `nalgebra_linalg*` crates (the
//! facade `nalgebra` re-exports every item at its 0.1.0 path). Features (docs/SPLIT.md §4):
//! `hessenberg`, `bidiagonal`, `schur` (on `hessenberg`), `exp`, as in `nalgebra`.

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
