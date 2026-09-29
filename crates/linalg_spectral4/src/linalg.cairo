//! The spectral decompositions of `nalgebra_linalg_spectral4` (upstream `nalgebra::linalg`) up to
//! 4x4: `hessenberg`, `symmetric_tridiagonal`, `balancing`, `bidiagonal`, `schur`, `eigen`, `exp`,
//! `pow`; the rest of the module is in the other `nalgebra_linalg*` crates (the facade `nalgebra`
//! re-exports every item at its 0.1.0 path). Features (docs/SPLIT.md §4): `hessenberg`
//! (`hessenberg`, `symmetric_tridiagonal`, `balancing`), `bidiagonal`, `schur` (`schur`, `eigen`,
//! on `hessenberg`), `exp` (`exp`, `pow`), as in `nalgebra`.

#[cfg(feature: 'hessenberg')]
pub mod balancing;
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
