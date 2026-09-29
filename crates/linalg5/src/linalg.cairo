//! The decompositions of `nalgebra_linalg5` (upstream `nalgebra::linalg`) of dimension 5: `qr`
//! (`Qr` of every shape with 5 rows or columns and none larger), `svd` (the SVD of the same shapes)
//! and `symmetric_eigen5`; the rest of the module is in the other `nalgebra_linalg*` crates and in
//! the facade `nalgebra`, which re-exports every item at its 0.1.0 path (`Lu6` is in
//! `nalgebra_static6_wide`; there is no `Lu5`, `Cholesky5` or `Udu5` in 0.1.0). Features
//! (docs/SPLIT.md §4): `qr`, `eigen`, `svd` (on `eigen`), as in `nalgebra`.

#[cfg(feature: 'qr')]
pub mod qr;
#[cfg(feature: 'svd')]
pub mod svd;
#[cfg(feature: 'eigen')]
pub mod symmetric_eigen5;
