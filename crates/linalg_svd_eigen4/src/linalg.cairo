//! The decompositions of `nalgebra_linalg_svd_eigen4` (upstream `nalgebra::linalg`): the symmetric
//! eigen decomposition up to 4x4 (`symmetric_eigen1` .. `symmetric_eigen4`) and the SVD of the
//! shapes up to 4 rows and columns (`svd`, `svd2`, `svd3`); the rest of the module is in the other
//! `nalgebra_linalg*` crates (the facade `nalgebra` re-exports every item at its 0.1.0 path).
//! Features (docs/SPLIT.md §4): `eigen`, `svd` (on `eigen`), as in `nalgebra`.

#[cfg(feature: 'svd')]
pub mod svd;
#[cfg(feature: 'svd')]
pub mod svd2;
#[cfg(feature: 'svd')]
pub mod svd3;
#[cfg(feature: 'eigen')]
pub mod symmetric_eigen1;
#[cfg(feature: 'eigen')]
pub mod symmetric_eigen2;
#[cfg(feature: 'eigen')]
pub mod symmetric_eigen3;
#[cfg(feature: 'eigen')]
pub mod symmetric_eigen4;
