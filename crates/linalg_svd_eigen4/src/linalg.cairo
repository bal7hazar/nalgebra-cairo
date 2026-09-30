//! The decompositions of `nalgebra_linalg_svd_eigen4` (upstream `nalgebra::linalg`) of dimension 4:
//! `svd`, `symmetric_eigen4`; the crate-internal kernels are under `internal`; the rest of the
//! module is in the other `nalgebra_linalg*` crates, `nalgebra_types*`, `nalgebra_static6_wide`
//! (`Lu6`) and the facade `nalgebra`, which re-exports every item at its 0.1.0 path. Features
//! (docs/SPLIT.md §4): `eigen` (`linalg::symmetric_eigen*`); `svd` (`linalg::svd*` (on `eigen`)),
//! as in `nalgebra`.

#[cfg(feature: 'svd')]
pub mod svd;
#[cfg(feature: 'eigen')]
pub mod symmetric_eigen4;
