//! The decompositions of `nalgebra_linalg_svd_eigen2` (upstream `nalgebra::linalg`) of dimension 2
//! (and 1): `svd`, `svd2`, `symmetric_eigen1`, `symmetric_eigen2`; the crate-internal kernels are
//! under `internal`; the rest of the module is in the other `nalgebra_linalg*` crates,
//! `nalgebra_types*`, `nalgebra_static6_wide` (`Lu6`) and the facade `nalgebra`, which re-exports
//! every item at its 0.1.0 path. Features (docs/SPLIT.md §4): `eigen`
//! (`linalg::symmetric_eigen*`);
//! `svd` (`linalg::svd*` (on `eigen`)), as in `nalgebra`.

#[cfg(feature: 'svd')]
pub mod svd;
#[cfg(feature: 'svd')]
pub mod svd2;
#[cfg(feature: 'eigen')]
pub mod symmetric_eigen1;
#[cfg(feature: 'eigen')]
pub mod symmetric_eigen2;
