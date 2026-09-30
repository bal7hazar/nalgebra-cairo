//! The decompositions of `nalgebra_linalg_core` (upstream `nalgebra::linalg`) shared by every
//! dimension: `balancing`, `householder`, `householder_steps`; the crate-internal kernels are under
//! `internal`; the rest of the module is in the other `nalgebra_linalg*` crates, `nalgebra_types*`,
//! `nalgebra_static6_wide` (`Lu6`) and the facade `nalgebra`, which re-exports every item at its
//! 0.1.0 path. Features (docs/SPLIT.md §4): `svd` (`linalg::svd*` (on `eigen`)); `hessenberg`
//! (`linalg::hessenberg`, `linalg::symmetric_tridiagonal`, `linalg::balancing`,
//! `linalg::householder_steps`); `exp` (`linalg::exp`, `linalg::pow`), as in `nalgebra`.

#[cfg(feature: 'hessenberg')]
pub mod balancing;
pub mod householder;
#[cfg(feature: 'hessenberg')]
pub mod householder_steps;
