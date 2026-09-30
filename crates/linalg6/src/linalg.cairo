//! The decompositions of `nalgebra_linalg6` (upstream `nalgebra::linalg`) of dimension 6:
//! `cholesky`, `cholesky_update`, `inverse`, `qr`, `udu`; the crate-internal kernels are under
//! `internal`; the rest of the module is in the other `nalgebra_linalg*` crates, `nalgebra_types*`,
//! `nalgebra_static6_wide` (`Lu6`) and the facade `nalgebra`, which re-exports every item at its
//! 0.1.0 path. Features (docs/SPLIT.md §4): `qr` (`linalg::qr`); `cholesky_update`
//! (`linalg::cholesky_update`), as in `nalgebra`.

pub mod cholesky;
#[cfg(feature: 'cholesky_update')]
pub mod cholesky_update;
pub mod inverse;
#[cfg(feature: 'qr')]
pub mod qr;
pub mod udu;
