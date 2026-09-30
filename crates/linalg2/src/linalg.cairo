//! The decompositions of `nalgebra_linalg2` (upstream `nalgebra::linalg`) of dimension 2 (and 1):
//! `cholesky`, `inverse`, `lu`, `qr`, `udu`; the crate-internal kernels are under `internal`; the
//! rest of the module is in the other `nalgebra_linalg*` crates, `nalgebra_types*`,
//! `nalgebra_static6_wide` (`Lu6`) and the facade `nalgebra`, which re-exports every item at its
//! 0.1.0 path. Features (docs/SPLIT.md §4): `qr` (`linalg::qr`), as in `nalgebra`.

pub mod cholesky;
pub mod inverse;
pub mod lu;
#[cfg(feature: 'qr')]
pub mod qr;
pub mod udu;
