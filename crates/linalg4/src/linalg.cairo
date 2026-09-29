//! The decompositions of `nalgebra_linalg4` (upstream `nalgebra::linalg`) up to 4x4: `lu` (`Lu2`
//! .. `Lu4`), `cholesky`, `cholesky_update`, `ldlt` / `udu`, `qr`, `inverse`, and the building
//! blocks `householder`, `householder_steps`, `householder_kernels`; the rest of the
//! module is in the other `nalgebra_linalg*` crates (the facade `nalgebra` re-exports every item at
//! its 0.1.0 path). Features (docs/SPLIT.md §4): `qr`, `cholesky_update`, `hessenberg`
//! (`householder_steps`), as in `nalgebra`.

pub mod cholesky;
#[cfg(feature: 'cholesky_update')]
pub mod cholesky_update;
pub mod householder;
pub mod householder_kernels;
#[cfg(feature: 'hessenberg')]
pub mod householder_steps;
pub mod inverse;
pub mod ldlt;
pub mod lu;
#[cfg(feature: 'qr')]
pub mod qr;
pub mod udu;
