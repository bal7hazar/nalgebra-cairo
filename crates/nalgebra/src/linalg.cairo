//! Matrix decompositions (upstream `nalgebra::linalg`), unrolled for the static sizes of
//! `base` — 2, 3, 4 and the 6 of spatial algebra (DESIGN D4, D6).
//!
//! Every decomposition here is **closed form or fixed cost**: no convergence loop, no iteration
//! count, no tolerance parameter. Gas is therefore a constant of the type, which is what a proof
//! system needs; accuracy is a measured property, reported in the doc comment of each
//! decomposition and checked against `tools/oracle` in the tests.
//!
//! - `cholesky`: `A = L·Lᵀ` for a symmetric POSITIVE-DEFINITE matrix (upstream `Cholesky`);
//! - `ldlt`: `A = L·D·Lᵀ` with `L` unit lower triangular, for any symmetric matrix whose
//!   leading principal minors are non-zero (upstream's mirror is `UDU`). No square root, a cheaper
//!   `solve`, and indefinite matrices are accepted: DESIGN D6 prefers it whenever the factor is
//!   only a means to solve a system;
//! - `symmetric_eigen2` / `symmetric_eigen3`: eigen decomposition of symmetric matrices (closed
//!   form in 2D, fixed-sweep Jacobi in 3D).
//!
//! Factorisations expose the same surface: `new(matrix) -> Option`, `l`, `solve`, `inverse`,
//! `determinant` (plus `d` for `LDLᵀ`). The input is a `SymMatrix2` / `SymMatrix3` at sizes 2 and
//! 3, and — there being no `SymMatrix4` / `SymMatrix6` — a `Matrix4` / `Matrix6` whose LOWER
//! triangle is read, like upstream.

pub mod cholesky;
#[cfg(test)]
mod eigen_test_utils;
#[cfg(test)]
mod factor_test_utils;
pub mod ldlt;
#[cfg(test)]
mod oracle_cholesky;
#[cfg(test)]
mod oracle_symmetric_eigen;
#[cfg(test)]
mod oracle_udu;
pub mod symmetric_eigen2;
pub mod symmetric_eigen3;

pub use cholesky::{
    Cholesky2, Cholesky2Trait, Cholesky3, Cholesky3Trait, Cholesky4, Cholesky4Trait, Cholesky6,
    Cholesky6Trait,
};
pub use ldlt::{Ldlt2, Ldlt2Trait, Ldlt3, Ldlt3Trait, Ldlt4, Ldlt4Trait, Ldlt6, Ldlt6Trait};
pub use symmetric_eigen2::{SymmetricEigen2, SymmetricEigen2Trait};
pub use symmetric_eigen3::{SymmetricEigen3, SymmetricEigen3Trait};
