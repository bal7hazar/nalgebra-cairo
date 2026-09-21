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
//! - `lu`: `P·A = L·U` with partial pivoting for any square matrix (upstream `LU`).
//! - `qr`: `A = Q·R` with `Q` orthonormal and `R` upper triangular with a non-negative diagonal
//!   (upstream `QR`, unpacked convention), sizes 2, 3 and 4, by modified Gram-Schmidt;
//! - `svd2` / `svd3`: `M = U·Σ·Vᵀ`, the pseudo-inverse, the least-squares solve and the polar
//!   decomposition `M = R·P` (upstream `SVD`), built on the symmetric eigen decomposition of
//!   `MᵀM` (DESIGN D6).

pub mod cholesky;
pub mod ldlt;
pub mod lu;
#[cfg(test)]
mod oracle_cholesky;
#[cfg(test)]
mod oracle_svd;
#[cfg(test)]
mod oracle_symmetric_eigen;
#[cfg(test)]
mod oracle_udu;
pub mod qr;
pub mod svd2;
pub mod svd3;
pub mod symmetric_eigen2;
pub mod symmetric_eigen3;

pub use cholesky::{
    Cholesky2, Cholesky2Trait, Cholesky3, Cholesky3Trait, Cholesky4, Cholesky4Trait, Cholesky6,
    Cholesky6Trait,
};
pub use ldlt::{Ldlt2, Ldlt2Trait, Ldlt3, Ldlt3Trait, Ldlt4, Ldlt4Trait, Ldlt6, Ldlt6Trait};
pub use lu::{
    Lu2, Lu2Trait, Lu3, Lu3Trait, Lu4, Lu4Trait, Lu6, Lu6Trait, Matrix2LuTrait, Matrix3LuTrait,
    Matrix4LuTrait, Matrix6LuTrait, Perm2, Perm3, Perm4, Perm6, PermTrait,
};
pub use qr::{
    Matrix2QrTrait, Matrix3QrTrait, Matrix4QrTrait, Qr2, Qr2Trait, Qr3, Qr3Trait, Qr4, Qr4Trait,
};
pub use svd2::{Matrix2SvdTrait, Svd2, Svd2Trait};
pub use svd3::{Matrix3SvdTrait, Svd3, Svd3Trait};
pub use symmetric_eigen2::{SymmetricEigen2, SymmetricEigen2Trait};
pub use symmetric_eigen3::{SymmetricEigen3, SymmetricEigen3Trait};
