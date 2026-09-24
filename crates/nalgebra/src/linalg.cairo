//! Matrix decompositions (upstream `nalgebra::linalg`), unrolled for the static sizes of
//! `base` — 2, 3, 4 and the 6 of spatial algebra (DESIGN D4, D6).
//!
//! Every decomposition here is **closed form or fixed cost**: no convergence loop, no iteration
//! count, no tolerance parameter. Gas is therefore a constant of the type, which is what a proof
//! system needs; accuracy is a measured property, reported in the doc comment of each
//! decomposition and checked against `tools/oracle` in the tests.
//!
//! - `cholesky`: `A = L·Lᵀ` for a symmetric POSITIVE-DEFINITE matrix (upstream `Cholesky`);
//! - `udu`: `A = U·D·Uᵀ` with `U` unit upper triangular, for any symmetric matrix whose
//! trailing
//!   principal minors are non-zero (upstream `UDU`: the fields `u`, `d`, `new`, `d_matrix`). Its
//!   kernel is the crate-internal `ldlt` (`A = L·D·Lᵀ`, no square root, indefinite matrices
//!   accepted, DESIGN D6) applied to the reversed matrix;
//! - `symmetric_eigen2` / `symmetric_eigen3`: eigen decomposition of symmetric matrices (closed
//!   form in 2D, fixed-sweep Jacobi in 3D).
//!
//! Like upstream, the symmetric factorisations take a full `MatrixN` and read ONE triangle: the
//! LOWER one for `Cholesky` and `SymmetricEigen`, the UPPER one for `UDU`.
//! - `lu`: `P·A = L·U` with partial pivoting for any square matrix (upstream `LU`).
//! - `qr`: `A = Q·R` with `Q` orthonormal and `R` upper triangular with a non-negative diagonal
//!   (upstream `QR`, unpacked convention), sizes 2, 3 and 4, by modified Gram-Schmidt;
//! - `svd2` / `svd3`: `M = U·Σ·Vᵀ`, the pseudo-inverse, the least-squares solve and the left
//! polar
//!   decomposition `M = P·U` (upstream `SVD`), built on the symmetric eigen decomposition of
//!   `MᵀM` (DESIGN D6).

pub mod cholesky;
pub(crate) mod ldlt;
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
pub mod udu;

pub use cholesky::{
    Cholesky2, Cholesky2Trait, Cholesky3, Cholesky3Trait, Cholesky4, Cholesky4Trait, Cholesky6,
    Cholesky6Trait,
};
pub use lu::{
    Lu2, Lu2Trait, Lu3, Lu3Trait, Lu4, Lu4Trait, Lu6, Lu6Trait, Matrix2LuTrait, Matrix3LuTrait,
    Matrix4LuTrait, Matrix6LuTrait, Perm2, Perm2Trait, Perm3, Perm3Trait, Perm4, Perm4Trait, Perm6,
    Perm6Trait,
};
pub use qr::{
    Matrix2QrTrait, Matrix3QrTrait, Matrix4QrTrait, Qr2, Qr2Trait, Qr3, Qr3Trait, Qr4, Qr4Trait,
};
pub use svd2::{Matrix2SvdTrait, Svd2, Svd2Trait};
pub use svd3::{Matrix3SvdTrait, Svd3, Svd3Trait};
pub use symmetric_eigen2::{Matrix2SymmetricEigenTrait, SymmetricEigen2, SymmetricEigen2Trait};
pub use symmetric_eigen3::{Matrix3SymmetricEigenTrait, SymmetricEigen3, SymmetricEigen3Trait};
pub use udu::{Udu2, Udu2Trait, Udu3, Udu3Trait, Udu4, Udu4Trait, Udu6, Udu6Trait};
