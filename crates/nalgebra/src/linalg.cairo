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
//!   polar decomposition `M = P·U` (upstream `SVD`), built on the symmetric eigen decomposition
//!   of `MᵀM` (DESIGN D6).

pub mod cholesky;
#[cfg(feature: 'cholesky_update')]
pub mod cholesky_update;
pub mod givens;
pub mod householder;
pub mod inverse;
pub(crate) mod ldlt;
pub mod lu;
pub mod lu_steps;
#[cfg(test)]
mod oracle_cholesky;
#[cfg(and(test, feature: 'svd'))]
mod oracle_svd;
#[cfg(and(test, feature: 'eigen'))]
mod oracle_symmetric_eigen;
#[cfg(test)]
mod oracle_udu;
pub mod permutation_sequence;
#[cfg(feature: 'qr')]
pub mod qr;
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
#[cfg(feature: 'eigen')]
pub mod symmetric_eigen5;
#[cfg(feature: 'eigen')]
pub mod symmetric_eigen6;
pub mod udu;

pub use cholesky::{
    Cholesky2, Cholesky2Trait, Cholesky3, Cholesky3Trait, Cholesky4, Cholesky4Trait, Cholesky6,
    Cholesky6Trait, Matrix2CholeskyTrait, Matrix3CholeskyTrait, Matrix4CholeskyTrait,
    Matrix6CholeskyTrait,
};
#[cfg(feature: 'cholesky_update')]
pub use cholesky_update::{
    Cholesky2UpdateTrait, Cholesky3UpdateTrait, Cholesky4UpdateTrait, Cholesky6UpdateTrait,
};
pub use givens::{GivensRotate, GivensRotateRows, GivensRotation, GivensRotationTrait};
pub use householder::reflection_axis_mut;
pub use inverse::{
    Matrix2InverseTrait, Matrix3InverseTrait, Matrix4InverseTrait, Matrix6InverseTrait,
};
pub use lu::{
    Lu2, Lu2Trait, Lu3, Lu3Trait, Lu4, Lu4Trait, Lu6, Lu6Trait, Matrix2LuTrait, Matrix3LuTrait,
    Matrix4LuTrait, Matrix6LuTrait, Perm2, Perm2Trait, Perm3, Perm3Trait, Perm4, Perm4Trait, Perm6,
    Perm6Trait,
};
pub use lu_steps::{gauss_step, gauss_step_swap, try_invert_to};
pub use permutation_sequence::{PermuteColumns, PermuteRows};
#[cfg(feature: 'qr')]
pub use qr::{
    Matrix1QrTrait, Matrix2QrTrait, Matrix2x3QrTrait, Matrix2x4QrTrait, Matrix2x5QrTrait,
    Matrix2x6QrTrait, Matrix3QrTrait, Matrix3x2QrTrait, Matrix3x4QrTrait, Matrix3x5QrTrait,
    Matrix3x6QrTrait, Matrix4QrTrait, Matrix4x2QrTrait, Matrix4x3QrTrait, Matrix4x5QrTrait,
    Matrix4x6QrTrait, Matrix5QrTrait, Matrix5x2QrTrait, Matrix5x3QrTrait, Matrix5x4QrTrait,
    Matrix5x6QrTrait, Matrix6QrTrait, Matrix6x2QrTrait, Matrix6x3QrTrait, Matrix6x4QrTrait,
    Matrix6x5QrTrait, Qr1, Qr1Trait, Qr1x2, Qr1x2Trait, Qr1x3, Qr1x3Trait, Qr1x4, Qr1x4Trait, Qr1x5,
    Qr1x5Trait, Qr1x6, Qr1x6Trait, Qr2, Qr2Trait, Qr2x1, Qr2x1Trait, Qr2x3, Qr2x3Trait, Qr2x4,
    Qr2x4Trait, Qr2x5, Qr2x5Trait, Qr2x6, Qr2x6Trait, Qr3, Qr3Trait, Qr3x1, Qr3x1Trait, Qr3x2,
    Qr3x2Trait, Qr3x4, Qr3x4Trait, Qr3x5, Qr3x5Trait, Qr3x6, Qr3x6Trait, Qr4, Qr4Trait, Qr4x1,
    Qr4x1Trait, Qr4x2, Qr4x2Trait, Qr4x3, Qr4x3Trait, Qr4x5, Qr4x5Trait, Qr4x6, Qr4x6Trait, Qr5,
    Qr5Trait, Qr5x1, Qr5x1Trait, Qr5x2, Qr5x2Trait, Qr5x3, Qr5x3Trait, Qr5x4, Qr5x4Trait, Qr5x6,
    Qr5x6Trait, Qr6, Qr6Trait, Qr6x1, Qr6x1Trait, Qr6x2, Qr6x2Trait, Qr6x3, Qr6x3Trait, Qr6x4,
    Qr6x4Trait, Qr6x5, Qr6x5Trait, RowVector2QrTrait, RowVector3QrTrait, RowVector4QrTrait,
    RowVector5QrTrait, RowVector6QrTrait, Vector2QrTrait, Vector3QrTrait, Vector4QrTrait,
    Vector5QrTrait, Vector6QrTrait,
};
#[cfg(feature: 'svd')]
pub use svd::{
    Matrix1SvdTrait, Matrix2x3SvdTrait, Matrix2x4SvdTrait, Matrix2x5SvdTrait, Matrix2x6SvdTrait,
    Matrix3x2SvdTrait, Matrix3x4SvdTrait, Matrix3x5SvdTrait, Matrix3x6SvdTrait, Matrix4SvdTrait,
    Matrix4x2SvdTrait, Matrix4x3SvdTrait, Matrix4x5SvdTrait, Matrix4x6SvdTrait, Matrix5SvdTrait,
    Matrix5x2SvdTrait, Matrix5x3SvdTrait, Matrix5x4SvdTrait, Matrix5x6SvdTrait, Matrix6SvdTrait,
    Matrix6x2SvdTrait, Matrix6x3SvdTrait, Matrix6x4SvdTrait, Matrix6x5SvdTrait, RowVector2SvdTrait,
    RowVector3SvdTrait, RowVector4SvdTrait, RowVector5SvdTrait, RowVector6SvdTrait, Svd1, Svd1Trait,
    Svd1x2, Svd1x2Trait, Svd1x3, Svd1x3Trait, Svd1x4, Svd1x4Trait, Svd1x5, Svd1x5Trait, Svd1x6,
    Svd1x6Trait, Svd2x1, Svd2x1Trait, Svd2x3, Svd2x3Trait, Svd2x4, Svd2x4Trait, Svd2x5, Svd2x5Trait,
    Svd2x6, Svd2x6Trait, Svd3x1, Svd3x1Trait, Svd3x2, Svd3x2Trait, Svd3x4, Svd3x4Trait, Svd3x5,
    Svd3x5Trait, Svd3x6, Svd3x6Trait, Svd4, Svd4Trait, Svd4x1, Svd4x1Trait, Svd4x2, Svd4x2Trait,
    Svd4x3, Svd4x3Trait, Svd4x5, Svd4x5Trait, Svd4x6, Svd4x6Trait, Svd5, Svd5Trait, Svd5x1,
    Svd5x1Trait, Svd5x2, Svd5x2Trait, Svd5x3, Svd5x3Trait, Svd5x4, Svd5x4Trait, Svd5x6, Svd5x6Trait,
    Svd6, Svd6Trait, Svd6x1, Svd6x1Trait, Svd6x2, Svd6x2Trait, Svd6x3, Svd6x3Trait, Svd6x4,
    Svd6x4Trait, Svd6x5, Svd6x5Trait, Vector2SvdTrait, Vector3SvdTrait, Vector4SvdTrait,
    Vector5SvdTrait, Vector6SvdTrait,
};
#[cfg(feature: 'svd')]
pub use svd2::{Matrix2SvdTrait, Svd2, Svd2Trait, svd_ordered2};
#[cfg(feature: 'svd')]
pub use svd3::{Matrix3SvdTrait, Svd3, Svd3Trait, svd_ordered3};
#[cfg(feature: 'eigen')]
pub use symmetric_eigen1::{Matrix1SymmetricEigenTrait, SymmetricEigen1, SymmetricEigen1Trait};
#[cfg(feature: 'eigen')]
pub use symmetric_eigen2::{
    Matrix2SymmetricEigenTrait, SymmetricEigen2, SymmetricEigen2Trait, wilkinson_shift,
};
#[cfg(feature: 'eigen')]
pub use symmetric_eigen3::{Matrix3SymmetricEigenTrait, SymmetricEigen3, SymmetricEigen3Trait};
#[cfg(feature: 'eigen')]
pub use symmetric_eigen4::{Matrix4SymmetricEigenTrait, SymmetricEigen4, SymmetricEigen4Trait};
#[cfg(feature: 'eigen')]
pub use symmetric_eigen5::{Matrix5SymmetricEigenTrait, SymmetricEigen5, SymmetricEigen5Trait};
#[cfg(feature: 'eigen')]
pub use symmetric_eigen6::{Matrix6SymmetricEigenTrait, SymmetricEigen6, SymmetricEigen6Trait};
pub use udu::{
    Matrix2UduTrait, Matrix3UduTrait, Matrix4UduTrait, Matrix6UduTrait, Udu2, Udu2Trait, Udu3,
    Udu3Trait, Udu4, Udu4Trait, Udu6, Udu6Trait,
};
