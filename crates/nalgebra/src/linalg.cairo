//! Matrix decompositions (upstream `nalgebra::linalg`), unrolled for the static shapes of `base`
//! (DESIGN D4, D6).
//!
//! Every decomposition here is **closed form or fixed cost**: no convergence loop, no iteration
//! count, no tolerance parameter. Gas is therefore a constant of the type, which is what a proof
//! system needs; accuracy is a measured property, reported in the doc comment of each
//! decomposition and checked against `tools/oracle` in the tests.
//!
//! - `cholesky`: `A = L·Lᵀ` for a symmetric POSITIVE-DEFINITE matrix (upstream `Cholesky`), 2,
//! 3,
//!   4, 6; `cholesky_update`: its rank-one update and column insertion / removal;
//! - `udu`: `A = U·D·Uᵀ` with `U` unit upper triangular (upstream `UDU`), on the crate-internal
//!   `ldlt` kernel (`A = L·D·Lᵀ`, no square root, indefinite matrices accepted) of the reversed
//!   matrix;
//! - `lu`: `P·A = L·U` with partial pivoting (upstream `LU`), 2, 3, 4, 6, and the permutation
//!   sequences `Perm1..6` (upstream `PermutationSequence`); `full_piv_lu`: `P·A·Q = L·U` with
//!   full pivoting of every shape (upstream `FullPivLU`);
//! - `qr`: `A = Q·R` of every shape by modified Gram-Schmidt (upstream `QR`, unpacked
//!   convention); `col_piv_qr`: `A·P = Q·R` with column pivoting of every shape by Householder
//!   reflections, upstream's storage (upstream `ColPivQR`);
//! - `lblt`: the Bunch-Kaufman `P·A·Pᵀ = L·B·Lᵀ` of the symmetric squares (upstream
//! `LBLT`);
//! - `symmetric_eigen1..6`: eigen decomposition of symmetric matrices (closed form in 2D,
//!   fixed-sweep Jacobi beyond); `svd*`: `M = U·Σ·Vᵀ` of every shape, pseudo-inverse, rank,
//!   polar decomposition (upstream `SVD`), on the symmetric eigen decomposition of `MᵀM`;
//! - `givens`, `householder`, `lu_steps`, `inverse`, `permutation_sequence`: upstream's building
//!   blocks and free functions.
//!
//! Like upstream, the symmetric factorisations take a full `MatrixN` and read ONE triangle: the
//! LOWER one for `Cholesky`, `SymmetricEigen` and `LBLT`, the UPPER one for `UDU`.
//!
//! Scarb features (DESIGN D9, all in `default`): `eigen` (`symmetric_eigen*`), `svd` (`svd*`, on
//! `eigen`), `qr`, `cholesky_update`, `full_piv_lu`, `col_piv_qr`, `lblt`. Nothing ungated uses
//! them: the only item of `linalg` the rest of the crate uses is `Lu6` (`Matrix6::determinant` /
//! `try_inverse`), and `lu` / `cholesky` / `udu` stay ungated.

#[cfg(feature: 'balancing')]
pub mod balancing;
#[cfg(feature: 'bidiagonal')]
pub mod bidiagonal;
pub mod cholesky;
#[cfg(feature: 'cholesky_update')]
pub mod cholesky_update;
#[cfg(feature: 'col_piv_qr')]
pub mod col_piv_qr;
#[cfg(feature: 'schur')]
pub mod eigen;
#[cfg(feature: 'full_piv_lu')]
pub mod full_piv_lu;
pub mod givens;
#[cfg(feature: 'hessenberg')]
pub mod hessenberg;
pub mod householder;
pub(crate) mod householder_kernels;
#[cfg(feature: 'hessenberg')]
pub mod householder_steps;
pub mod inverse;
#[cfg(feature: 'lblt')]
pub mod lblt;
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
#[cfg(feature: 'schur')]
pub mod schur;
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
#[cfg(feature: 'symmetric_tridiagonal')]
pub mod symmetric_tridiagonal;
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
#[cfg(feature: 'col_piv_qr')]
pub use col_piv_qr::{
    ColPivQr1, ColPivQr1Trait, ColPivQr1x2, ColPivQr1x2Trait, ColPivQr1x3, ColPivQr1x3Trait,
    ColPivQr1x4, ColPivQr1x4Trait, ColPivQr1x5, ColPivQr1x5Trait, ColPivQr1x6, ColPivQr1x6Trait,
    ColPivQr2, ColPivQr2Trait, ColPivQr2x1, ColPivQr2x1Trait, ColPivQr2x3, ColPivQr2x3Trait,
    ColPivQr2x4, ColPivQr2x4Trait, ColPivQr2x5, ColPivQr2x5Trait, ColPivQr2x6, ColPivQr2x6Trait,
    ColPivQr3, ColPivQr3Trait, ColPivQr3x1, ColPivQr3x1Trait, ColPivQr3x2, ColPivQr3x2Trait,
    ColPivQr3x4, ColPivQr3x4Trait, ColPivQr3x5, ColPivQr3x5Trait, ColPivQr3x6, ColPivQr3x6Trait,
    ColPivQr4, ColPivQr4Trait, ColPivQr4x1, ColPivQr4x1Trait, ColPivQr4x2, ColPivQr4x2Trait,
    ColPivQr4x3, ColPivQr4x3Trait, ColPivQr4x5, ColPivQr4x5Trait, ColPivQr4x6, ColPivQr4x6Trait,
    ColPivQr5, ColPivQr5Trait, ColPivQr5x1, ColPivQr5x1Trait, ColPivQr5x2, ColPivQr5x2Trait,
    ColPivQr5x3, ColPivQr5x3Trait, ColPivQr5x4, ColPivQr5x4Trait, ColPivQr5x6, ColPivQr5x6Trait,
    ColPivQr6, ColPivQr6Trait, ColPivQr6x1, ColPivQr6x1Trait, ColPivQr6x2, ColPivQr6x2Trait,
    ColPivQr6x3, ColPivQr6x3Trait, ColPivQr6x4, ColPivQr6x4Trait, ColPivQr6x5, ColPivQr6x5Trait,
    Matrix1ColPivQrTrait, Matrix2ColPivQrTrait, Matrix2x3ColPivQrTrait, Matrix2x4ColPivQrTrait,
    Matrix2x5ColPivQrTrait, Matrix2x6ColPivQrTrait, Matrix3ColPivQrTrait, Matrix3x2ColPivQrTrait,
    Matrix3x4ColPivQrTrait, Matrix3x5ColPivQrTrait, Matrix3x6ColPivQrTrait, Matrix4ColPivQrTrait,
    Matrix4x2ColPivQrTrait, Matrix4x3ColPivQrTrait, Matrix4x5ColPivQrTrait, Matrix4x6ColPivQrTrait,
    Matrix5ColPivQrTrait, Matrix5x2ColPivQrTrait, Matrix5x3ColPivQrTrait, Matrix5x4ColPivQrTrait,
    Matrix5x6ColPivQrTrait, Matrix6ColPivQrTrait, Matrix6x2ColPivQrTrait, Matrix6x3ColPivQrTrait,
    Matrix6x4ColPivQrTrait, Matrix6x5ColPivQrTrait, RowVector2ColPivQrTrait,
    RowVector3ColPivQrTrait, RowVector4ColPivQrTrait, RowVector5ColPivQrTrait,
    RowVector6ColPivQrTrait, Vector2ColPivQrTrait, Vector3ColPivQrTrait, Vector4ColPivQrTrait,
    Vector5ColPivQrTrait, Vector6ColPivQrTrait,
};
#[cfg(feature: 'full_piv_lu')]
pub use full_piv_lu::{
    FullPivLu1, FullPivLu1Trait, FullPivLu1x2, FullPivLu1x2Trait, FullPivLu1x3, FullPivLu1x3Trait,
    FullPivLu1x4, FullPivLu1x4Trait, FullPivLu1x5, FullPivLu1x5Trait, FullPivLu1x6,
    FullPivLu1x6Trait, FullPivLu2, FullPivLu2Trait, FullPivLu2x1, FullPivLu2x1Trait, FullPivLu2x3,
    FullPivLu2x3Trait, FullPivLu2x4, FullPivLu2x4Trait, FullPivLu2x5, FullPivLu2x5Trait,
    FullPivLu2x6, FullPivLu2x6Trait, FullPivLu3, FullPivLu3Trait, FullPivLu3x1, FullPivLu3x1Trait,
    FullPivLu3x2, FullPivLu3x2Trait, FullPivLu3x4, FullPivLu3x4Trait, FullPivLu3x5,
    FullPivLu3x5Trait, FullPivLu3x6, FullPivLu3x6Trait, FullPivLu4, FullPivLu4Trait, FullPivLu4x1,
    FullPivLu4x1Trait, FullPivLu4x2, FullPivLu4x2Trait, FullPivLu4x3, FullPivLu4x3Trait,
    FullPivLu4x5, FullPivLu4x5Trait, FullPivLu4x6, FullPivLu4x6Trait, FullPivLu5, FullPivLu5Trait,
    FullPivLu5x1, FullPivLu5x1Trait, FullPivLu5x2, FullPivLu5x2Trait, FullPivLu5x3,
    FullPivLu5x3Trait, FullPivLu5x4, FullPivLu5x4Trait, FullPivLu5x6, FullPivLu5x6Trait, FullPivLu6,
    FullPivLu6Trait, FullPivLu6x1, FullPivLu6x1Trait, FullPivLu6x2, FullPivLu6x2Trait, FullPivLu6x3,
    FullPivLu6x3Trait, FullPivLu6x4, FullPivLu6x4Trait, FullPivLu6x5, FullPivLu6x5Trait,
    Matrix1FullPivLuTrait, Matrix2FullPivLuTrait, Matrix2x3FullPivLuTrait, Matrix2x4FullPivLuTrait,
    Matrix2x5FullPivLuTrait, Matrix2x6FullPivLuTrait, Matrix3FullPivLuTrait,
    Matrix3x2FullPivLuTrait, Matrix3x4FullPivLuTrait, Matrix3x5FullPivLuTrait,
    Matrix3x6FullPivLuTrait, Matrix4FullPivLuTrait, Matrix4x2FullPivLuTrait,
    Matrix4x3FullPivLuTrait, Matrix4x5FullPivLuTrait, Matrix4x6FullPivLuTrait,
    Matrix5FullPivLuTrait, Matrix5x2FullPivLuTrait, Matrix5x3FullPivLuTrait,
    Matrix5x4FullPivLuTrait, Matrix5x6FullPivLuTrait, Matrix6FullPivLuTrait,
    Matrix6x2FullPivLuTrait, Matrix6x3FullPivLuTrait, Matrix6x4FullPivLuTrait,
    Matrix6x5FullPivLuTrait, RowVector2FullPivLuTrait, RowVector3FullPivLuTrait,
    RowVector4FullPivLuTrait, RowVector5FullPivLuTrait, RowVector6FullPivLuTrait,
    Vector2FullPivLuTrait, Vector3FullPivLuTrait, Vector4FullPivLuTrait, Vector5FullPivLuTrait,
    Vector6FullPivLuTrait,
};
pub use givens::{GivensRotate, GivensRotateRows, GivensRotation, GivensRotationTrait};
pub use householder::reflection_axis_mut;
pub use inverse::{
    Matrix2InverseTrait, Matrix3InverseTrait, Matrix4InverseTrait, Matrix6InverseTrait,
};
#[cfg(feature: 'lblt')]
pub use lblt::{
    Lblt1, Lblt1Trait, Lblt2, Lblt2Trait, Lblt3, Lblt3Trait, Lblt4, Lblt4Trait, Lblt5, Lblt5Trait,
    Lblt6, Lblt6Trait, Matrix1LbltTrait, Matrix2LbltTrait, Matrix3LbltTrait, Matrix4LbltTrait,
    Matrix5LbltTrait, Matrix6LbltTrait,
};
pub use lu::{
    Lu2, Lu2Trait, Lu3, Lu3Trait, Lu4, Lu4Trait, Lu6, Lu6Trait, Matrix2LuTrait, Matrix3LuTrait,
    Matrix4LuTrait, Matrix6LuTrait, Perm1, Perm1Trait, Perm2, Perm2Trait, Perm3, Perm3Trait, Perm4,
    Perm4Trait, Perm5, Perm5Trait, Perm6, Perm6Trait,
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
