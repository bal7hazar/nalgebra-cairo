//! Statically sized vectors and matrices (upstream `nalgebra::base`).
//!
//! The 36 static shapes (`Matrix1..6`, `MatrixRxC`, `Vector2..6`, `RowVector2..6`), their upstream
//! aliases, the `MatrixMul` / `MatrixTrMul` products and their private kernels are written by
//! `tools/shapegen/shapegen.py`, like the block between the `shapegen` markers below.

#[cfg(test)]
pub(crate) mod matrix_test_utils;
#[cfg(test)]
mod oracle_dim6_matrix;
#[cfg(test)]
mod oracle_dim6_vector;
#[cfg(test)]
mod oracle_matrix2;
#[cfg(test)]
mod oracle_matrix2_inverse;
#[cfg(test)]
mod oracle_matrix3;
#[cfg(test)]
mod oracle_matrix3_inverse;
#[cfg(test)]
mod oracle_matrix4;
#[cfg(test)]
mod oracle_matrix4_inverse;
#[cfg(test)]
mod oracle_sym_matrix;
pub mod point2;
pub mod point3;
pub(crate) mod sym_matrix2;
pub(crate) mod sym_matrix3;
pub mod unit;

pub use point2::Point2;
pub use point3::Point3;
pub use unit::Unit;

// shapegen: begin
mod kernels;
pub mod matrix1;
pub mod matrix2;
pub mod matrix2x3;
pub mod matrix2x4;
pub mod matrix2x5;
pub mod matrix2x6;
pub mod matrix3;
pub mod matrix3x2;
pub mod matrix3x4;
pub mod matrix3x5;
pub mod matrix3x6;
pub mod matrix4;
pub mod matrix4x2;
pub mod matrix4x3;
pub mod matrix4x5;
pub mod matrix4x6;
pub mod matrix5;
pub mod matrix5x2;
pub mod matrix5x3;
pub mod matrix5x4;
pub mod matrix5x6;
pub mod matrix6;
pub mod matrix6x2;
pub mod matrix6x3;
pub mod matrix6x4;
pub mod matrix6x5;
pub mod matrix_mul;
pub mod matrix_tr_mul;
pub mod row_vector2;
pub mod row_vector3;
pub mod row_vector4;
pub mod row_vector5;
pub mod row_vector6;
pub mod vector2;
pub mod vector3;
pub mod vector4;
pub mod vector5;
pub mod vector6;
pub use matrix1::{Matrix1, Matrix1Trait, RowVector1, UnitVector1, Vector1};
pub use matrix2::{Matrix2, Matrix2Trait};
pub use matrix2x3::{Matrix2x3, Matrix2x3Trait};
pub use matrix2x4::{Matrix2x4, Matrix2x4Trait};
pub use matrix2x5::{Matrix2x5, Matrix2x5Trait};
pub use matrix2x6::{Matrix2x6, Matrix2x6Trait};
pub use matrix3::{Matrix3, Matrix3Trait};
pub use matrix3x2::{Matrix3x2, Matrix3x2Trait};
pub use matrix3x4::{Matrix3x4, Matrix3x4Trait};
pub use matrix3x5::{Matrix3x5, Matrix3x5Trait};
pub use matrix3x6::{Matrix3x6, Matrix3x6Trait};
pub use matrix4::{Matrix4, Matrix4Trait};
pub use matrix4x2::{Matrix4x2, Matrix4x2Trait};
pub use matrix4x3::{Matrix4x3, Matrix4x3Trait};
pub use matrix4x5::{Matrix4x5, Matrix4x5Trait};
pub use matrix4x6::{Matrix4x6, Matrix4x6Trait};
pub use matrix5::{Matrix5, Matrix5Trait};
pub use matrix5x2::{Matrix5x2, Matrix5x2Trait};
pub use matrix5x3::{Matrix5x3, Matrix5x3Trait};
pub use matrix5x4::{Matrix5x4, Matrix5x4Trait};
pub use matrix5x6::{Matrix5x6, Matrix5x6Trait};
pub use matrix6::{Matrix6, Matrix6Trait};
pub use matrix6x2::{Matrix6x2, Matrix6x2Trait};
pub use matrix6x3::{Matrix6x3, Matrix6x3Trait};
pub use matrix6x4::{Matrix6x4, Matrix6x4Trait};
pub use matrix6x5::{Matrix6x5, Matrix6x5Trait};
pub use matrix_mul::MatrixMul;
pub use matrix_tr_mul::MatrixTrMul;
pub use row_vector2::{Matrix1x2, RowVector2, RowVector2Trait};
pub use row_vector3::{Matrix1x3, RowVector3, RowVector3Trait};
pub use row_vector4::{Matrix1x4, RowVector4, RowVector4Trait};
pub use row_vector5::{Matrix1x5, RowVector5, RowVector5Trait};
pub use row_vector6::{Matrix1x6, RowVector6, RowVector6Trait};
pub use vector2::{Matrix2x1, UnitVector2, Vector2, Vector2Trait};
pub use vector3::{Matrix3x1, UnitVector3, Vector3, Vector3Trait};
pub use vector4::{Matrix4x1, UnitVector4, Vector4, Vector4Trait};
pub use vector5::{Matrix5x1, UnitVector5, Vector5, Vector5Trait};
pub use vector6::{Matrix6x1, UnitVector6, Vector6, Vector6Trait};
// shapegen: end
