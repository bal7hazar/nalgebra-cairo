//! Statically sized vectors and matrices (upstream `nalgebra::base`).
//!
//! The 36 static shapes (`Matrix1..6`, `MatrixRxC`, `Vector2..6`, `RowVector2..6`), their upstream
//! aliases, the `MatrixMul` / `MatrixTrMul` products and their private kernels are written by
//! `tools/shapegen/shapegen.py`, like the block between the `shapegen` markers below.

#[cfg(test)]
pub(crate) mod matrix_test_utils;
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
#[cfg(feature: 'dynamic')]
pub mod dynamic;
pub mod point2;
pub mod point3;
pub(crate) mod sym_matrix2;
pub(crate) mod sym_matrix3;
pub mod unit;

#[cfg(feature: 'dynamic')]
pub use dynamic::{
    DMatrix, DMatrixTrait, DVector, DVectorTrait, Matrix1xX, Matrix2xX, Matrix3xX, Matrix4xX,
    Matrix5xX, Matrix6xX, MatrixXx1, MatrixXx2, MatrixXx3, MatrixXx4, MatrixXx5, MatrixXx6,
    RowDVector, RowDVectorTrait,
};
pub use point2::Point2;
pub use point3::Point3;
pub use unit::{Normed, Unit, UnitTrait};

// shapegen: begin
#[cfg(feature: 'blas')]
pub mod blas;
pub mod cg;
pub mod errors;
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
pub mod matrix_index;
pub mod matrix_kronecker;
pub mod matrix_mul;
pub mod matrix_tr_mul;
pub mod matrix_view;
pub mod norm;
pub mod point_swizzle;
pub mod row_vector2;
pub mod row_vector3;
pub mod row_vector4;
pub mod row_vector5;
pub mod row_vector6;
pub mod solve;
#[cfg(feature: 'statistics')]
pub mod statistics;
mod transpose;
pub mod vector2;
pub mod vector3;
pub mod vector4;
pub mod vector5;
pub mod vector6;
#[cfg(feature: 'blas')]
pub use blas::{
    Matrix1BlasTrait, Matrix2BlasTrait, Matrix2x3BlasTrait, Matrix2x4BlasTrait, Matrix2x5BlasTrait,
    Matrix2x6BlasTrait, Matrix3BlasTrait, Matrix3x2BlasTrait, Matrix3x4BlasTrait,
    Matrix3x5BlasTrait, Matrix3x6BlasTrait, Matrix4BlasTrait, Matrix4x2BlasTrait,
    Matrix4x3BlasTrait, Matrix4x5BlasTrait, Matrix4x6BlasTrait, Matrix5BlasTrait,
    Matrix5x2BlasTrait, Matrix5x3BlasTrait, Matrix5x4BlasTrait, Matrix5x6BlasTrait,
    Matrix6BlasTrait, Matrix6x2BlasTrait, Matrix6x3BlasTrait, Matrix6x4BlasTrait,
    Matrix6x5BlasTrait, MatrixGemm, MatrixGemmTr, MatrixGemv, MatrixGemvTr, MatrixQuadform,
    MatrixQuadformTr, RowVector2BlasTrait, RowVector3BlasTrait, RowVector4BlasTrait,
    RowVector5BlasTrait, RowVector6BlasTrait, Vector2BlasTrait, Vector3BlasTrait, Vector4BlasTrait,
    Vector5BlasTrait, Vector6BlasTrait,
};
pub use cg::{
    Matrix1CgTrait, Matrix2CgTrait, Matrix3CgAngleTrait, Matrix3CgTrait, Matrix4CgAngleTrait,
    Matrix4CgTrait, Matrix5CgTrait, Matrix6CgTrait,
};
pub use matrix1::{
    Matrix1, Matrix1AngleTrait, Matrix1Trait, RowVector1, UnitVector1, UnitVector1AngleTrait,
    UnitVector1Trait, Vector1,
};
pub use matrix2::{Matrix2, Matrix2AngleTrait, Matrix2Trait};
pub use matrix2x3::{Matrix2x3, Matrix2x3AngleTrait, Matrix2x3Trait};
pub use matrix2x4::{Matrix2x4, Matrix2x4AngleTrait, Matrix2x4Trait};
pub use matrix2x5::{Matrix2x5, Matrix2x5AngleTrait, Matrix2x5Trait};
pub use matrix2x6::{Matrix2x6, Matrix2x6AngleTrait, Matrix2x6Trait};
pub use matrix3::{Matrix3, Matrix3AngleTrait, Matrix3Trait};
pub use matrix3x2::{Matrix3x2, Matrix3x2AngleTrait, Matrix3x2Trait};
pub use matrix3x4::{Matrix3x4, Matrix3x4AngleTrait, Matrix3x4Trait};
pub use matrix3x5::{Matrix3x5, Matrix3x5AngleTrait, Matrix3x5Trait};
pub use matrix3x6::{Matrix3x6, Matrix3x6AngleTrait, Matrix3x6Trait};
pub use matrix4::{Matrix4, Matrix4AngleTrait, Matrix4Trait};
pub use matrix4x2::{Matrix4x2, Matrix4x2AngleTrait, Matrix4x2Trait};
pub use matrix4x3::{Matrix4x3, Matrix4x3AngleTrait, Matrix4x3Trait};
pub use matrix4x5::{Matrix4x5, Matrix4x5AngleTrait, Matrix4x5Trait};
pub use matrix4x6::{Matrix4x6, Matrix4x6AngleTrait, Matrix4x6Trait};
pub use matrix5::{Matrix5, Matrix5AngleTrait, Matrix5Trait};
pub use matrix5x2::{Matrix5x2, Matrix5x2AngleTrait, Matrix5x2Trait};
pub use matrix5x3::{Matrix5x3, Matrix5x3AngleTrait, Matrix5x3Trait};
pub use matrix5x4::{Matrix5x4, Matrix5x4AngleTrait, Matrix5x4Trait};
pub use matrix5x6::{Matrix5x6, Matrix5x6AngleTrait, Matrix5x6Trait};
pub use matrix6::{Matrix6, Matrix6AngleTrait, Matrix6Trait};
pub use matrix6x2::{Matrix6x2, Matrix6x2AngleTrait, Matrix6x2Trait};
pub use matrix6x3::{Matrix6x3, Matrix6x3AngleTrait, Matrix6x3Trait};
pub use matrix6x4::{Matrix6x4, Matrix6x4AngleTrait, Matrix6x4Trait};
pub use matrix6x5::{Matrix6x5, Matrix6x5AngleTrait, Matrix6x5Trait};
pub use matrix_index::MatrixIndex;
pub use matrix_kronecker::MatrixKronecker;
pub use matrix_mul::MatrixMul;
pub use matrix_tr_mul::MatrixTrMul;
pub use matrix_view::{ColumnPart, FixedColumns, FixedResize, FixedRows, FixedView, RowPart};
pub use norm::{EuclideanNorm, LpNorm, Norm, OneNorm, UniformNorm};
pub use point_swizzle::{
    Point1SwizzleTrait, Point2SwizzleTrait, Point3SwizzleTrait, Point4SwizzleTrait,
    Point5SwizzleTrait, Point6SwizzleTrait,
};
pub use row_vector2::{Matrix1x2, RowVector2, RowVector2AngleTrait, RowVector2Trait};
pub use row_vector3::{Matrix1x3, RowVector3, RowVector3AngleTrait, RowVector3Trait};
pub use row_vector4::{Matrix1x4, RowVector4, RowVector4AngleTrait, RowVector4Trait};
pub use row_vector5::{Matrix1x5, RowVector5, RowVector5AngleTrait, RowVector5Trait};
pub use row_vector6::{Matrix1x6, RowVector6, RowVector6AngleTrait, RowVector6Trait};
pub use solve::MatrixSolve;
#[cfg(feature: 'statistics')]
pub use statistics::{
    Matrix1StatisticsTrait, Matrix2StatisticsTrait, Matrix2x3StatisticsTrait,
    Matrix2x4StatisticsTrait, Matrix2x5StatisticsTrait, Matrix2x6StatisticsTrait,
    Matrix3StatisticsTrait, Matrix3x2StatisticsTrait, Matrix3x4StatisticsTrait,
    Matrix3x5StatisticsTrait, Matrix3x6StatisticsTrait, Matrix4StatisticsTrait,
    Matrix4x2StatisticsTrait, Matrix4x3StatisticsTrait, Matrix4x5StatisticsTrait,
    Matrix4x6StatisticsTrait, Matrix5StatisticsTrait, Matrix5x2StatisticsTrait,
    Matrix5x3StatisticsTrait, Matrix5x4StatisticsTrait, Matrix5x6StatisticsTrait,
    Matrix6StatisticsTrait, Matrix6x2StatisticsTrait, Matrix6x3StatisticsTrait,
    Matrix6x4StatisticsTrait, Matrix6x5StatisticsTrait, RowVector2StatisticsTrait,
    RowVector3StatisticsTrait, RowVector4StatisticsTrait, RowVector5StatisticsTrait,
    RowVector6StatisticsTrait, Vector2StatisticsTrait, Vector3StatisticsTrait,
    Vector4StatisticsTrait, Vector5StatisticsTrait, Vector6StatisticsTrait,
};
pub use vector2::{
    Matrix2x1, UnitVector2, UnitVector2AngleTrait, UnitVector2Trait, Vector2, Vector2AngleTrait,
    Vector2Trait,
};
pub use vector3::{
    Matrix3x1, UnitVector3, UnitVector3AngleTrait, UnitVector3Trait, Vector3, Vector3AngleTrait,
    Vector3Trait,
};
pub use vector4::{
    Matrix4x1, UnitVector4, UnitVector4AngleTrait, UnitVector4Trait, Vector4, Vector4AngleTrait,
    Vector4Trait,
};
pub use vector5::{
    Matrix5x1, UnitVector5, UnitVector5AngleTrait, UnitVector5Trait, Vector5, Vector5AngleTrait,
    Vector5Trait,
};
pub use vector6::{
    Matrix6x1, UnitVector6, UnitVector6AngleTrait, UnitVector6Trait, Vector6, Vector6AngleTrait,
    Vector6Trait,
};
// shapegen: end
