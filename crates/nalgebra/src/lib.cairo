//! Linear algebra for provable game physics: a Cairo port of the Rust `nalgebra` crate.
//!
//! Modules mirror upstream: `base` (vectors, matrices), `geometry` (rotations, isometries),
//! `linalg` (decompositions). See `docs/DESIGN.md` and `docs/ROADMAP.md`.

pub mod base;
pub mod geometry;
pub mod linalg;

// shapegen: begin
pub use base::{
    ColumnPart, EuclideanNorm, FixedColumns, FixedResize, FixedRows, FixedView, LpNorm, Matrix1,
    Matrix1AngleTrait, Matrix1CgTrait, Matrix1Trait, Matrix1x2, Matrix1x3, Matrix1x4, Matrix1x5,
    Matrix1x6, Matrix2, Matrix2AngleTrait, Matrix2CgTrait, Matrix2Trait, Matrix2x1, Matrix2x3,
    Matrix2x3AngleTrait, Matrix2x3Trait, Matrix2x4, Matrix2x4AngleTrait, Matrix2x4Trait, Matrix2x5,
    Matrix2x5AngleTrait, Matrix2x5Trait, Matrix2x6, Matrix2x6AngleTrait, Matrix2x6Trait, Matrix3,
    Matrix3AngleTrait, Matrix3CgAngleTrait, Matrix3CgTrait, Matrix3Trait, Matrix3x1, Matrix3x2,
    Matrix3x2AngleTrait, Matrix3x2Trait, Matrix3x4, Matrix3x4AngleTrait, Matrix3x4Trait, Matrix3x5,
    Matrix3x5AngleTrait, Matrix3x5Trait, Matrix3x6, Matrix3x6AngleTrait, Matrix3x6Trait, Matrix4,
    Matrix4AngleTrait, Matrix4CgAngleTrait, Matrix4CgTrait, Matrix4Trait, Matrix4x1, Matrix4x2,
    Matrix4x2AngleTrait, Matrix4x2Trait, Matrix4x3, Matrix4x3AngleTrait, Matrix4x3Trait, Matrix4x5,
    Matrix4x5AngleTrait, Matrix4x5Trait, Matrix4x6, Matrix4x6AngleTrait, Matrix4x6Trait, Matrix5,
    Matrix5AngleTrait, Matrix5CgTrait, Matrix5Trait, Matrix5x1, Matrix5x2, Matrix5x2AngleTrait,
    Matrix5x2Trait, Matrix5x3, Matrix5x3AngleTrait, Matrix5x3Trait, Matrix5x4, Matrix5x4AngleTrait,
    Matrix5x4Trait, Matrix5x6, Matrix5x6AngleTrait, Matrix5x6Trait, Matrix6, Matrix6AngleTrait,
    Matrix6CgTrait, Matrix6Trait, Matrix6x1, Matrix6x2, Matrix6x2AngleTrait, Matrix6x2Trait,
    Matrix6x3, Matrix6x3AngleTrait, Matrix6x3Trait, Matrix6x4, Matrix6x4AngleTrait, Matrix6x4Trait,
    Matrix6x5, Matrix6x5AngleTrait, Matrix6x5Trait, MatrixIndex, MatrixKronecker, MatrixMul,
    MatrixSolve, MatrixTrMul, Norm, OneNorm, Point1SwizzleTrait, Point2SwizzleTrait,
    Point3SwizzleTrait, Point4SwizzleTrait, Point5SwizzleTrait, Point6SwizzleTrait, RowPart,
    RowVector1, RowVector2, RowVector2AngleTrait, RowVector2Trait, RowVector3, RowVector3AngleTrait,
    RowVector3Trait, RowVector4, RowVector4AngleTrait, RowVector4Trait, RowVector5,
    RowVector5AngleTrait, RowVector5Trait, RowVector6, RowVector6AngleTrait, RowVector6Trait,
    UniformNorm, UnitVector1, UnitVector1AngleTrait, UnitVector1Trait, UnitVector2,
    UnitVector2AngleTrait, UnitVector2Trait, UnitVector3, UnitVector3AngleTrait, UnitVector3Trait,
    UnitVector4, UnitVector4AngleTrait, UnitVector4Trait, UnitVector5, UnitVector5AngleTrait,
    UnitVector5Trait, UnitVector6, UnitVector6AngleTrait, UnitVector6Trait, Vector1, Vector2,
    Vector2AngleTrait, Vector2Trait, Vector3, Vector3AngleTrait, Vector3Trait, Vector4,
    Vector4AngleTrait, Vector4Trait, Vector5, Vector5AngleTrait, Vector5Trait, Vector6,
    Vector6AngleTrait, Vector6Trait,
};
#[cfg(feature: 'dynamic')]
pub use base::{
    DMatrix, DMatrixTrait, DVector, DVectorTrait, InsertFixedColumns, InsertFixedRows,
    Matrix1DynamicTrait, Matrix1xX, Matrix2DynamicTrait, Matrix2x3DynamicTrait,
    Matrix2x4DynamicTrait, Matrix2x5DynamicTrait, Matrix2x6DynamicTrait, Matrix2xX,
    Matrix3DynamicTrait, Matrix3x2DynamicTrait, Matrix3x4DynamicTrait, Matrix3x5DynamicTrait,
    Matrix3x6DynamicTrait, Matrix3xX, Matrix4DynamicTrait, Matrix4x2DynamicTrait,
    Matrix4x3DynamicTrait, Matrix4x5DynamicTrait, Matrix4x6DynamicTrait, Matrix4xX,
    Matrix5DynamicTrait, Matrix5x2DynamicTrait, Matrix5x3DynamicTrait, Matrix5x4DynamicTrait,
    Matrix5x6DynamicTrait, Matrix5xX, Matrix6DynamicTrait, Matrix6x2DynamicTrait,
    Matrix6x3DynamicTrait, Matrix6x4DynamicTrait, Matrix6x5DynamicTrait, Matrix6xX, MatrixXx1,
    MatrixXx2, MatrixXx3, MatrixXx4, MatrixXx5, MatrixXx6, RemoveFixedColumns, RemoveFixedRows,
    RowDVector, RowDVectorTrait, RowVector2DynamicTrait, RowVector3DynamicTrait,
    RowVector4DynamicTrait, RowVector5DynamicTrait, RowVector6DynamicTrait, Vector2DynamicTrait,
    Vector3DynamicTrait, Vector4DynamicTrait, Vector5DynamicTrait, Vector6DynamicTrait,
};
#[cfg(feature: 'blas')]
pub use base::{
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
#[cfg(feature: 'statistics')]
pub use base::{
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
// shapegen: end
pub use base::{Normed, Point2, Point3, Unit, UnitTrait};
pub use geometry::{
    AbstractRotation, Affine2, Affine2Trait, Affine3, Affine3Trait, DualQuaternion,
    DualQuaternionTrait, Isometry2, Isometry2AngleTrait, Isometry2Trait, Isometry3,
    Isometry3AngleTrait, Isometry3DualQuaternionTrait, Isometry3Trait, IsometryMatrix2,
    IsometryMatrix2AngleTrait, IsometryMatrix2Trait, IsometryMatrix3, IsometryMatrix3AngleTrait,
    IsometryMatrix3Trait, Matrix4OrthographicTrait, Matrix4PerspectiveTrait, Orthographic3,
    Orthographic3AngleTrait, Orthographic3Trait, Perspective3, Perspective3AngleTrait,
    Perspective3Trait, Point1, Point1Trait, Point2ExtTrait, Point2Index, Point2PartialOrd,
    Point3ExtTrait, Point3Index, Point3PartialOrd, Point4, Point4Trait, Point5, Point5Trait, Point6,
    Point6Trait, Projective2, Projective2Trait, Projective3, Projective3Trait, Quaternion,
    QuaternionTrait, QuaternionTranscendentalTrait, Reflection1, Reflection1Columns,
    Reflection1Rows, Reflection1Trait, Reflection2, Reflection2Columns, Reflection2Rows,
    Reflection2Trait, Reflection3, Reflection3Columns, Reflection3Rows, Reflection3Trait,
    Reflection4, Reflection4Columns, Reflection4Rows, Reflection4Trait, Reflection5,
    Reflection5Columns, Reflection5Rows, Reflection5Trait, Reflection6, Reflection6Columns,
    Reflection6Rows, Reflection6Trait, Rotation2, Rotation2AngleTrait, Rotation2Trait, Rotation3,
    Rotation3AngleTrait, Rotation3Trait, Scale1, Scale1Trait, Scale2, Scale2Trait, Scale3,
    Scale3Trait, Scale4, Scale4Trait, Scale5, Scale5Trait, Scale6, Scale6Trait, Similarity2,
    Similarity2AngleTrait, Similarity2Trait, Similarity3, Similarity3AngleTrait, Similarity3Trait,
    SimilarityMatrix2, SimilarityMatrix2AngleTrait, SimilarityMatrix2Trait, SimilarityMatrix3,
    SimilarityMatrix3AngleTrait, SimilarityMatrix3Trait, Transform2, Transform2Trait, Transform3,
    Transform3Trait, TransformDiv, TransformMul, TransformSetCategory, Translation1,
    Translation1Trait, Translation2, Translation2Trait, Translation3,
    Translation3DualQuaternionTrait, Translation3Trait, Translation4, Translation4Trait,
    Translation5, Translation5Trait, Translation6, Translation6Trait, UnitComplex,
    UnitComplexAngleTrait, UnitComplexTrait, UnitDualQuaternion, UnitDualQuaternionAngleTrait,
    UnitDualQuaternionTrait, UnitQuaternion, UnitQuaternionAngleTrait,
    UnitQuaternionDualQuaternionTrait, UnitQuaternionTrait,
};
pub use linalg::{
    Cholesky2, Cholesky2Trait, Cholesky3, Cholesky3Trait, Cholesky4, Cholesky4Trait, Cholesky6,
    Cholesky6Trait, GivensRotate, GivensRotateRows, GivensRotation, GivensRotationTrait, Lu2,
    Lu2Trait, Lu3, Lu3Trait, Lu4, Lu4Trait, Lu6, Lu6Trait, Matrix2CholeskyTrait,
    Matrix2InverseTrait, Matrix2LuTrait, Matrix2QrTrait, Matrix2SvdTrait, Matrix2UduTrait,
    Matrix3CholeskyTrait, Matrix3InverseTrait, Matrix3LuTrait, Matrix3QrTrait, Matrix3SvdTrait,
    Matrix3UduTrait, Matrix4CholeskyTrait, Matrix4InverseTrait, Matrix4LuTrait, Matrix4QrTrait,
    Matrix4UduTrait, Matrix6CholeskyTrait, Matrix6InverseTrait, Matrix6LuTrait, Matrix6UduTrait,
    Perm2, Perm2Trait, Perm3, Perm3Trait, Perm4, Perm4Trait, Perm6, Perm6Trait, PermuteColumns,
    PermuteRows, Qr2, Qr2Trait, Qr3, Qr3Trait, Qr4, Qr4Trait, Svd2, Svd2Trait, Svd3, Svd3Trait,
    SymmetricEigen2, SymmetricEigen2Trait, SymmetricEigen3, SymmetricEigen3Trait,
};
