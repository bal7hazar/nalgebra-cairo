//! Linear algebra for provable game physics: a Cairo port of the Rust `nalgebra` crate.
//!
//! Modules mirror upstream: `base` (vectors, matrices), `geometry` (rotations, isometries),
//! `linalg` (decompositions), and behind their features `sparse` (compressed sparse column
//! matrices) and `io` (Matrix Market parsing). The free functions of upstream's crate root
//! (`nalgebra::distance`, `convert`...) are in `root` and the construction macros
//! (`nalgebra::matrix!`...) in `macros` (feature `macros`), both re-exported here. See
//! `docs/DESIGN.md` and `docs/ROADMAP.md`.

pub mod base;
pub mod geometry;
#[cfg(feature: 'io')]
pub mod io;
pub mod linalg;
#[cfg(feature: 'macros')]
pub mod macros;
pub mod root;
#[cfg(feature: 'sparse')]
pub mod sparse;

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
    Convolution, DMatrix, DMatrixTrait, DVector, DVectorTrait, InsertFixedColumns, InsertFixedRows,
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
    Matrix2InverseTrait, Matrix2LuTrait, Matrix2UduTrait, Matrix3CholeskyTrait, Matrix3InverseTrait,
    Matrix3LuTrait, Matrix3UduTrait, Matrix4CholeskyTrait, Matrix4InverseTrait, Matrix4LuTrait,
    Matrix4UduTrait, Matrix6CholeskyTrait, Matrix6InverseTrait, Matrix6LuTrait, Matrix6UduTrait,
    Perm2, Perm2Trait, Perm3, Perm3Trait, Perm4, Perm4Trait, Perm6, Perm6Trait, PermuteColumns,
    PermuteRows,
};
#[cfg(feature: 'cholesky_update')]
pub use linalg::{
    Cholesky2UpdateTrait, Cholesky3UpdateTrait, Cholesky4UpdateTrait, Cholesky6UpdateTrait,
};
#[cfg(feature: 'col_piv_qr')]
pub use linalg::{
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
pub use linalg::{
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
#[cfg(feature: 'lblt')]
pub use linalg::{
    Lblt1, Lblt1Trait, Lblt2, Lblt2Trait, Lblt3, Lblt3Trait, Lblt4, Lblt4Trait, Lblt5, Lblt5Trait,
    Lblt6, Lblt6Trait, Matrix1LbltTrait, Matrix2LbltTrait, Matrix3LbltTrait, Matrix4LbltTrait,
    Matrix5LbltTrait, Matrix6LbltTrait,
};
#[cfg(feature: 'qr')]
pub use linalg::{
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
pub use linalg::{
    Matrix1SvdTrait, Matrix2SvdTrait, Matrix2x3SvdTrait, Matrix2x4SvdTrait, Matrix2x5SvdTrait,
    Matrix2x6SvdTrait, Matrix3SvdTrait, Matrix3x2SvdTrait, Matrix3x4SvdTrait, Matrix3x5SvdTrait,
    Matrix3x6SvdTrait, Matrix4SvdTrait, Matrix4x2SvdTrait, Matrix4x3SvdTrait, Matrix4x5SvdTrait,
    Matrix4x6SvdTrait, Matrix5SvdTrait, Matrix5x2SvdTrait, Matrix5x3SvdTrait, Matrix5x4SvdTrait,
    Matrix5x6SvdTrait, Matrix6SvdTrait, Matrix6x2SvdTrait, Matrix6x3SvdTrait, Matrix6x4SvdTrait,
    Matrix6x5SvdTrait, RowVector2SvdTrait, RowVector3SvdTrait, RowVector4SvdTrait,
    RowVector5SvdTrait, RowVector6SvdTrait, Svd1, Svd1Trait, Svd1x2, Svd1x2Trait, Svd1x3,
    Svd1x3Trait, Svd1x4, Svd1x4Trait, Svd1x5, Svd1x5Trait, Svd1x6, Svd1x6Trait, Svd2, Svd2Trait,
    Svd2x1, Svd2x1Trait, Svd2x3, Svd2x3Trait, Svd2x4, Svd2x4Trait, Svd2x5, Svd2x5Trait, Svd2x6,
    Svd2x6Trait, Svd3, Svd3Trait, Svd3x1, Svd3x1Trait, Svd3x2, Svd3x2Trait, Svd3x4, Svd3x4Trait,
    Svd3x5, Svd3x5Trait, Svd3x6, Svd3x6Trait, Svd4, Svd4Trait, Svd4x1, Svd4x1Trait, Svd4x2,
    Svd4x2Trait, Svd4x3, Svd4x3Trait, Svd4x5, Svd4x5Trait, Svd4x6, Svd4x6Trait, Svd5, Svd5Trait,
    Svd5x1, Svd5x1Trait, Svd5x2, Svd5x2Trait, Svd5x3, Svd5x3Trait, Svd5x4, Svd5x4Trait, Svd5x6,
    Svd5x6Trait, Svd6, Svd6Trait, Svd6x1, Svd6x1Trait, Svd6x2, Svd6x2Trait, Svd6x3, Svd6x3Trait,
    Svd6x4, Svd6x4Trait, Svd6x5, Svd6x5Trait, Vector2SvdTrait, Vector3SvdTrait, Vector4SvdTrait,
    Vector5SvdTrait, Vector6SvdTrait,
};
#[cfg(feature: 'eigen')]
pub use linalg::{
    SymmetricEigen1, SymmetricEigen1Trait, SymmetricEigen2, SymmetricEigen2Trait, SymmetricEigen3,
    SymmetricEigen3Trait, SymmetricEigen4, SymmetricEigen4Trait, SymmetricEigen5,
    SymmetricEigen5Trait, SymmetricEigen6, SymmetricEigen6Trait,
};
#[cfg(and(feature: 'macros', feature: 'dynamic'))]
pub use macros::{dmatrix, dvector};
#[cfg(feature: 'macros')]
pub use macros::{matrix, point, stack, vector};
pub use root::{
    abs, center, clamp, convert, convert_ref, convert_ref_unchecked, convert_unchecked, distance,
    distance_squared, inf, inf_sup, is_convertible, max, min, one, partial_clamp, partial_cmp,
    partial_ge, partial_gt, partial_le, partial_lt, partial_max, partial_min, partial_sort2, sup,
    try_convert, try_convert_ref, wrap, zero,
};
