//! Linear algebra for provable game physics: a Cairo port of the Rust `nalgebra` crate.
//!
//! Modules mirror upstream: `base` (vectors, matrices), `geometry` (rotations, isometries),
//! `linalg` (decompositions). See `docs/DESIGN.md` and `docs/ROADMAP.md`.

pub mod base;
pub mod geometry;
pub mod linalg;

// shapegen: begin
pub use base::{
    EuclideanNorm, LpNorm, Matrix1, Matrix1AngleTrait, Matrix1Trait, Matrix1x2, Matrix1x3,
    Matrix1x4, Matrix1x5, Matrix1x6, Matrix2, Matrix2AngleTrait, Matrix2Trait, Matrix2x1, Matrix2x3,
    Matrix2x3AngleTrait, Matrix2x3Trait, Matrix2x4, Matrix2x4AngleTrait, Matrix2x4Trait, Matrix2x5,
    Matrix2x5AngleTrait, Matrix2x5Trait, Matrix2x6, Matrix2x6AngleTrait, Matrix2x6Trait, Matrix3,
    Matrix3AngleTrait, Matrix3Trait, Matrix3x1, Matrix3x2, Matrix3x2AngleTrait, Matrix3x2Trait,
    Matrix3x4, Matrix3x4AngleTrait, Matrix3x4Trait, Matrix3x5, Matrix3x5AngleTrait, Matrix3x5Trait,
    Matrix3x6, Matrix3x6AngleTrait, Matrix3x6Trait, Matrix4, Matrix4AngleTrait, Matrix4Trait,
    Matrix4x1, Matrix4x2, Matrix4x2AngleTrait, Matrix4x2Trait, Matrix4x3, Matrix4x3AngleTrait,
    Matrix4x3Trait, Matrix4x5, Matrix4x5AngleTrait, Matrix4x5Trait, Matrix4x6, Matrix4x6AngleTrait,
    Matrix4x6Trait, Matrix5, Matrix5AngleTrait, Matrix5Trait, Matrix5x1, Matrix5x2,
    Matrix5x2AngleTrait, Matrix5x2Trait, Matrix5x3, Matrix5x3AngleTrait, Matrix5x3Trait, Matrix5x4,
    Matrix5x4AngleTrait, Matrix5x4Trait, Matrix5x6, Matrix5x6AngleTrait, Matrix5x6Trait, Matrix6,
    Matrix6AngleTrait, Matrix6Trait, Matrix6x1, Matrix6x2, Matrix6x2AngleTrait, Matrix6x2Trait,
    Matrix6x3, Matrix6x3AngleTrait, Matrix6x3Trait, Matrix6x4, Matrix6x4AngleTrait, Matrix6x4Trait,
    Matrix6x5, Matrix6x5AngleTrait, Matrix6x5Trait, MatrixIndex, MatrixMul, MatrixTrMul, Norm,
    OneNorm, RowVector1, RowVector2, RowVector2AngleTrait, RowVector2Trait, RowVector3,
    RowVector3AngleTrait, RowVector3Trait, RowVector4, RowVector4AngleTrait, RowVector4Trait,
    RowVector5, RowVector5AngleTrait, RowVector5Trait, RowVector6, RowVector6AngleTrait,
    RowVector6Trait, UniformNorm, UnitVector1, UnitVector1AngleTrait, UnitVector1Trait, UnitVector2,
    UnitVector2AngleTrait, UnitVector2Trait, UnitVector3, UnitVector3AngleTrait, UnitVector3Trait,
    UnitVector4, UnitVector4AngleTrait, UnitVector4Trait, UnitVector5, UnitVector5AngleTrait,
    UnitVector5Trait, UnitVector6, UnitVector6AngleTrait, UnitVector6Trait, Vector1, Vector2,
    Vector2AngleTrait, Vector2Trait, Vector3, Vector3AngleTrait, Vector3Trait, Vector4,
    Vector4AngleTrait, Vector4Trait, Vector5, Vector5AngleTrait, Vector5Trait, Vector6,
    Vector6AngleTrait, Vector6Trait,
};
// shapegen: end
pub use base::{Normed, Point2, Point3, Unit, UnitTrait};
pub use geometry::{
    AbstractRotation, Isometry2, Isometry2AngleTrait, Isometry2Trait, Isometry3,
    Isometry3AngleTrait, Isometry3Trait, IsometryMatrix2, IsometryMatrix2AngleTrait,
    IsometryMatrix2Trait, IsometryMatrix3, IsometryMatrix3AngleTrait, IsometryMatrix3Trait, Point1,
    Point1Trait, Point2ExtTrait, Point2Index, Point2PartialOrd, Point3ExtTrait, Point3Index,
    Point3PartialOrd, Point4, Point4Trait, Point5, Point5Trait, Point6, Point6Trait, Quaternion,
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
    SimilarityMatrix3AngleTrait, SimilarityMatrix3Trait, Translation1, Translation1Trait,
    Translation2, Translation2Trait, Translation3, Translation3Trait, Translation4,
    Translation4Trait, Translation5, Translation5Trait, Translation6, Translation6Trait,
    UnitComplex, UnitComplexAngleTrait, UnitComplexTrait, UnitQuaternion, UnitQuaternionAngleTrait,
    UnitQuaternionTrait,
};
pub use linalg::{
    Cholesky2, Cholesky2Trait, Cholesky3, Cholesky3Trait, Cholesky4, Cholesky4Trait, Cholesky6,
    Cholesky6Trait, Lu2, Lu2Trait, Lu3, Lu3Trait, Lu4, Lu4Trait, Lu6, Lu6Trait, Matrix2LuTrait,
    Matrix2QrTrait, Matrix2SvdTrait, Matrix3LuTrait, Matrix3QrTrait, Matrix3SvdTrait,
    Matrix4LuTrait, Matrix4QrTrait, Matrix6LuTrait, Perm2, Perm2Trait, Perm3, Perm3Trait, Perm4,
    Perm4Trait, Perm6, Perm6Trait, Qr2, Qr2Trait, Qr3, Qr3Trait, Qr4, Qr4Trait, Svd2, Svd2Trait,
    Svd3, Svd3Trait, SymmetricEigen2, SymmetricEigen2Trait, SymmetricEigen3, SymmetricEigen3Trait,
};
