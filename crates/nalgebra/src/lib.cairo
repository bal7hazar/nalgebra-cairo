//! Linear algebra for provable game physics: a Cairo port of the Rust `nalgebra` crate.
//!
//! Modules mirror upstream: `base` (vectors, matrices), `geometry` (rotations, isometries),
//! `linalg` (decompositions). See `docs/DESIGN.md` and `docs/ROADMAP.md`.

pub mod base;
pub mod geometry;
pub mod linalg;

// shapegen: begin
pub use base::{
    Matrix1, Matrix1Trait, Matrix1x2, Matrix1x3, Matrix1x4, Matrix1x5, Matrix1x6, Matrix2,
    Matrix2Trait, Matrix2x1, Matrix2x3, Matrix2x3Trait, Matrix2x4, Matrix2x4Trait, Matrix2x5,
    Matrix2x5Trait, Matrix2x6, Matrix2x6Trait, Matrix3, Matrix3Trait, Matrix3x1, Matrix3x2,
    Matrix3x2Trait, Matrix3x4, Matrix3x4Trait, Matrix3x5, Matrix3x5Trait, Matrix3x6, Matrix3x6Trait,
    Matrix4, Matrix4Trait, Matrix4x1, Matrix4x2, Matrix4x2Trait, Matrix4x3, Matrix4x3Trait,
    Matrix4x5, Matrix4x5Trait, Matrix4x6, Matrix4x6Trait, Matrix5, Matrix5Trait, Matrix5x1,
    Matrix5x2, Matrix5x2Trait, Matrix5x3, Matrix5x3Trait, Matrix5x4, Matrix5x4Trait, Matrix5x6,
    Matrix5x6Trait, Matrix6, Matrix6Trait, Matrix6x1, Matrix6x2, Matrix6x2Trait, Matrix6x3,
    Matrix6x3Trait, Matrix6x4, Matrix6x4Trait, Matrix6x5, Matrix6x5Trait, MatrixMul, MatrixTrMul,
    RowVector1, RowVector2, RowVector2Trait, RowVector3, RowVector3Trait, RowVector4,
    RowVector4Trait, RowVector5, RowVector5Trait, RowVector6, RowVector6Trait, UnitVector1,
    UnitVector2, UnitVector3, UnitVector4, UnitVector5, UnitVector6, Vector1, Vector2, Vector2Trait,
    Vector3, Vector3Trait, Vector4, Vector4Trait, Vector5, Vector5Trait, Vector6, Vector6Trait,
};
// shapegen: end
pub use base::{Point2, Point3, Unit};
pub use geometry::{
    AbstractRotation, Isometry2, Isometry2AngleTrait, Isometry2Trait, Isometry3,
    Isometry3AngleTrait, Isometry3Trait, IsometryMatrix2, IsometryMatrix2AngleTrait,
    IsometryMatrix2Trait, IsometryMatrix3, IsometryMatrix3AngleTrait, IsometryMatrix3Trait, Point1,
    Point1Trait, Point2ExtTrait, Point2Index, Point2PartialOrd, Point3ExtTrait, Point3Index,
    Point3PartialOrd, Point4, Point4Trait, Point5, Point5Trait, Point6, Point6Trait, Quaternion,
    QuaternionTrait, QuaternionTranscendentalTrait, Rotation2, Rotation2AngleTrait, Rotation2Trait,
    Rotation3, Rotation3AngleTrait, Rotation3Trait, Similarity2, Similarity2AngleTrait,
    Similarity2Trait, Similarity3, Similarity3AngleTrait, Similarity3Trait, SimilarityMatrix2,
    SimilarityMatrix2AngleTrait, SimilarityMatrix2Trait, SimilarityMatrix3,
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
