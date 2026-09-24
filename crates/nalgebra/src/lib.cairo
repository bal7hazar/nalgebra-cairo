//! Linear algebra for provable game physics: a Cairo port of the Rust `nalgebra` crate.
//!
//! Modules mirror upstream: `base` (vectors, matrices), `geometry` (rotations, isometries),
//! `linalg` (decompositions). See `docs/DESIGN.md` and `docs/ROADMAP.md`.

pub mod base;
pub mod geometry;
pub mod linalg;

pub use base::{Matrix2, Matrix3, Matrix4, Point2, Point3, Unit, Vector2, Vector3, Vector4};
pub use base::{
    Matrix2Trait, Matrix3Trait, Matrix4Trait, Matrix6, Matrix6Trait, Vector6, Vector6Trait,
};
pub use geometry::{
    Isometry2, Isometry2AngleTrait, Isometry2Trait, Isometry3, Isometry3AngleTrait, Isometry3Trait,
    Quaternion, QuaternionTrait, Rotation2, Rotation2AngleTrait, Rotation2Trait, Rotation3,
    Rotation3AngleTrait, Rotation3Trait, Similarity2, Similarity2AngleTrait, Similarity2Trait,
    Similarity3, Similarity3AngleTrait, Similarity3Trait, Translation2, Translation2Trait,
    Translation3, Translation3Trait, UnitComplex, UnitComplexAngleTrait, UnitComplexTrait,
    UnitQuaternion, UnitQuaternionAngleTrait, UnitQuaternionTrait,
};
pub use linalg::{
    Cholesky2, Cholesky2Trait, Cholesky3, Cholesky3Trait, Cholesky4, Cholesky4Trait, Cholesky6,
    Cholesky6Trait, Lu2, Lu2Trait, Lu3, Lu3Trait, Lu4, Lu4Trait, Lu6, Lu6Trait, Matrix2LuTrait,
    Matrix2QrTrait, Matrix2SvdTrait, Matrix3LuTrait, Matrix3QrTrait, Matrix3SvdTrait,
    Matrix4LuTrait, Matrix4QrTrait, Matrix6LuTrait, Perm2, Perm2Trait, Perm3, Perm3Trait, Perm4,
    Perm4Trait, Perm6, Perm6Trait, Qr2, Qr2Trait, Qr3, Qr3Trait, Qr4, Qr4Trait, Svd2, Svd2Trait,
    Svd3, Svd3Trait, SymmetricEigen2, SymmetricEigen2Trait, SymmetricEigen3, SymmetricEigen3Trait,
};
