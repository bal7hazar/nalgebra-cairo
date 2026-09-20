//! Linear algebra for provable game physics: a Cairo port of the Rust `nalgebra` crate.
//!
//! Modules mirror upstream: `base` (vectors, matrices), `geometry` (rotations, isometries),
//! `linalg` (decompositions). See `docs/DESIGN.md` and `docs/ROADMAP.md`.

pub mod base;
pub mod geometry;
pub mod linalg;

pub use base::{Matrix2, Matrix3, Matrix4, Point2, Point3, Unit, Vector2, Vector3, Vector4};
pub use base::{
    Matrix2Trait, Matrix3Trait, Matrix4Trait, Matrix6, Matrix6Trait, SymMatrix2, SymMatrix2Trait,
    SymMatrix3, SymMatrix3Trait, Vector6, Vector6Trait,
};
pub use geometry::{
    Quaternion, QuaternionTrait, Rotation2, Rotation2AngleTrait, Rotation2Trait, Rotation3,
    Rotation3AngleTrait, Rotation3Trait, UnitComplex, UnitComplexAngleTrait, UnitComplexTrait,
    UnitQuaternion, UnitQuaternionAngleTrait, UnitQuaternionTrait,
};
pub use linalg::{
    Cholesky2, Cholesky2Trait, Cholesky3, Cholesky3Trait, Cholesky4, Cholesky4Trait, Cholesky6,
    Cholesky6Trait, Ldlt2, Ldlt2Trait, Ldlt3, Ldlt3Trait, Ldlt4, Ldlt4Trait, Ldlt6, Ldlt6Trait, Lu2,
    Lu2Trait, Lu3, Lu3Trait, Lu4, Lu4Trait, Lu6, Lu6Trait, Matrix2LuTrait, Matrix3LuTrait,
    Matrix4LuTrait, Matrix6LuTrait, Perm2, Perm3, Perm4, Perm6, PermTrait, SymmetricEigen2,
    SymmetricEigen2Trait, SymmetricEigen3, SymmetricEigen3Trait,
};
