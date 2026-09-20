//! Linear algebra for provable game physics: a Cairo port of the Rust `nalgebra` crate.
//!
//! Modules mirror upstream: `base` (vectors, matrices), `geometry` (rotations, isometries),
//! `linalg` (decompositions). See `docs/DESIGN.md` and `docs/ROADMAP.md`.

pub mod base;
pub mod linalg;

pub use base::{Matrix2, Matrix3, Matrix4, Point2, Point3, Unit, Vector2, Vector3, Vector4};
pub use base::{
    Matrix2Trait, Matrix3Trait, Matrix4Trait, Matrix6, Matrix6Trait, SymMatrix2, SymMatrix2Trait,
    SymMatrix3, SymMatrix3Trait, Vector6, Vector6Trait,
};
pub use linalg::{SymmetricEigen2, SymmetricEigen2Trait, SymmetricEigen3, SymmetricEigen3Trait};
