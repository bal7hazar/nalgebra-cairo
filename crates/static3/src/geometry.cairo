//! Rotations, translations and rigid-body transformations of dimension 2 and 3 (upstream
//! `nalgebra::geometry`; the facade `nalgebra` re-exports them at the same paths).

pub mod abstract_rotation;
pub mod dual_quaternion;
pub mod isometry2;
pub mod isometry3;
pub mod isometry_matrix2;
pub mod isometry_matrix3;
pub mod point;
pub mod quaternion;
pub mod rotation2;
pub mod rotation3;
pub mod similarity2;
pub mod similarity3;
pub mod similarity_matrix2;
pub mod similarity_matrix3;
pub mod translation2;
pub mod translation3;
pub mod unit_complex;
pub mod unit_dual_quaternion;
pub mod unit_quaternion;

// the generated shapes name `crate::geometry::Rotation2` / `Rotation3` (0.1.0's re-exports)
pub use rotation2::Rotation2;
pub use rotation3::Rotation3;
