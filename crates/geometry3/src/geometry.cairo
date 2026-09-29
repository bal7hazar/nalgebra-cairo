pub mod abstract_rotation;
pub mod dual_quaternion;
pub mod isometry3;
pub mod isometry_matrix3;
pub mod point;
pub mod quaternion;
pub mod rotation3;
pub mod scale3;
pub mod similarity3;
pub mod similarity_matrix3;
pub mod translation3;
pub mod unit_dual_quaternion;
pub mod unit_quaternion;

// the generated `cg` names `crate::geometry::Rotation3Trait` & co. (0.1.0's re-exports)
pub use isometry_matrix3::IsometryMatrix3Trait;
pub use rotation3::{Rotation3AngleTrait, Rotation3Trait};
