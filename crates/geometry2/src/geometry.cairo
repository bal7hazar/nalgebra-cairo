pub mod isometry2;
pub mod isometry_matrix2;
pub mod point;
pub mod point1;
pub mod rotation2;
pub mod scale1;
pub mod scale2;
pub mod similarity2;
pub mod similarity_matrix2;
pub mod translation1;
pub mod translation2;
pub mod unit_complex;

// the generated `cg` names `crate::geometry::Rotation2Trait` & co. (0.1.0's re-exports)
pub use rotation2::{Rotation2AngleTrait, Rotation2Trait};
