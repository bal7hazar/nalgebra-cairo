//! Rotations, translations and rigid-body transformations (upstream `nalgebra::geometry`).

pub mod isometry2;
pub mod isometry3;
pub mod quaternion;
pub mod rotation2;
pub mod rotation3;
pub mod similarity2;
pub mod similarity3;
pub mod translation2;
pub mod translation3;
pub mod unit_complex;
pub mod unit_quaternion;

pub use isometry2::{Isometry2, Isometry2AngleTrait, Isometry2Trait};
pub use isometry3::{Isometry3, Isometry3AngleTrait, Isometry3Trait};
pub use quaternion::{Quaternion, QuaternionTrait};
pub use rotation2::{Rotation2, Rotation2AngleTrait, Rotation2Trait};
pub use rotation3::{Rotation3, Rotation3AngleTrait, Rotation3Trait};
pub use similarity2::{Similarity2, Similarity2AngleTrait, Similarity2Trait};
pub use similarity3::{Similarity3, Similarity3AngleTrait, Similarity3Trait};
pub use translation2::{Translation2, Translation2Trait};
pub use translation3::{Translation3, Translation3Trait};
pub use unit_complex::{UnitComplex, UnitComplexAngleTrait, UnitComplexTrait};
pub use unit_quaternion::{UnitQuaternion, UnitQuaternionAngleTrait, UnitQuaternionTrait};
