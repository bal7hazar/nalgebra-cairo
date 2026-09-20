//! Rotations, translations and rigid-body transformations (upstream `nalgebra::geometry`).

pub mod quaternion;
pub mod rotation2;
pub mod rotation3;
pub mod unit_complex;
pub mod unit_quaternion;

pub use quaternion::{Quaternion, QuaternionTrait};
pub use rotation2::{Rotation2, Rotation2AngleTrait, Rotation2Trait};
pub use rotation3::{Rotation3, Rotation3AngleTrait, Rotation3Trait};
pub use unit_complex::{UnitComplex, UnitComplexAngleTrait, UnitComplexTrait};
pub use unit_quaternion::{UnitQuaternion, UnitQuaternionAngleTrait, UnitQuaternionTrait};
