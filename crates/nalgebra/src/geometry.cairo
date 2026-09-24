//! Rotations, translations and rigid-body transformations (upstream `nalgebra::geometry`).

pub mod abstract_rotation;
pub mod isometry2;
pub mod isometry3;
pub mod point;
pub mod point1;
pub mod point4;
pub mod point5;
pub mod point6;
pub mod quaternion;
pub mod rotation2;
pub mod rotation3;
pub mod similarity2;
pub mod similarity3;
pub mod translation1;
pub mod translation2;
pub mod translation3;
pub mod translation4;
pub mod translation5;
pub mod translation6;
pub mod unit_complex;
pub mod unit_quaternion;
pub use abstract_rotation::AbstractRotation;

pub use isometry2::{Isometry2, Isometry2AngleTrait, Isometry2Trait};
pub use isometry3::{Isometry3, Isometry3AngleTrait, Isometry3Trait};
pub use point::{
    Point2ExtTrait, Point2Index, Point2PartialOrd, Point3ExtTrait, Point3Index, Point3PartialOrd,
};
pub use point1::{Point1, Point1Trait};
pub use point4::{Point4, Point4Trait};
pub use point5::{Point5, Point5Trait};
pub use point6::{Point6, Point6Trait};
pub use quaternion::{Quaternion, QuaternionTrait, QuaternionTranscendentalTrait};
pub use rotation2::{Rotation2, Rotation2AngleTrait, Rotation2Trait};
pub use rotation3::{Rotation3, Rotation3AngleTrait, Rotation3Trait};
pub use similarity2::{Similarity2, Similarity2AngleTrait, Similarity2Trait};
pub use similarity3::{Similarity3, Similarity3AngleTrait, Similarity3Trait};
pub use translation1::{Translation1, Translation1Trait};
pub use translation2::{Translation2, Translation2Trait};
pub use translation3::{Translation3, Translation3Trait};
pub use translation4::{Translation4, Translation4Trait};
pub use translation5::{Translation5, Translation5Trait};
pub use translation6::{Translation6, Translation6Trait};
pub use unit_complex::{UnitComplex, UnitComplexAngleTrait, UnitComplexTrait};
pub use unit_quaternion::{UnitQuaternion, UnitQuaternionAngleTrait, UnitQuaternionTrait};
