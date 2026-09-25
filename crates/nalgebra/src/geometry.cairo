//! Rotations, translations and rigid-body transformations (upstream `nalgebra::geometry`).

pub mod abstract_rotation;
pub mod dual_quaternion;
pub mod isometry2;
pub mod isometry3;
pub mod isometry_matrix2;
pub mod isometry_matrix3;
pub mod orthographic3;
pub mod perspective3;
pub mod point;
pub mod point1;
pub mod point4;
pub mod point5;
pub mod point6;
pub mod quaternion;
pub mod reflection1;
pub mod reflection2;
pub mod reflection3;
pub mod reflection4;
pub mod reflection5;
pub mod reflection6;
pub mod rotation2;
pub mod rotation3;
pub mod scale1;
pub mod scale2;
pub mod scale3;
pub mod scale4;
pub mod scale5;
pub mod scale6;
pub mod similarity2;
pub mod similarity3;
pub mod similarity_matrix2;
pub mod similarity_matrix3;
pub mod translation1;
pub mod translation2;
pub mod translation3;
pub mod translation4;
pub mod translation5;
pub mod translation6;
pub mod unit_complex;
pub mod unit_dual_quaternion;
pub mod unit_quaternion;
pub use abstract_rotation::AbstractRotation;
pub use dual_quaternion::{DualQuaternion, DualQuaternionTrait};

pub use isometry2::{Isometry2, Isometry2AngleTrait, Isometry2Trait};
pub use isometry3::{Isometry3, Isometry3AngleTrait, Isometry3Trait};
pub use isometry_matrix2::{IsometryMatrix2, IsometryMatrix2AngleTrait, IsometryMatrix2Trait};
pub use isometry_matrix3::{IsometryMatrix3, IsometryMatrix3AngleTrait, IsometryMatrix3Trait};
pub use orthographic3::{
    Matrix4OrthographicTrait, Orthographic3, Orthographic3AngleTrait, Orthographic3Trait,
};
pub use perspective3::{
    Matrix4PerspectiveTrait, Perspective3, Perspective3AngleTrait, Perspective3Trait,
};
pub use point::{
    Point2ExtTrait, Point2Index, Point2PartialOrd, Point3ExtTrait, Point3Index, Point3PartialOrd,
};
pub use point1::{Point1, Point1Trait};
pub use point4::{Point4, Point4Trait};
pub use point5::{Point5, Point5Trait};
pub use point6::{Point6, Point6Trait};
pub use quaternion::{Quaternion, QuaternionTrait, QuaternionTranscendentalTrait};
pub use reflection1::{Reflection1, Reflection1Columns, Reflection1Rows, Reflection1Trait};
pub use reflection2::{Reflection2, Reflection2Columns, Reflection2Rows, Reflection2Trait};
pub use reflection3::{Reflection3, Reflection3Columns, Reflection3Rows, Reflection3Trait};
pub use reflection4::{Reflection4, Reflection4Columns, Reflection4Rows, Reflection4Trait};
pub use reflection5::{Reflection5, Reflection5Columns, Reflection5Rows, Reflection5Trait};
pub use reflection6::{Reflection6, Reflection6Columns, Reflection6Rows, Reflection6Trait};
pub use rotation2::{Rotation2, Rotation2AngleTrait, Rotation2Trait};
pub use rotation3::{Rotation3, Rotation3AngleTrait, Rotation3Trait};
pub use scale1::{Scale1, Scale1Trait};
pub use scale2::{Scale2, Scale2Trait};
pub use scale3::{Scale3, Scale3Trait};
pub use scale4::{Scale4, Scale4Trait};
pub use scale5::{Scale5, Scale5Trait};
pub use scale6::{Scale6, Scale6Trait};
pub use similarity2::{Similarity2, Similarity2AngleTrait, Similarity2Trait};
pub use similarity3::{Similarity3, Similarity3AngleTrait, Similarity3Trait};
pub use similarity_matrix2::{
    SimilarityMatrix2, SimilarityMatrix2AngleTrait, SimilarityMatrix2Trait,
};
pub use similarity_matrix3::{
    SimilarityMatrix3, SimilarityMatrix3AngleTrait, SimilarityMatrix3Trait,
};
pub use translation1::{Translation1, Translation1Trait};
pub use translation2::{Translation2, Translation2Trait};
pub use translation3::{Translation3, Translation3Trait};
pub use translation4::{Translation4, Translation4Trait};
pub use translation5::{Translation5, Translation5Trait};
pub use translation6::{Translation6, Translation6Trait};
pub use unit_complex::{UnitComplex, UnitComplexAngleTrait, UnitComplexTrait};
pub use unit_dual_quaternion::{
    Isometry3DualQuaternionTrait, Translation3DualQuaternionTrait, UnitDualQuaternion,
    UnitDualQuaternionAngleTrait, UnitDualQuaternionTrait, UnitQuaternionDualQuaternionTrait,
};
pub use unit_quaternion::{UnitQuaternion, UnitQuaternionAngleTrait, UnitQuaternionTrait};
