//! `Similarity3`: a 3D uniform scaling, followed by a rotation, followed by a translation
//! (upstream `nalgebra::Similarity3`, which is `Similarity<T, UnitQuaternion<T>, 3>`).
//!
//! `sim * p = translation + scaling * (rotation * p)`. The scaling factor must be nonzero:
//! constructors and scaling mutators panic with `nalgebra: zero scale` on zero.
//!
//! Numeric contract: quaternion rotations go through the `UnitQuaternion` fused kernels; the final
//! scale-plus-translation is fused into one wide accumulation per component. This preserves
//! upstream's fixed-point-observable order (`rotate`, then `scale`, then `translate`) while
//! avoiding a checked scalar addition after the scale.
pub use nalgebra_geometry3::geometry::similarity3::{
    Similarity3, Similarity3AngleImpl, Similarity3AngleTrait, Similarity3Default, Similarity3Div,
    Similarity3DivAssign, Similarity3DivAssignIsometry3, Similarity3Impl, Similarity3Mul,
    Similarity3MulAssign, Similarity3MulAssignIsometry3, Similarity3MulAssignTranslation3,
    Similarity3One, Similarity3Trait, errors,
};
