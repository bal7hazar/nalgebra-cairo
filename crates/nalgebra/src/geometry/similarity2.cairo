//! `Similarity2`: a 2D uniform scaling, followed by a rotation, followed by a translation
//! (upstream `nalgebra::Similarity2`, which is `Similarity<T, UnitComplex<T>, 2>`).
//!
//! `sim * p = translation + scaling * (rotation * p)`. The scaling factor must be nonzero:
//! constructors and scaling mutators panic with `nalgebra: zero scale` on zero.
//!
//! Numeric contract: rotations go through the `UnitComplex` fused kernels; the final
//! scale-plus-translation is fused into one wide accumulation per component. This preserves
//! upstream's fixed-point-observable order (`rotate`, then `scale`, then `translate`) while
//! avoiding a checked scalar addition after the scale.

pub use nalgebra_static3::geometry::similarity2::*;
