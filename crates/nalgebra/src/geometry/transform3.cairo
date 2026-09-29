//! `Transform3`: a general transformation: any homogeneous matrix, possibly singular. `try_inverse`
//! only (no `inverse`), and `transform_point` / `transform_vector` divide by the homogeneous
//! coordinate. Upstream: `nalgebra::Transform3`, i.e. `Transform<T, TGeneral, 3>`
//! (`geometry/transform*.rs`), WP 8.4-P11a. The design of the six transform types (one Cairo
//! struct per upstream alias, upstream's category rules) is in `transform.cairo`.
//!
//! - `Transform3Trait` / `Transform3Impl`: construction, accessors, inverse, transforms, the
//! products by the
//!   other geometry types (`mul_<rhs>` / `div_<rhs>`), comparisons;
//! - the operator / conversion impls: `*` with a `Transform3`, `Default`, `One`, `Index<(usize,
//! usize)>`,
//!   `Into` / `TryInto` (upstream's `SubsetOf` / `From`);
//! - the category-changing products are `TransformMul` / `TransformDiv`, `set_category` is
//!   `TransformSetCategory` (`transform.cairo`).
//!
//! Numeric contract (AGENTS.md): every sum of products is one fused `Real` kernel (one floor per
//! output scalar), every quotient correctly rounded; overflow panics.
pub use nalgebra_transform3::geometry::transform3::{
    Matrix4FromTransform3, Transform3, Transform3Default, Transform3FromAffine3,
    Transform3FromIsometry3, Transform3FromIsometryMatrix3, Transform3FromProjective3,
    Transform3FromRotation3, Transform3FromScale3, Transform3FromSimilarity3,
    Transform3FromSimilarityMatrix3, Transform3FromTranslation3, Transform3FromUnitDualQuaternion,
    Transform3FromUnitQuaternion, Transform3Impl, Transform3Index, Transform3Mul, Transform3One,
    Transform3Trait, Transform3TryFromMatrix4,
};
