//! `Projective3`: an invertible homogeneous matrix. `inverse` exists, and `transform_point` /
//! `transform_vector` divide by the homogeneous coordinate. Upstream: `nalgebra::Projective3`, i.e.
//! `Transform<T, TProjective, 3>`
//! (`geometry/transform*.rs`), WP 8.4-P11a. The design of the six transform types (one Cairo
//! struct per upstream alias, upstream's category rules) is in `transform.cairo`.
//!
//! - `Projective3Trait` / `Projective3Impl`: construction, accessors, inverse, transforms, the
//! products by the
//!   other geometry types (`mul_<rhs>` / `div_<rhs>`), comparisons;
//! - the operator / conversion impls: `*` / `/` with a `Projective3`, `Default`, `One`,
//! `Index<(usize, usize)>`,
//!   `Into` / `TryInto` (upstream's `SubsetOf` / `From`);
//! - the category-changing products are `TransformMul` / `TransformDiv`, `set_category` is
//!   `TransformSetCategory` (`transform.cairo`).
//!
//! Numeric contract (AGENTS.md): every sum of products is one fused `Real` kernel (one floor per
//! output scalar), every quotient correctly rounded; overflow panics.
pub use nalgebra_transform3::geometry::projective3::{
    Matrix4FromProjective3, Projective3, Projective3Default, Projective3Div, Projective3FromAffine3,
    Projective3FromIsometry3, Projective3FromIsometryMatrix3, Projective3FromRotation3,
    Projective3FromScale3, Projective3FromSimilarity3, Projective3FromSimilarityMatrix3,
    Projective3FromTranslation3, Projective3FromUnitDualQuaternion, Projective3FromUnitQuaternion,
    Projective3Impl, Projective3Index, Projective3Mul, Projective3One, Projective3Trait,
    Projective3TryFromMatrix4, Projective3TryFromTransform3,
};
