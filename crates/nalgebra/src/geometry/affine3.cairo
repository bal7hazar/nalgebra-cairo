//! `Affine3`: an invertible homogeneous matrix whose last row is `(0, .., 0, 1)`. `inverse` exists
//! (by blocks), and `transform_point` / `transform_vector` ignore the last row (no normalizer).
//! Upstream: `nalgebra::Affine3`, i.e. `Transform<T, TAffine, 3>`
//! (`geometry/transform*.rs`), WP 8.4-P11a. The design of the six transform types (one Cairo
//! struct per upstream alias, upstream's category rules) is in `transform.cairo`.
//!
//! - `Affine3Trait` / `Affine3Impl`: construction, accessors, inverse, transforms, the products by
//! the
//!   other geometry types (`mul_<rhs>` / `div_<rhs>`), comparisons;
//! - the operator / conversion impls: `*` / `/` with a `Affine3`, `Default`, `One`, `Index<(usize,
//! usize)>`,
//!   `Into` / `TryInto` (upstream's `SubsetOf` / `From`);
//! - the category-changing products are `TransformMul` / `TransformDiv`, `set_category` is
//!   `TransformSetCategory` (`transform.cairo`).
//!
//! Numeric contract (AGENTS.md): every sum of products is one fused `Real` kernel (one floor per
//! output scalar), every quotient correctly rounded; overflow panics.
pub use nalgebra_transform3::geometry::affine3::{
    Affine3, Affine3Default, Affine3Div, Affine3FromIsometry3, Affine3FromIsometryMatrix3,
    Affine3FromRotation3, Affine3FromScale3, Affine3FromSimilarity3, Affine3FromSimilarityMatrix3,
    Affine3FromTranslation3, Affine3FromUnitDualQuaternion, Affine3FromUnitQuaternion, Affine3Impl,
    Affine3Index, Affine3Mul, Affine3One, Affine3Trait, Affine3TryFromMatrix4,
    Affine3TryFromProjective3, Affine3TryFromTransform3, Matrix4FromAffine3,
};
