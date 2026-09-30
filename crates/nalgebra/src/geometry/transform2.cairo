//! `Transform2`: a general transformation: any homogeneous matrix, possibly singular. `try_inverse`
//! only (no `inverse`), and `transform_point` / `transform_vector` divide by the homogeneous
//! coordinate. Upstream: `nalgebra::Transform2`, i.e. `Transform<T, TGeneral, 2>`
//! (`geometry/transform*.rs`), WP 8.4-P11a. The design of the six transform types (one Cairo
//! struct per upstream alias, upstream's category rules) is in `transform.cairo`.
//!
//! - `Transform2Trait` / `Transform2Impl`: construction, accessors, inverse, transforms, the
//! products by the
//!   other geometry types (`mul_<rhs>` / `div_<rhs>`), comparisons;
//! - the operator / conversion impls: `*` with a `Transform2`, `Default`, `One`, `Index<(usize,
//! usize)>`,
//!   `Into` / `TryInto` (upstream's `SubsetOf` / `From`);
//! - the category-changing products are `TransformMul` / `TransformDiv`, `set_category` is
//!   `TransformSetCategory` (`transform.cairo`).
//!
//! Numeric contract (AGENTS.md): every sum of products is one fused `Real` kernel (one floor per
//! output scalar), every quotient correctly rounded; overflow panics.
pub use nalgebra_transform2::geometry::transform2::{
    Matrix3FromTransform2, Transform2, Transform2Default, Transform2FromAffine2,
    Transform2FromIsometry2, Transform2FromIsometryMatrix2, Transform2FromProjective2,
    Transform2FromRotation2, Transform2FromScale2, Transform2FromSimilarity2,
    Transform2FromSimilarityMatrix2, Transform2FromTranslation2, Transform2FromUnitComplex,
    Transform2Impl, Transform2Index, Transform2Mul, Transform2One, Transform2Trait,
    Transform2TryFromMatrix3,
};
