//! `Projective2`: an invertible homogeneous matrix. `inverse` exists, and `transform_point` /
//! `transform_vector` divide by the homogeneous coordinate. Upstream: `nalgebra::Projective2`, i.e.
//! `Transform<T, TProjective, 2>`
//! (`geometry/transform*.rs`), WP 8.4-P11a. The design of the six transform types (one Cairo
//! struct per upstream alias, upstream's category rules) is in `transform.cairo`.
//!
//! - `Projective2Trait` / `Projective2Impl`: construction, accessors, inverse, transforms, the
//! products by the
//!   other geometry types (`mul_<rhs>` / `div_<rhs>`), comparisons;
//! - the operator / conversion impls: `*` / `/` with a `Projective2`, `Default`, `One`,
//! `Index<(usize, usize)>`,
//!   `Into` / `TryInto` (upstream's `SubsetOf` / `From`);
//! - the category-changing products are `TransformMul` / `TransformDiv`, `set_category` is
//!   `TransformSetCategory` (`transform.cairo`).
//!
//! Numeric contract (AGENTS.md): every sum of products is one fused `Real` kernel (one floor per
//! output scalar), every quotient correctly rounded; overflow panics.
pub use nalgebra_transform2::geometry::projective2::{
    Matrix3FromProjective2, Projective2, Projective2Default, Projective2Div, Projective2FromAffine2,
    Projective2FromIsometry2, Projective2FromIsometryMatrix2, Projective2FromRotation2,
    Projective2FromScale2, Projective2FromSimilarity2, Projective2FromSimilarityMatrix2,
    Projective2FromTranslation2, Projective2FromUnitComplex, Projective2Impl, Projective2Index,
    Projective2Mul, Projective2One, Projective2Trait, Projective2TryFromMatrix3,
    Projective2TryFromTransform2,
};
