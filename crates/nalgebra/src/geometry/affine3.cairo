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

pub use nalgebra_geometry4::geometry::affine3::*;
