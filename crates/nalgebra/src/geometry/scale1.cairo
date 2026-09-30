//! `Scale1`: a 1-dimensional non-uniform scale (upstream `nalgebra::Scale1`, i.e. `Scale<T, 1>`),
//! WP 8.4-P10.
//!
//! A scale is the vector of the factors multiplied to the coordinates of a point, one per axis (the
//! diagonal of its homogeneous matrix). Written from one template for the sizes 1 to 6; the vector
//! is the matching shape `Matrix1`. A scale acts on POINTS (`transform_point`) and on vectors
//! (`mul_vector`) alike; its product with another scale, the product with a scalar (`scale`) and
//! the inverse are component-wise.
//!
//! Numeric contract (AGENTS.md): every product is one floored fixed-point multiplication per
//! component, every inverse one correctly rounded reciprocal (`Real::recip`, to nearest, ties to
//! even, like `f64 /`); overflow panics, nothing wraps. `inverse_unchecked` is unchecked only in
//! upstream's sense (no zero test): a zero factor panics with `Fixed: division by zero`.
pub use nalgebra_geometry2::geometry::scale1::*;
