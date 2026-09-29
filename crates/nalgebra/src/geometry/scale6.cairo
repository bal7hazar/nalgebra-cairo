//! `Scale6`: a 6-dimensional non-uniform scale (upstream `nalgebra::Scale6`, i.e. `Scale<T, 6>`),
//! WP 8.4-P10.
//!
//! A scale is the vector of the factors multiplied to the coordinates of a point, one per axis (the
//! diagonal of its homogeneous matrix). Written from one template for the sizes 1 to 6; the vector
//! is the matching shape `Vector6`. A scale acts on POINTS (`transform_point`) and on vectors
//! (`mul_vector`) alike; its product with another scale, the product with a scalar (`scale`) and
//! the inverse are component-wise.
//!
//! Numeric contract (AGENTS.md): every product is one floored fixed-point multiplication per
//! component, every inverse one correctly rounded reciprocal (`Real::recip`, to nearest, ties to
//! even, like `f64 /`); overflow panics, nothing wraps. `inverse_unchecked` is unchecked only in
//! upstream's sense (no zero test): a zero factor panics with `Fixed: division by zero`.
//!
//! There is no `to_homogeneous` (nor `From<Scale6> for Matrix7`): the homogeneous matrix of a 6D
//! scale is 7x7, and the static shapes stop at 6 (DESIGN D4), like `Translation6`. Out of scope
//! for 0.1.0 by owner ruling (issue #41).

pub use nalgebra_geometry6::geometry::scale6::*;
