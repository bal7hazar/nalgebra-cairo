//! `Translation6`: a 6-dimensional translation (upstream `nalgebra::Translation6`, i.e.
//! `Translation<T, 6>`), WP 8.4-P09a.
//!
//! The same API as `Translation2` / `Translation3` with the WP 8.4-P09a completion, written from
//! one template for the sizes 1, 4, 5 and 6 (the vector is the matching shape `Vector6`). Every
//! operation is an exact addition, subtraction or negation: nothing rounds; overflow panics.
//!
//! There is no `to_homogeneous`: the homogeneous matrix of a 6D translation is 7x7, and the static
//! shapes stop at 6 (DESIGN D4). Out of scope for 0.1.0 by owner ruling (issue #41).

pub use nalgebra_geometry6::geometry::translation6::*;
