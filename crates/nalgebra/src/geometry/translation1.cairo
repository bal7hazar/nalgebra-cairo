//! `Translation1`: a 1-dimensional translation (upstream `nalgebra::Translation1`, i.e.
//! `Translation<T, 1>`), WP 8.4-P09a.
//!
//! The same API as `Translation2` / `Translation3` with the WP 8.4-P09a completion, written from
//! one template for the sizes 1, 4, 5 and 6 (the vector is the matching shape `Matrix1`). Every
//! operation is an exact addition, subtraction or negation: nothing rounds; overflow panics.
pub use nalgebra_core::geometry::translation1::*;
pub use nalgebra_geometry2::geometry::translation1::*;
