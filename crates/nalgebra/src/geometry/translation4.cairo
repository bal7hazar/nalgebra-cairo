//! `Translation4`: a 4-dimensional translation (upstream `nalgebra::Translation4`, i.e.
//! `Translation<T, 4>`), WP 8.4-P09a.
//!
//! The same API as `Translation2` / `Translation3` with the WP 8.4-P09a completion, written from
//! one template for the sizes 1, 4, 5 and 6 (the vector is the matching shape `Vector4`). Every
//! operation is an exact addition, subtraction or negation: nothing rounds; overflow panics.
pub use nalgebra_geometry4::geometry::translation4::*;
pub use nalgebra_types4::geometry::translation4::*;
