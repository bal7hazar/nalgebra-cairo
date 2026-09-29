//! `Translation5`: a 5-dimensional translation (upstream `nalgebra::Translation5`, i.e.
//! `Translation<T, 5>`), WP 8.4-P09a.
//!
//! The same API as `Translation2` / `Translation3` with the WP 8.4-P09a completion, written from
//! one template for the sizes 1, 4, 5 and 6 (the vector is the matching shape `Vector5`). Every
//! operation is an exact addition, subtraction or negation: nothing rounds; overflow panics.

pub use nalgebra_geometry6::geometry::translation5::{
    Translation5, Translation5Div, Translation5DivAssign, Translation5FromArray,
    Translation5FromPoint, Translation5FromVector, Translation5Impl, Translation5IntoArray,
    Translation5Mul, Translation5MulAssign, Translation5One, Translation5Trait,
};
