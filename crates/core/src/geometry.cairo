//! The geometry types of `nalgebra_core` (upstream `nalgebra::geometry`): `Point1`, `Point4`,
//! `Translation1`, `Translation4` and the shared point panic messages; the rest of the module lives
//! above (the facade `nalgebra` re-exports it whole at `nalgebra::geometry`).

pub mod point;
pub mod point1;
pub mod point4;
pub mod translation1;
pub mod translation4;

// the generated shapes name `crate::geometry::Translation1` (0.1.0's root re-export)
pub use translation1::{Translation1, Translation1Trait};
