//! The geometry of `nalgebra_shapes5` (upstream `nalgebra::geometry`): the methods of `Point4` and
//! `Translation4` (`Point4Trait`, `Translation4Trait`: they build dimension-5 types); the types are
//! in `nalgebra_core`, the rest of the module above (the facade `nalgebra` re-exports it whole at
//! `nalgebra::geometry`).

pub mod point4;
pub mod translation4;

// the generated shapes name `crate::geometry::Translation4Trait` (0.1.0's root re-export)
pub use translation4::Translation4Trait;
