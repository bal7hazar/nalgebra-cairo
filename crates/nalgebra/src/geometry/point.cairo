//! Points (upstream `nalgebra::Point`, `geometry/point*.rs`): the panic messages shared by every
//! point type, and the WP 8.4-P09a completion of `Point2` / `Point3`.
//!
//! `Point2` / `Point3` live in `crate::base` (they predate the geometry module); `Point1`,
//! `Point4`, `Point5` and `Point6` live in `geometry::point1` ... `geometry::point6`, with the
//! whole API in one trait each. The completion of `Point2` / `Point3` is here, in
//! `Point2ExtTrait` / `Point3ExtTrait`: the approximate comparisons (`relative_eq`, `ulps_eq`),
//! `cast`, `from_slice`, `len` / `is_empty` / `stride`, `min_value` / `max_value` (upstream's
//! `Bounded`), and the impls `p[i]` (`Index<usize>`) and the component-wise partial order
//! (`PartialOrd`). Cairo finds an impl of a core trait in the module of its type or through an
//! import, so `Point2Index` / `Point2PartialOrd` (and the `Point3` ones) must be imported where
//! they are used.

pub use nalgebra_core::geometry::point::*;
pub use nalgebra_geometry2::geometry::point::*;
pub use nalgebra_geometry3::geometry::point::*;
pub use nalgebra_types2::base::point2::{Point2Index, Point2PartialOrd};
pub use nalgebra_types3::base::point3::{Point3Index, Point3PartialOrd};
