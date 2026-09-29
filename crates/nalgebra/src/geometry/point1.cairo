//! `Point1`: a 1-dimensional point (upstream `nalgebra::Point1`, i.e. `Point<T, 1>`), WP 8.4-P09a.
//!
//! The same API as `Point2` / `Point3` (`crate::base::point2`, which predate the move of points to
//! `geometry`) with the WP 8.4-P09a completion, written from one template for the sizes 1, 4, 5
//! and 6. The coordinates are the named fields of the matching vector shape `Matrix1`
//! (`x`), which is what `coords()` returns; see `crate::base::point3` for the table
//! of the heterogeneous operators (`sub_point`, `add_vector`, `scale`...).
//!
//! Numeric contract (AGENTS.md): every operation is exact except the divisions (`unscale`,
//! `from_homogeneous`: correctly rounded) and `lerp` (one fused kernel per coordinate); nothing
//! wraps silently.
pub use nalgebra_core::geometry::point1::*;
pub use nalgebra_geometry2::geometry::point1::*;
