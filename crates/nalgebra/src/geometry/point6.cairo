//! `Point6`: a 6-dimensional point (upstream `nalgebra::Point6`, i.e. `Point<T, 6>`), WP 8.4-P09a.
//!
//! The same API as `Point2` / `Point3` (`crate::base::point2`, which predate the move of points to
//! `geometry`) with the WP 8.4-P09a completion, written from one template for the sizes 1, 4, 5
//! and 6. The coordinates are the named fields of the matching vector shape `Vector6`
//! (`x, y, z, w, a, b`), which is what `coords()` returns; see `crate::base::point3` for the table
//! of the heterogeneous operators (`sub_point`, `add_vector`, `scale`...).
//! There is no `to_homogeneous` / `from_homogeneous`: the homogeneous coordinates of a 6D point
//! form a 7-vector, and the static shapes stop at 6 (DESIGN D4). Out of scope for 0.1.0 by owner
//! ruling (issue #41).
//!
//! Numeric contract (AGENTS.md): every operation is exact except the divisions (`unscale`,
//! `from_homogeneous`: correctly rounded) and `lerp` (one fused kernel per coordinate); nothing
//! wraps silently.

pub use nalgebra_geometry6::geometry::point6::*;
