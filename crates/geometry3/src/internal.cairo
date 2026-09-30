//! Internal, no stability promise: the items that `nalgebra` 0.1.0 keeps crate-private but that
//! the packages above `nalgebra_geometry3` use once the library is split (docs/SPLIT.md §12.3).
//! Each module mirrors the module of the same path in `nalgebra_geometry3` (`internal::base::x`
//! holds the crate-private items of `base::x`). They may change or disappear in any release; the
//! facade `nalgebra` never re-exports them.

pub mod base;
pub mod geometry;
