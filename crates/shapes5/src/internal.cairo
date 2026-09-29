//! Internal, no stability promise: the items that `nalgebra` 0.1.0 keeps crate-private but that
//! the packages above `nalgebra_shapes5` use once the library is split (docs/SPLIT.md §12.3). Each
//! module mirrors the module of the same path in `nalgebra_shapes5` (`internal::base::matrix5`
//! holds the crate-private items of `base::matrix5`). They may change or disappear in any release;
//! the facade `nalgebra` never re-exports them.

pub mod base;

pub mod linalg;
