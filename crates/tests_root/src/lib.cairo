//! Package `nalgebra_tests_root` (WP 8.6-P21): tests and gas benchmarks, through the public API,
//! of upstream's crate-root functions (`nalgebra::distance`, `convert`, `partial_cmp`...), of the
//! construction macros (`matrix!`, `vector!`, `point!`, `dmatrix!`, `dvector!`, `stack!`) and of
//! the `core::iter::Sum` / `Product` impls of the static shapes and of `DMatrix` / `DVector`.
//!
//! The package enables no Cairo experimental feature: it is also the measurement that a
//! dependent needs none to call the macros or `iter.sum()` (README, "Features").

#[cfg(test)]
mod benches;
#[cfg(test)]
mod functions;
#[cfg(test)]
mod iter;
#[cfg(test)]
mod macros;
