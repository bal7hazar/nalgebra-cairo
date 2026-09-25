//! Package `nalgebra_tests_geometry_scale` (WP 8.4-P10): the tests and gas benchmarks of `Scale1`
//! ..
//! `Scale6` and `Reflection1` .. `Reflection6` through the public API of `nalgebra`. A package of
//! its own so that the other geometry test packages stay under their compile budget (AGENTS.md);
//! not published.
//!
//! `oracle.cairo` is emitted from `tools/oracle` (committed vectors, at most 2 cases per
//! distribution):
//!
//! ```text
//! cargo run --release -- emit-cairo scale_reflection --from vectors --max-per-dist 2 \
//!     --out crates/tests_geometry_scale/src/oracle.cairo
//! ```

#[cfg(test)]
mod common;
#[cfg(test)]
mod oracle;
#[cfg(test)]
mod reflection;
#[cfg(test)]
mod scale;
