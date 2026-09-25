//! Package `nalgebra_tests_geometry_projections` (WP 8.4-P11b): the tests and gas benchmarks of
//! `Perspective3` and `Orthographic3` (construction, accessors, setters, projections, inverse,
//! conversions, `Matrix4::new_perspective` / `new_orthographic`), through the public API of
//! `nalgebra`. A package of its own so that the other geometry test packages stay under their
//! compile budget (AGENTS.md); not published.
//!
//! `oracle.cairo` is emitted from `tools/oracle` (committed vectors, at most 4 cases per
//! distribution):
//!
//! ```text
//! cargo run --release -- emit-cairo projections --from vectors --max-per-dist 4 \
//!     --out crates/tests_geometry_projections/src/oracle.cairo
//! ```

#[cfg(test)]
mod common;
#[cfg(test)]
mod oracle;
#[cfg(test)]
mod orthographic3;
#[cfg(test)]
mod perspective3;
