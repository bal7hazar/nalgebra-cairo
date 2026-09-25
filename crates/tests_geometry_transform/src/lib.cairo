//! Package `nalgebra_tests_geometry_transform` (WP 8.4-P11a): the tests and gas benchmarks of
//! `Transform2/3`, `Projective2/3` and `Affine2/3` (inverses, transforms, the products with every
//! geometry type, the category rules and conversions, the standard traits) and of
//! `Perspective3` / `Orthographic3::to_projective`, through the public API of `nalgebra`. A
//! package of its own so that the other geometry test packages stay under their compile budget
//! (AGENTS.md); not published.
//!
//! `oracle.cairo` is emitted from `tools/oracle` (committed vectors, at most 4 cases per
//! distribution):
//!
//! ```text
//! cargo run --release -- emit-cairo transform --from vectors --max-per-dist 4 \
//!     --out crates/tests_geometry_transform/src/oracle.cairo
//! ```

#[cfg(test)]
mod affine;
#[cfg(test)]
mod common;
#[cfg(test)]
mod operators;
#[cfg(test)]
mod oracle;
#[cfg(test)]
mod projective;
