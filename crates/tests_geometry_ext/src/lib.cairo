//! Package `nalgebra_tests_geometry_ext` (WP 8.4-P09a): the tests and gas benchmarks of the
//! completion of `Rotation2` / `Rotation3`, of the point types (`Point1` .. `Point6`), of the
//! translation types (`Translation1` .. `Translation6`) and of `AbstractRotation`, through the
//! public API of `nalgebra`. A package of its own so that `nalgebra_tests_geometry` stays under
//! its compile budget (AGENTS.md); not published.
//!
//! `oracle.cairo` is emitted from `tools/oracle` (committed vectors, at most 8 cases per
//! distribution):
//!
//! ```text
//! cargo run --release -- emit-cairo rotation_matrix_completion --from vectors --max-per-dist 8 \
//!     --out crates/tests_geometry_ext/src/oracle.cairo
//! ```

#[cfg(test)]
mod abstract_rotation;
#[cfg(test)]
mod oracle;
#[cfg(test)]
mod point;
#[cfg(test)]
mod rotation2;
#[cfg(test)]
mod rotation3;
#[cfg(test)]
mod translation;
