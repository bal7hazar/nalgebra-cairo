//! Package `nalgebra_tests_geometry_dual` (WP 8.4-P12): the tests and gas benchmarks of
//! `DualQuaternion` and `UnitDualQuaternion` (construction, algebra, inverse, transforms,
//! interpolation, approximate comparisons, the products with `UnitQuaternion`, `Translation3` and
//! `Isometry3`, the conversions), through the public API of `nalgebra`. A package of its own so
//! that the other geometry test packages stay under their compile budget (AGENTS.md); not
//! published.
//!
//! `oracle.cairo` is emitted from `tools/oracle` (committed vectors, at most 2 cases per
//! distribution):
//!
//! ```text
//! cargo run --release -- emit-cairo dual_quaternion --from vectors --max-per-dist 2 \
//!     --out crates/tests_geometry_dual/src/oracle.cairo
//! ```

#[cfg(test)]
mod common;
#[cfg(test)]
mod dual_quaternion;
#[cfg(test)]
mod inplace;
#[cfg(test)]
mod oracle;
#[cfg(test)]
mod unit_dual_quaternion;
