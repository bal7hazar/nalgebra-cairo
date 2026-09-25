//! Package `nalgebra_tests_geometry_poses` (WP 8.4-P09b): the tests and gas benchmarks of the
//! rotation-matrix poses `IsometryMatrix2/3` / `SimilarityMatrix2/3`, of the P09b completion of
//! `Isometry2/3` / `Similarity2/3` (divisions, `rotation_wrt_point`, `look_at_lh`, the mixed
//! products, the approximate comparisons, `cast`, `Default` / `One`, the conversions and the
//! compound assignments), of the `Rotation2/3` / `Translation2/3` operators whose outputs are the
//! rotation-matrix poses, and of the WP's fidelity fixes (the exponential of a real quaternion,
//! `Rotation2::renormalize`), through the public API of `nalgebra`. A package of its own so that
//! `nalgebra_tests_geometry` / `nalgebra_tests_geometry_ext` stay under their compile budget
//! (AGENTS.md); not published.
//!
//! `oracle.cairo` is emitted from `tools/oracle` (committed vectors, at most 2 cases per
//! distribution):
//!
//! ```text
//! cargo run --release -- emit-cairo pose_completion --from vectors --max-per-dist 2 \
//!     --out crates/tests_geometry_poses/src/oracle.cairo
//! ```

#[cfg(test)]
mod common;
#[cfg(test)]
mod isometry_matrix;
#[cfg(test)]
mod oracle;
#[cfg(test)]
mod poses;
#[cfg(test)]
mod similarity_matrix;
