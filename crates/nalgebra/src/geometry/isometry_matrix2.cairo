//! `IsometryMatrix2`: a 2D rigid-body transform whose rotation is a `Rotation2` MATRIX (upstream
//! `nalgebra::IsometryMatrix2`, which is `Isometry<T, Rotation2<T>, 2>`).
//!
//! Upstream has one generic `Isometry<T, R, D>`; Cairo has no generic-rotation struct with the
//! fused kernels of each rotation type, so the rotation-matrix instance is a struct of its own with
//! the same method set as `Isometry2` (whose rotation is a `UnitComplex`), the same field names
//! (`rotation`, `translation`) and the same semantics. The two convert into each other with
//! `.into()` (`nalgebra::convert` upstream, exact both ways: the complex IS the first column).
//!
//! - `IsometryMatrix2Trait` / `IsometryMatrix2Impl`: everything algebraic (composition, inverse,
//!   transforms, the in-place `append_*_mut`, the heterogeneous operators as `mul_<rhs>` /
//!   `div_<rhs>` methods, homogeneous form, comparisons, `cast`), over a `Real` scalar;
//! - `IsometryMatrix2AngleTrait` / `IsometryMatrix2AngleImpl`: the constructors through an angle
//! and
//!   `lerp_slerp`, which additionally need `simba::scalar::Transcendental`;
//! - the operator / conversion impls (`*`, `/`, `*=`, `/=`, `Default`, `One`, `From`).
//!
//! **Prefer `Isometry2` in hot code**: a composition costs 17 440 gas here against 10 350 for
//! `Isometry2` (four fused kernels for the rotation instead of two), and the transforms cost
//! exactly the same, bit for bit (`transform_point` 4 060 either way;
//! `nalgebra_tests_geometry_poses`
//! benches). Every "rotate then translate" goes through the fused `rotate_translate` kernel (one
//! rounding per component), bit for bit what rotating then adding gives, since
//! `floor(x + t) = floor(x) + t` for an integral `t` in raw units, for 24 % less gas
//! (`bench_isometry_matrix2_transform_point__alt_rotate_then_add`).
//!
//! Numeric contract (AGENTS.md): every sum of products goes through a fused `Real` kernel (one
//! floor rounding and one overflow check per output scalar); nothing wraps silently.
pub use nalgebra_geometry2::geometry::isometry_matrix2::*;
