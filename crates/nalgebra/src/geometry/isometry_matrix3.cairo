//! `IsometryMatrix3`: a 3D rigid-body transform whose rotation is a `Rotation3` MATRIX (upstream
//! `nalgebra::IsometryMatrix3`, which is `Isometry<T, Rotation3<T>, 3>`).
//!
//! The rotation-matrix instance of upstream's generic `Isometry<T, R, D>` is a struct of its own,
//! with the method set, field names (`rotation`, `translation`) and semantics of `Isometry3`
//! (whose rotation is a `UnitQuaternion`). The two convert into each other with `.into()`
//! (`nalgebra::convert` upstream): quaternion to matrix is upstream's 24-product form, matrix to
//! quaternion is Shepperd's method (one square root, three divisions), so the round trip is
//! exact only to a few ulp.
//!
//! - `IsometryMatrix3Trait` / `IsometryMatrix3Impl`: everything algebraic (composition, inverse,
//!   transforms, the in-place `append_*_mut`, the heterogeneous operators as `mul_<rhs>` /
//!   `div_<rhs>` methods, the observer / look-at frames, homogeneous form, comparisons, `cast`),
//!   over a `Real` scalar;
//! - `IsometryMatrix3AngleTrait` / `IsometryMatrix3AngleImpl`: the constructors from a rotation
//!   vector and the spherical interpolation, which additionally need
//!   `simba::scalar::Transcendental`;
//! - the operator / conversion impls (`*`, `/`, `*=`, `/=`, `Default`, `One`, `From`).
//!
//! **The matrix form pays off when a pose transforms vectors**: `transform_point` costs 6 840 gas
//! here against 23 270 for `Isometry3` (9 products against 15 plus the quaternion's extra
//! roundings), and a composition 34 120 against 35 320; the quaternion form is the one to
//! renormalise and interpolate (`nalgebra_tests_geometry_poses` benches). Every "rotate then
//! translate" goes through the fused `rotate_translate` kernel (one rounding per component), bit
//! for bit what rotating then adding gives, for 22 % less gas
//! (`bench_isometry_matrix3_transform_point__alt_rotate_then_add`).
//!
//! Numeric contract (AGENTS.md): every sum of products goes through a fused `Real` kernel (one
//! floor rounding and one overflow check per output scalar); nothing wraps silently.
pub use nalgebra_geometry3::geometry::isometry_matrix3::*;
