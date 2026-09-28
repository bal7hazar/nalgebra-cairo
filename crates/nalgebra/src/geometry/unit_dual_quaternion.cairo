//! `UnitDualQuaternion`: a 3D rigid-body transform as a dual quaternion of unit norm (upstream
//! `nalgebra::UnitDualQuaternion`, WP 8.4-P12).
//!
//! `real` is the rotation (a unit quaternion) and `dual = t · real / 2` encodes the translation
//! `t`: the same transform as `Isometry3 { rotation: real, translation: t }` (`to_isometry` /
//! `from_isometry` convert between the two).
//!
//! - `UnitDualQuaternionTrait` / `UnitDualQuaternionImpl`: construction (`from_parts`,
//!   `from_isometry`, `from_rotation`), the `Unit` wrapper methods (`new_unchecked`,
//!   `new_normalize`, `try_new`, `into_inner`, `renormalize*`), conjugate / inverse, the parts
//!   (`rotation`, `translation`, `to_isometry`, `to_homogeneous`), the transforms, `lerp` /
//!   `nlerp`, the approximate comparisons and the products with `DualQuaternion`,
//!   `UnitQuaternion`, `Translation3` and `Isometry3` (`mul_<rhs>` / `div_<rhs>`: Cairo's
//!   `Mul` / `Div` are homogeneous) — everything algebraic, for any `simba::scalar::Real`;
//! - `UnitDualQuaternionAngleTrait` / `UnitDualQuaternionAngleImpl`: the screw-linear
//!   interpolation `sclerp` / `try_sclerp`, which additionally needs
//!   `simba::scalar::Transcendental`;
//! - `UnitQuaternionDualQuaternionTrait`, `Translation3DualQuaternionTrait`,
//!   `Isometry3DualQuaternionTrait`: upstream's `q * dq`, `t * dq`, `iso * dq` and their
//!   divisions, as methods of the left operand;
//! - `*`, `/`, unary `-`, `Default`, `One` and the conversions (`Isometry3` / `Matrix4` /
//!   `Similarity3` from a unit dual quaternion; a unit dual quaternion from an `Isometry3`, a
//!   `UnitQuaternion`, a `Rotation3` or a `Translation3`): their impls live in this module.
//!
//! Upstream this type is `Unit<DualQuaternion<T>>`; here it is a dedicated wrapper struct with the
//! same invariant (like `UnitQuaternion`), so that its operators live in this module. The
//! invariant `|real| = 1`, `real · dual* + dual · real* = 0` is a CONTRACT: `new_unchecked` does
//! not check it, and the dual quaternions produced here satisfy it within a few ulp, not exactly.
//! `(r, d)` and `(-r, -d)` are the same transform; no operation normalises the sign except
//! `sclerp` (shortest path), exactly like upstream.
//!
//! Numeric contract (AGENTS.md): every sum of products goes through a fused `Real` kernel (one
//! floor rounding and one overflow check per output scalar); nothing wraps silently. The halving
//! of `from_parts` (`dual = t · real / 2`) and the doubling of `translation` (`t = 2 · dual ·
//! real*`) are folded into their accumulations (`Real::wide_mul_scalar`), so each output component
//! is floored ONCE for the whole expression.

pub use nalgebra_static3::geometry::unit_dual_quaternion::*;
