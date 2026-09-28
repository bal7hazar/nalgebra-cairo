//! `Isometry2`: a 2D rigid-body transform, a rotation followed by a translation (upstream
//! `nalgebra::Isometry2`, which is `Isometry<T, UnitComplex<T>, 2>`).
//!
//! `iso * p = rotation * p + translation`. This is THE pose type of a physics engine: every body
//! position, collider placement and contact frame is an isometry (docs/research/01, §3).
//!
//! - `Isometry2Trait` / `Isometry2Impl`: construction from parts, composition, inverse, `inv_mul`,
//!   transforms, the in-place `append_*_mut`, the operator forms `mul_translation` /
//!   `mul_unit_complex` (Cairo's `Mul` is homogeneous) and `to_homogeneous` — everything that is
//!   algebraic, hence available for any `simba::scalar::Real` scalar (the fused kernels, the
//!   renormalisation of the rotation part and the trigonometry-free interpolation are
//!   crate-internal, WP 8.0);
//! - `Isometry2AngleTrait` / `Isometry2AngleImpl`: the constructors and the interpolation that go
//!   through an ANGLE (`new`, `rotation`, `lerp_slerp`), which additionally need
//!   `simba::scalar::Transcendental`;
//! - `a * b` (composition), `a / b`, `*=` / `/=`, `Default`, `One` and the conversions from a
//!   `Translation2`, a vector, a point or an array (and into a `Similarity2`): their impls live in
//!   this module, where the compiler finds them without any import. The rotation-MATRIX instance
//!   of upstream's generic `Isometry` is `IsometryMatrix2` (`.into()` converts between the two).
//!
//! The rotation is a `UnitComplex`, never a `Rotation2`: the two hold the same information, but the
//! complex form composes for 4 000 gas against 10 260 for the matrix and transforms a vector for
//! exactly the same price (see the module documentation of `rotation2`). Use
//! `to_homogeneous` when a matrix is what the consumer wants.
//!
//! Accuracy: the translation is carried EXACTLY through every operation whose rotation is the
//! identity, and the rotated parts inherit the 1-ulp floor of the `UnitComplex` kernels. Every
//! "rotate then translate" goes through the fused `rotate_translate` kernel (one rounding per
//! component). It gives the same bits as rotating and adding afterwards — `floor(x + t) =
//! floor(x) + t` for an integral `t` in raw units — for 23 % less gas, since the addition of a
//! `Fixed` pays an overflow check the accumulator does not
//! (`bench_isometry2_transform_point__alt_rotate_then_add`,
//! `test_transform_point_fused_and_composed_agree_bit_for_bit`).
//!
//! Numeric contract (AGENTS.md): every sum of products goes through a fused `Real` kernel (one
//! floor rounding and one overflow check per output scalar); nothing wraps silently.

#[cfg(test)]
mod benches;
#[cfg(test)]
mod tests;
pub use nalgebra_static3::geometry::isometry2::*;
// the crate-private helpers the in-crate tests use (`nalgebra_static3::internal`)
#[cfg(test)]
use nalgebra_static3::internal::geometry::isometry2::{Isometry2InternalTrait};
