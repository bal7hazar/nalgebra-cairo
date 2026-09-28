//! `Isometry3`: a 3D rigid-body transform, a rotation followed by a translation (upstream
//! `nalgebra::Isometry3`, which is `Isometry<T, UnitQuaternion<T>, 3>`).
//!
//! `iso * p = rotation · p + translation`. This is THE pose type of a physics engine: every body
//! position, collider placement and contact frame is an isometry, and `inv_mul` / `transform_*` /
//! `inverse_transform_*` are the four calls rapier makes in its inner loops (docs/research/01,
//! §3).
//!
//! - `Isometry3Trait` / `Isometry3Impl`: construction from parts, composition, inverse, `inv_mul`,
//!   transforms, the in-place `append_*_mut`, the operator forms `mul_translation` /
//!   `mul_unit_quaternion` (Cairo's `Mul` is homogeneous), `to_homogeneous` and the observer
//!   frames — everything that is algebraic, hence available for any `simba::scalar::Real` scalar
//!   (the fused kernels, the renormalisation of the rotation part and the trigonometry-free
//!   interpolation are crate-internal, WP 8.0);
//! - `Isometry3AngleTrait` / `Isometry3AngleImpl`: the constructors that take a rotation VECTOR
//!   (`new`, `rotation`) and the spherical interpolation (`lerp_slerp`, `try_lerp_slerp`), which
//!   additionally need `simba::scalar::Transcendental`;
//! - `a * b` (composition), `a / b`, `*=` / `/=`, `Default`, `One` and the conversions from a
//!   `Translation3`, a vector, a point or an array (and into a `Similarity3`): their impls live in
//!   this module, where the compiler finds them without any import. The rotation-MATRIX instance
//!   of upstream's generic `Isometry` is `IsometryMatrix3` (`.into()` converts between the two).
//!
//! **Representation of the rotation.** The quaternion form is the right one for a pose that is
//! composed and renormalised every step: composition costs 11 860 gas against 23 310 for a
//! `Rotation3`, and rotating ONE vector costs 23 230 against 23 530 + 6 650 through the matrix.
//! The break-even measured in `unit_quaternion` is TWO vectors: transforming ONE point costs
//! 24 430 gas here against 35 540 through `to_rotation_matrix` + `Matrix3 · Vector3`
//! (`bench_isometry3_transform_point__alt_rotation_matrix`), but the matrix is then free for the
//! next points. A body that transforms two points or more with the same pose per step should
//! build its `Rotation3` once (`iso.rotation.to_rotation_matrix()`, 23 530) or go through
//! `to_homogeneous` (23 830); this type never materialises a matrix internally.
//!
//! Accuracy: every "rotate then translate" goes through the fused `rotate_translate` kernel (one
//! rounding for the whole `w·t + u×t + v + translation`). It gives the same bits as rotating and
//! adding afterwards — `floor(x + t) = floor(x) + t` for an integral `t` in raw units — for 5 %
//! less gas, since the addition of a `Fixed` pays an overflow check the accumulator does not
//! (`bench_isometry3_transform_point__alt_rotate_then_add`,
//! `test_transform_point_fused_and_composed_agree_bit_for_bit`).
//!
//! Numeric contract (AGENTS.md): every sum of products goes through a fused `Real` kernel (one
//! floor rounding and one overflow check per output scalar); nothing wraps silently.

#[cfg(test)]
mod benches;
#[cfg(test)]
mod tests;
// an explicit list: the conversions and products of the shapes into this module's type (0.1.0:
// `base::matrix*`, `base::row_vector*`) sit in this module of `nalgebra_static3`, the module of
// their geometry type (docs/SPLIT.md §3.2, §12.6)
pub use nalgebra_static3::geometry::isometry3::{
    Isometry3, Isometry3AngleImpl, Isometry3AngleTrait, Isometry3Default, Isometry3Div,
    Isometry3DivAssign, Isometry3FromArray, Isometry3FromPoint3, Isometry3FromTranslation,
    Isometry3FromVector3, Isometry3Impl, Isometry3Mul, Isometry3MulAssign,
    Isometry3MulAssignTranslation3, Isometry3One, Isometry3Trait, Similarity3FromIsometry3,
};
// the crate-private helpers the in-crate tests use (`nalgebra_static3::internal`)
#[cfg(test)]
use nalgebra_static3::internal::geometry::isometry3::{Isometry3InternalTrait};
