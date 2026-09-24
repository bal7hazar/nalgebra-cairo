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
//! - `a * b` (composition) and the conversion from a `Translation3`: their impls live in this
//!   module, where the compiler finds them without any import.
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

use simba::scalar::{Real, Transcendental};
use crate::base::matrix4::Matrix4;
use crate::base::point3::Point3;
use crate::base::vector3::{Vector3, Vector3Trait};
use crate::geometry::unit_quaternion::UnitQuaternionInternalTrait;
use super::quaternion::Quaternion;
use super::rotation3::{Rotation3, Rotation3Trait};
use super::translation3::{Translation3, Translation3Trait};
use super::unit_quaternion::{UnitQuaternion, UnitQuaternionAngleTrait, UnitQuaternionTrait};

#[cfg(test)]
mod benches;
#[cfg(test)]
mod oracle;
#[cfg(test)]
mod tests;

/// A 3D direct isometry: the rotation `rotation` followed by the translation `translation`. The
/// field names and their order are upstream's (`Isometry { rotation, translation }`); the parts
/// are read through the public fields, like upstream (the trait methods `translation` and
/// `rotation` are upstream's CONSTRUCTORS, not accessors).
#[derive(Copy, Drop, PartialEq, Serde, Debug, Hash)]
pub struct Isometry3<T> {
    pub rotation: UnitQuaternion<T>,
    pub translation: Translation3<T>,
}

/// Operations of `Isometry3<T>` that need no trigonometry, over a `Real` scalar. By value,
/// unrolled, no loop.
#[generate_trait]
pub impl Isometry3Impl<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of Isometry3Trait<T> {
    // --- construction and parts ---------------------------------------------------------------

    /// The identity isometry (no rotation, no translation). Exact. Upstream:
    /// `Isometry3::identity`.
    #[inline(always)]
    fn identity() -> Isometry3<T> {
        Isometry3 {
            rotation: UnitQuaternionTrait::identity(), translation: Translation3Trait::identity(),
        }
    }

    /// The isometry `translation ∘ rotation` (rotate first, then translate). Exact (two copies).
    /// Upstream: `Isometry3::from_parts`.
    #[inline(always)]
    fn from_parts(translation: Translation3<T>, rotation: UnitQuaternion<T>) -> Isometry3<T> {
        Isometry3 { rotation, translation }
    }

    /// The pure translation by `(x, y, z)`. Exact. Upstream: `Isometry3::translation`.
    #[inline(always)]
    fn translation(x: T, y: T, z: T) -> Isometry3<T> {
        Isometry3 {
            rotation: UnitQuaternionTrait::identity(),
            translation: Translation3 { vector: Vector3 { x, y, z } },
        }
    }

    // --- inverse and composition --------------------------------------------------------------

    /// The inverse isometry: the rotation is conjugated (three negations, exact) and the
    /// translation becomes `rotation⁻¹ · (-translation)` — upstream's order, which matters in
    /// fixed point since `floor(-x) ≠ -floor(x)`. One conjugate rotation (15 products, 3
    /// roundings). Panics on overflow (`-MIN` of a translation component). Upstream: `inverse`.
    fn inverse(self: Isometry3<T>) -> Isometry3<T> {
        let v = Vector3 {
            x: -self.translation.vector.x,
            y: -self.translation.vector.y,
            z: -self.translation.vector.z,
        };
        Isometry3 {
            rotation: self.rotation.conjugate(),
            translation: Translation3 { vector: self.rotation.inverse_transform_vector(v) },
        }
    }

    /// `self⁻¹ * other`, the RELATIVE pose rapier computes for every contact and joint
    /// (docs/research/01, §3.3), WITHOUT materialising the inverse: the translation is
    /// `rotation⁻¹ · (other.translation - self.translation)` (one exact subtraction and one
    /// conjugate rotation, `UnitQuaternion::inverse_transform_vector`) and the rotation is
    /// `rotation⁻¹ · other.rotation` (`UnitQuaternion::conj_mul`: one Hamilton product with the
    /// conjugate's signs folded in, no negation, no division).
    ///
    /// Measured 40 110 gas against 61 320 for `self.inverse() * other`, 1.53x: the inverse rotates
    /// its own translation for nothing, and the composition then rotates the other one
    /// (`bench_isometry3_inv_mul__alt_inverse_then_mul`). The two agree to a few ulp but NOT bit
    /// for bit — the loser rounds the intermediate `rotation⁻¹ · (-translation)`
    /// (`test_inv_mul_alt_inverse_then_mul_differs_by_rounding`). Folding the conjugate's signs
    /// into both kernels saves 1 200 gas over the conjugate-first formulation, for the same bits
    /// (`bench_isometry3_inv_mul__alt_conjugate_then_mul`,
    /// `test_inv_mul_fused_matches_conjugate_then_mul`). Upstream: `inv_mul`.
    fn inv_mul(self: Isometry3<T>, other: Isometry3<T>) -> Isometry3<T> {
        let d = Vector3 {
            x: other.translation.vector.x - self.translation.vector.x,
            y: other.translation.vector.y - self.translation.vector.y,
            z: other.translation.vector.z - self.translation.vector.z,
        };
        Isometry3 {
            rotation: self.rotation.conj_mul(other.rotation),
            translation: Translation3 { vector: self.rotation.inverse_transform_vector(d) },
        }
    }

    // --- transforms ----------------------------------------------------------------------------

    /// `self * p = rotation · p + translation`, through `rotate_translate`: one rounding per
    /// component. Panics on overflow. Upstream: `transform_point` (`iso * p`).
    fn transform_point(self: Isometry3<T>, p: Point3<T>) -> Point3<T> {
        let c = Isometry3InternalTrait::rotate_translate(
            self.rotation, Vector3 { x: p.x, y: p.y, z: p.z }, self.translation.vector,
        );
        Point3 { x: c.x, y: c.y, z: c.z }
    }

    /// `rotation · v`: an isometry acts on a DISPLACEMENT by its rotation only (the translation
    /// cancels in `(p + v) - p`). Upstream: `transform_vector` (`iso * v`).
    #[inline(always)]
    fn transform_vector(self: Isometry3<T>, v: Vector3<T>) -> Vector3<T> {
        self.rotation.transform_vector(v)
    }

    /// `self⁻¹ * p = rotation⁻¹ · (p - translation)`: one exact subtraction, then the
    /// conjugate applied directly — no inverse isometry and no inverse rotation is ever built.
    /// Upstream:
    /// `inverse_transform_point`.
    fn inverse_transform_point(self: Isometry3<T>, p: Point3<T>) -> Point3<T> {
        let d = Point3 {
            x: p.x - self.translation.vector.x,
            y: p.y - self.translation.vector.y,
            z: p.z - self.translation.vector.z,
        };
        self.rotation.inverse_transform_point(d)
    }

    /// `rotation⁻¹ · v`: the inverse acting on a displacement, the conjugate applied directly
    /// (`UnitQuaternion::inverse_transform_vector`, as cheap as `transform_vector`: the
    /// conjugate's signs are folded into the sandwich). Upstream:
    /// `inverse_transform_vector`.
    #[inline(always)]
    fn inverse_transform_vector(self: Isometry3<T>, v: Vector3<T>) -> Vector3<T> {
        self.rotation.inverse_transform_vector(v)
    }

    // --- append (in place) and the operator forms ----------------------------------------------

    /// `Translation(t) ∘ self`: the same rotation, the translation shifted by `t` (an exact
    /// addition), in place. Upstream: `append_translation_mut`.
    #[inline(always)]
    fn append_translation_mut(ref self: Isometry3<T>, t: Translation3<T>) {
        self =
            Isometry3 {
                rotation: self.rotation,
                translation: Translation3 {
                    vector: Vector3 {
                        x: self.translation.vector.x + t.vector.x,
                        y: self.translation.vector.y + t.vector.y,
                        z: self.translation.vector.z + t.vector.z,
                    },
                },
            };
    }

    /// `self * t = self ∘ Translation(t)`: the same rotation, the translation shifted by
    /// `rotation · t` (one `rotate_translate`). Upstream: `Mul<Translation3> for Isometry3` — a
    /// named method because Cairo's `Mul` is homogeneous (a documented rename,
    /// `scripts/api_parity.py`).
    fn mul_translation(self: Isometry3<T>, t: Translation3<T>) -> Isometry3<T> {
        Isometry3 {
            rotation: self.rotation,
            translation: Translation3 {
                vector: Isometry3InternalTrait::rotate_translate(
                    self.rotation, t.vector, self.translation.vector,
                ),
            },
        }
    }

    /// `Rotation(r) ∘ self`: a rotation about the ORIGIN applied after `self`, so the translation
    /// is rotated too (`r · translation`) and the rotation becomes `r · rotation`, in place.
    /// Upstream:
    /// `append_rotation_mut`.
    fn append_rotation_mut(ref self: Isometry3<T>, r: UnitQuaternion<T>) {
        self =
            Isometry3 {
                rotation: r * self.rotation,
                translation: Translation3 { vector: r.transform_vector(self.translation.vector) },
            };
    }

    /// `self * r = self ∘ Rotation(r)`: a rotation applied BEFORE `self`, which leaves the
    /// translation untouched. One Hamilton product. Upstream: `Mul<UnitQuaternion> for Isometry3`
    /// — a named method because Cairo's `Mul` is homogeneous (a documented rename,
    /// `scripts/api_parity.py`).
    #[inline(always)]
    fn mul_unit_quaternion(self: Isometry3<T>, r: UnitQuaternion<T>) -> Isometry3<T> {
        Isometry3 { rotation: self.rotation * r, translation: self.translation }
    }

    /// The rotation `r` applied about the point `p` (which stays fixed): the translation becomes
    /// `r · (translation - p) + p`, the rotation `r · rotation`, in place. One exact subtraction,
    /// one `rotate_translate` and one Hamilton product. Upstream:
    /// `append_rotation_wrt_point_mut`.
    fn append_rotation_wrt_point_mut(ref self: Isometry3<T>, r: UnitQuaternion<T>, p: Point3<T>) {
        let d = Vector3 {
            x: self.translation.vector.x - p.x,
            y: self.translation.vector.y - p.y,
            z: self.translation.vector.z - p.z,
        };
        self =
            Isometry3 {
                rotation: r * self.rotation,
                translation: Translation3 {
                    vector: Isometry3InternalTrait::rotate_translate(
                        r, d, Vector3 { x: p.x, y: p.y, z: p.z },
                    ),
                },
            };
    }

    /// The rotation `r` applied about the isometry's own centre (the point `translation`): the
    /// translation is unchanged and the rotation becomes `r · rotation`, i.e. one Hamilton product
    /// and nothing else (`append_rotation_wrt_point_mut` with `p = translation`, where the
    /// subtraction and the addition cancel exactly). Upstream: `append_rotation_wrt_center_mut`.
    #[inline(always)]
    fn append_rotation_wrt_center_mut(ref self: Isometry3<T>, r: UnitQuaternion<T>) {
        self = Isometry3 { rotation: r * self.rotation, translation: self.translation };
    }

    // --- observer frames ------------------------------------------------------------------------

    /// The isometry placing an observer at `eye` whose local `z` axis points at `target` and whose
    /// local `y` axis is as close as possible to `up`: the translation is `eye` and the rotation is
    /// `Rotation3::face_towards(target - eye, up)` converted to a quaternion (Shepperd's method).
    ///
    /// Goes through the matrix because `Rotation3` is where `face_towards` is implemented (three
    /// normalisations and one cross product); the conversion adds one square root and three
    /// divisions. `up` MUST not be parallel to `target - eye` (the normalisation would divide by
    /// zero). Upstream: `Isometry3::face_towards` (which builds the quaternion directly from the
    /// matrix as well).
    fn face_towards(eye: Point3<T>, target: Point3<T>, up: Vector3<T>) -> Isometry3<T> {
        let dir = Vector3 { x: target.x - eye.x, y: target.y - eye.y, z: target.z - eye.z };
        let r: Rotation3<T> = Rotation3Trait::face_towards(dir, up);
        Isometry3 {
            rotation: UnitQuaternionTrait::from_rotation_matrix(r),
            translation: Translation3 { vector: Vector3 { x: eye.x, y: eye.y, z: eye.z } },
        }
    }

    /// The view transform of a right-handed look-at camera at `eye` looking at `target`: the
    /// INVERSE frame of `face_towards`, i.e. the isometry taking world coordinates to camera
    /// coordinates (`target - eye` is mapped onto the negative `z` axis). Its translation is
    /// `rotation · (-eye)`. `up` MUST not be parallel to `target - eye`. Upstream:
    /// `Isometry3::look_at_rh`.
    fn look_at_rh(eye: Point3<T>, target: Point3<T>, up: Vector3<T>) -> Isometry3<T> {
        let dir = Vector3 { x: target.x - eye.x, y: target.y - eye.y, z: target.z - eye.z };
        let r: Rotation3<T> = Rotation3Trait::look_at_rh(dir, up);
        let q: UnitQuaternion<T> = UnitQuaternionTrait::from_rotation_matrix(r);
        let neg = Vector3 { x: -eye.x, y: -eye.y, z: -eye.z };
        Isometry3 { rotation: q, translation: Translation3 { vector: q.transform_vector(neg) } }
    }

    // --- conversions, renormalisation, comparison, interpolation -------------------------------

    /// The isometry as a 4x4 homogeneous matrix: the rotation block (upstream's quaternion-to-
    /// matrix form, 24 products, entries within 2 ulp), the translation in the last column and
    /// `m44 = 1`. Upstream: `to_homogeneous`.
    fn to_homogeneous(self: Isometry3<T>) -> Matrix4<T> {
        let m = self.rotation.to_rotation_matrix().matrix;
        Matrix4 {
            m11: m.m11,
            m21: m.m21,
            m31: m.m31,
            m41: R::zero(),
            m12: m.m12,
            m22: m.m22,
            m32: m.m32,
            m42: R::zero(),
            m13: m.m13,
            m23: m.m23,
            m33: m.m33,
            m43: R::zero(),
            m14: self.translation.vector.x,
            m24: self.translation.vector.y,
            m34: self.translation.vector.z,
            m44: R::one(),
        }
    }

    /// `true` when the two translations and the two rotations are within `ulps` smallest units
    /// (raw units for fixed point) of each other, component by component; cannot overflow. Note
    /// that `-rotation` is the same rotation and is NOT `abs_diff_eq` to it. Upstream:
    /// `approx::AbsDiffEq::abs_diff_eq`, the tolerance being counted in ulp instead of a float
    /// epsilon (DESIGN D3).
    #[inline(always)]
    fn abs_diff_eq(self: Isometry3<T>, other: Isometry3<T>, ulps: u64) -> bool {
        self.translation.abs_diff_eq(other.translation, ulps)
            && self.rotation.abs_diff_eq(other.rotation, ulps)
    }
}

/// Crate-internal kernels of `Isometry3<T>` (WP 8.0: the public API is strictly upstream's): the
/// fused `rotate_translate` behind every "rotate then translate" (DESIGN D6), the renormalisation
/// of the rotation part (upstream renormalizes `iso.rotation` itself, in place) and the
/// trigonometry-free `lerp_nlerp` (upstream has `lerp_slerp` only).
#[generate_trait]
pub(crate) impl Isometry3InternalImpl<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of Isometry3InternalTrait<T> {
    /// `r · v + t`, the kernel every "rotate then translate" of this type goes through: the
    /// quaternion sandwich `v + 2w·(u × v) + 2u × (u × v)` of
    /// `UnitQuaternion::transform_vector` with the translation folded into its wide accumulation,
    /// so each output component is floored ONCE for the whole expression (15 products, 3
    /// roundings, no division).
    ///
    /// It gives the same bits as rotating and adding afterwards, since `floor(x + t) =
    /// floor(x) + t` for an integral `t` in raw units, for 24 430 gas instead of 25 750 (5 %
    /// cheaper): a `Fixed` addition costs an overflow check (540 gas) that the accumulator does
    /// not pay. It also cannot overflow on the intermediate rotated vector, only on the result.
    /// Duplicating the sandwich (instead of calling `UnitQuaternion::transform_vector` and
    /// adding) is the price of that single rounding; it is written once here and reused by
    /// `transform_point`, `Mul`, `mul_translation` and `append_rotation_wrt_point_mut`.
    /// Evidence: `bench_isometry3_transform_point__alt_rotate_then_add` and
    /// `test_transform_point_fused_and_composed_agree_bit_for_bit`.
    ///
    /// Not an upstream method: upstream writes `rotation * v + translation`, which in fixed point
    /// is exactly this kernel. Panics on overflow of an intermediate doubling (`|v|` above about
    /// `2^30`).
    fn rotate_translate(r: UnitQuaternion<T>, v: Vector3<T>, t: Vector3<T>) -> Vector3<T> {
        let u = r.imag();
        let c = u.cross(v);
        let d = Vector3 { x: c.x + c.x, y: c.y + c.y, z: c.z + c.z };
        let uxd = u.cross(d);
        let w = r.quaternion.w;
        Vector3 {
            x: R::wide_rescale(
                R::wide_add(
                    R::wide_add(R::wide_add(R::wide_add_prod(R::wide_zero(), w, d.x), uxd.x), v.x),
                    t.x,
                ),
            ),
            y: R::wide_rescale(
                R::wide_add(
                    R::wide_add(R::wide_add(R::wide_add_prod(R::wide_zero(), w, d.y), uxd.y), v.y),
                    t.y,
                ),
            ),
            z: R::wide_rescale(
                R::wide_add(
                    R::wide_add(R::wide_add(R::wide_add_prod(R::wide_zero(), w, d.z), uxd.z), v.z),
                    t.z,
                ),
            ),
        }
    }
    /// Renormalises the rotation exactly (`UnitQuaternion::renormalize`: one norm and four exactly
    /// correctly rounded divisions), leaving the translation untouched. Panics with
    /// `Fixed: division by zero` on a zero rotation. Upstream: `Unit::renormalize` applied to the
    /// rotation part (upstream has no `Isometry::renormalize`).
    #[inline(always)]
    fn renormalize(self: Isometry3<T>) -> Isometry3<T> {
        let mut rotation = self.rotation;
        let _ = rotation.renormalize();
        Isometry3 { rotation, translation: self.translation }
    }
    /// Renormalises the rotation with one Newton step (`UnitQuaternion::renormalize_fast`: no
    /// square root, no division, 15 % cheaper than `renormalize`), for a norm already within about
    /// `2^-16` of 1 — which is what a pose composed every step drifts to. This is the call a
    /// rigid-body integrator makes once per body per step. Upstream: `Unit::renormalize_fast`
    /// applied to the rotation part.
    #[inline(always)]
    fn renormalize_fast(self: Isometry3<T>) -> Isometry3<T> {
        let mut rotation = self.rotation;
        rotation.renormalize_fast();
        Isometry3 { rotation, translation: self.translation }
    }
    /// Interpolation WITHOUT trigonometry: the translations linearly, the rotations by
    /// `UnitQuaternion::nlerp` (four fused lerps, one norm, four divisions). `t` is not clamped.
    ///
    /// Measured 31 610 gas against 91 650 for `lerp_slerp` (2.9x), which pays an `acos` and two
    /// `sin` (`bench_isometry3_lerp_slerp__*`). Like upstream's `nlerp` it does NOT take the
    /// shortest arc and its angular velocity is not constant (the chord is walked at constant
    /// speed): for the small relative rotations of one physics step the difference is far below an
    /// ulp, for rendering between two distant poses it is visible. Panics with
    /// `Fixed: division by zero` when the interpolated quaternion vanishes (exactly opposite
    /// rotations at `t = 1/2`).
    ///
    /// Upstream has no `Isometry3::lerp_nlerp`; this is `lerp_slerp` with `nlerp` in place of
    /// `slerp`.
    fn lerp_nlerp(self: Isometry3<T>, other: Isometry3<T>, t: T) -> Isometry3<T> {
        Isometry3 {
            rotation: self.rotation.nlerp(other.rotation, t),
            translation: Translation3 {
                vector: Vector3 {
                    x: R::lerp(self.translation.vector.x, other.translation.vector.x, t),
                    y: R::lerp(self.translation.vector.y, other.translation.vector.y, t),
                    z: R::lerp(self.translation.vector.z, other.translation.vector.z, t),
                },
            },
        }
    }
}

/// Operations of `Isometry3<T>` that need trigonometry, hence their own trait: scalars may
/// implement `Real` only (the whole rapier hot path — composition, `inv_mul`, transforms — is
/// in `Isometry3Trait`, and `UnitQuaternion::renormalize_fast` on the rotation, none of which needs
/// trigonometry).
#[generate_trait]
pub impl Isometry3AngleImpl<
    T,
    impl R: Real<T>,
    impl Tr: Transcendental<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of Isometry3AngleTrait<T> {
    /// The isometry that rotates by the rotation vector `axisangle` (direction = axis, length =
    /// angle, the exponential map) then translates by `translation`: one
    /// `UnitQuaternion::from_scaled_axis` (one `sin_cos`, one division). A zero `axisangle` gives
    /// a pure translation. Upstream: `Isometry3::new(translation, axisangle)`.
    fn new(translation: Vector3<T>, axisangle: Vector3<T>) -> Isometry3<T> {
        Isometry3 {
            rotation: UnitQuaternionAngleTrait::from_scaled_axis(axisangle),
            translation: Translation3 { vector: translation },
        }
    }

    /// The pure rotation by the rotation vector `axisangle` about the origin. Upstream:
    /// `Isometry3::rotation`.
    fn rotation(axisangle: Vector3<T>) -> Isometry3<T> {
        Isometry3 {
            rotation: UnitQuaternionAngleTrait::from_scaled_axis(axisangle),
            translation: Translation3Trait::identity(),
        }
    }

    /// Interpolation between two poses: the translations linearly, the rotations spherically
    /// (`UnitQuaternion::slerp`, shortest arc, constant angular velocity). `t` is not clamped.
    ///
    /// 91 650 gas, dominated by one `acos` and two `sin`. Panics with `nalgebra: ambiguous slerp`
    /// only for a scalar whose resolution makes `sqrt(1 - cos²)` vanish (never in Q32.32, see
    /// `UnitQuaternion::slerp`). The trigonometry-free alternative, `UnitQuaternion::nlerp` on the
    /// rotation and a lerp on the translation, costs 31 610 (`bench_isometry3_lerp_slerp__*`).
    /// Upstream:
    /// `Isometry3::lerp_slerp`.
    fn lerp_slerp(self: Isometry3<T>, other: Isometry3<T>, t: T) -> Isometry3<T> {
        Isometry3 {
            rotation: self.rotation.slerp(other.rotation, t),
            translation: Translation3 {
                vector: Vector3 {
                    x: R::lerp(self.translation.vector.x, other.translation.vector.x, t),
                    y: R::lerp(self.translation.vector.y, other.translation.vector.y, t),
                    z: R::lerp(self.translation.vector.z, other.translation.vector.z, t),
                },
            },
        }
    }

    /// `lerp_slerp`, or `None` when the two rotations are closer than `epsilon` (in scalar units,
    /// not upstream's relative float epsilon) after the shortest-arc flip, where the interpolation
    /// direction is ill-conditioned — see `UnitQuaternion::try_slerp`. With `epsilon = 0` the
    /// result is always `Some`. Upstream: `Isometry3::try_lerp_slerp`.
    fn try_lerp_slerp(
        self: Isometry3<T>, other: Isometry3<T>, t: T, epsilon: T,
    ) -> Option<Isometry3<T>> {
        match self.rotation.try_slerp(other.rotation, t, epsilon) {
            Some(rotation) => Some(
                Isometry3 {
                    rotation,
                    translation: Translation3 {
                        vector: Vector3 {
                            x: R::lerp(self.translation.vector.x, other.translation.vector.x, t),
                            y: R::lerp(self.translation.vector.y, other.translation.vector.y, t),
                            z: R::lerp(self.translation.vector.z, other.translation.vector.z, t),
                        },
                    },
                },
            ),
            None => None,
        }
    }
}

/// `a * b`: the composition of two isometries, `b` applied first: the rotation is the Hamilton
/// product of the rotations and the translation is `a.translation + a.rotation · b.translation`
/// (one `rotate_translate`, so one rounding per component). Upstream: `Mul`.
pub impl Isometry3Mul<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of Mul<Isometry3<T>> {
    fn mul(lhs: Isometry3<T>, rhs: Isometry3<T>) -> Isometry3<T> {
        Isometry3 {
            rotation: lhs.rotation * rhs.rotation,
            translation: Translation3 {
                vector: Isometry3InternalTrait::rotate_translate(
                    lhs.rotation, rhs.translation.vector, lhs.translation.vector,
                ),
            },
        }
    }
}

/// `t.into()`: the pure translation as an isometry. Upstream: `From<Translation3> for Isometry3`.
pub impl Isometry3FromTranslation<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>,
> of Into<Translation3<T>, Isometry3<T>> {
    #[inline(always)]
    fn into(self: Translation3<T>) -> Isometry3<T> {
        Isometry3 {
            rotation: UnitQuaternion {
                quaternion: Quaternion { i: R::zero(), j: R::zero(), k: R::zero(), w: R::one() },
            },
            translation: self,
        }
    }
}
