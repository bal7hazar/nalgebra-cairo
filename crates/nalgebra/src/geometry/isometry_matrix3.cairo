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

use core::num::traits::One;
use core::ops::{DivAssign, MulAssign};
use simba::scalar::{Real, Transcendental};
use crate::base::matrix3::Matrix3;
use crate::base::matrix4::Matrix4;
use crate::base::point3::Point3;
use crate::base::unit::Unit;
use crate::base::vector3::Vector3;
use super::isometry3::Isometry3;
use super::rotation3::{Rotation3, Rotation3AngleTrait, Rotation3Trait};
use super::similarity_matrix3::{SimilarityMatrix3, SimilarityMatrix3Trait};
use super::translation3::{Translation3, Translation3Trait};
use super::unit_quaternion::UnitQuaternionTrait;

/// A 3D direct isometry whose rotation is stored as a matrix: the rotation `rotation` followed by
/// the translation `translation` (upstream's field names and order).
#[derive(Copy, Drop, PartialEq, Serde, Debug, Hash)]
pub struct IsometryMatrix3<T> {
    pub rotation: Rotation3<T>,
    pub translation: Translation3<T>,
}

/// Operations of `IsometryMatrix3<T>` that need no trigonometry, over a `Real` scalar. By value,
/// unrolled, no loop.
#[generate_trait]
pub impl IsometryMatrix3Impl<
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
> of IsometryMatrix3Trait<T> {
    // --- construction and parts ---------------------------------------------------------------

    /// The identity isometry. Exact. Upstream: `Isometry::identity`.
    #[inline(always)]
    fn identity() -> IsometryMatrix3<T> {
        IsometryMatrix3 {
            rotation: Rotation3Trait::identity(), translation: Translation3Trait::identity(),
        }
    }

    /// The isometry `translation ∘ rotation`. Exact. Upstream: `Isometry::from_parts`.
    #[inline(always)]
    fn from_parts(translation: Translation3<T>, rotation: Rotation3<T>) -> IsometryMatrix3<T> {
        IsometryMatrix3 { rotation, translation }
    }

    /// The pure translation by `(x, y, z)`. Exact (upstream's `Rotation3::new` of a zero vector
    /// is the identity exactly). Upstream: `IsometryMatrix3::translation`.
    #[inline(always)]
    fn translation(x: T, y: T, z: T) -> IsometryMatrix3<T> {
        IsometryMatrix3 {
            rotation: Rotation3Trait::identity(),
            translation: Translation3 { vector: Vector3 { x, y, z } },
        }
    }

    /// The rotation `r` about the point `p`: translation `r · (-p) + p` through one
    /// `rotate_translate` (upstream: `r.transform_vector(-p) + p`, the same bits). Upstream:
    /// `Isometry::rotation_wrt_point`.
    #[inline(always)]
    fn rotation_wrt_point(r: Rotation3<T>, p: Point3<T>) -> IsometryMatrix3<T> {
        IsometryMatrix3 {
            rotation: r,
            translation: Translation3 {
                vector: IsometryMatrix3InternalTrait::rotate_translate(
                    r, Vector3 { x: -p.x, y: -p.y, z: -p.z }, Vector3 { x: p.x, y: p.y, z: p.z },
                ),
            },
        }
    }

    /// The isometry placing an observer at `eye` whose local `z` axis points at `target`
    /// (`Rotation3::face_towards(target - eye, up)`), translated by `eye`. `up` MUST not be
    /// parallel to `target - eye`. Upstream: `IsometryMatrix3::face_towards`.
    fn face_towards(eye: Point3<T>, target: Point3<T>, up: Vector3<T>) -> IsometryMatrix3<T> {
        let dir = Vector3 { x: target.x - eye.x, y: target.y - eye.y, z: target.z - eye.z };
        IsometryMatrix3 {
            rotation: Rotation3Trait::face_towards(dir, up),
            translation: Translation3 { vector: Vector3 { x: eye.x, y: eye.y, z: eye.z } },
        }
    }

    /// Deprecated alias of `face_towards`. Upstream: `IsometryMatrix3::new_observer_frame`.
    #[inline(always)]
    fn new_observer_frame(eye: Point3<T>, target: Point3<T>, up: Vector3<T>) -> IsometryMatrix3<T> {
        Self::face_towards(eye, target, up)
    }

    /// The right-handed view transform: rotation `Rotation3::look_at_rh(target - eye, up)`,
    /// translation `rotation · (-eye)` (one `Matrix3 * Vector3`). Upstream:
    /// `IsometryMatrix3::look_at_rh`.
    fn look_at_rh(eye: Point3<T>, target: Point3<T>, up: Vector3<T>) -> IsometryMatrix3<T> {
        let dir = Vector3 { x: target.x - eye.x, y: target.y - eye.y, z: target.z - eye.z };
        let r: Rotation3<T> = Rotation3Trait::look_at_rh(dir, up);
        let neg = Vector3 { x: -eye.x, y: -eye.y, z: -eye.z };
        IsometryMatrix3 {
            rotation: r, translation: Translation3 { vector: r.transform_vector(neg) },
        }
    }

    /// The left-handed view transform: rotation `Rotation3::look_at_lh(target - eye, up)`,
    /// translation `rotation · (-eye)`. Upstream: `IsometryMatrix3::look_at_lh`.
    fn look_at_lh(eye: Point3<T>, target: Point3<T>, up: Vector3<T>) -> IsometryMatrix3<T> {
        let dir = Vector3 { x: target.x - eye.x, y: target.y - eye.y, z: target.z - eye.z };
        let r: Rotation3<T> = Rotation3Trait::look_at_lh(dir, up);
        let neg = Vector3 { x: -eye.x, y: -eye.y, z: -eye.z };
        IsometryMatrix3 {
            rotation: r, translation: Translation3 { vector: r.transform_vector(neg) },
        }
    }

    // --- inverse and composition --------------------------------------------------------------

    /// The inverse isometry: the transposed rotation (exact) and the translation
    /// `rotationᵀ · (-translation)` (upstream's order). Upstream: `inverse`.
    #[inline(always)]
    fn inverse(self: IsometryMatrix3<T>) -> IsometryMatrix3<T> {
        let t = self.translation.vector;
        let v = Vector3 { x: -t.x, y: -t.y, z: -t.z };
        IsometryMatrix3 {
            rotation: self.rotation.inverse(),
            translation: Translation3 { vector: self.rotation.inverse_transform_vector(v) },
        }
    }

    /// `self⁻¹ * other` without materialising the inverse: translation `rotationᵀ ·
    /// (other.translation - self.translation)`, rotation `rotationᵀ · other.rotation`. Upstream:
    /// `inv_mul`.
    fn inv_mul(self: IsometryMatrix3<T>, other: IsometryMatrix3<T>) -> IsometryMatrix3<T> {
        let (a, b) = (self.translation.vector, other.translation.vector);
        let d = Vector3 { x: b.x - a.x, y: b.y - a.y, z: b.z - a.z };
        IsometryMatrix3 {
            rotation: self.rotation.inverse() * other.rotation,
            translation: Translation3 { vector: self.rotation.inverse_transform_vector(d) },
        }
    }

    // --- transforms ----------------------------------------------------------------------------

    /// `self * p = rotation · p + translation`, one `rotate_translate`. Upstream:
    /// `transform_point`.
    #[inline(always)]
    fn transform_point(self: IsometryMatrix3<T>, p: Point3<T>) -> Point3<T> {
        let c = IsometryMatrix3InternalTrait::rotate_translate(
            self.rotation, Vector3 { x: p.x, y: p.y, z: p.z }, self.translation.vector,
        );
        Point3 { x: c.x, y: c.y, z: c.z }
    }

    /// `rotation · v`. Upstream: `transform_vector`.
    #[inline(always)]
    fn transform_vector(self: IsometryMatrix3<T>, v: Vector3<T>) -> Vector3<T> {
        self.rotation.transform_vector(v)
    }

    /// `rotation · v` for a unit vector, not renormalised. Upstream: `Mul<Unit<Vector3>> for
    /// Isometry`.
    #[inline(always)]
    fn transform_unit_vector(self: IsometryMatrix3<T>, v: Unit<Vector3<T>>) -> Unit<Vector3<T>> {
        Unit { value: self.rotation.transform_vector(v.value) }
    }

    /// `self⁻¹ * p = rotationᵀ · (p - translation)`. Upstream: `inverse_transform_point`.
    #[inline(always)]
    fn inverse_transform_point(self: IsometryMatrix3<T>, p: Point3<T>) -> Point3<T> {
        let t = self.translation.vector;
        self.rotation.inverse_transform_point(Point3 { x: p.x - t.x, y: p.y - t.y, z: p.z - t.z })
    }

    /// `rotationᵀ · v`. Upstream: `inverse_transform_vector`.
    #[inline(always)]
    fn inverse_transform_vector(self: IsometryMatrix3<T>, v: Vector3<T>) -> Vector3<T> {
        self.rotation.inverse_transform_vector(v)
    }

    /// `rotationᵀ · v` for a unit vector, not renormalised. Upstream:
    /// `inverse_transform_unit_vector`.
    #[inline(always)]
    fn inverse_transform_unit_vector(
        self: IsometryMatrix3<T>, v: Unit<Vector3<T>>,
    ) -> Unit<Vector3<T>> {
        Unit { value: self.rotation.inverse_transform_vector(v.value) }
    }

    // --- append (in place) and the operator forms ----------------------------------------------

    /// `Translation(t) ∘ self`, in place (exact additions). Upstream: `append_translation_mut`.
    #[inline(always)]
    fn append_translation_mut(ref self: IsometryMatrix3<T>, t: Translation3<T>) {
        let v = self.translation.vector;
        self =
            IsometryMatrix3 {
                rotation: self.rotation,
                translation: Translation3 {
                    vector: Vector3 {
                        x: v.x + t.vector.x, y: v.y + t.vector.y, z: v.z + t.vector.z,
                    },
                },
            };
    }

    /// `self * t`: the translation shifted by `rotation · t` (one `rotate_translate`). Upstream:
    /// `Mul<Translation> for Isometry`.
    #[inline(always)]
    fn mul_translation(self: IsometryMatrix3<T>, t: Translation3<T>) -> IsometryMatrix3<T> {
        IsometryMatrix3 {
            rotation: self.rotation,
            translation: Translation3 {
                vector: IsometryMatrix3InternalTrait::rotate_translate(
                    self.rotation, t.vector, self.translation.vector,
                ),
            },
        }
    }

    /// `Rotation(r) ∘ self`, in place: rotation `r · rotation`, translation `r · translation`.
    /// Upstream: `append_rotation_mut`.
    #[inline(always)]
    fn append_rotation_mut(ref self: IsometryMatrix3<T>, r: Rotation3<T>) {
        self =
            IsometryMatrix3 {
                rotation: r * self.rotation,
                translation: Translation3 { vector: r.transform_vector(self.translation.vector) },
            };
    }

    /// The rotation `r` applied about the point `p`, in place: translation
    /// `r · (translation - p) + p`, rotation `r · rotation`. Upstream:
    /// `append_rotation_wrt_point_mut`.
    fn append_rotation_wrt_point_mut(ref self: IsometryMatrix3<T>, r: Rotation3<T>, p: Point3<T>) {
        let t = self.translation.vector;
        let d = Vector3 { x: t.x - p.x, y: t.y - p.y, z: t.z - p.z };
        self =
            IsometryMatrix3 {
                rotation: r * self.rotation,
                translation: Translation3 {
                    vector: IsometryMatrix3InternalTrait::rotate_translate(
                        r, d, Vector3 { x: p.x, y: p.y, z: p.z },
                    ),
                },
            };
    }

    /// The rotation `r` applied about the isometry's own centre, in place (only the rotation
    /// changes). Upstream: `append_rotation_wrt_center_mut`.
    #[inline(always)]
    fn append_rotation_wrt_center_mut(ref self: IsometryMatrix3<T>, r: Rotation3<T>) {
        self = IsometryMatrix3 { rotation: r * self.rotation, translation: self.translation };
    }

    /// `self * r`: rotation `rotation · r`, the translation unchanged. Upstream: `Mul<Rotation>
    /// for Isometry<T, Rotation<T, D>, D>`.
    #[inline(always)]
    fn mul_rotation(self: IsometryMatrix3<T>, r: Rotation3<T>) -> IsometryMatrix3<T> {
        IsometryMatrix3 { rotation: self.rotation * r, translation: self.translation }
    }

    /// `self / r = self * r⁻¹`. Upstream: `Div<Rotation> for Isometry<T, Rotation<T, D>, D>`.
    #[inline(always)]
    fn div_rotation(self: IsometryMatrix3<T>, r: Rotation3<T>) -> IsometryMatrix3<T> {
        IsometryMatrix3 { rotation: self.rotation / r, translation: self.translation }
    }

    /// `self * sim`: the similarity `(self * sim.isometry, sim.scaling)`. Upstream:
    /// `Mul<Similarity> for Isometry`.
    #[inline(always)]
    fn mul_similarity(self: IsometryMatrix3<T>, sim: SimilarityMatrix3<T>) -> SimilarityMatrix3<T> {
        SimilarityMatrix3 { isometry: self * sim.isometry, scaling: sim.scaling }
    }

    /// `self / sim = self * sim⁻¹` (upstream's formula). Upstream: `Div<Similarity> for
    /// Isometry`.
    #[inline(always)]
    fn div_similarity(self: IsometryMatrix3<T>, sim: SimilarityMatrix3<T>) -> SimilarityMatrix3<T> {
        Self::mul_similarity(self, sim.inverse())
    }

    // --- conversions and comparisons -----------------------------------------------------------

    /// The 4x4 homogeneous matrix: the rotation block, the translation in the last column, then
    /// `(0, 0, 0, 1)`. Exact (copies). Upstream: `to_homogeneous`.
    fn to_homogeneous(self: IsometryMatrix3<T>) -> Matrix4<T> {
        let m = self.rotation.matrix;
        let t = self.translation.vector;
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
            m14: t.x,
            m24: t.y,
            m34: t.z,
            m44: R::one(),
        }
    }

    /// Alias of `to_homogeneous`. Upstream: `Isometry::to_matrix`.
    #[inline(always)]
    fn to_matrix(self: IsometryMatrix3<T>) -> Matrix4<T> {
        Self::to_homogeneous(self)
    }

    /// `true` when the translations and the rotation matrices are within `ulps` of each other,
    /// component by component. Upstream: `approx::AbsDiffEq::abs_diff_eq` (DESIGN D3).
    #[inline(always)]
    fn abs_diff_eq(self: IsometryMatrix3<T>, other: IsometryMatrix3<T>, ulps: u64) -> bool {
        self.translation.abs_diff_eq(other.translation, ulps)
            && self.rotation.abs_diff_eq(other.rotation, ulps)
    }

    /// `relative_eq` of the translations and of the rotations. Upstream:
    /// `approx::RelativeEq::relative_eq` (DESIGN D3).
    fn relative_eq(
        self: IsometryMatrix3<T>, other: IsometryMatrix3<T>, epsilon: u64, max_relative: T,
    ) -> bool {
        self.translation.relative_eq(other.translation, epsilon, max_relative)
            && self.rotation.relative_eq(other.rotation, epsilon, max_relative)
    }

    /// `ulps_eq` of the translations and of the rotations. Upstream: `approx::UlpsEq::ulps_eq`
    /// (DESIGN D3).
    fn ulps_eq(
        self: IsometryMatrix3<T>, other: IsometryMatrix3<T>, epsilon: u64, max_ulps: u32,
    ) -> bool {
        self.translation.ulps_eq(other.translation, epsilon, max_ulps)
            && self.rotation.ulps_eq(other.rotation, epsilon, max_ulps)
    }

    /// The same isometry with every scalar converted by `Into<T, U>`. Upstream:
    /// `IsometryMatrix3::cast` (and `SubsetOf<Isometry>`).
    fn cast<U, +Into<T, U>, +Drop<U>>(self: IsometryMatrix3<T>) -> IsometryMatrix3<U> {
        IsometryMatrix3 { rotation: self.rotation.cast(), translation: self.translation.cast() }
    }
}

/// Crate-internal kernel of `IsometryMatrix3<T>`: the fused "rotate then translate".
#[generate_trait]
pub(crate) impl IsometryMatrix3InternalImpl<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>,
> of IsometryMatrix3InternalTrait<T> {
    /// `r · v + t`: the three products AND the translation accumulated exactly, floored once per
    /// component; the same bits as `r.transform_vector(v) + t`, without the overflow check of the
    /// `Fixed` addition. Not an upstream method.
    #[inline(always)]
    fn rotate_translate(r: Rotation3<T>, v: Vector3<T>, t: Vector3<T>) -> Vector3<T> {
        let m = r.matrix;
        Vector3 {
            x: Self::row(m.m11, m.m12, m.m13, v, t.x),
            y: Self::row(m.m21, m.m22, m.m23, v, t.y),
            z: Self::row(m.m31, m.m32, m.m33, v, t.z),
        }
    }

    /// `a·v.x + b·v.y + c·v.z + t`, floored once.
    #[inline(always)]
    fn row(a: T, b: T, c: T, v: Vector3<T>, t: T) -> T {
        R::wide_rescale(
            R::wide_add(
                R::wide_add_prod(
                    R::wide_add_prod(R::wide_add_prod(R::wide_zero(), a, v.x), b, v.y), c, v.z,
                ),
                t,
            ),
        )
    }
}

/// Operations of `IsometryMatrix3<T>` that need trigonometry, hence their own trait.
#[generate_trait]
pub impl IsometryMatrix3AngleImpl<
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
> of IsometryMatrix3AngleTrait<T> {
    /// The isometry that rotates by the rotation vector `axisangle` (`Rotation3::new`, the
    /// exponential map) then translates by `translation`. Upstream:
    /// `IsometryMatrix3::new(translation, axisangle)`.
    #[inline(always)]
    fn new(translation: Vector3<T>, axisangle: Vector3<T>) -> IsometryMatrix3<T> {
        IsometryMatrix3 {
            rotation: Rotation3AngleTrait::new(axisangle),
            translation: Translation3 { vector: translation },
        }
    }

    /// The pure rotation by the rotation vector `axisangle`. Upstream:
    /// `IsometryMatrix3::rotation`.
    #[inline(always)]
    fn rotation(axisangle: Vector3<T>) -> IsometryMatrix3<T> {
        IsometryMatrix3 {
            rotation: Rotation3AngleTrait::new(axisangle),
            translation: Translation3Trait::identity(),
        }
    }

    /// The translations interpolated linearly, the rotations by `Rotation3::slerp` (shortest
    /// arc). `t` is not clamped. Upstream: `IsometryMatrix3::lerp_slerp`.
    fn lerp_slerp(self: IsometryMatrix3<T>, other: IsometryMatrix3<T>, t: T) -> IsometryMatrix3<T> {
        IsometryMatrix3 {
            rotation: self.rotation.slerp(other.rotation, t),
            translation: IsometryMatrix3AngleInternalTrait::lerp_translation(
                self.translation, other.translation, t,
            ),
        }
    }

    /// `lerp_slerp`, or `None` when the two rotations are closer than `epsilon` (in scalar units)
    /// after the shortest-arc flip (`Rotation3::try_slerp`). Upstream:
    /// `IsometryMatrix3::try_lerp_slerp`.
    fn try_lerp_slerp(
        self: IsometryMatrix3<T>, other: IsometryMatrix3<T>, t: T, epsilon: T,
    ) -> Option<IsometryMatrix3<T>> {
        match self.rotation.try_slerp(other.rotation, t, epsilon) {
            Some(rotation) => Some(
                IsometryMatrix3 {
                    rotation,
                    translation: IsometryMatrix3AngleInternalTrait::lerp_translation(
                        self.translation, other.translation, t,
                    ),
                },
            ),
            None => None,
        }
    }
}

/// Crate-internal helper of `IsometryMatrix3AngleTrait`.
#[generate_trait]
pub(crate) impl IsometryMatrix3AngleInternalImpl<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>,
> of IsometryMatrix3AngleInternalTrait<T> {
    /// One fused `lerp` per component.
    #[inline(always)]
    fn lerp_translation(a: Translation3<T>, b: Translation3<T>, t: T) -> Translation3<T> {
        Translation3 {
            vector: Vector3 {
                x: R::lerp(a.vector.x, b.vector.x, t),
                y: R::lerp(a.vector.y, b.vector.y, t),
                z: R::lerp(a.vector.z, b.vector.z, t),
            },
        }
    }
}

/// `a * b`, `b` applied first: rotation `a.rotation · b.rotation` (27 products, one rounding per
/// entry), translation `a.translation + a.rotation · b.translation` (one `rotate_translate`).
/// Upstream: `Mul<Isometry> for Isometry`.
pub impl IsometryMatrix3Mul<
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
> of Mul<IsometryMatrix3<T>> {
    fn mul(lhs: IsometryMatrix3<T>, rhs: IsometryMatrix3<T>) -> IsometryMatrix3<T> {
        IsometryMatrix3 {
            rotation: lhs.rotation * rhs.rotation,
            translation: Translation3 {
                vector: IsometryMatrix3InternalTrait::rotate_translate(
                    lhs.rotation, rhs.translation.vector, lhs.translation.vector,
                ),
            },
        }
    }
}

/// `a / b = a * b⁻¹` (upstream's formula). Upstream: `Div<Isometry> for Isometry`.
pub impl IsometryMatrix3Div<
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
> of Div<IsometryMatrix3<T>> {
    #[inline(always)]
    fn div(lhs: IsometryMatrix3<T>, rhs: IsometryMatrix3<T>) -> IsometryMatrix3<T> {
        lhs * rhs.inverse()
    }
}

/// `iso *= t`: `iso = iso * t`. Upstream: `MulAssign<Translation> for Isometry`.
pub impl IsometryMatrix3MulAssignTranslation3<
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
> of MulAssign<IsometryMatrix3<T>, Translation3<T>> {
    #[inline(always)]
    fn mul_assign(ref self: IsometryMatrix3<T>, rhs: Translation3<T>) {
        self = self.mul_translation(rhs);
    }
}

/// `a *= b`: `a = a * b`. Upstream: `MulAssign<Isometry> for Isometry`.
pub impl IsometryMatrix3MulAssign<
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
> of MulAssign<IsometryMatrix3<T>, IsometryMatrix3<T>> {
    #[inline(always)]
    fn mul_assign(ref self: IsometryMatrix3<T>, rhs: IsometryMatrix3<T>) {
        self = self * rhs;
    }
}

/// `a /= b`: `a = a * b⁻¹`. Upstream: `DivAssign<Isometry> for Isometry`.
pub impl IsometryMatrix3DivAssign<
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
> of DivAssign<IsometryMatrix3<T>, IsometryMatrix3<T>> {
    #[inline(always)]
    fn div_assign(ref self: IsometryMatrix3<T>, rhs: IsometryMatrix3<T>) {
        self = self * rhs.inverse();
    }
}

/// `Default::default()`: the identity. Upstream: `Default for Isometry`.
pub impl IsometryMatrix3Default<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>,
> of Default<IsometryMatrix3<T>> {
    #[inline(always)]
    fn default() -> IsometryMatrix3<T> {
        let (o, l) = (R::zero(), R::one());
        IsometryMatrix3 {
            rotation: Rotation3 {
                matrix: Matrix3 {
                    m11: l, m21: o, m31: o, m12: o, m22: l, m32: o, m13: o, m23: o, m33: l,
                },
            },
            translation: Translation3 { vector: Vector3 { x: o, y: o, z: o } },
        }
    }
}

/// `One::one()`: the identity; `is_one` compares with it exactly. Upstream: `num::One for
/// Isometry`.
pub impl IsometryMatrix3One<
    T, impl R: Real<T>, +PartialEq<T>, +Copy<T>, +Drop<T>,
> of One<IsometryMatrix3<T>> {
    #[inline(always)]
    fn one() -> IsometryMatrix3<T> {
        IsometryMatrix3Default::<T>::default()
    }

    #[inline(always)]
    fn is_one(self: @IsometryMatrix3<T>) -> bool {
        *self == IsometryMatrix3Default::<T>::default()
    }

    #[inline(always)]
    fn is_non_one(self: @IsometryMatrix3<T>) -> bool {
        !Self::is_one(self)
    }
}

/// `t.into()`: the pure translation. Upstream: `From<Translation> for Isometry`.
pub impl IsometryMatrix3FromTranslation3<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>,
> of Into<Translation3<T>, IsometryMatrix3<T>> {
    #[inline(always)]
    fn into(self: Translation3<T>) -> IsometryMatrix3<T> {
        let mut iso = IsometryMatrix3Default::<T>::default();
        iso.translation = self;
        iso
    }
}

/// `v.into()`: the pure translation by `v`. Upstream: `From<SVector<T, 3>> for Isometry`.
pub impl IsometryMatrix3FromVector3<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>,
> of Into<Vector3<T>, IsometryMatrix3<T>> {
    #[inline(always)]
    fn into(self: Vector3<T>) -> IsometryMatrix3<T> {
        let mut iso = IsometryMatrix3Default::<T>::default();
        iso.translation = Translation3 { vector: self };
        iso
    }
}

/// `p.into()`: the pure translation by the coordinates of `p`. Upstream: `From<Point<T, 3>> for
/// Isometry`.
pub impl IsometryMatrix3FromPoint3<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>,
> of Into<Point3<T>, IsometryMatrix3<T>> {
    #[inline(always)]
    fn into(self: Point3<T>) -> IsometryMatrix3<T> {
        let mut iso = IsometryMatrix3Default::<T>::default();
        iso.translation = Translation3 { vector: Vector3 { x: self.x, y: self.y, z: self.z } };
        iso
    }
}

/// `[x, y, z].into()`: the pure translation by `(x, y, z)`. Upstream: `From<[T; 3]> for
/// Isometry`.
pub impl IsometryMatrix3FromArray<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>,
> of Into<[T; 3], IsometryMatrix3<T>> {
    #[inline(always)]
    fn into(self: [T; 3]) -> IsometryMatrix3<T> {
        let [x, y, z] = self;
        let mut iso = IsometryMatrix3Default::<T>::default();
        iso.translation = Translation3 { vector: Vector3 { x, y, z } };
        iso
    }
}

/// `iso.into()`: the same isometry with its rotation as a unit quaternion (Shepperd's method,
/// `UnitQuaternion::from_rotation_matrix`). Upstream: `SubsetOf<Isometry3> for IsometryMatrix3`
/// (`nalgebra::convert`).
pub impl Isometry3FromIsometryMatrix3<
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
> of Into<IsometryMatrix3<T>, Isometry3<T>> {
    #[inline(always)]
    fn into(self: IsometryMatrix3<T>) -> Isometry3<T> {
        Isometry3 {
            rotation: UnitQuaternionTrait::from_rotation_matrix(self.rotation),
            translation: self.translation,
        }
    }
}

/// `iso.into()`: the same isometry with its rotation as a matrix (upstream's quaternion-to-matrix
/// form, `UnitQuaternion::to_rotation_matrix`). Upstream: `SubsetOf<IsometryMatrix3> for
/// Isometry3` (`nalgebra::convert`).
pub impl IsometryMatrix3FromIsometry3<
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
> of Into<Isometry3<T>, IsometryMatrix3<T>> {
    #[inline(always)]
    fn into(self: Isometry3<T>) -> IsometryMatrix3<T> {
        IsometryMatrix3 {
            rotation: UnitQuaternionTrait::to_rotation_matrix(self.rotation),
            translation: self.translation,
        }
    }
}

/// `iso.into()`: the similarity of scaling 1. Upstream: `SubsetOf<Similarity> for Isometry`
/// (`nalgebra::convert`).
pub impl SimilarityMatrix3FromIsometryMatrix3<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>,
> of Into<IsometryMatrix3<T>, SimilarityMatrix3<T>> {
    #[inline(always)]
    fn into(self: IsometryMatrix3<T>) -> SimilarityMatrix3<T> {
        SimilarityMatrix3 { isometry: self, scaling: R::one() }
    }
}
