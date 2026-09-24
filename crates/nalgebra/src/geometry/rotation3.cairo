//! `Rotation3`: a 3D rotation stored as a 3x3 orthonormal matrix (upstream `nalgebra::Rotation3`).
//!
//! - `Rotation3Trait` / `Rotation3Impl`: constructors, transforms, composition, conversions to and
//!   from `UnitQuaternion`, renormalisation — everything that needs no trigonometry;
//! - `Rotation3AngleTrait` / `Rotation3AngleImpl`: axis-angle and Euler-angle constructors and
//!   extractors, which additionally need `simba::scalar::Transcendental`;
//! - the operator `*` (composition) lives in this module, where the compiler finds it without any
//!   import.
//!
//! The matrix form is the right representation when several vectors are transformed with the same
//! rotation: `transform_vector` is one `Matrix3 * Vector3` (9 products, 6 650 gas) against 15
//! products (23 230) for `UnitQuaternion::transform_vector`, and building the matrix from a
//! quaternion costs 23 530 — so the break-even is TWO vectors per step (measured,
//! `bench_unit_quaternion_transform_vector_x2__*`). Composition is the other way round: 23 310 for
//! a matrix product against 11 860 for a Hamilton product, and a matrix composed repeatedly drifts
//! out of orthonormality, which `renormalize` (39 170) has to fix.
//!
//! The invariant (orthonormal, determinant +1) is a CONTRACT, like upstream's
//! `from_matrix_unchecked`: nothing checks it, and `inverse` is implemented as the transpose.
//!
//! Numeric contract (AGENTS.md): every sum of products goes through a fused `Real` kernel (one
//! floor rounding and one overflow check per output scalar); nothing wraps silently.

use simba::scalar::{Real, Transcendental};
use crate::base::matrix3::{Matrix3, Matrix3Trait};
use crate::base::matrix4::Matrix4;
use crate::base::point3::Point3;
use crate::base::unit::{Unit, UnitTrait};
use crate::base::vector3::{Vector3, Vector3Trait};
use super::unit_quaternion::{UnitQuaternion, UnitQuaternionTrait};

#[cfg(test)]
mod benches;
#[cfg(test)]
mod oracle;
#[cfg(test)]
mod tests;

/// A 3D rotation as an orthonormal 3x3 matrix of determinant +1.
#[derive(Copy, Drop, PartialEq, Serde, Debug, Hash)]
pub struct Rotation3<T> {
    pub matrix: Matrix3<T>,
}

/// Methods of `Rotation3<T>` that need no trigonometry. By value, unrolled, no loop.
#[generate_trait]
pub impl Rotation3Impl<
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
> of Rotation3Trait<T> {
    // --- constructors and parts --------------------------------------------------------------

    /// The identity rotation. Exact. Upstream: `Rotation3::identity`.
    #[inline(always)]
    fn identity() -> Rotation3<T> {
        Rotation3 { matrix: Matrix3Trait::identity() }
    }

    /// Wraps `m` WITHOUT checking that it is a rotation matrix: the caller guarantees it is
    /// orthonormal with determinant +1. Upstream: `Rotation3::from_matrix_unchecked`.
    #[inline(always)]
    fn from_matrix_unchecked(m: Matrix3<T>) -> Rotation3<T> {
        Rotation3 { matrix: m }
    }

    /// The underlying matrix (a copy: everything is by value here). Upstream: `matrix`
    /// (`into_inner`).
    #[inline(always)]
    fn matrix(self: Rotation3<T>) -> Matrix3<T> {
        self.matrix
    }

    /// The inverse rotation, which for an orthonormal matrix is the transpose: exact, no division.
    /// Upstream: `inverse` (= `transpose`).
    #[inline(always)]
    fn inverse(self: Rotation3<T>) -> Rotation3<T> {
        Rotation3 { matrix: self.matrix.transpose() }
    }

    /// Alias of `inverse` (the transpose of an orthonormal matrix is its inverse). Upstream:
    /// `transpose`.
    #[inline(always)]
    fn transpose(self: Rotation3<T>) -> Rotation3<T> {
        Rotation3 { matrix: self.matrix.transpose() }
    }

    /// The rotation as a homogeneous 4x4 matrix: the rotation block, a zero translation and
    /// `m44 = 1`. Exact. Upstream: `to_homogeneous`.
    #[inline(always)]
    fn to_homogeneous(self: Rotation3<T>) -> Matrix4<T> {
        let m = self.matrix;
        Matrix4 {
            m11: m.m11,
            m21: m.m21,
            m31: m.m31,
            m41: R::ZERO,
            m12: m.m12,
            m22: m.m22,
            m32: m.m32,
            m42: R::ZERO,
            m13: m.m13,
            m23: m.m23,
            m33: m.m33,
            m43: R::ZERO,
            m14: R::ZERO,
            m24: R::ZERO,
            m34: R::ZERO,
            m44: R::ONE,
        }
    }

    // --- transforms --------------------------------------------------------------------------

    /// `self · v`: one `Matrix3 * Vector3`, each component one fused `sum_prod3` (9 products, 3
    /// roundings). Cheaper than `UnitQuaternion::transform_vector` (15 products), which is why a
    /// body transforming several vectors per step should keep its `Rotation3`. Upstream:
    /// `transform_vector` (`r * v`).
    #[inline(always)]
    fn transform_vector(self: Rotation3<T>, v: Vector3<T>) -> Vector3<T> {
        self.matrix.mul_vec(v)
    }

    /// `transform_vector` of the point's coordinates (a rotation fixes the origin). Upstream:
    /// `transform_point` (`r * p`).
    #[inline(always)]
    fn transform_point(self: Rotation3<T>, p: Point3<T>) -> Point3<T> {
        let c = self.matrix.mul_vec(Vector3 { x: p.x, y: p.y, z: p.z });
        Point3 { x: c.x, y: c.y, z: c.z }
    }

    /// `self⁻¹ · v = selfᵀ · v`: one `tr_mul_vec`, as cheap as `transform_vector` (the
    /// transpose is free, it is just another access pattern). Upstream: `inverse_transform_vector`.
    #[inline(always)]
    fn inverse_transform_vector(self: Rotation3<T>, v: Vector3<T>) -> Vector3<T> {
        self.matrix.tr_mul_vec(v)
    }

    /// `inverse_transform_vector` of the point's coordinates. Upstream:
    /// `inverse_transform_point`.
    #[inline(always)]
    fn inverse_transform_point(self: Rotation3<T>, p: Point3<T>) -> Point3<T> {
        let c = self.matrix.tr_mul_vec(Vector3 { x: p.x, y: p.y, z: p.z });
        Point3 { x: c.x, y: c.y, z: c.z }
    }

    // --- conversions with `UnitQuaternion` --------------------------------------------------

    // --- axis, composition, construction from vectors ---------------------------------------

    /// The rotation axis as a unit vector, or `None` when the rotation is the identity or a half
    /// turn about an undetermined axis (the antisymmetric part vanishes at `angle = π` too, where
    /// upstream returns `None` as well).
    ///
    /// The axis is the antisymmetric part `(m32 - m23, m13 - m31, m21 - m12)` (exact, `= 2 sin θ`
    /// times the axis) normalised. Upstream: `axis`.
    #[inline(always)]
    fn axis(self: Rotation3<T>) -> Option<Unit<Vector3<T>>> {
        let m = self.matrix;
        let v = Vector3 { x: m.m32 - m.m23, y: m.m13 - m.m31, z: m.m21 - m.m12 };
        UnitTrait::try_new(v, R::ZERO)
    }

    /// The shortest rotation taking the direction of `a` to the direction of `b`, or `None` when
    /// they are exactly antiparallel; the identity for a zero-length input.
    ///
    /// Computed through `UnitQuaternion::rotation_between` (algebraic, no trigonometry) and
    /// converted: 74 470 gas against 107 500 for upstream's `from_axis_angle(a × b, acos(a ·
    /// b))`, which pays an `acos` and a `sin_cos` to reconstruct an angle that cancels out — and
    /// it is more accurate (`bench_rotation3_rotation_between__alt_axis_angle`). Upstream:
    /// `Rotation3::rotation_between`.
    fn rotation_between(a: Vector3<T>, b: Vector3<T>) -> Option<Rotation3<T>> {
        match UnitQuaternionTrait::rotation_between(a, b) {
            Some(q) => Some(UnitQuaternionTrait::to_rotation_matrix(q)),
            None => None,
        }
    }

    /// The rotation whose third column (the local `z` axis) is the direction of `dir` and whose
    /// second column is as close as possible to `up`: `z = dir/|dir|`, `x = (up × z)/|up × z|`,
    /// `y = z × x`. `up` MUST not be parallel to `dir` (the normalisation would divide by zero).
    /// Upstream: `Rotation3::face_towards` (`new_observer_frame`).
    fn face_towards(dir: Vector3<T>, up: Vector3<T>) -> Rotation3<T> {
        let zaxis = dir.normalize();
        let xaxis = up.cross(zaxis).normalize();
        let yaxis = zaxis.cross(xaxis);
        Rotation3 { matrix: Matrix3Trait::from_columns(xaxis, yaxis, zaxis) }
    }

    /// The rotation of a right-handed look-at camera: the transpose of `face_towards(-dir, up)`, so
    /// that `dir` is mapped onto the NEGATIVE `z` axis. `up` MUST not be parallel to `dir`.
    /// Upstream: `Rotation3::look_at_rh`.
    fn look_at_rh(dir: Vector3<T>, up: Vector3<T>) -> Rotation3<T> {
        let neg = Vector3 { x: -dir.x, y: -dir.y, z: -dir.z };
        Rotation3 { matrix: Self::face_towards(neg, up).matrix.transpose() }
    }

    // --- renormalisation and comparison -----------------------------------------------------

    /// Restores orthonormality after repeated composition, in place, by Gram-Schmidt on the
    /// columns: `x = c1/|c1|`, `y = (c2 - (x·c2)·x)/|...|`, `z = x × y`. Two norms, six
    /// divisions, one dot product and one cross product: 39 170 gas, against 70 250 for the round
    /// trip through a quaternion (`from_rotation_matrix`, `renormalize_fast`, `to_rotation_matrix`)
    /// and 78 320 for one Newton step of the polar decomposition `R·(3I - RᵀR)/2` (54 products,
    /// and it only halves the error instead of renormalizing exactly). Both are kept as benchmarks
    /// (`bench_rotation3_renormalize__alt_*`).
    ///
    /// The result is orthonormal to within a few ulp and is the closest rotation to `self` only to
    /// first order (Gram-Schmidt privileges the first column, unlike the polar decomposition).
    /// Panics with `Fixed: division by zero` on a singular matrix. Upstream:
    /// `Rotation::renormalize`
    /// (which uses a QR decomposition).
    fn renormalize(ref self: Rotation3<T>) {
        let m = self.matrix;
        let x = Vector3 { x: m.m11, y: m.m21, z: m.m31 }.normalize();
        let c2 = Vector3 { x: m.m12, y: m.m22, z: m.m32 };
        let d = -Vector3Trait::dot(x, c2);
        let y = Vector3 {
            x: R::mul_add(d, x.x, c2.x), y: R::mul_add(d, x.y, c2.y), z: R::mul_add(d, x.z, c2.z),
        }
            .normalize();
        let z = x.cross(y);
        self = Rotation3 { matrix: Matrix3Trait::from_columns(x, y, z) };
    }

    /// `true` when every entry is within `ulps` smallest units (raw units for fixed point) of the
    /// matching entry of `other`; cannot overflow. Upstream: `approx::AbsDiffEq::abs_diff_eq`, the
    /// tolerance being counted in ulp instead of a float epsilon (DESIGN D3).
    #[inline(always)]
    fn abs_diff_eq(self: Rotation3<T>, other: Rotation3<T>, ulps: u64) -> bool {
        self.matrix.abs_diff_eq(other.matrix, ulps)
    }
}

/// Crate-internal by-value forms of the in-place `renormalize` (WP 8.0: the
/// public methods are upstream's `&mut self` ones), for the tests and the value-style call sites.
#[generate_trait]
pub(crate) impl Rotation3InternalImpl<
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
> of Rotation3InternalTrait<T> {
    /// `self` renormalized exactly (`Rotation3Trait::renormalize`), by value.
    #[inline(always)]
    fn renormalized(self: Rotation3<T>) -> Rotation3<T> {
        let mut r = self;
        Rotation3Trait::renormalize(ref r);
        r
    }
}

/// Methods of `Rotation3<T>` that need trigonometry, hence their own trait: scalars may implement
/// `Real` only (see `Vector3AngleTrait`).
#[generate_trait]
pub impl Rotation3AngleImpl<
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
> of Rotation3AngleTrait<T> {
    /// The rotation of `angle` radians about `axis` (Rodrigues' formula, upstream's form):
    /// `R = cos θ·I + (1 - cos θ)·u uᵀ + sin θ·[u]×`. One `sin_cos`, then 3 shared
    /// products and one fused kernel per entry (21 products). `axis` MUST be a unit vector; a zero
    /// angle gives the identity exactly (upstream branches on it too).
    ///
    /// Measured 49 070 gas, marginally cheaper than `UnitQuaternion::from_axis_angle` followed by
    /// `to_rotation_matrix` (49 610) and more accurate, since it does not go through a half angle
    /// (`bench_rotation3_from_axis_angle__alt_quaternion`). Upstream:
    /// `Rotation3::from_axis_angle`.
    fn from_axis_angle(axis: Unit<Vector3<T>>, angle: T) -> Rotation3<T> {
        if angle == R::ZERO {
            return Rotation3Trait::identity();
        }
        let u = axis.value;
        let (s, c) = Tr::sin_cos(angle);
        let omc = R::ONE - c;
        // Shared products of the symmetric part (each rounded once, as upstream).
        let (uxy, uxz, uyz) = (u.x * u.y, u.x * u.z, u.y * u.z);
        Rotation3 {
            matrix: Matrix3 {
                m11: R::mul_add(R::sqr(u.x), omc, c),
                m21: R::sum_prod2(uxy, omc, u.z, s),
                m31: R::diff_prod(uxz, omc, u.y, s),
                m12: R::diff_prod(uxy, omc, u.z, s),
                m22: R::mul_add(R::sqr(u.y), omc, c),
                m32: R::sum_prod2(uyz, omc, u.x, s),
                m13: R::sum_prod2(uxz, omc, u.y, s),
                m23: R::diff_prod(uyz, omc, u.x, s),
                m33: R::mul_add(R::sqr(u.z), omc, c),
            },
        }
    }

    /// The rotation whose axis is the direction of `axisangle` and whose angle is its length (the
    /// exponential map). A zero vector gives the identity (upstream divides by the zero norm and
    /// relies on the `angle == 0` branch of `from_axis_angle`; fixed point must test first).
    /// Upstream: `Rotation3::from_scaled_axis` (`Rotation3::new`).
    fn from_scaled_axis(axisangle: Vector3<T>) -> Rotation3<T> {
        let n = Vector3Trait::norm(axisangle);
        if n == R::ZERO {
            return Rotation3Trait::identity();
        }
        Self::from_axis_angle(UnitTrait::new_unchecked(Vector3Trait::unscale(axisangle, n)), n)
    }

    /// The rotation `Rz(yaw) · Ry(pitch) · Rx(roll)` (upstream's convention). Three `sin_cos`,
    /// two shared products, then one fused kernel per entry. Upstream:
    /// `Rotation3::from_euler_angles`.
    fn from_euler_angles(roll: T, pitch: T, yaw: T) -> Rotation3<T> {
        let (sr, cr) = Tr::sin_cos(roll);
        let (sp, cp) = Tr::sin_cos(pitch);
        let (sy, cy) = Tr::sin_cos(yaw);
        let (spsr, spcr) = (sp * sr, sp * cr);
        Rotation3 {
            matrix: Matrix3 {
                m11: cy * cp,
                m21: sy * cp,
                m31: -sp,
                m12: R::diff_prod(cy, spsr, sy, cr),
                m22: R::sum_prod2(sy, spsr, cy, cr),
                m32: cp * sr,
                m13: R::sum_prod2(cy, spcr, sy, sr),
                m23: R::diff_prod(sy, spcr, cy, sr),
                m33: cp * cr,
            },
        }
    }

    /// The rotation angle, in `[0, π]`: `acos((trace - 1) / 2)` like upstream, the argument being
    /// clamped to `[-1, 1]` (a rounded rotation matrix can have a trace slightly outside, where
    /// upstream's `acos` would return NaN and ours would panic with `Fixed: acos domain`).
    ///
    /// Cheap (18 750 gas) but `acos` amplifies the error near `0` and `π`, where its derivative is
    /// `1/θ`: at `θ = 2^-10` the result is off by 1 024 ulp, and below `2^-16` it returns exactly
    /// 0.
    /// Converting to a quaternion and taking `2·atan2` is accurate to 2 ulp everywhere but costs
    /// 45 130 (2.4x); use it when small angles matter (`bench_rotation3_angle__alt_quaternion`,
    /// `test_angle_alt_quaternion_is_more_accurate_near_zero`). Upstream: `Rotation3::angle`.
    fn angle(self: Rotation3<T>) -> T {
        let m = self.matrix;
        let half = (m.m11 + m.m22 + m.m33 - R::ONE) * R::HALF;
        Tr::acos(R::clamp(half, R::NEG_ONE, R::ONE))
    }

    /// The rotation vector `axis · angle` (the logarithmic map), or the zero vector when the axis
    /// is undetermined (identity or half turn). Upstream: `Rotation3::scaled_axis`.
    fn scaled_axis(self: Rotation3<T>) -> Vector3<T> {
        match Rotation3Trait::axis(self) {
            Some(axis) => axis.value.scale(Self::angle(self)),
            None => Vector3 { x: R::ZERO, y: R::ZERO, z: R::ZERO },
        }
    }

    /// The Tait-Bryan angles `(roll, pitch, yaw)` of the rotation, with
    /// `self = Rz(yaw)·Ry(pitch)·Rx(roll)`: `pitch = -asin(m31)` in `[-π/2, π/2]`,
    /// `roll = atan2(m32, m33)`, `yaw = atan2(m21, m11)`.
    ///
    /// Unlike upstream the entries are NOT divided by `cos(pitch)` first: the divisions cancel
    /// inside `atan2` (that factor is `>= 0`), which saves four divisions and a `cos`. At the
    /// gimbal lock `|m31| >= 1` the angles are not separable: `yaw = 0` and `roll` carries the
    /// whole rotation, like upstream. One `asin` and two `atan2`. Upstream:
    /// `Rotation3::euler_angles`.
    fn euler_angles(self: Rotation3<T>) -> (T, T, T) {
        let m = self.matrix;
        if R::abs(m.m31) < R::ONE {
            (Tr::atan2(m.m32, m.m33), -Tr::asin(m.m31), Tr::atan2(m.m21, m.m11))
        } else if R::is_negative(m.m31) {
            (Tr::atan2(m.m23, m.m22), R::FRAC_PI_2, R::ZERO)
        } else {
            (-Tr::atan2(m.m23, m.m22), -R::FRAC_PI_2, R::ZERO)
        }
    }

    /// `rotation_between` with the angle multiplied by `s`. Upstream's algorithm (one `acos`, one
    /// `from_axis_angle`), which is the only way to scale an angle. `None` in the antiparallel
    /// case. Upstream: `Rotation3::scaled_rotation_between`.
    fn scaled_rotation_between(a: Vector3<T>, b: Vector3<T>, s: T) -> Option<Rotation3<T>> {
        let na = a.try_normalize(R::ZERO);
        let nb = b.try_normalize(R::ZERO);
        match (na, nb) {
            (
                Some(u), Some(v),
            ) => {
                let c = u.cross(v);
                let d = R::clamp(u.dot(v), R::NEG_ONE, R::ONE);
                match UnitTrait::try_new(c, R::ZERO) {
                    Some(axis) => Some(Self::from_axis_angle(axis, Tr::acos(d) * s)),
                    None => if R::is_negative(d) {
                        None
                    } else {
                        Some(Rotation3Trait::identity())
                    },
                }
            },
            _ => Some(Rotation3Trait::identity()),
        }
    }
}

/// The composition of two rotations: the product of the matrices, `lhs` applied last (27 products,
/// each entry one fused `sum_prod3`). Repeated composition drifts out of orthonormality:
/// `renormalize` restores it. Upstream: `Mul`.
pub impl Rotation3Mul<
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
> of Mul<Rotation3<T>> {
    #[inline(always)]
    fn mul(lhs: Rotation3<T>, rhs: Rotation3<T>) -> Rotation3<T> {
        Rotation3 { matrix: lhs.matrix * rhs.matrix }
    }
}

/// `q.into()`: the matrix of a unit quaternion. Upstream: `From<UnitQuaternion> for Rotation3`.
pub impl Rotation3FromUnitQuaternion<
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
> of Into<UnitQuaternion<T>, Rotation3<T>> {
    #[inline(always)]
    fn into(self: UnitQuaternion<T>) -> Rotation3<T> {
        UnitQuaternionTrait::to_rotation_matrix(self)
    }
}

/// `r.into()`: the unit quaternion of a rotation matrix. Upstream:
/// `From<Rotation3> for UnitQuaternion`.
pub impl UnitQuaternionFromRotation3<
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
> of Into<Rotation3<T>, UnitQuaternion<T>> {
    #[inline(always)]
    fn into(self: Rotation3<T>) -> UnitQuaternion<T> {
        UnitQuaternionTrait::from_rotation_matrix(self)
    }
}
