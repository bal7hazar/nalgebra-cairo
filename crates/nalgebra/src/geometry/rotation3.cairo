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

use core::num::traits::One;
use core::ops::Index;
use simba::scalar::{Real, Transcendental};
use crate::base::matrix3::{Matrix3, Matrix3Trait};
use crate::base::matrix4::Matrix4;
use crate::base::point3::Point3;
use crate::base::unit::{Unit, UnitTrait};
use crate::base::vector3::{Vector3, Vector3Trait};
use crate::base::{MatrixMul, MatrixTrMul};
use super::isometry3::Isometry3;
use super::isometry_matrix3::{IsometryMatrix3, IsometryMatrix3Trait};
use super::quaternion::ApproxEqTrait;
use super::similarity3::Similarity3;
use super::similarity_matrix3::{SimilarityMatrix3, SimilarityMatrix3Trait};
use super::translation3::Translation3;
use super::unit_quaternion::{UnitQuaternion, UnitQuaternionAngleTrait, UnitQuaternionTrait};

#[cfg(test)]
mod benches;
#[cfg(test)]
mod tests;

/// Panic messages of `Rotation3` (stable API).
pub mod errors {
    /// `r[(i, j)]` with `i > 2` or `j > 2`.
    pub const INDEX_OUT_OF_BOUNDS: felt252 = 'nalgebra: index out of bounds';
    /// `euler_angles_ordered` with a first axis not orthogonal to the other two (upstream's
    /// `assert_relative_eq!(.., epsilon = 1e-6)`).
    pub const AXES_NOT_ORTHOGONAL: felt252 = 'nalgebra: axes not orthogonal';
}

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
            m41: R::zero(),
            m12: m.m12,
            m22: m.m22,
            m32: m.m32,
            m42: R::zero(),
            m13: m.m13,
            m23: m.m23,
            m33: m.m33,
            m43: R::zero(),
            m14: R::zero(),
            m24: R::zero(),
            m34: R::zero(),
            m44: R::one(),
        }
    }

    // --- transforms --------------------------------------------------------------------------

    /// `self · v`: one `Matrix3 * Vector3`, each component one fused `sum_prod3` (9 products, 3
    /// roundings). Cheaper than `UnitQuaternion::transform_vector` (15 products), which is why a
    /// body transforming several vectors per step should keep its `Rotation3`. Upstream:
    /// `transform_vector` (`r * v`).
    #[inline(always)]
    fn transform_vector(self: Rotation3<T>, v: Vector3<T>) -> Vector3<T> {
        self.matrix.mul_mat(v)
    }

    /// `transform_vector` of the point's coordinates (a rotation fixes the origin). Upstream:
    /// `transform_point` (`r * p`).
    #[inline(always)]
    fn transform_point(self: Rotation3<T>, p: Point3<T>) -> Point3<T> {
        let c = self.matrix.mul_mat(Vector3 { x: p.x, y: p.y, z: p.z });
        Point3 { x: c.x, y: c.y, z: c.z }
    }

    /// `self⁻¹ · v = selfᵀ · v`: one `tr_mul`, as cheap as `transform_vector` (the
    /// transpose is free, it is just another access pattern). Upstream: `inverse_transform_vector`.
    #[inline(always)]
    fn inverse_transform_vector(self: Rotation3<T>, v: Vector3<T>) -> Vector3<T> {
        self.matrix.tr_mul(v)
    }

    /// `inverse_transform_vector` of the point's coordinates. Upstream:
    /// `inverse_transform_point`.
    #[inline(always)]
    fn inverse_transform_point(self: Rotation3<T>, p: Point3<T>) -> Point3<T> {
        let c = self.matrix.tr_mul(Vector3 { x: p.x, y: p.y, z: p.z });
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
        UnitTrait::try_new(v, R::zero())
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

    // --- P09a completion: parts, unit vectors, observer frames ---------------------------------

    /// Alias of `matrix`. Upstream: `Rotation::into_inner`.
    #[inline(always)]
    fn into_inner(self: Rotation3<T>) -> Matrix3<T> {
        self.matrix
    }

    /// Deprecated alias of `into_inner`. Upstream: `Rotation::unwrap`.
    #[inline(always)]
    fn unwrap(self: Rotation3<T>) -> Matrix3<T> {
        self.matrix
    }

    /// The rotation whose matrix has the columns `basis[0]`, `basis[1]`, `basis[2]`, WITHOUT
    /// checking that they are orthonormal. Exact. Upstream: `Rotation3::from_basis_unchecked`.
    #[inline(always)]
    fn from_basis_unchecked(basis: [Vector3<T>; 3]) -> Rotation3<T> {
        let [x, y, z] = basis;
        Rotation3 { matrix: Matrix3Trait::from_columns(x, y, z) }
    }

    /// The rotation taking `self` to `other`: `other * self⁻¹`, one matrix product with the
    /// transpose (27 products, one rounding per entry). Upstream: `rotation_to`.
    #[inline(always)]
    fn rotation_to(self: Rotation3<T>, other: Rotation3<T>) -> Rotation3<T> {
        Rotation3 { matrix: other.matrix * self.matrix.transpose() }
    }

    /// `self * v` for a unit vector: `transform_vector` of its value, re-wrapped WITHOUT
    /// renormalising. Upstream: `Mul<Unit<Vector3>> for Rotation3` (`r * v`).
    #[inline(always)]
    fn transform_unit_vector(self: Rotation3<T>, v: Unit<Vector3<T>>) -> Unit<Vector3<T>> {
        Unit { value: self.matrix.mul_mat(v.value) }
    }

    /// `self⁻¹ * v` for a unit vector: `inverse_transform_vector` of its value, not
    /// renormalised. Upstream: `inverse_transform_unit_vector`.
    #[inline(always)]
    fn inverse_transform_unit_vector(self: Rotation3<T>, v: Unit<Vector3<T>>) -> Unit<Vector3<T>> {
        Unit { value: self.matrix.tr_mul(v.value) }
    }

    /// The left-handed look-at rotation, `face_towards(dir, up).inverse()`: `dir` is mapped to the
    /// POSITIVE `z` axis. `up` MUST not be parallel to `dir`. Upstream: `Rotation3::look_at_lh`.
    #[inline(always)]
    fn look_at_lh(dir: Vector3<T>, up: Vector3<T>) -> Rotation3<T> {
        Rotation3 { matrix: Self::face_towards(dir, up).matrix.transpose() }
    }

    /// Deprecated alias of `face_towards`. Upstream: `Rotation3::new_observer_frames`.
    #[inline(always)]
    fn new_observer_frames(dir: Vector3<T>, up: Vector3<T>) -> Rotation3<T> {
        Self::face_towards(dir, up)
    }

    // --- P09a completion: heterogeneous operators (Cairo's operator traits are homogeneous) ----

    /// `self * q`: the unit quaternion `from_rotation_matrix(self) * q` (Shepperd's method, then
    /// one Hamilton product), upstream's formula. Upstream: `Mul<UnitQuaternion> for Rotation3`.
    #[inline(always)]
    fn mul_unit_quaternion(self: Rotation3<T>, q: UnitQuaternion<T>) -> UnitQuaternion<T> {
        UnitQuaternionTrait::from_rotation_matrix(self) * q
    }

    /// `self / q = self * q⁻¹`: `from_rotation_matrix(self) / q` (one fused product with the
    /// conjugate). Upstream: `Div<UnitQuaternion> for Rotation3`.
    #[inline(always)]
    fn div_unit_quaternion(self: Rotation3<T>, q: UnitQuaternion<T>) -> UnitQuaternion<T> {
        UnitQuaternionTrait::from_rotation_matrix(self) / q
    }

    // --- P09a completion: comparisons and casts -------------------------------------------------

    /// `true` when every entry is `relative_eq` to the matching entry of `other` (within
    /// `epsilon` ulp, or `max_relative` times the larger magnitude; see
    /// `QuaternionTrait::relative_eq`). Upstream: `approx::RelativeEq::relative_eq` (DESIGN D3).
    fn relative_eq(self: Rotation3<T>, other: Rotation3<T>, epsilon: u64, max_relative: T) -> bool {
        let (a, b) = (self.matrix, other.matrix);
        ApproxEqTrait::relative_eq(a.m11, b.m11, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(a.m21, b.m21, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(a.m31, b.m31, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(a.m12, b.m12, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(a.m22, b.m22, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(a.m32, b.m32, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(a.m13, b.m13, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(a.m23, b.m23, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(a.m33, b.m33, epsilon, max_relative)
    }

    /// `true` when every entry is `ulps_eq` to the matching entry of `other` (within `epsilon`
    /// ulp, or `max_ulps` ulp without crossing zero; see `QuaternionTrait::ulps_eq`). Upstream:
    /// `approx::UlpsEq::ulps_eq` (DESIGN D3).
    fn ulps_eq(self: Rotation3<T>, other: Rotation3<T>, epsilon: u64, max_ulps: u32) -> bool {
        let (a, b) = (self.matrix, other.matrix);
        ApproxEqTrait::ulps_eq(a.m11, b.m11, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(a.m21, b.m21, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(a.m31, b.m31, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(a.m12, b.m12, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(a.m22, b.m22, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(a.m32, b.m32, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(a.m13, b.m13, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(a.m23, b.m23, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(a.m33, b.m33, epsilon, max_ulps)
    }

    /// The same rotation with every entry converted by `Into<T, U>` (the identity for the single
    /// scalar `Fixed`). Upstream: `cast` (and `SubsetOf<Rotation3<U>>`).
    fn cast<U, +Into<T, U>, +Drop<U>>(self: Rotation3<T>) -> Rotation3<U> {
        let m = self.matrix;
        Rotation3 {
            matrix: Matrix3 {
                m11: m.m11.into(),
                m21: m.m21.into(),
                m31: m.m31.into(),
                m12: m.m12.into(),
                m22: m.m22.into(),
                m32: m.m32.into(),
                m13: m.m13.into(),
                m23: m.m23.into(),
                m33: m.m33.into(),
            },
        }
    }

    // --- P09b completion: operators with the rotation-matrix isometries / similarities ----------

    /// `self * t`: the isometry of rotation `self` and translation `self · t` (one
    /// `Matrix3 * Vector3`). Upstream: `Mul<Translation> for Rotation` (output `IsometryMatrix3`).
    #[inline(always)]
    fn mul_translation(self: Rotation3<T>, t: Translation3<T>) -> IsometryMatrix3<T> {
        IsometryMatrix3 {
            rotation: self, translation: Translation3 { vector: self.matrix.mul_mat(t.vector) },
        }
    }

    /// `self * iso`: translation `self · iso.translation`, rotation `self · iso.rotation`.
    /// Upstream: `Mul<Isometry<T, Rotation3<T>, 3>> for Rotation3`.
    #[inline(always)]
    fn mul_isometry(self: Rotation3<T>, iso: IsometryMatrix3<T>) -> IsometryMatrix3<T> {
        IsometryMatrix3 {
            rotation: self * iso.rotation,
            translation: Translation3 { vector: self.matrix.mul_mat(iso.translation.vector) },
        }
    }

    /// `self / iso = self * iso⁻¹` (upstream's formula: the inverse is materialised). Upstream:
    /// `Div<Isometry<T, Rotation3<T>, 3>> for Rotation3`.
    #[inline(always)]
    fn div_isometry(self: Rotation3<T>, iso: IsometryMatrix3<T>) -> IsometryMatrix3<T> {
        Self::mul_isometry(self, IsometryMatrix3Trait::inverse(iso))
    }

    /// `self * sim`: the similarity `(self * sim.isometry, sim.scaling)`. Upstream:
    /// `Mul<Similarity<T, Rotation3<T>, 3>> for Rotation3`.
    #[inline(always)]
    fn mul_similarity(self: Rotation3<T>, sim: SimilarityMatrix3<T>) -> SimilarityMatrix3<T> {
        SimilarityMatrix3 { isometry: Self::mul_isometry(self, sim.isometry), scaling: sim.scaling }
    }

    /// `self / sim = self * sim⁻¹` (upstream's formula). Upstream:
    /// `Div<Similarity<T, Rotation3<T>, 3>> for Rotation3`.
    #[inline(always)]
    fn div_similarity(self: Rotation3<T>, sim: SimilarityMatrix3<T>) -> SimilarityMatrix3<T> {
        Self::mul_similarity(self, SimilarityMatrix3Trait::inverse(sim))
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
        if angle == R::zero() {
            return Rotation3Trait::identity();
        }
        let u = axis.value;
        let (s, c) = Tr::sin_cos(angle);
        let omc = R::one() - c;
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
        if n == R::zero() {
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
        let half = (m.m11 + m.m22 + m.m33 - R::one()) * R::HALF;
        Tr::acos(R::clamp(half, R::NEG_ONE, R::one()))
    }

    /// The rotation vector `axis · angle` (the logarithmic map), or the zero vector when the axis
    /// is undetermined (identity or half turn). Upstream: `Rotation3::scaled_axis`.
    fn scaled_axis(self: Rotation3<T>) -> Vector3<T> {
        match Rotation3Trait::axis(self) {
            Some(axis) => axis.value.scale(Self::angle(self)),
            None => Vector3 { x: R::zero(), y: R::zero(), z: R::zero() },
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
        if R::abs(m.m31) < R::one() {
            (Tr::atan2(m.m32, m.m33), -Tr::asin(m.m31), Tr::atan2(m.m21, m.m11))
        } else if R::is_sign_negative(m.m31) {
            (Tr::atan2(m.m23, m.m22), R::frac_pi_2(), R::zero())
        } else {
            (-Tr::atan2(m.m23, m.m22), -R::frac_pi_2(), R::zero())
        }
    }

    /// `rotation_between` with the angle multiplied by `s`. Upstream's algorithm (one `acos`, one
    /// `from_axis_angle`), which is the only way to scale an angle. `None` in the antiparallel
    /// case. Upstream: `Rotation3::scaled_rotation_between`.
    fn scaled_rotation_between(a: Vector3<T>, b: Vector3<T>, s: T) -> Option<Rotation3<T>> {
        let na = a.try_normalize(R::zero());
        let nb = b.try_normalize(R::zero());
        match (na, nb) {
            (
                Some(u), Some(v),
            ) => {
                let c = u.cross(v);
                let d = R::clamp(u.dot(v), R::NEG_ONE, R::one());
                match UnitTrait::try_new(c, R::zero()) {
                    Some(axis) => Some(Self::from_axis_angle(axis, Tr::acos(d) * s)),
                    None => if R::is_sign_negative(d) {
                        None
                    } else {
                        Some(Rotation3Trait::identity())
                    },
                }
            },
            _ => Some(Rotation3Trait::identity()),
        }
    }

    // --- P09a completion ------------------------------------------------------------------------

    /// The rotation whose axis is the direction of `axisangle` and whose angle is its length:
    /// `from_scaled_axis` (the identity for a zero vector, like upstream's `from_axis_angle` with
    /// a zero angle). Upstream: `Rotation3::new`.
    #[inline(always)]
    fn new(axisangle: Vector3<T>) -> Rotation3<T> {
        Self::from_scaled_axis(axisangle)
    }

    /// `(axis, angle)` with the angle in `[0, π]`, or `None` when the axis is undetermined
    /// (identity or half turn): `axis()` and `angle()`, upstream's formula. Upstream:
    /// `Rotation3::axis_angle`.
    fn axis_angle(self: Rotation3<T>) -> Option<(Unit<Vector3<T>>, T)> {
        match Rotation3Trait::axis(self) {
            Some(axis) => Some((axis, Self::angle(self))),
            None => None,
        }
    }

    /// The angle of the rotation taking `self` to `other`, in `[0, π]`: upstream's
    /// `rotation_to(other).angle()`, i.e. `acos((tr(other · selfᵀ) - 1) / 2)`, WITHOUT forming
    /// the product: the trace `Σ other_ij · self_ij` is ONE fused kernel of nine products, and
    /// `(tr - 1) / 2` is floored once from the exact accumulator, where the product floors its
    /// three diagonal entries and `angle` floors the halving too: 32 230 gas against 79 950
    /// (`bench_rotation3_angle_to__alt_rotation_to_angle`, 2.5x). The argument is clamped to
    /// `[-1, 1]` like `angle` (see there for the accuracy of `acos` near `0` and `π`).
    /// Upstream: `Rotation3::angle_to`.
    fn angle_to(self: Rotation3<T>, other: Rotation3<T>) -> T {
        let (a, b) = (self.matrix, other.matrix);
        let acc = R::wide_add_prod(R::wide_add_prod(R::wide_zero(), a.m11, b.m11), a.m21, b.m21);
        let acc = R::wide_add_prod(R::wide_add_prod(acc, a.m31, b.m31), a.m12, b.m12);
        let acc = R::wide_add_prod(R::wide_add_prod(acc, a.m22, b.m22), a.m32, b.m32);
        let acc = R::wide_add_prod(R::wide_add_prod(acc, a.m13, b.m13), a.m23, b.m23);
        let acc = R::wide_sub(R::wide_add_prod(acc, a.m33, b.m33), R::one());
        Tr::acos(R::clamp(R::wide_mul_scalar(acc, R::HALF), R::NEG_ONE, R::one()))
    }

    /// The rotation raised to the real power `n`: `from_axis_angle(axis, n · angle)`, upstream's
    /// formula. When the axis is undetermined (the identity or a half turn) upstream returns
    /// `-I` if `m11 < 0` and the identity otherwise, and so does this port — `-I` is NOT a
    /// rotation (its determinant is `-1`): an upstream quirk kept for fidelity (PLAN M8).
    /// Upstream: `Rotation3::powf`.
    fn powf(self: Rotation3<T>, n: T) -> Rotation3<T> {
        match Rotation3Trait::axis(self) {
            Some(axis) => Self::from_axis_angle(axis, Self::angle(self) * n),
            None => if R::is_sign_negative(self.matrix.m11) && self.matrix.m11 != R::zero() {
                let (o, m) = (R::zero(), R::NEG_ONE);
                Rotation3 {
                    matrix: Matrix3 {
                        m11: m, m21: o, m31: o, m12: o, m22: m, m32: o, m13: o, m23: o, m33: m,
                    },
                }
            } else {
                Rotation3Trait::identity()
            },
        }
    }

    /// Deprecated alias of `euler_angles`. Upstream: `Rotation3::to_euler_angles`.
    #[inline(always)]
    fn to_euler_angles(self: Rotation3<T>) -> (T, T, T) {
        Self::euler_angles(self)
    }

    /// The angles `[θ1, θ2, θ3]` of the decomposition of `self` about the axes `seq` (intrinsic,
    /// or extrinsic when `extrinsic`), and whether they are observable (`false` at a gimbal lock,
    /// where the first angle — the last one when extrinsic — is set to zero). Upstream's
    /// algorithm (Shuster & Markley, "General formula for extracting the Euler angles"), step
    /// for step: `λ = atan2((n1 × n2) · n3, n1 · n3)`, `O = C · self · Cᵀ · R1(λ)` with
    /// `C = [n2, n1 × n2, n1]ᵀ` (three fused matrix products), `θ2 = acos(O33) + λ`, the two
    /// other angles by `atan2` of entries of `O`, then upstream's range adjustments (every angle
    /// in `[-π, π]`, `θ2` in `[0, π]` for a symmetric sequence and in `[-π/2, π/2]`
    /// otherwise).
    ///
    /// `O33` is clamped to `[-1, 1]` before the `acos` (a rounded product can exceed `1` by an
    /// ulp; `angle` does the same). The gimbal-lock threshold is upstream's `1e-6` rad
    /// (`from_ratio(1, 10^6)`). The axes are compared EXACTLY for the symmetric test
    /// (`seq[0] == seq[2]`), like upstream. Panics with `nalgebra: axes not orthogonal` when
    /// `|n1 · n2|` or `|n3 · n1|` exceeds `1e-6` (upstream's assertion). Upstream:
    /// `Rotation3::euler_angles_ordered`.
    fn euler_angles_ordered(
        self: Rotation3<T>, seq: [Unit<Vector3<T>>; 3], extrinsic: bool,
    ) -> ([T; 3], bool) {
        let eps = R::from_ratio(1, 1000000);
        let [s0, s1, s2] = seq;
        let (n1, n2, n3) = if extrinsic {
            (s2.value, s1.value, s0.value)
        } else {
            (s0.value, s1.value, s2.value)
        };
        if R::abs(n1.dot(n2)) > eps || R::abs(n3.dot(n1)) > eps {
            core::panic_with_felt252(errors::AXES_NOT_ORTHOGONAL);
        }
        let n12 = n1.cross(n2);
        let s1 = n12.dot(n3);
        let c1 = n1.dot(n3);
        let lambda = Tr::atan2(s1, c1);
        // `C` has the rows `n2`, `n1 × n2`, `n1`.
        let c = Matrix3 {
            m11: n2.x,
            m21: n12.x,
            m31: n1.x,
            m12: n2.y,
            m22: n12.y,
            m32: n1.y,
            m13: n2.z,
            m23: n12.z,
            m33: n1.z,
        };
        let (z, one) = (R::zero(), R::one());
        let r1l = Matrix3 {
            m11: one, m21: z, m31: z, m12: z, m22: c1, m32: -s1, m13: z, m23: s1, m33: c1,
        };
        let o = c * self.matrix * (c.transpose() * r1l);
        let a1 = Tr::acos(R::clamp(o.m33, R::NEG_ONE, one));
        let safe1 = R::abs(a1) >= eps;
        let safe2 = R::abs(a1 - R::pi()) >= eps;
        let observable = safe1 && safe2;
        let mut a1 = a1 + lambda;
        let (mut a0, mut a2) = (z, z);
        if observable {
            a0 = Tr::atan2(o.m13, -o.m23);
            a2 = Tr::atan2(o.m31, o.m32);
        } else if extrinsic {
            a2 =
                if !safe1 {
                    Tr::atan2(o.m12 - o.m21, o.m11 + o.m22)
                } else {
                    -Tr::atan2(o.m12 + o.m21, o.m11 - o.m22)
                };
        } else {
            a0 =
                if !safe1 {
                    Tr::atan2(o.m12 - o.m21, o.m11 + o.m22)
                } else {
                    Tr::atan2(o.m12 + o.m21, o.m11 - o.m22)
                };
        }
        let adjust = if s0 == s2 {
            a1 < z || a1 > R::pi()
        } else {
            a1 < -R::frac_pi_2() || a1 > R::frac_pi_2()
        };
        if adjust && observable {
            a0 = a0 + R::pi();
            a1 = lambda + lambda - a1;
            a2 = a2 - R::pi();
        }
        let (a0, a1, a2) = (
            Rotation3AngleInternalTrait::<T>::wrap_pi(a0),
            Rotation3AngleInternalTrait::<T>::wrap_pi(a1),
            Rotation3AngleInternalTrait::<T>::wrap_pi(a2),
        );
        if extrinsic {
            ([a2, a1, a0], observable)
        } else {
            ([a0, a1, a2], observable)
        }
    }

    /// `from_matrix_eps(m, default_epsilon, 0, identity)`: the rotation closest to `m`, in closed
    /// form. Upstream: `Rotation3::from_matrix`.
    #[inline(always)]
    fn from_matrix(m: Matrix3<T>) -> Rotation3<T> {
        Self::from_matrix_eps(m, R::default_epsilon(), 0, Rotation3Trait::identity())
    }

    /// The rotation part of `m` (the rotation `R` maximising `tr(Rᵀ m)`), computed as
    /// `UnitQuaternion::from_matrix_eps(m, eps, max_iter, guess.into())` expanded back into a
    /// matrix (one `to_rotation_matrix`): `max_iter = 0` is the closed-form LIMIT of
    /// upstream's iteration, `max_iter > 0` runs upstream's Müller iteration BOUNDED by
    /// `FROM_MATRIX_MAX_ITER` (see `UnitQuaternionAngleTrait::from_matrix_eps` for both, their
    /// costs and the measurements). Upstream iterates on matrices (`R ← exp([ω]×) · R`, 27
    /// products per composition, plus Rodrigues' formula); the quaternion form follows the same
    /// path (`test_from_matrix_alt_matrix_iteration_agrees`) at 1 404 700 gas for 8 iterations
    /// against 1 625 860 (`bench_rotation3_from_matrix__alt_matrix_iterate_8`), and it stays on
    /// the rotation group between the steps. The closed form costs 529 600. Upstream:
    /// `Rotation3::from_matrix_eps`.
    fn from_matrix_eps(
        m: Matrix3<T>, eps: T, max_iter: usize, guess: Rotation3<T>,
    ) -> Rotation3<T> {
        let q = UnitQuaternionAngleTrait::from_matrix_eps(
            m, eps, max_iter, UnitQuaternionTrait::from_rotation_matrix(guess),
        );
        UnitQuaternionTrait::to_rotation_matrix(q)
    }

    /// Spherical interpolation, upstream's formula: the `slerp` of the two unit quaternions
    /// (`from_rotation_matrix` of both, shortest arc), expanded back into a matrix. `t = 0` gives
    /// `self` and `t = 1` gives `other` within the rounding of the round trip. Upstream:
    /// `Rotation3::slerp`.
    fn slerp(self: Rotation3<T>, other: Rotation3<T>, t: T) -> Rotation3<T> {
        let q = UnitQuaternionAngleTrait::slerp(
            UnitQuaternionTrait::from_rotation_matrix(self),
            UnitQuaternionTrait::from_rotation_matrix(other),
            t,
        );
        UnitQuaternionTrait::to_rotation_matrix(q)
    }

    /// `slerp`, or `None` when the rotations are nearly aligned (`UnitQuaternion::try_slerp`,
    /// `epsilon` in scalar units). Upstream: `Rotation3::try_slerp`.
    fn try_slerp(
        self: Rotation3<T>, other: Rotation3<T>, t: T, epsilon: T,
    ) -> Option<Rotation3<T>> {
        match UnitQuaternionAngleTrait::try_slerp(
            UnitQuaternionTrait::from_rotation_matrix(self),
            UnitQuaternionTrait::from_rotation_matrix(other),
            t,
            epsilon,
        ) {
            Some(q) => Some(UnitQuaternionTrait::to_rotation_matrix(q)),
            None => None,
        }
    }
}

/// Crate-internal kernels of `Rotation3AngleTrait`.
#[generate_trait]
pub(crate) impl Rotation3AngleInternalImpl<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Sub<T>, +Neg<T>, +PartialOrd<T>,
> of Rotation3AngleInternalTrait<T> {
    /// `a` brought into `[-π, π]` by at most one turn (upstream's final loop of
    /// `euler_angles_ordered`).
    #[inline(always)]
    fn wrap_pi(a: T) -> T {
        if a < -R::pi() {
            a + R::two_pi()
        } else if a > R::pi() {
            a - R::two_pi()
        } else {
            a
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

/// `a / b = a * b⁻¹`: the product with the transpose (27 products, one rounding per entry).
/// Upstream: `Div<Rotation3>`.
pub impl Rotation3Div<
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
> of Div<Rotation3<T>> {
    #[inline(always)]
    fn div(lhs: Rotation3<T>, rhs: Rotation3<T>) -> Rotation3<T> {
        Rotation3 { matrix: lhs.matrix * rhs.matrix.transpose() }
    }
}

/// `Default::default()`: the identity rotation. Upstream: `Default for Rotation3`.
pub impl Rotation3Default<T, impl R: Real<T>, +Copy<T>, +Drop<T>> of Default<Rotation3<T>> {
    #[inline(always)]
    fn default() -> Rotation3<T> {
        let (o, l) = (R::zero(), R::one());
        Rotation3 {
            matrix: Matrix3 {
                m11: l, m21: o, m31: o, m12: o, m22: l, m32: o, m13: o, m23: o, m33: l,
            },
        }
    }
}

/// `One::one()`: the identity rotation; `is_one` compares with the identity matrix exactly.
/// Upstream: `num::One for Rotation3`.
pub impl Rotation3One<T, impl R: Real<T>, +PartialEq<T>, +Copy<T>, +Drop<T>> of One<Rotation3<T>> {
    #[inline(always)]
    fn one() -> Rotation3<T> {
        Rotation3Default::<T>::default()
    }

    #[inline(always)]
    fn is_one(self: @Rotation3<T>) -> bool {
        *self == Rotation3Default::<T>::default()
    }

    #[inline(always)]
    fn is_non_one(self: @Rotation3<T>) -> bool {
        !Self::is_one(self)
    }
}

/// `r[(i, j)]`: the entry of row `i` and column `j` of the matrix. Panics with
/// `nalgebra: index out of bounds` for `i > 2` or `j > 2`. Upstream: `Index<(usize, usize)> for
/// Rotation`.
pub impl Rotation3Index<T, +Copy<T>, +Drop<T>> of Index<Rotation3<T>, (usize, usize)> {
    type Target = T;

    fn index(ref self: Rotation3<T>, index: (usize, usize)) -> T {
        let m = self.matrix;
        let (i, j) = index;
        let row = match i {
            0 => (m.m11, m.m12, m.m13),
            1 => (m.m21, m.m22, m.m23),
            2 => (m.m31, m.m32, m.m33),
            _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
        };
        let (a, b, c) = row;
        match j {
            0 => a,
            1 => b,
            2 => c,
            _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
        }
    }
}

/// `r.into()`: the isometry of rotation `r` (converted by `from_rotation_matrix`) and zero
/// translation. Upstream: `SubsetOf<Isometry3> for Rotation3` (`nalgebra::convert(r)`).
pub impl Isometry3FromRotation3<
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
> of Into<Rotation3<T>, Isometry3<T>> {
    #[inline(always)]
    fn into(self: Rotation3<T>) -> Isometry3<T> {
        Isometry3 {
            rotation: UnitQuaternionTrait::from_rotation_matrix(self),
            translation: Translation3 {
                vector: Vector3 { x: R::zero(), y: R::zero(), z: R::zero() },
            },
        }
    }
}

/// `r.into()`: the similarity of rotation `r`, zero translation and scaling 1. Upstream:
/// `SubsetOf<Similarity3> for Rotation3` (`nalgebra::convert(r)`).
pub impl Similarity3FromRotation3<
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
> of Into<Rotation3<T>, Similarity3<T>> {
    #[inline(always)]
    fn into(self: Rotation3<T>) -> Similarity3<T> {
        Similarity3 { isometry: self.into(), scaling: R::one() }
    }
}
