//! `UnitQuaternion`: a 3D rotation as a quaternion of unit norm (upstream
//! `nalgebra::UnitQuaternion`).
//!
//! - `UnitQuaternionTrait` / `UnitQuaternionImpl`: construction, products, transforms, conversions
//!   to and from a rotation matrix, renormalisation and the integration step — everything that
//!   needs no trigonometry;
//! - `UnitQuaternionAngleTrait` / `UnitQuaternionAngleImpl`: axis-angle, Euler angles, `slerp` and
//!   `powf`, which additionally need `simba::scalar::Transcendental`;
//! - the operators `*` (composition) and unary `-` live in this module, where the compiler finds
//!   them without any import.
//!
//! The split in two traits mirrors `Vector3Trait` / `Vector3AngleTrait`: a scalar may implement
//! `Real` without `Transcendental`, and then the whole rapier hot path is still available
//! (`q * q`, `transform_vector`, `inverse_transform_vector`, `append_axisangle_linearized`,
//! `renormalize_fast`) — it needs no trigonometry at all.
//!
//! Upstream this type is `Unit<Quaternion<T>>`; here it is a dedicated wrapper struct with the same
//! invariant, so that the operators and the rotation methods live in this module (a Cairo operator
//! impl is only found in the module of its type, and `Unit` lives in `base::unit`). The invariant
//! `|q| = 1` is a CONTRACT: `new_unchecked` does not check it, and the quaternions produced here
//! have a norm of `1` within about 2 ulp, not exactly `1` (fixed point cannot do better).
//! `renormalize_fast` (one Newton step, no square root) brings a drifted rotation back.
//!
//! `q` and `-q` are the same rotation. No operation normalises the sign, except `slerp` (shortest
//! arc) and `axis` / `angle` / `scaled_axis` / `axis_angle` (which report the `w >= 0`
//! representative, i.e. an angle in `[0, π]`), exactly like upstream.
//!
//! Numeric contract (AGENTS.md): every sum of products goes through a fused `Real` kernel (one
//! floor rounding and one overflow check per output scalar); nothing wraps silently.

use core::num::traits::One;
use simba::scalar::{Real, Transcendental};
use crate::base::matrix3::{Matrix3, Matrix3Trait};
use crate::base::matrix4::Matrix4;
use crate::base::point3::Point3;
use crate::base::unit::{Unit, UnitTrait};
use crate::base::vector3::{Vector3, Vector3Trait};
use super::isometry3::Isometry3;
use super::quaternion::{
    Quaternion, QuaternionInternalTrait, QuaternionTrait, QuaternionTranscendentalTrait,
};
use super::rotation3::{Rotation3, Rotation3Trait};
use super::similarity3::Similarity3;
use super::translation3::Translation3;

#[cfg(test)]
mod benches;
#[cfg(test)]
mod ext_benches;
#[cfg(test)]
mod ext_oracle;
#[cfg(test)]
mod ext_tests;
#[cfg(test)]
mod oracle;
#[cfg(test)]
mod tests;

/// Panic messages of `UnitQuaternion` (stable API).
pub mod errors {
    /// `mean_of` of an empty span (upstream asserts that the sum of the outer products is not
    /// zero).
    pub const EMPTY_MEAN: felt252 = 'nalgebra: mean of nothing';
}

/// Upper bound of the Müller iterations of `from_matrix_eps` when `max_iter > 0` (`max_iter = 0`,
/// upstream's "until convergence", computes the limit in closed form instead). Measured on the
/// oracle set (`test_from_matrix_eps_iterations_on_the_oracle_set`): from the identity, 17 of the
/// 24 cases reach the fixed point in 21 to 57 iterations (the convergence is linear) and 7 need
/// more than 64; the bound keeps the worst case finite (about 100 000 gas per iteration).
pub const FROM_MATRIX_MAX_ITER: usize = 64;

/// Upper bound of the successive perturbations `from_matrix_eps` tries at a stationary point
/// before accepting it as the maximum (upstream loops until the distance changes by more than
/// one ulp).
pub const FROM_MATRIX_MAX_PERTURBATIONS: usize = 4;

/// Number of normalised squarings `S ← S² / tr(S²)` of the `mean_of` matrix: the dominant
/// eigenvector is extracted from `S^(2^k)`, whose other eigenvalues have shrunk by
/// `(λ₂ / λ₁)^(2^k)`. Measured on the oracle set
/// (`test_mean_of_squarings_on_the_oracle_set`).
pub const MEAN_OF_SQUARINGS: usize = 12;

/// A 3D rotation: a quaternion of unit norm. Nothing enforces the invariant: build it with
/// `new_normalize` / `try_new` / `from_axis_angle` / ..., or with `new_unchecked` when the
/// quaternion is known to be normalized. Upstream: `UnitQuaternion<T>` (`= Unit<Quaternion<T>>`).
#[derive(Copy, Drop, PartialEq, Serde, Debug, Hash)]
pub struct UnitQuaternion<T> {
    pub quaternion: Quaternion<T>,
}

/// Rotation operations of `UnitQuaternion<T>` that need no trigonometry. By value, unrolled, no
/// loop.
#[generate_trait]
pub impl UnitQuaternionImpl<
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
> of UnitQuaternionTrait<T> {
    // --- construction and parts --------------------------------------------------------------

    /// The identity rotation `1 + 0i + 0j + 0k`. Exact. Upstream: `UnitQuaternion::identity`.
    #[inline(always)]
    fn identity() -> UnitQuaternion<T> {
        UnitQuaternion {
            quaternion: Quaternion { i: R::zero(), j: R::zero(), k: R::zero(), w: R::one() },
        }
    }

    /// Wraps `q` WITHOUT normalizing it: the caller guarantees a unit norm. Upstream:
    /// `UnitQuaternion::new_unchecked` (`Unit::new_unchecked`).
    #[inline(always)]
    fn new_unchecked(q: Quaternion<T>) -> UnitQuaternion<T> {
        UnitQuaternion { quaternion: q }
    }

    /// `q / |q|`: the norm (floored once), then one correctly rounded division per component, so
    /// the error is about `1 + 1 / |q|` ulp per component whatever the magnitude of `q`. Panics
    /// with `Fixed: division by zero` on a zero quaternion. Upstream:
    /// `UnitQuaternion::new_normalize` (`from_quaternion`, `Unit::new_normalize`).
    #[inline(always)]
    fn new_normalize(q: Quaternion<T>) -> UnitQuaternion<T> {
        UnitQuaternion { quaternion: q.normalize() }
    }

    /// `Some(new_normalize(q))`, or `None` when `|q| <= min_norm`. With `min_norm >= 0` it never
    /// divides by zero. Upstream: `Unit::try_new`.
    #[inline(always)]
    fn try_new(q: Quaternion<T>, min_norm: T) -> Option<UnitQuaternion<T>> {
        let n = q.norm();
        if n <= min_norm {
            None
        } else {
            Some(UnitQuaternion { quaternion: q.unscale(n) })
        }
    }

    /// The underlying quaternion (a copy: everything is by value here). Upstream:
    /// `UnitQuaternion::quaternion` (`as_ref`).
    #[inline(always)]
    fn quaternion(self: UnitQuaternion<T>) -> Quaternion<T> {
        self.quaternion
    }

    /// Alias of `quaternion`. Upstream: `Unit::into_inner`.
    #[inline(always)]
    fn into_inner(self: UnitQuaternion<T>) -> Quaternion<T> {
        self.quaternion
    }

    /// The imaginary part `(i, j, k)` = `axis · sin(angle / 2)`. Upstream: `imag` (through
    /// `Deref`), used by rapier's angular-velocity code.
    #[inline(always)]
    fn imag(self: UnitQuaternion<T>) -> Vector3<T> {
        Vector3 { x: self.quaternion.i, y: self.quaternion.j, z: self.quaternion.k }
    }

    /// The real part `cos(angle / 2)`. Upstream: `scalar` (through `Deref`).
    #[inline(always)]
    fn scalar(self: UnitQuaternion<T>) -> T {
        self.quaternion.w
    }

    // --- conjugate, inverse, composition ----------------------------------------------------

    /// The conjugate `(w, -i, -j, -k)`: the inverse rotation. Exact; panics on overflow (`-MIN`).
    /// Upstream: `conjugate`.
    #[inline(always)]
    fn conjugate(self: UnitQuaternion<T>) -> UnitQuaternion<T> {
        UnitQuaternion { quaternion: self.quaternion.conjugate() }
    }

    /// The inverse rotation, which for a unit quaternion is the conjugate (no division, unlike
    /// `Quaternion::try_inverse`). Exact. Upstream: `inverse`.
    #[inline(always)]
    fn inverse(self: UnitQuaternion<T>) -> UnitQuaternion<T> {
        UnitQuaternion { quaternion: self.quaternion.conjugate() }
    }

    /// The rotation `r` such that `r · self = other`, i.e. `other · self⁻¹`: one Hamilton
    /// product on the conjugate, no division. Upstream: `rotation_to` (`other / self`).
    #[inline(always)]
    fn rotation_to(self: UnitQuaternion<T>, other: UnitQuaternion<T>) -> UnitQuaternion<T> {
        UnitQuaternion { quaternion: other.quaternion * self.quaternion.conjugate() }
    }

    /// Dot product of the two quaternions = `cos(angle between them / 2)` up to the sign, fused
    /// (floored once). Cannot overflow. Upstream: `dot` (through `Deref`).
    #[inline(always)]
    fn dot(self: UnitQuaternion<T>, other: UnitQuaternion<T>) -> T {
        self.quaternion.dot(other.quaternion)
    }

    // --- renormalization --------------------------------------------------------------------

    /// Renormalizes exactly, in place: `self` becomes `new_normalize(self.quaternion)`, i.e. one
    /// norm and one exactly correctly rounded division per component, and the norm it had is
    /// returned. Use it when the norm may be far from 1 (after `nlerp`, after an unnormalized
    /// construction). Panics on a zero quaternion. Upstream: `Unit::renormalize` (`&mut self`,
    /// returns the previous norm).
    #[inline(always)]
    fn renormalize(ref self: UnitQuaternion<T>) -> T {
        let n = self.quaternion.norm();
        self = UnitQuaternion { quaternion: self.quaternion.unscale(n) };
        n
    }

    /// Renormalizes, in place, a quaternion whose norm is already close to 1 (accumulated rounding
    /// of repeated composition, as in rapier's per-step update): one Newton step for the inverse
    /// square root, `q · (3 - |q|²) / 2`, with the factor as ONE fused kernel (`mul_add(|q|²,
    /// -1/2, 3/2)`, bit-identical to upstream's `1/2 · (3 - |q|²)`) and one product per
    /// component. No square root, no division: 11 700 gas against 13 780 for the exact
    /// `renormalize`, 15 % cheaper (`bench_unit_quaternion_renormalize_fast__*`; a `Fixed` division
    /// costs 2 800 and a product 1 750, so trading four divisions and a square root for four
    /// products saves only that much).
    ///
    /// With `|q|² = 1 + e` the new squared norm is `1 - 3e²/4 + e³/4`: the error is squared at
    /// every step, so one step is exact to the last ulp for `|e| < 2^-16` and the norm stays within
    /// about 2 ulp below 1. A zero quaternion stays zero (no panic). Panics on overflow when
    /// `|q|²` does not fit (norm above about 46 340). Upstream: `Unit::renormalize_fast`.
    #[inline(always)]
    fn renormalize_fast(ref self: UnitQuaternion<T>) {
        let f = R::mul_add(self.quaternion.norm_squared(), -R::HALF, R::HALF + R::one());
        self = UnitQuaternion { quaternion: self.quaternion.scale(f) };
    }

    /// `true` when the four components are within `ulps` smallest units (raw units for fixed point)
    /// of `other`'s. Note that `q` and `-q` are the same rotation and are NOT `abs_diff_eq`: use
    /// `angle_to` for a rotation distance. Upstream: `approx::AbsDiffEq::abs_diff_eq`, the
    /// tolerance being counted in ulp instead of a float epsilon (DESIGN D3).
    #[inline(always)]
    fn abs_diff_eq(self: UnitQuaternion<T>, other: UnitQuaternion<T>, ulps: u64) -> bool {
        self.quaternion.abs_diff_eq(other.quaternion, ulps)
    }

    // --- axis --------------------------------------------------------------------------------

    /// The rotation axis as a unit vector, or `None` when the rotation is the identity (a zero
    /// imaginary part, i.e. an angle of `0` or `2π`). The sign convention is upstream's: the axis
    /// is flipped when `w < 0`, so that the matching `angle()` lies in `[0, π]`.
    ///
    /// The axis is `imag / |imag|` (one norm, then one correctly rounded division per component),
    /// so its error is about `1 + 1 / |imag|` ulp per component: the axis of a rotation by a very
    /// small angle is poorly determined (`|imag| = sin(angle / 2)`), which is why `scaled_axis` is
    /// the right output for integration. Upstream: `axis`.
    fn axis(self: UnitQuaternion<T>) -> Option<Unit<Vector3<T>>> {
        let v = Self::imag(self);
        let v = if R::is_sign_negative(self.quaternion.w) {
            Vector3 { x: -v.x, y: -v.y, z: -v.z }
        } else {
            v
        };
        UnitTrait::try_new(v, R::zero())
    }

    // --- conversions to / from a rotation matrix --------------------------------------------

    /// The rotation as a 3x3 matrix, in upstream's form `(w² + i² - j² - k², 2(ij - wk), ...)`
    /// (equal to the `1 - 2(j² + k²)` form only when `|q|` is exactly 1, which fixed point cannot
    /// guarantee; upstream's form stays exact for a quaternion that has drifted).
    ///
    /// Each of the nine entries is ONE fused kernel — four products for a diagonal entry
    /// (`Real::wide_*`, signs folded into the accumulation), two for an off-diagonal one (the
    /// factor 2 folded into a doubled operand, which is exact) — so every entry is floored once:
    /// 24 products, 9 roundings, no division, no trigonometry, entries within 2 ulp of the exactly
    /// rounded matrix (oracle tolerance: 4 ulp).
    ///
    /// Measured 22 370 gas. The `1 - 2(j² + k²)` diagonal would cost 21 770 (-2.7 %) but is only
    /// valid for an exactly unit quaternion, and turns a drifted one into a matrix that is not even
    /// a similarity (`bench_unit_quaternion_to_rotation_matrix__alt_one_minus`,
    /// `test_to_rotation_matrix_alt_one_minus_differs_by_the_norm_defect`): upstream's form is
    /// kept.
    ///
    /// The conversion costs as much as 3.5 `Rotation3` rotations, so the break-even against
    /// `transform_vector` (23 230) is TWO vectors: 1 vector 23 230 against 32 340, 2 vectors 47 580
    /// against 37 870, 3 vectors 70 910 against 44 440
    /// (`bench_unit_quaternion_transform_vector_x2__*` and `_x3__*`). A body that transforms two
    /// vectors or more per step should cache its `Rotation3`.
    /// Upstream: `to_rotation_matrix`.
    fn to_rotation_matrix(self: UnitQuaternion<T>) -> Rotation3<T> {
        let Quaternion { i, j, k, w } = self.quaternion;
        // Exact doublings (|component| <= 1 for a unit quaternion).
        let i2 = i + i;
        let j2 = j + j;
        let k2 = k + k;
        // Diagonal: w² ± i² ± j² ± k², one wide accumulation each (`Real::Wide` is not
        // `Copy`, so the w² term is accumulated three times: one felt multiplication each).
        let m11 = R::wide_rescale(
            R::wide_sub_prod(
                R::wide_sub_prod(
                    R::wide_add_prod(R::wide_add_prod(R::wide_zero(), w, w), i, i), j, j,
                ),
                k,
                k,
            ),
        );
        let m22 = R::wide_rescale(
            R::wide_sub_prod(
                R::wide_add_prod(
                    R::wide_sub_prod(R::wide_add_prod(R::wide_zero(), w, w), i, i), j, j,
                ),
                k,
                k,
            ),
        );
        let m33 = R::wide_rescale(
            R::wide_add_prod(
                R::wide_sub_prod(
                    R::wide_sub_prod(R::wide_add_prod(R::wide_zero(), w, w), i, i), j, j,
                ),
                k,
                k,
            ),
        );
        Rotation3 {
            matrix: Matrix3 {
                m11,
                m21: R::sum_prod2(i2, j, w, k2),
                m31: R::diff_prod(i2, k, w, j2),
                m12: R::diff_prod(i2, j, w, k2),
                m22,
                m32: R::sum_prod2(w, i2, j2, k),
                m13: R::sum_prod2(w, j2, i2, k),
                m23: R::diff_prod(j2, k, w, i2),
                m33,
            },
        }
    }

    /// The unit quaternion of a rotation matrix, by Shepperd's method: the branch with the largest
    /// of `w²`, `i²`, `j²`, `k²` is selected by comparisons on the trace and the diagonal, so
    /// the square root is always taken of a value `>= 1` and the three divisions by `2·sqrt(...)
    /// >= 2`
    /// lose no precision. One square root, one product and three divisions.
    ///
    /// The sign is upstream's (the selected branch fixes it; `w > 0` only in the trace branch):
    /// `q` and `-q` are the same rotation, and the oracle vectors are matched in sign too. `r` MUST
    /// be a rotation matrix; for one that has drifted, `Rotation3::renormalize` first. Upstream:
    /// `UnitQuaternion::from_rotation_matrix`.
    fn from_rotation_matrix(r: Rotation3<T>) -> UnitQuaternion<T> {
        let m = r.matrix;
        let tr = m.m11 + m.m22 + m.m33;
        if R::is_sign_positive(tr) {
            // d = sqrt(1 + tr) in (1, 2]; upstream's `denom` is 2d and its `quarter · denom` is
            // d/2.
            let d = R::sqrt(R::one() + tr);
            let denom = d + d;
            UnitQuaternion {
                quaternion: {
                    let (i, j, k) = R::div3(m.m32 - m.m23, m.m13 - m.m31, m.m21 - m.m12, denom);
                    Quaternion { i, j, k, w: d * R::HALF }
                },
            }
        } else if m.m11 > m.m22 && m.m11 > m.m33 {
            let d = R::sqrt(R::one() + m.m11 - m.m22 - m.m33);
            let denom = d + d;
            UnitQuaternion {
                quaternion: {
                    let (j, k, w) = R::div3(m.m12 + m.m21, m.m13 + m.m31, m.m32 - m.m23, denom);
                    Quaternion { i: d * R::HALF, j, k, w }
                },
            }
        } else if m.m22 > m.m33 {
            let d = R::sqrt(R::one() + m.m22 - m.m11 - m.m33);
            let denom = d + d;
            UnitQuaternion {
                quaternion: {
                    let (i, k, w) = R::div3(m.m12 + m.m21, m.m23 + m.m32, m.m13 - m.m31, denom);
                    Quaternion { i, j: d * R::HALF, k, w }
                },
            }
        } else {
            let d = R::sqrt(R::one() + m.m33 - m.m11 - m.m22);
            let denom = d + d;
            UnitQuaternion {
                quaternion: {
                    let (i, j, w) = R::div3(m.m13 + m.m31, m.m23 + m.m32, m.m21 - m.m12, denom);
                    Quaternion { i, j, k: d * R::HALF, w }
                },
            }
        }
    }

    /// The rotation as a homogeneous 4x4 matrix (the rotation block, a zero translation and
    /// `m44 = 1`). Upstream: `to_homogeneous`.
    #[inline(always)]
    fn to_homogeneous(self: UnitQuaternion<T>) -> Matrix4<T> {
        Rotation3Trait::to_homogeneous(Self::to_rotation_matrix(self))
    }

    // --- transforms --------------------------------------------------------------------------

    /// `q · v · q⁻¹`, the rotated vector, through the expanded form
    /// `v + 2w·(u × v) + 2u × (u × v)` with `u = imag(q)`: two cross products (each component
    /// one fused `diff_prod`), exact doublings, and one wide accumulation per output component
    /// (`w·t + cross + v`, floored once). 15 products, 9 roundings, no division.
    ///
    /// Measured 23 230 gas, against 24 720 for the naive sandwich `q · (0, v) · q⁻¹` (32
    /// products, only 6 % dearer, because a fused Hamilton product is cheap) and 32 340 for
    /// converting to a matrix and multiplying. It is the cheapest way to rotate ONE vector; from
    /// two vectors on, one `to_rotation_matrix` and `Rotation3::transform_vector` win
    /// (`bench_unit_quaternion_transform_vector*__*`).
    ///
    /// Panics on overflow of an intermediate doubling (`|v|` above about `2^30`). Upstream:
    /// `transform_vector` (`q * v`).
    fn transform_vector(self: UnitQuaternion<T>, v: Vector3<T>) -> Vector3<T> {
        let u = Self::imag(self);
        let c = u.cross(v);
        let t = Vector3 { x: c.x + c.x, y: c.y + c.y, z: c.z + c.z };
        let uxt = u.cross(t);
        let w = self.quaternion.w;
        Vector3 {
            x: R::wide_rescale(
                R::wide_add(R::wide_add(R::wide_add_prod(R::wide_zero(), w, t.x), uxt.x), v.x),
            ),
            y: R::wide_rescale(
                R::wide_add(R::wide_add(R::wide_add_prod(R::wide_zero(), w, t.y), uxt.y), v.y),
            ),
            z: R::wide_rescale(
                R::wide_add(R::wide_add(R::wide_add_prod(R::wide_zero(), w, t.z), uxt.z), v.z),
            ),
        }
    }

    /// `transform_vector` of the point's coordinates (a rotation fixes the origin). Upstream:
    /// `transform_point` (`q * p`).
    #[inline(always)]
    fn transform_point(self: UnitQuaternion<T>, p: Point3<T>) -> Point3<T> {
        let c = Self::transform_vector(self, Vector3 { x: p.x, y: p.y, z: p.z });
        Point3 { x: c.x, y: c.y, z: c.z }
    }

    /// `q⁻¹ · v · q`: the sandwich of `transform_vector` on the conjugate, with the
    /// conjugate's signs folded into the operand order instead of negating the imaginary part `u`:
    /// `(-u) × v = v × u` and `(-u) × t = t × u` are the same exact products, so
    /// `t = 2·(v × u)` and the result `v + w·t + t × u` are bit-identical to
    /// `transform_vector(conjugate(self), v)` without its three negations
    /// (`bench_unit_quaternion_inverse_transform_vector__alt_conjugate_then_transform`,
    /// `test_inverse_transform_vector_fused_matches_conjugate_then_transform`). A component of
    /// the quaternion equal to the scalar's `MIN` no longer panics on the negation. 15 products,
    /// 9 roundings, like `transform_vector`: 23 230 gas against 23 830 with the negations.
    /// Upstream: `inverse_transform_vector`.
    fn inverse_transform_vector(self: UnitQuaternion<T>, v: Vector3<T>) -> Vector3<T> {
        let u = Self::imag(self);
        let c = v.cross(u);
        let t = Vector3 { x: c.x + c.x, y: c.y + c.y, z: c.z + c.z };
        let txu = t.cross(u);
        let w = self.quaternion.w;
        Vector3 {
            x: R::wide_rescale(
                R::wide_add(R::wide_add(R::wide_add_prod(R::wide_zero(), w, t.x), txu.x), v.x),
            ),
            y: R::wide_rescale(
                R::wide_add(R::wide_add(R::wide_add_prod(R::wide_zero(), w, t.y), txu.y), v.y),
            ),
            z: R::wide_rescale(
                R::wide_add(R::wide_add(R::wide_add_prod(R::wide_zero(), w, t.z), txu.z), v.z),
            ),
        }
    }

    /// `inverse_transform_vector` of the point's coordinates. Upstream: `inverse_transform_point`.
    #[inline(always)]
    fn inverse_transform_point(self: UnitQuaternion<T>, p: Point3<T>) -> Point3<T> {
        let c = Self::inverse_transform_vector(self, Vector3 { x: p.x, y: p.y, z: p.z });
        Point3 { x: c.x, y: c.y, z: c.z }
    }

    // --- rotation between two vectors -------------------------------------------------------

    /// The shortest rotation taking the direction of `a` to the direction of `b`, or `None` when
    /// they are exactly antiparallel (the axis is then undefined), like upstream. A zero-length
    /// input gives the identity.
    ///
    /// Trigonometry-free, unlike upstream (`from_axis_angle(a × b / |a × b|, acos(a · b))`):
    /// with the normalised inputs `u`, `v`, the quaternion `(1 + u·v, u × v)` normalised is
    /// exactly `(cos(θ/2), axis·sin(θ/2))`, since `|(1 + cos θ, sin θ·axis)| = 2·cos(θ/2)`.
    /// Measured 49 440 gas against 85 760 for upstream's path
    /// (`bench_unit_quaternion_rotation_between__*`): the `acos` and the `sin_cos` are saved, and
    /// the axis and the angle are never materialised.
    ///
    /// Accuracy degrades as the two directions get antiparallel (`1 + u·v → 0`), where the
    /// rotation itself is ill-conditioned. `None` is returned only when `u × v` floors to exactly
    /// zero AND `u · v < 0`; a pair that is antiparallel only to within rounding still gives the
    /// (correct)
    /// half turn about the rounded cross product. Upstream: `UnitQuaternion::rotation_between`.
    fn rotation_between(a: Vector3<T>, b: Vector3<T>) -> Option<UnitQuaternion<T>> {
        match (a.try_normalize(R::zero()), b.try_normalize(R::zero())) {
            (Some(u), Some(v)) => Self::rotation_between_axis(Unit { value: u }, Unit { value: v }),
            // A zero-length input has no direction: upstream returns the identity.
            _ => Some(Self::identity()),
        }
    }

    /// `rotation_between` of two UNIT vectors (no normalisation): the same trigonometry-free
    /// kernel, `(1 + a·b, a × b)` normalised, and `None` when `a × b` floors to zero with
    /// `a · b < 0` (exactly antiparallel). Upstream computes `scaled_rotation_between_axis(a, b,
    /// 1)` through `acos` and `from_axis_angle`; the two agree to the oracle tolerance. Upstream:
    /// `UnitQuaternion::rotation_between_axis`.
    fn rotation_between_axis(
        a: Unit<Vector3<T>>, b: Unit<Vector3<T>>,
    ) -> Option<UnitQuaternion<T>> {
        let (u, v) = (a.value, b.value);
        let c = u.cross(v);
        if c.is_zero() {
            if R::is_sign_negative(Vector3Trait::dot(u, v)) {
                // A half turn about an undefined axis: not a simple rotation.
                None
            } else {
                Some(Self::identity())
            }
        } else {
            let q = Quaternion { i: c.x, j: c.y, k: c.z, w: R::one() + Vector3Trait::dot(u, v) };
            Some(Self::new_normalize(q))
        }
    }

    // --- interpolation and integration ------------------------------------------------------

    /// Normalised linear interpolation: `lerp` of the two quaternions, then an exact
    /// normalisation. Unlike `slerp` it does NOT take the shortest arc (upstream does not either)
    /// and its angular velocity is not constant, but it needs no trigonometry (4 fused lerps, one
    /// norm, 4 divisions). `t` is not clamped.
    ///
    /// Panics with `Fixed: division by zero` when the interpolated quaternion is zero, which
    /// happens only for exactly opposite rotations at `t = 1/2`. Upstream: `nlerp`.
    #[inline(always)]
    fn nlerp(self: UnitQuaternion<T>, other: UnitQuaternion<T>, t: T) -> UnitQuaternion<T> {
        Self::new_normalize(self.quaternion.lerp(other.quaternion, t))
    }

    /// The rotation integrated by the angular increment `axisangle` (a rotation vector, e.g.
    /// `ω · dt`), linearised: `q + (1/2)·(0, axisangle)·q`, then normalised. This is rapier's
    /// per-step orientation update: no trigonometry at all, second-order accurate in
    /// `|axisangle|` (the exact update `from_scaled_axis(axisangle) * q` costs a `sin_cos` and a
    /// full Hamilton product).
    ///
    /// The reduced Hamilton product by a pure quaternion is unrolled with the addition of `q`
    /// folded into the accumulation: three products and one exact addition per component, one
    /// rounding each (12 products). The normalisation is EXACT (`new_normalize`) like upstream, not
    /// `renormalize_fast`: the norm of the sum is `sqrt(1 + |axisangle|²/4)`, too far from 1 for a
    /// single Newton step (at `|axisangle| = 1` the fast step would be off by 4e-3).
    /// Upstream: `append_axisangle_linearized`.
    fn append_axisangle_linearized(
        self: UnitQuaternion<T>, axisangle: Vector3<T>,
    ) -> UnitQuaternion<T> {
        let Quaternion { i, j, k, w } = self.quaternion;
        let (a, b, c) = (axisangle.x * R::HALF, axisangle.y * R::HALF, axisangle.z * R::HALF);
        // q + (0, a, b, c) · q, component by component.
        let nw = R::wide_sub_prod(R::wide_add(R::wide_zero(), w), a, i);
        let nw = R::wide_rescale(R::wide_sub_prod(R::wide_sub_prod(nw, b, j), c, k));
        let ni = R::wide_add_prod(R::wide_add(R::wide_zero(), i), a, w);
        let ni = R::wide_rescale(R::wide_sub_prod(R::wide_add_prod(ni, b, k), c, j));
        let nj = R::wide_add_prod(R::wide_add(R::wide_zero(), j), b, w);
        let nj = R::wide_rescale(R::wide_add_prod(R::wide_sub_prod(nj, a, k), c, i));
        let nk = R::wide_add_prod(R::wide_add(R::wide_zero(), k), c, w);
        let nk = R::wide_rescale(R::wide_sub_prod(R::wide_add_prod(nk, a, j), b, i));
        Self::new_normalize(Quaternion { i: ni, j: nj, k: nk, w: nw })
    }

    // --- P08 completion: construction and conversions ------------------------------------------

    /// Alias of `new_normalize`. Upstream: `UnitQuaternion::from_quaternion`.
    #[inline(always)]
    fn from_quaternion(q: Quaternion<T>) -> UnitQuaternion<T> {
        UnitQuaternion { quaternion: q.normalize() }
    }

    /// The rotation whose matrix has the columns `basis[0]`, `basis[1]`, `basis[2]`, WITHOUT
    /// checking that they are orthonormal: `from_rotation_matrix` of that matrix (Shepperd's
    /// method). Upstream: `UnitQuaternion::from_basis_unchecked`.
    fn from_basis_unchecked(basis: [Vector3<T>; 3]) -> UnitQuaternion<T> {
        let [x, y, z] = basis;
        Self::from_rotation_matrix(Rotation3 { matrix: Matrix3Trait::from_columns(x, y, z) })
    }

    /// The same rotation with every component converted by `Into<T, U>` (the identity for the
    /// single scalar `Fixed`). Upstream: `cast` (and `SubsetOf<UnitQuaternion<U>>`).
    fn cast<U, +Into<T, U>, +Drop<U>>(self: UnitQuaternion<T>) -> UnitQuaternion<U> {
        UnitQuaternion { quaternion: self.quaternion.cast() }
    }

    /// `self.quaternion.lerp(other.quaternion, t)`: the plain linear interpolation of the two
    /// quaternions, NOT a unit quaternion (see `nlerp`). Upstream: `UnitQuaternion::lerp`.
    #[inline(always)]
    fn lerp(self: UnitQuaternion<T>, other: UnitQuaternion<T>, t: T) -> Quaternion<T> {
        self.quaternion.lerp(other.quaternion, t)
    }

    /// `relative_eq` of the quaternions (component-wise, tolerances in ulp; `q` and `-q` are not
    /// equal for it). Upstream: `approx::RelativeEq::relative_eq`.
    #[inline(always)]
    fn relative_eq(
        self: UnitQuaternion<T>, other: UnitQuaternion<T>, epsilon: u64, max_relative: T,
    ) -> bool {
        self.quaternion.relative_eq(other.quaternion, epsilon, max_relative)
    }

    /// `ulps_eq` of the quaternions (component-wise, tolerances in ulp). Upstream:
    /// `approx::UlpsEq::ulps_eq`.
    #[inline(always)]
    fn ulps_eq(
        self: UnitQuaternion<T>, other: UnitQuaternion<T>, epsilon: u64, max_ulps: u32,
    ) -> bool {
        self.quaternion.ulps_eq(other.quaternion, epsilon, max_ulps)
    }

    // --- P08 completion: observer frames -------------------------------------------------------

    /// The rotation mapping the `z` axis to the direction of `dir`, with the `y` axis as close as
    /// possible to `up`: `from_rotation_matrix(Rotation3::face_towards(dir, up))`, like upstream.
    /// `up` MUST not be parallel to `dir` (`Fixed: division by zero`). Upstream:
    /// `UnitQuaternion::face_towards`.
    #[inline(always)]
    fn face_towards(dir: Vector3<T>, up: Vector3<T>) -> UnitQuaternion<T> {
        Self::from_rotation_matrix(Rotation3Trait::face_towards(dir, up))
    }

    /// Deprecated alias of `face_towards`. Upstream: `UnitQuaternion::new_observer_frames`.
    #[inline(always)]
    fn new_observer_frames(dir: Vector3<T>, up: Vector3<T>) -> UnitQuaternion<T> {
        Self::face_towards(dir, up)
    }

    /// The right-handed look-at rotation, `face_towards(-dir, up).inverse()`: `dir` is mapped to
    /// the NEGATIVE `z` axis. Upstream: `UnitQuaternion::look_at_rh`.
    #[inline(always)]
    fn look_at_rh(dir: Vector3<T>, up: Vector3<T>) -> UnitQuaternion<T> {
        let neg = Vector3 { x: -dir.x, y: -dir.y, z: -dir.z };
        Self::face_towards(neg, up).conjugate()
    }

    /// The left-handed look-at rotation, `face_towards(dir, up).inverse()`: `dir` is mapped to the
    /// POSITIVE `z` axis. Upstream: `UnitQuaternion::look_at_lh`.
    #[inline(always)]
    fn look_at_lh(dir: Vector3<T>, up: Vector3<T>) -> UnitQuaternion<T> {
        Self::face_towards(dir, up).conjugate()
    }

    // --- P08 completion: unit vectors ----------------------------------------------------------

    /// `self * v` for a unit vector: `transform_vector` of its value, re-wrapped WITHOUT
    /// renormalising (a rotation preserves the norm, up to the rounding of the three components).
    /// Upstream: `Mul<Unit<Vector3>>` (`q * v`).
    #[inline(always)]
    fn transform_unit_vector(self: UnitQuaternion<T>, v: Unit<Vector3<T>>) -> Unit<Vector3<T>> {
        Unit { value: Self::transform_vector(self, v.value) }
    }

    /// `self⁻¹ * v` for a unit vector: `inverse_transform_vector` of its value, not
    /// renormalised. Upstream: `inverse_transform_unit_vector`.
    #[inline(always)]
    fn inverse_transform_unit_vector(
        self: UnitQuaternion<T>, v: Unit<Vector3<T>>,
    ) -> Unit<Vector3<T>> {
        Unit { value: Self::inverse_transform_vector(self, v.value) }
    }

    // --- P08 completion: heterogeneous operators (Cairo's operator traits are homogeneous) -----

    /// `self * r` with a rotation matrix: `self * from_rotation_matrix(r)`, a unit quaternion.
    /// Upstream: `Mul<Rotation3> for UnitQuaternion` (`q * r`).
    #[inline(always)]
    fn mul_rotation(self: UnitQuaternion<T>, r: Rotation3<T>) -> UnitQuaternion<T> {
        UnitQuaternion { quaternion: self.quaternion * Self::from_rotation_matrix(r).quaternion }
    }

    /// `self / r = self * r⁻¹` with a rotation matrix: `from_rotation_matrix(r)`, then the fused
    /// product by its conjugate. Upstream: `Div<Rotation3> for UnitQuaternion` (`q / r`).
    #[inline(always)]
    fn div_rotation(self: UnitQuaternion<T>, r: Rotation3<T>) -> UnitQuaternion<T> {
        UnitQuaternion {
            quaternion: self.quaternion.mul_conj(Self::from_rotation_matrix(r).quaternion),
        }
    }

    /// `self * t`: the isometry that translates by `t` then rotates, i.e. rotation `self` and
    /// translation `self · t` (one `transform_vector`). Upstream: `Mul<Translation3> for
    /// UnitQuaternion` (`q * t`, an `Isometry3`).
    #[inline(always)]
    fn mul_translation(self: UnitQuaternion<T>, t: Translation3<T>) -> Isometry3<T> {
        Isometry3 {
            rotation: self,
            translation: Translation3 { vector: Self::transform_vector(self, t.vector) },
        }
    }

    /// `self * iso`: rotation `self * iso.rotation`, translation `self · iso.translation` (one
    /// Hamilton product, one `transform_vector`). Upstream: `Mul<Isometry3> for UnitQuaternion`.
    #[inline(always)]
    fn mul_isometry(self: UnitQuaternion<T>, iso: Isometry3<T>) -> Isometry3<T> {
        Isometry3 {
            rotation: UnitQuaternion { quaternion: self.quaternion * iso.rotation.quaternion },
            translation: Translation3 {
                vector: Self::transform_vector(self, iso.translation.vector),
            },
        }
    }

    /// `self / iso = self * iso⁻¹`, WITHOUT forming the inverse isometry: with `q = self *
    /// iso.rotation⁻¹` (one fused product with the conjugate), the rotation is `q` and the
    /// translation `q · (-iso.translation)` — one vector rotation where upstream's `self *
    /// iso.inverse()` rotates twice (`bench_unit_quaternion_div_isometry__alt_inverse_then_mul`,
    /// agreeing to a few ulp: `test_div_isometry_alt_inverse_then_mul_agrees`). Panics on
    /// overflow (`-MIN` of a translation component). Upstream: `Div<Isometry3> for
    /// UnitQuaternion`.
    fn div_isometry(self: UnitQuaternion<T>, iso: Isometry3<T>) -> Isometry3<T> {
        let q = UnitQuaternion { quaternion: self.quaternion.mul_conj(iso.rotation.quaternion) };
        let t = iso.translation.vector;
        Isometry3 {
            rotation: q,
            translation: Translation3 {
                vector: Self::transform_vector(q, Vector3 { x: -t.x, y: -t.y, z: -t.z }),
            },
        }
    }

    /// `self * sim`: the isometry part composed like `mul_isometry`, the scaling unchanged.
    /// Upstream: `Mul<Similarity3> for UnitQuaternion`.
    #[inline(always)]
    fn mul_similarity(self: UnitQuaternion<T>, sim: Similarity3<T>) -> Similarity3<T> {
        Similarity3 { isometry: Self::mul_isometry(self, sim.isometry), scaling: sim.scaling }
    }

    /// `self / sim = self * sim⁻¹`, without forming the inverse: with `q = self *
    /// sim.rotation⁻¹`, the translation is `q · (-sim.translation) / sim.scaling` (one
    /// rotation, three correctly rounded divisions) and the scaling `1 / sim.scaling` (correctly
    /// rounded, as in `Similarity3::inverse`). Panics with `Fixed: division by zero` on a zero
    /// scaling.
    /// Upstream: `Div<Similarity3> for UnitQuaternion`.
    fn div_similarity(self: UnitQuaternion<T>, sim: Similarity3<T>) -> Similarity3<T> {
        let iso = Self::div_isometry(self, sim.isometry);
        let v = iso.translation.vector;
        let (x, y, z) = R::div3(v.x, v.y, v.z, sim.scaling);
        Similarity3 {
            isometry: Isometry3 {
                rotation: iso.rotation, translation: Translation3 { vector: Vector3 { x, y, z } },
            },
            scaling: R::div(R::one(), sim.scaling),
        }
    }

    // --- P08 completion: mean ------------------------------------------------------------------

    /// The mean rotation of a set of unit quaternions: the unit eigenvector of the largest
    /// eigenvalue of `M = Σ q qᵀ` (Oshman & Carmi 2006), which ignores the signs of the inputs
    /// (`q` and `-q` weigh the same).
    ///
    /// `M` is accumulated exactly (ten wide sums, a loop over the span: the input is dynamic) and
    /// scaled by `1 / n` (its trace is `n` for unit inputs), then its dominant eigenvector is
    /// extracted by `MEAN_OF_SQUARINGS` normalised squarings `S ← S² / tr(S²)` — a fixed
    /// number of unrolled steps (40 + 16 products, one reciprocal each), after which `S` is the
    /// rank-one projector `v vᵀ` to within `(λ₂ / λ₁)^4096` — and its column of largest
    /// diagonal is normalised. Upstream runs a symmetric eigendecomposition (at most 10 QR sweeps);
    /// there is no 4x4 eigensolver here, and the dominant eigenvector is all the mean needs.
    ///
    /// **Deviation:** upstream builds its result with `Quaternion::new(v[0], v[1], v[2], v[3])`
    /// from an eigenvector stored in `(i, j, k, w)` order, which permutes the components (its mean
    /// of identities is the half turn about `z`); the mean here is the eigenvector itself. The
    /// oracle compares with upstream's result un-permuted. The sign of the result is the one that
    /// makes its largest component positive. When the two largest eigenvalues are equal the mean
    /// is not unique (upstream picks one as well). Panics with `nalgebra: mean of nothing` on an
    /// empty span. Upstream: `UnitQuaternion::mean_of`.
    fn mean_of(unit_quaternions: Span<UnitQuaternion<T>>) -> UnitQuaternion<T> {
        let n = unit_quaternions.len();
        if n == 0 {
            core::panic_with_felt252(errors::EMPTY_MEAN);
        }
        let s = UnitQuaternionInternalTrait::<T>::outer_sum(unit_quaternions, n);
        let s = UnitQuaternionInternalTrait::<T>::dominant_projector(s);
        UnitQuaternionInternalTrait::<T>::dominant_column(s)
    }
}

/// Crate-internal kernels of `UnitQuaternion<T>` (WP 8.0: the public API is strictly upstream's):
/// the fused `conj_mul` of `Isometry3::inv_mul` (upstream writes `self.inverse() * other`), and the
/// by-value forms of the in-place `renormalize` / `renormalize_fast` for the tests and the
/// value-style call sites.
#[generate_trait]
pub(crate) impl UnitQuaternionInternalImpl<
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
> of UnitQuaternionInternalTrait<T> {
    /// `self` renormalized exactly (`UnitQuaternionTrait::renormalize`), by value.
    #[inline(always)]
    fn renormalized(self: UnitQuaternion<T>) -> UnitQuaternion<T> {
        let mut r = self;
        let _ = UnitQuaternionTrait::renormalize(ref r);
        r
    }

    /// `self` renormalized by one Newton step (`UnitQuaternionTrait::renormalize_fast`), by value.
    #[inline(always)]
    fn renormalized_fast(self: UnitQuaternion<T>) -> UnitQuaternion<T> {
        let mut r = self;
        UnitQuaternionTrait::renormalize_fast(ref r);
        r
    }

    /// `self⁻¹ · other` (= `self.conjugate() * other`): the rotation `other` expressed in the
    /// frame of `self`, as ONE fused Hamilton product with the conjugate's signs folded in
    /// (`QuaternionTrait::conj_mul`) — bit-identical to `self.inverse() * other`, three negations
    /// cheaper, and a component equal to the scalar's `MIN` no longer panics. Upstream has no
    /// direct equivalent: it replaces `self.inverse() * other` (as in `Isometry3::inv_mul`).
    #[inline(always)]
    fn conj_mul(self: UnitQuaternion<T>, other: UnitQuaternion<T>) -> UnitQuaternion<T> {
        UnitQuaternion { quaternion: self.quaternion.conj_mul(other.quaternion) }
    }

    /// `Σ q qᵀ / n` over the span, in `(w, i, j, k)` order: ten exact wide sums (one loop over
    /// the dynamic input), each scaled by the rounded `1 / n` with ONE rounding
    /// (`Real::wide_mul_scalar`). `n` must be the length of the span, at least 1.
    fn outer_sum(unit_quaternions: Span<UnitQuaternion<T>>, n: usize) -> Sym4<T> {
        let mut span = unit_quaternions;
        let (mut ww, mut wi, mut wj, mut wk) = (
            R::wide_zero(), R::wide_zero(), R::wide_zero(), R::wide_zero(),
        );
        let (mut ii, mut ij, mut ik) = (R::wide_zero(), R::wide_zero(), R::wide_zero());
        let (mut jj, mut jk, mut kk) = (R::wide_zero(), R::wide_zero(), R::wide_zero());
        while let Some(q) = span.pop_front() {
            let Quaternion { i, j, k, w } = (*q).quaternion;
            ww = R::wide_add_prod(ww, w, w);
            wi = R::wide_add_prod(wi, w, i);
            wj = R::wide_add_prod(wj, w, j);
            wk = R::wide_add_prod(wk, w, k);
            ii = R::wide_add_prod(ii, i, i);
            ij = R::wide_add_prod(ij, i, j);
            ik = R::wide_add_prod(ik, i, k);
            jj = R::wide_add_prod(jj, j, j);
            jk = R::wide_add_prod(jk, j, k);
            kk = R::wide_add_prod(kk, k, k);
        }
        let r = R::from_ratio(1, n.into());
        Sym4 {
            ww: R::wide_mul_scalar(ww, r),
            wi: R::wide_mul_scalar(wi, r),
            wj: R::wide_mul_scalar(wj, r),
            wk: R::wide_mul_scalar(wk, r),
            ii: R::wide_mul_scalar(ii, r),
            ij: R::wide_mul_scalar(ij, r),
            ik: R::wide_mul_scalar(ik, r),
            jj: R::wide_mul_scalar(jj, r),
            jk: R::wide_mul_scalar(jk, r),
            kk: R::wide_mul_scalar(kk, r),
        }
    }

    /// One normalised squaring `S² / tr(S²)` of a symmetric positive semi-definite matrix of
    /// trace 1: `tr(S²) = ‖S‖²_F` (16 products, one wide sum) is inverted once (it is in
    /// `[1/4, 1]`, so its reciprocal is in `[1, 4]`), then every entry of `S²` is one wide sum of
    /// four products scaled by that reciprocal with ONE rounding. The result has trace 1 again
    /// (to rounding). Out of line: `mean_of` calls it `MEAN_OF_SQUARINGS` times.
    fn normalized_square(s: Sym4<T>) -> Sym4<T> {
        let Sym4 { ww, wi, wj, wk, ii, ij, ik, jj, jk, kk } = s;
        let t = R::wide_add_prod(R::wide_add_prod(R::wide_zero(), ww, ww), ii, ii);
        let t = R::wide_add_prod(R::wide_add_prod(t, jj, jj), kk, kk);
        let t = R::wide_add_prod(R::wide_add_prod(t, wi, wi), wi, wi);
        let t = R::wide_add_prod(R::wide_add_prod(t, wj, wj), wj, wj);
        let t = R::wide_add_prod(R::wide_add_prod(t, wk, wk), wk, wk);
        let t = R::wide_add_prod(R::wide_add_prod(t, ij, ij), ij, ij);
        let t = R::wide_add_prod(R::wide_add_prod(t, ik, ik), ik, ik);
        let t = R::wide_add_prod(R::wide_add_prod(t, jk, jk), jk, jk);
        let r = R::recip(R::wide_rescale(t));
        Sym4 {
            ww: Self::row_col(ww, ww, wi, wi, wj, wj, wk, wk, r),
            wi: Self::row_col(ww, wi, wi, ii, wj, ij, wk, ik, r),
            wj: Self::row_col(ww, wj, wi, ij, wj, jj, wk, jk, r),
            wk: Self::row_col(ww, wk, wi, ik, wj, jk, wk, kk, r),
            ii: Self::row_col(wi, wi, ii, ii, ij, ij, ik, ik, r),
            ij: Self::row_col(wi, wj, ii, ij, ij, jj, ik, jk, r),
            ik: Self::row_col(wi, wk, ii, ik, ij, jk, ik, kk, r),
            jj: Self::row_col(wj, wj, ij, ij, jj, jj, jk, jk, r),
            jk: Self::row_col(wj, wk, ij, ik, jj, jk, jk, kk, r),
            kk: Self::row_col(wk, wk, ik, ik, jk, jk, kk, kk, r),
        }
    }

    /// `(a0·b0 + a1·b1 + a2·b2 + a3·b3) · r`, the exact sum scaled with ONE rounding.
    #[inline(always)]
    fn row_col(a0: T, b0: T, a1: T, b1: T, a2: T, b2: T, a3: T, b3: T, r: T) -> T {
        let acc = R::wide_add_prod(R::wide_add_prod(R::wide_zero(), a0, b0), a1, b1);
        R::wide_mul_scalar(R::wide_add_prod(R::wide_add_prod(acc, a2, b2), a3, b3), r)
    }

    /// `MEAN_OF_SQUARINGS` (12) normalised squarings, unrolled: `S^4096` normalised, the
    /// projector on the dominant eigenvector to within `(λ₂ / λ₁)^4096`.
    fn dominant_projector(s: Sym4<T>) -> Sym4<T> {
        let s = Self::normalized_square(Self::normalized_square(Self::normalized_square(s)));
        let s = Self::normalized_square(Self::normalized_square(Self::normalized_square(s)));
        let s = Self::normalized_square(Self::normalized_square(Self::normalized_square(s)));
        Self::normalized_square(Self::normalized_square(Self::normalized_square(s)))
    }

    /// The rotation maximising `tr(Rᵀ m)` (closest to `m` in Frobenius norm): the unit
    /// eigenvector of the largest eigenvalue of Horn's symmetric matrix `K(m)`, for which
    /// `tr(R(q)ᵀ m) = qᵀ K q` (`(w, i, j, k)` order: `K_ww = tr m`, `K_wi = m32 - m23`, ...,
    /// `K_ii = m11 - m22 - m33`, `K_ij = m12 + m21`, ...; every entry an exact sum of entries of
    /// `m`). `K` has trace 0 and spectral radius at most `‖K‖_F = 2‖m‖_F`, so `K +
    /// 2‖m‖_F·I` is positive semi-definite with the same dominant eigenvector; scaled by `1 /
    /// (8‖m‖_F)` it has trace 1, and `dominant_projector` / `dominant_column` extract the
    /// eigenvector (the eigenvalue ratio is at most `(λ₂ + c) / (λ₁ + c)` with `λ₁ - λ₂
    /// = 2(σ₂ + σ₃)`, e.g. 0.85 for a condition number of 8). The identity for the zero
    /// matrix. Upstream's sign convention is applied by the caller.
    fn closest_rotation(m: Matrix3<T>) -> UnitQuaternion<T> {
        let acc = R::wide_add_prod(R::wide_add_prod(R::wide_zero(), m.m11, m.m11), m.m21, m.m21);
        let acc = R::wide_add_prod(R::wide_add_prod(acc, m.m31, m.m31), m.m12, m.m12);
        let acc = R::wide_add_prod(R::wide_add_prod(acc, m.m22, m.m22), m.m32, m.m32);
        let acc = R::wide_add_prod(R::wide_add_prod(acc, m.m13, m.m13), m.m23, m.m23);
        let f = R::wide_sqrt(R::wide_add_prod(acc, m.m33, m.m33));
        if f == R::zero() {
            return UnitQuaternionTrait::identity();
        }
        let c = f + f;
        let r = R::recip(c + c + c + c);
        let s = Sym4 {
            ww: (m.m11 + m.m22 + m.m33 + c) * r,
            wi: (m.m32 - m.m23) * r,
            wj: (m.m13 - m.m31) * r,
            wk: (m.m21 - m.m12) * r,
            ii: (m.m11 - m.m22 - m.m33 + c) * r,
            ij: (m.m12 + m.m21) * r,
            ik: (m.m13 + m.m31) * r,
            jj: (m.m22 - m.m11 - m.m33 + c) * r,
            jk: (m.m23 + m.m32) * r,
            kk: (m.m33 - m.m11 - m.m22 + c) * r,
        };
        Self::dominant_column(Self::dominant_projector(s))
    }

    /// `±q` with upstream's `from_rotation_matrix` (Shepperd) sign: `w > 0` when the trace of the
    /// rotation `3w² - |v|²` is positive, otherwise the component of largest magnitude among
    /// `i`, `j`, `k` (first in that order on ties, as Shepperd's branches) is made positive.
    /// Upstream's `from_matrix_eps` ends with that conversion, so its result carries that sign.
    fn shepperd_sign(q: UnitQuaternion<T>) -> UnitQuaternion<T> {
        let Quaternion { i, j, k, w } = q.quaternion;
        let tr = R::wide_add_prod(R::wide_add_prod(R::wide_zero(), w, w), w, w);
        let tr = R::wide_sub_prod(R::wide_sub_prod(R::wide_add_prod(tr, w, w), i, i), j, j);
        let tr = R::wide_rescale(R::wide_sub_prod(tr, k, k));
        let (ai, aj, ak) = (R::abs(i), R::abs(j), R::abs(k));
        let positive = if tr > R::zero() {
            R::is_sign_positive(w)
        } else if ai > aj && ai > ak {
            R::is_sign_positive(i)
        } else if aj > ak {
            R::is_sign_positive(j)
        } else {
            R::is_sign_positive(k)
        };
        if positive {
            q
        } else {
            UnitQuaternion { quaternion: -q.quaternion }
        }
    }

    /// The normalised column of largest diagonal entry of a rank-one projector `v vᵀ`: `±v`, the
    /// sign making its largest component positive.
    fn dominant_column(s: Sym4<T>) -> UnitQuaternion<T> {
        let Sym4 { ww, wi, wj, wk, ii, ij, ik, jj, jk, kk } = s;
        let q = if ww >= ii && ww >= jj && ww >= kk {
            Quaternion { i: wi, j: wj, k: wk, w: ww }
        } else if ii >= jj && ii >= kk {
            Quaternion { i: ii, j: ij, k: ik, w: wi }
        } else if jj >= kk {
            Quaternion { i: ij, j: jj, k: jk, w: wj }
        } else {
            Quaternion { i: ik, j: jk, k: kk, w: wk }
        };
        UnitQuaternion { quaternion: q.normalize() }
    }
}

/// A symmetric 4x4 matrix in `(w, i, j, k)` order, upper triangle: the working type of
/// `UnitQuaternion::mean_of`. Crate-internal.
#[derive(Copy, Drop, PartialEq, Debug)]
pub(crate) struct Sym4<T> {
    pub ww: T,
    pub wi: T,
    pub wj: T,
    pub wk: T,
    pub ii: T,
    pub ij: T,
    pub ik: T,
    pub jj: T,
    pub jk: T,
    pub kk: T,
}

/// Rotation operations of `UnitQuaternion<T>` that need trigonometry, hence their own trait:
/// scalars may implement `Real` only (see `Vector3AngleTrait`).
///
/// Every function here costs at least one transcendental call (`sin_cos` 16 800, `atan2` 15 400,
/// `acos` 11 400, `asin` 11 400 gas), which dominates the arithmetic. The rapier hot path uses none
/// of them except `from_scaled_axis` / `scaled_axis` at the user-facing boundary.
#[generate_trait]
pub impl UnitQuaternionAngleImpl<
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
> of UnitQuaternionAngleTrait<T> {
    // --- axis-angle and Euler-angle constructors --------------------------------------------

    /// The rotation of `angle` radians about `axis`: `(cos(angle/2), axis·sin(angle/2))`. One
    /// `sin_cos` (of `angle · 1/2`, which floors exactly like upstream's `angle / 2`) and three
    /// products. `axis` MUST be a unit vector. Upstream: `from_axis_angle`.
    fn from_axis_angle(axis: Unit<Vector3<T>>, angle: T) -> UnitQuaternion<T> {
        let (s, c) = Tr::sin_cos(angle * R::HALF);
        let v = axis.value;
        UnitQuaternion { quaternion: Quaternion { i: v.x * s, j: v.y * s, k: v.z * s, w: c } }
    }

    /// The rotation whose axis is the direction of `axisangle` and whose angle is its length (a
    /// rotation vector, the exponential map). A zero vector gives the identity.
    ///
    /// Upstream's `exp` of the pure quaternion `axisangle/2`, i.e. `(cos n, h·sin(n)/n)` with
    /// `h = axisangle/2` and `n = |h|`: ONE division for the common factor `sin(n)/n` like
    /// upstream, then three products (34 920 gas). Normalising `h` first (three divisions) costs
    /// 40 560, 16 % more, for at most 1 ulp
    /// (`bench_unit_quaternion_from_scaled_axis__alt_unit_axis`).
    ///
    /// No cheaper algebraic path exists: `sin_cos` is already a range reduction plus one Horner
    /// evaluation (DESIGN D6), and a dedicated small-angle series would need a branch and would
    /// still round `1 - n²/2`. Upstream: `from_scaled_axis` (`UnitQuaternion::new`).
    fn from_scaled_axis(axisangle: Vector3<T>) -> UnitQuaternion<T> {
        let h = Vector3 {
            x: axisangle.x * R::HALF, y: axisangle.y * R::HALF, z: axisangle.z * R::HALF,
        };
        let n = R::norm3(h.x, h.y, h.z);
        if n == R::zero() {
            return UnitQuaternionTrait::identity();
        }
        let (s, c) = Tr::sin_cos(n);
        let f = R::div(s, n);
        UnitQuaternion { quaternion: Quaternion { i: h.x * f, j: h.y * f, k: h.z * f, w: c } }
    }

    /// The rotation `Rz(yaw) · Ry(pitch) · Rx(roll)` (upstream's convention: intrinsic Tait-Bryan
    /// angles, roll applied first). Three `sin_cos`, four products shared between the four
    /// components, each of which is one fused kernel of two products. Upstream:
    /// `from_euler_angles`.
    fn from_euler_angles(roll: T, pitch: T, yaw: T) -> UnitQuaternion<T> {
        let (sr, cr) = Tr::sin_cos(roll * R::HALF);
        let (sp, cp) = Tr::sin_cos(pitch * R::HALF);
        let (sy, cy) = Tr::sin_cos(yaw * R::HALF);
        let (crcp, srsp) = (cr * cp, sr * sp);
        let (srcp, crsp) = (sr * cp, cr * sp);
        UnitQuaternion {
            quaternion: Quaternion {
                i: R::diff_prod(srcp, cy, crsp, sy),
                j: R::sum_prod2(crsp, cy, srcp, sy),
                k: R::diff_prod(crcp, sy, srsp, cy),
                w: R::sum_prod2(crcp, cy, srsp, sy),
            },
        }
    }

    // --- axis, angle, Euler angles ----------------------------------------------------------

    /// The rotation angle, in `[0, π]`: `2·atan2(|imag|, |w|)` like upstream. One norm and one
    /// `atan2` (35 490 gas); the doubling is an exact addition.
    ///
    /// `2·acos(|w|)` would cost 32 170 (-9 %) but collapses near the ends: at an angle of `2^-20`
    /// rad `|w|` rounds to 1 and it returns 0, while `atan2` is exact to 4 ulp
    /// (`bench_unit_quaternion_angle__alt_acos`, `test_angle_alt_acos_loses_precision_near_zero`).
    /// Upstream: `angle`.
    fn angle(self: UnitQuaternion<T>) -> T {
        let q = self.quaternion;
        let n = R::norm3(q.i, q.j, q.k);
        let half = Tr::atan2(n, R::abs(q.w));
        half + half
    }

    /// `(axis, angle)`, or `None` when the rotation is the identity (the axis is then undefined and
    /// the angle would be `0`). The axis is flipped so that `angle` lies in `[0, π]` (upstream's
    /// convention). Upstream: `axis_angle`.
    fn axis_angle(self: UnitQuaternion<T>) -> Option<(Unit<Vector3<T>>, T)> {
        match UnitQuaternionTrait::axis(self) {
            Some(axis) => Some((axis, Self::angle(self))),
            None => None,
        }
    }

    /// The rotation vector `axis · angle` (the logarithmic map, the inverse of
    /// `from_scaled_axis`), or the zero vector for the identity. The norm of the imaginary part is
    /// computed once (upstream computes it twice, in `axis` and in `angle`): one norm, one `atan2`,
    /// three divisions and three products.
    ///
    /// The angle is in `[0, π]` and the axis is flipped accordingly, so a rotation by slightly
    /// more than `π` comes back as a rotation by slightly less than `π` about the opposite axis
    /// (upstream's convention).
    ///
    /// Measured 54 710 gas; folding `angle / |imag|` into one factor and three products costs
    /// 46 340 (-15 %) but rounds the factor first, which costs a few ulp on the result
    /// (`bench_unit_quaternion_scaled_axis__alt_factor`, `test_scaled_axis_alt_factor_*`). The
    /// exact divisions are kept, like `Vector3Trait::unscale`. Upstream: `scaled_axis`.
    fn scaled_axis(self: UnitQuaternion<T>) -> Vector3<T> {
        let q = self.quaternion;
        let n = R::norm3(q.i, q.j, q.k);
        if n == R::zero() {
            return Vector3 { x: R::zero(), y: R::zero(), z: R::zero() };
        }
        let half = Tr::atan2(n, R::abs(q.w));
        let angle = half + half;
        // axis = imag / |imag| (sign-corrected), then scaled by the angle, as upstream.
        if R::is_sign_negative(q.w) {
            let (x, y, z) = R::div3(-q.i, -q.j, -q.k, n);
            Vector3 { x: x * angle, y: y * angle, z: z * angle }
        } else {
            let (x, y, z) = R::div3(q.i, q.j, q.k, n);
            Vector3 { x: x * angle, y: y * angle, z: z * angle }
        }
    }

    /// The angle of the rotation taking `self` to `other`, in `[0, π]`:
    /// `rotation_to(other).angle()` (one Hamilton product, one norm, one `atan2`). It is a metric
    /// on rotations, unlike the component-wise `abs_diff_eq`: `q` and `-q` are the same rotation
    /// and are at angle `0` from each other. Upstream: `angle_to`.
    #[inline(always)]
    fn angle_to(self: UnitQuaternion<T>, other: UnitQuaternion<T>) -> T {
        Self::angle(UnitQuaternionTrait::rotation_to(self, other))
    }

    /// The Tait-Bryan angles `(roll, pitch, yaw)` such that
    /// `self = Rz(yaw)·Ry(pitch)·Rx(roll)`, with `pitch` in `[-π/2, π/2]` and the others in
    /// `(-π, π]`.
    ///
    /// Only the five entries of the rotation matrix that are needed are computed (one fused kernel
    /// each), then `pitch = -asin(m31)`, `roll = atan2(m32, m33)`, `yaw = atan2(m21, m11)`. Unlike
    /// upstream the entries are NOT divided by `cos(pitch)` first: that factor is `>= 0` and
    /// cancels inside `atan2`, so the division would only cost four divisions and a `cos`.
    ///
    /// At the gimbal lock `|m31| >= 1` (pitch = ±π/2) `roll` and `yaw` are not separable: `yaw`
    /// is set to `0` and `roll` carries the whole rotation, like upstream. One `asin` and two
    /// `atan2`.
    /// Upstream: `euler_angles` (`to_euler_angles`).
    fn euler_angles(self: UnitQuaternion<T>) -> (T, T, T) {
        let Quaternion { i, j, k, w } = self.quaternion;
        let i2 = i + i;
        let j2 = j + j;
        let k2 = k + k;
        let m31 = R::diff_prod(i2, k, w, j2);
        if R::abs(m31) < R::one() {
            let m33 = R::wide_rescale(
                R::wide_add_prod(
                    R::wide_sub_prod(
                        R::wide_sub_prod(R::wide_add_prod(R::wide_zero(), w, w), i, i), j, j,
                    ),
                    k,
                    k,
                ),
            );
            let m11 = R::wide_rescale(
                R::wide_sub_prod(
                    R::wide_sub_prod(
                        R::wide_add_prod(R::wide_add_prod(R::wide_zero(), w, w), i, i), j, j,
                    ),
                    k,
                    k,
                ),
            );
            let m32 = R::sum_prod2(w, i2, j2, k);
            let m21 = R::sum_prod2(i2, j, w, k2);
            (Tr::atan2(m32, m33), -Tr::asin(m31), Tr::atan2(m21, m11))
        } else {
            let m22 = R::wide_rescale(
                R::wide_sub_prod(
                    R::wide_add_prod(
                        R::wide_sub_prod(R::wide_add_prod(R::wide_zero(), w, w), i, i), j, j,
                    ),
                    k,
                    k,
                ),
            );
            let m23 = R::diff_prod(j2, k, w, i2);
            if R::is_sign_negative(m31) {
                (Tr::atan2(m23, m22), R::frac_pi_2(), R::zero())
            } else {
                (-Tr::atan2(m23, m22), -R::frac_pi_2(), R::zero())
            }
        }
    }

    // --- interpolation and powers -----------------------------------------------------------

    /// Spherical linear interpolation along the SHORTEST arc (`other` is negated when the dot
    /// product is negative, like upstream), with constant angular velocity. `t` is not clamped.
    ///
    /// One `acos`, two `sin`, one square root, two divisions and an exact normalisation: the
    /// dearest operation of the type by far — and one the physics engine never calls (`nlerp` is
    /// enough for render interpolation).
    ///
    /// `slerp` is TOTAL here, where upstream may panic with "ambiguous configuration": the
    /// shortest-arc flip makes `cos >= 0`, the early return covers `cos >= 1` (opposite rotations
    /// become identical ones), and `sqrt(1 - cos²)` cannot floor to zero for `cos < 1` in Q32.32
    /// (it is at least `2^-16`). `epsilon = 0` is passed to `try_slerp`, so the panic
    /// `nalgebra: ambiguous slerp` is kept only for scalars whose resolution would make
    /// `sqrt(1 - cos²)` vanish. Upstream: `slerp`.
    fn slerp(self: UnitQuaternion<T>, other: UnitQuaternion<T>, t: T) -> UnitQuaternion<T> {
        Self::try_slerp(self, other, t, R::zero()).expect('nalgebra: ambiguous slerp')
    }

    /// `slerp`, or `None` when `sin(half the angle between the rotations)` is `<= epsilon`. After
    /// the shortest-arc flip that angle is in `[0, π]`, so `epsilon` rejects NEARLY ALIGNED
    /// rotations, whose interpolation direction is ill-conditioned (`epsilon = 1/512` rejects pairs
    /// less than about 1/256 rad apart). `epsilon` is in scalar units, not upstream's relative
    /// float epsilon; with `epsilon = 0` the result is always `Some` (see `slerp`). Upstream:
    /// `try_slerp`.
    fn try_slerp(
        self: UnitQuaternion<T>, other: UnitQuaternion<T>, t: T, epsilon: T,
    ) -> Option<UnitQuaternion<T>> {
        // Shortest arc: flip `other` when the dot product is negative, so that cos >= 0.
        let d = UnitQuaternionTrait::dot(self, other);
        let (o, c) = if R::is_sign_negative(d) {
            (
                Quaternion {
                    i: -other.quaternion.i,
                    j: -other.quaternion.j,
                    k: -other.quaternion.k,
                    w: -other.quaternion.w,
                },
                -d,
            )
        } else {
            (other.quaternion, d)
        };
        if c >= R::one() {
            // The same rotation (up to rounding): nothing to interpolate.
            return Some(self);
        }
        let hang = Tr::acos(c);
        // sin of that half angle, from one fused `1 - c²` (floored once).
        let shang = R::sqrt(R::diff_prod(R::one(), R::one(), c, c));
        if shang <= epsilon {
            return None;
        }
        let ta = R::div(Tr::sin((R::one() - t) * hang), shang);
        let tb = R::div(Tr::sin(t * hang), shang);
        let s = self.quaternion;
        let q = Quaternion {
            i: R::sum_prod2(s.i, ta, o.i, tb),
            j: R::sum_prod2(s.j, ta, o.j, tb),
            k: R::sum_prod2(s.k, ta, o.k, tb),
            w: R::sum_prod2(s.w, ta, o.w, tb),
        };
        Some(UnitQuaternionTrait::new_normalize(q))
    }

    /// The rotation raised to the real power `n`: the same axis, `n` times the angle (`n · angle`
    /// is not reduced modulo `2π`, so a large `n` wraps the way the scalar's `sin_cos` does). The
    /// identity to any power is the identity. Upstream: `powf`.
    fn powf(self: UnitQuaternion<T>, n: T) -> UnitQuaternion<T> {
        match Self::axis_angle(self) {
            Some((axis, angle)) => Self::from_axis_angle(axis, angle * n),
            None => UnitQuaternionTrait::identity(),
        }
    }

    /// `rotation_between` with the angle multiplied by `s` (`s = 1` gives `rotation_between`).
    /// Upstream's algorithm, which needs the angle explicitly: one `acos` of the dot product
    /// (clamped to `[-1, 1]`, since a rounded dot product of two rounded unit vectors may exceed
    /// it)
    /// and one `from_axis_angle`. `None` in the same antiparallel case as `rotation_between`.
    /// Upstream: `scaled_rotation_between`.
    fn scaled_rotation_between(a: Vector3<T>, b: Vector3<T>, s: T) -> Option<UnitQuaternion<T>> {
        match (a.try_normalize(R::zero()), b.try_normalize(R::zero())) {
            (
                Some(u), Some(v),
            ) => Self::scaled_rotation_between_axis(Unit { value: u }, Unit { value: v }, s),
            _ => Some(UnitQuaternionTrait::identity()),
        }
    }

    /// `scaled_rotation_between` of two UNIT vectors (no normalisation): `from_axis_angle(a × b
    /// normalised, acos(a · b) · s)`, the dot product clamped to `[-1, 1]`; `None` when `a × b`
    /// floors to zero with `a · b < 0`, the identity when it floors to zero with `a · b >= 0`.
    /// Upstream: `UnitQuaternion::scaled_rotation_between_axis` (which tests `|a × b|` against
    /// `default_epsilon`, here 1 ulp: the same as the exact-zero test once the norm floors).
    fn scaled_rotation_between_axis(
        a: Unit<Vector3<T>>, b: Unit<Vector3<T>>, s: T,
    ) -> Option<UnitQuaternion<T>> {
        let (u, v) = (a.value, b.value);
        let c = u.cross(v);
        let d = R::clamp(Vector3Trait::dot(u, v), R::NEG_ONE, R::one());
        match UnitTrait::try_new(c, R::zero()) {
            Some(axis) => Some(Self::from_axis_angle(axis, Tr::acos(d) * s)),
            None => if R::is_sign_negative(d) {
                None
            } else {
                Some(UnitQuaternionTrait::identity())
            },
        }
    }

    // --- P08 completion ------------------------------------------------------------------------

    /// Alias of `from_scaled_axis` (bit-identical): the rotation of the rotation vector
    /// `axisangle`. Upstream's `new` is `exp` of `(0, axisangle / 2)` with the threshold
    /// `|axisangle / 2| <= default_epsilon` (1 ulp here) below which it returns the identity;
    /// `from_scaled_axis` tests exact zero instead, and at `|axisangle / 2| = 1` ulp both give the
    /// identity to within 1 ulp. Upstream: `UnitQuaternion::new`.
    #[inline(always)]
    fn new(axisangle: Vector3<T>) -> UnitQuaternion<T> {
        Self::from_scaled_axis(axisangle)
    }

    /// `from_scaled_axis` with a threshold: the identity when `|axisangle / 2| <= eps` (upstream's
    /// `exp_eps` of the pure quaternion `(0, axisangle / 2)`; with `eps = 0` this is
    /// `from_scaled_axis` bit for bit). Upstream: `UnitQuaternion::new_eps`.
    fn new_eps(axisangle: Vector3<T>, eps: T) -> UnitQuaternion<T> {
        let h = Vector3 {
            x: axisangle.x * R::HALF, y: axisangle.y * R::HALF, z: axisangle.z * R::HALF,
        };
        let n = R::norm3(h.x, h.y, h.z);
        if n <= eps {
            return UnitQuaternionTrait::identity();
        }
        let (s, c) = Tr::sin_cos(n);
        let f = R::div(s, n);
        UnitQuaternion { quaternion: Quaternion { i: h.x * f, j: h.y * f, k: h.z * f, w: c } }
    }

    /// Alias of `new_eps`. Upstream: `UnitQuaternion::from_scaled_axis_eps`.
    #[inline(always)]
    fn from_scaled_axis_eps(axisangle: Vector3<T>, eps: T) -> UnitQuaternion<T> {
        Self::new_eps(axisangle, eps)
    }

    /// `exp` of the underlying quaternion (a general quaternion, not a rotation: for a unit
    /// quaternion of real part `w`, `e^w · (cos|v|, v̂ sin|v|)`). Upstream:
    /// `UnitQuaternion::exp`.
    #[inline(always)]
    fn exp(self: UnitQuaternion<T>) -> Quaternion<T> {
        self.quaternion.exp()
    }

    /// The pure quaternion `(0, scaled_axis())`, i.e. `(0, axis · angle)` with the FULL angle,
    /// like upstream (the logarithm of the quaternion itself would be `(0, axis · angle / 2)`:
    /// upstream's `UnitQuaternion::ln` is `from_imag(axis * angle)`), and zero for the identity.
    /// Upstream:
    /// `UnitQuaternion::ln`.
    #[inline(always)]
    fn ln(self: UnitQuaternion<T>) -> Quaternion<T> {
        QuaternionTrait::from_imag(Self::scaled_axis(self))
    }

    /// Deprecated alias of `euler_angles`. Upstream: `UnitQuaternion::to_euler_angles`.
    #[inline(always)]
    fn to_euler_angles(self: UnitQuaternion<T>) -> (T, T, T) {
        Self::euler_angles(self)
    }

    /// `from_matrix_eps(m, default_epsilon, 0, identity)`: the rotation closest to `m`, in closed
    /// form (see `from_matrix_eps`). Upstream: `UnitQuaternion::from_matrix`.
    #[inline(always)]
    fn from_matrix(m: Matrix3<T>) -> UnitQuaternion<T> {
        Self::from_matrix_eps(m, R::default_epsilon(), 0, UnitQuaternionTrait::identity())
    }

    /// The rotation part of the matrix `m` (the rotation `R` maximising `tr(Rᵀ m)`, i.e. closest
    /// to `m` in Frobenius norm; the rotation factor of the polar decomposition when
    /// `det m > 0`).
    ///
    /// - `max_iter = 0` (upstream: iterate until convergence): the LIMIT is computed directly,
    ///   as the dominant eigenvector of Horn's 4x4 matrix `K(m)` (`qᵀ K q = tr(R(q)ᵀ m)`) by
    ///   `MEAN_OF_SQUARINGS` normalised squarings — a fixed cost of about 400 000 gas, no loop,
    ///   `guess` and `eps` unused. Upstream's iteration converges only LINEARLY: from the
    ///   identity, 21 to 57 iterations of about 100 000 gas each on 17 of the 24 oracle cases,
    ///   more than 64 on the other 7 (`test_from_matrix_eps_iterations_on_the_oracle_set`,
    ///   `bench_unit_quaternion_from_matrix__alt_iterate`).
    /// - `max_iter > 0`: upstream's algorithm, Müller et al.'s iteration ("A Robust Method to
    ///   Extract the Rotational Part of Deformations") from `guess`:
    ///   `ω = Σ_c r_c × m_c / (|Σ_c r_c · m_c| + ε)` over the columns, `R ← exp(ω) · R`,
    ///   until `|ω| <= eps` (in scalar units; the rounding noise of `ω` is a few ulp, so with
    ///   `eps = default_epsilon` = 1 ulp the loop runs to its bound), with upstream's perturbation
    ///   by `max(sqrt(eps), eps²)` radians about a cycling axis at a stationary point. The
    ///   rotation is carried as a unit quaternion: each iteration is one `to_rotation_matrix`, one
    ///   fused kernel per component of `ω` (six products) and for its denominator (nine), one
    ///   `norm3`, one `sin_cos`, four divisions and one Hamilton product, about 100 000 gas.
    ///   **Bounded:** at most `min(max_iter, FROM_MATRIX_MAX_ITER)` iterations and
    ///   `FROM_MATRIX_MAX_PERTURBATIONS` successive perturbations.
    ///
    /// Either way the result carries upstream's sign, the one of its final
    /// `from_rotation_matrix` (Shepperd's branches). The distance test of the perturbation squares
    /// the entries of `m - R`: panics on overflow for entries above about 15 000. Upstream:
    /// `UnitQuaternion::from_matrix_eps`.
    fn from_matrix_eps(
        m: Matrix3<T>, eps: T, max_iter: usize, guess: UnitQuaternion<T>,
    ) -> UnitQuaternion<T> {
        if max_iter == 0 {
            return UnitQuaternionInternalTrait::shepperd_sign(
                UnitQuaternionInternalTrait::closest_rotation(m),
            );
        }
        let (q, _) = UnitQuaternionAngleInternalTrait::from_matrix_eps_count(
            m, eps, max_iter, guess,
        );
        UnitQuaternionInternalTrait::shepperd_sign(q)
    }
}

/// Crate-internal kernels of `UnitQuaternionAngleTrait::from_matrix_eps`.
#[generate_trait]
pub(crate) impl UnitQuaternionAngleInternalImpl<
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
> of UnitQuaternionAngleInternalTrait<T> {
    /// Müller's iteration of `from_matrix_eps` (`max_iter > 0`, or 0 for the cap alone) and the
    /// number of iterations it ran (for the convergence tests). No sign convention applied.
    fn from_matrix_eps_count(
        m: Matrix3<T>, eps: T, max_iter: usize, guess: UnitQuaternion<T>,
    ) -> (UnitQuaternion<T>, usize) {
        let cap = if max_iter == 0 || max_iter > FROM_MATRIX_MAX_ITER {
            FROM_MATRIX_MAX_ITER
        } else {
            max_iter
        };
        let eps_dist = R::max(R::sqrt(eps), eps * eps);
        let mut q = guess.quaternion;
        // Perturbation axis: x, then z, then y (upstream's `yzx` swizzle of the x axis, cycled).
        let mut axis: u8 = 0;
        let mut iter: usize = 0;
        while iter < cap {
            iter += 1;
            let r = UnitQuaternionTrait::to_rotation_matrix(UnitQuaternion { quaternion: q })
                .matrix;
            let (x, y, z) = Self::muller_axis(r, m);
            let n = R::norm3(x, y, z);
            if n > eps {
                let (s, c) = Tr::sin_cos(n * R::HALF);
                let f = R::div(s, n);
                q = Quaternion { i: x * f, j: y * f, k: z * f, w: c } * q;
                continue;
            }
            // A stationary point: perturb to tell a maximum of `tr(Rᵀ m)` from a saddle.
            let d0 = Self::distance_squared(m, r);
            let (ps, pc) = Tr::sin_cos(eps_dist * R::HALF);
            let e = if axis == 0 {
                Quaternion { i: ps, j: R::zero(), k: R::zero(), w: pc }
            } else if axis == 1 {
                Quaternion { i: R::zero(), j: R::zero(), k: ps, w: pc }
            } else {
                Quaternion { i: R::zero(), j: ps, k: R::zero(), w: pc }
            };
            let mut p = q;
            let mut d1 = d0;
            let mut moved = false;
            let mut tries: usize = 0;
            while tries < FROM_MATRIX_MAX_PERTURBATIONS {
                tries += 1;
                p = p * e;
                d1 =
                    Self::distance_squared(
                        m,
                        UnitQuaternionTrait::to_rotation_matrix(UnitQuaternion { quaternion: p })
                            .matrix,
                    );
                if !R::abs_diff_eq(d0, d1, 1) {
                    moved = true;
                    break;
                }
            }
            if !moved || d0 < d1 {
                // The distance grows in the perturbed direction: a minimum, done.
                break;
            }
            axis = if axis == 2 {
                0
            } else {
                axis + 1
            };
            q = p;
        }
        (UnitQuaternionTrait::new_normalize(q), iter)
    }

    /// Müller's rotation vector `Σ_c r_c × m_c / (|Σ_c r_c · m_c| + ε)` (columns `c`, `ε` =
    /// `default_epsilon`): each component one fused kernel of six products and the denominator
    /// one of nine (exactly floored), then three correctly rounded divisions.
    fn muller_axis(r: Matrix3<T>, m: Matrix3<T>) -> (T, T, T) {
        let x = R::wide_sub_prod(R::wide_add_prod(R::wide_zero(), r.m21, m.m31), r.m31, m.m21);
        let x = R::wide_sub_prod(R::wide_add_prod(x, r.m22, m.m32), r.m32, m.m22);
        let x = R::wide_rescale(R::wide_sub_prod(R::wide_add_prod(x, r.m23, m.m33), r.m33, m.m23));
        let y = R::wide_sub_prod(R::wide_add_prod(R::wide_zero(), r.m31, m.m11), r.m11, m.m31);
        let y = R::wide_sub_prod(R::wide_add_prod(y, r.m32, m.m12), r.m12, m.m32);
        let y = R::wide_rescale(R::wide_sub_prod(R::wide_add_prod(y, r.m33, m.m13), r.m13, m.m33));
        let z = R::wide_sub_prod(R::wide_add_prod(R::wide_zero(), r.m11, m.m21), r.m21, m.m11);
        let z = R::wide_sub_prod(R::wide_add_prod(z, r.m12, m.m22), r.m22, m.m12);
        let z = R::wide_rescale(R::wide_sub_prod(R::wide_add_prod(z, r.m13, m.m23), r.m23, m.m13));
        let d = R::wide_add_prod(R::wide_add_prod(R::wide_zero(), r.m11, m.m11), r.m21, m.m21);
        let d = R::wide_add_prod(R::wide_add_prod(d, r.m31, m.m31), r.m12, m.m12);
        let d = R::wide_add_prod(R::wide_add_prod(d, r.m22, m.m22), r.m32, m.m32);
        let d = R::wide_add_prod(R::wide_add_prod(d, r.m13, m.m13), r.m23, m.m23);
        let d = R::wide_rescale(R::wide_add_prod(d, r.m33, m.m33));
        R::div3(x, y, z, R::abs(d) + R::default_epsilon())
    }

    /// `‖m - r‖²_F`: nine exact differences, one wide sum of squares, floored once.
    fn distance_squared(m: Matrix3<T>, r: Matrix3<T>) -> T {
        let (a, b, c) = (m.m11 - r.m11, m.m21 - r.m21, m.m31 - r.m31);
        let (d, e, f) = (m.m12 - r.m12, m.m22 - r.m22, m.m32 - r.m32);
        let (g, h, k) = (m.m13 - r.m13, m.m23 - r.m23, m.m33 - r.m33);
        let acc = R::wide_add_prod(R::wide_add_prod(R::wide_zero(), a, a), b, b);
        let acc = R::wide_add_prod(R::wide_add_prod(acc, c, c), d, d);
        let acc = R::wide_add_prod(R::wide_add_prod(acc, e, e), f, f);
        let acc = R::wide_add_prod(R::wide_add_prod(acc, g, g), h, h);
        R::wide_rescale(R::wide_add_prod(acc, k, k))
    }
}

/// The composition of two rotations: the Hamilton product of the quaternions, `lhs` applied last
/// (`(lhs * rhs).transform_vector(v) = lhs.transform_vector(rhs.transform_vector(v))`). The result
/// is a unit quaternion within the rounding of its four components (the norm is multiplicative);
/// `renormalize_fast` brings back a rotation composed many times. Upstream: `Mul`.
pub impl UnitQuaternionMul<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>,
> of Mul<UnitQuaternion<T>> {
    #[inline(always)]
    fn mul(lhs: UnitQuaternion<T>, rhs: UnitQuaternion<T>) -> UnitQuaternion<T> {
        UnitQuaternion { quaternion: lhs.quaternion * rhs.quaternion }
    }
}

/// `-q`: the opposite quaternion, which is the SAME rotation (the double cover of the rotation
/// group). Exact; panics on overflow (`-MIN`). Crate-internal (WP 8.0): upstream has `Neg` on
/// `Unit<Vector>` only; `-*q` there is a plain `Quaternion`.
pub(crate) impl UnitQuaternionNeg<T, +Neg<T>, +Copy<T>, +Drop<T>> of Neg<UnitQuaternion<T>> {
    #[inline(always)]
    fn neg(a: UnitQuaternion<T>) -> UnitQuaternion<T> {
        UnitQuaternion {
            quaternion: Quaternion {
                i: -a.quaternion.i, j: -a.quaternion.j, k: -a.quaternion.k, w: -a.quaternion.w,
            },
        }
    }
}

/// `a / b = a * b⁻¹`: the rotation `r` with `r * b = a`. ONE fused Hamilton product with `b`'s
/// conjugate signs folded into the accumulation (`QuaternionInternalTrait::mul_conj`): bit for
/// bit `a * b.inverse()`, without the three negations
/// (`bench_unit_quaternion_div__alt_inverse_then_mul`). Upstream: `Div for UnitQuaternion`.
pub impl UnitQuaternionDiv<
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
> of Div<UnitQuaternion<T>> {
    #[inline(always)]
    fn div(lhs: UnitQuaternion<T>, rhs: UnitQuaternion<T>) -> UnitQuaternion<T> {
        UnitQuaternion {
            quaternion: QuaternionInternalTrait::mul_conj(lhs.quaternion, rhs.quaternion),
        }
    }
}

/// `Default::default()`: the identity rotation. Upstream: `Default for UnitQuaternion`.
pub impl UnitQuaternionDefault<T, impl R: Real<T>, +Drop<T>> of Default<UnitQuaternion<T>> {
    #[inline(always)]
    fn default() -> UnitQuaternion<T> {
        UnitQuaternion {
            quaternion: Quaternion { i: R::zero(), j: R::zero(), k: R::zero(), w: R::one() },
        }
    }
}

/// `One::one()`: the identity rotation; `is_one` compares with `(1, 0, 0, 0)` exactly (`-1`, the
/// same rotation, is not `one`). Upstream: `num::One for UnitQuaternion`.
pub impl UnitQuaternionOne<
    T, impl R: Real<T>, +PartialEq<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>,
> of One<UnitQuaternion<T>> {
    #[inline(always)]
    fn one() -> UnitQuaternion<T> {
        UnitQuaternion {
            quaternion: Quaternion { i: R::zero(), j: R::zero(), k: R::zero(), w: R::one() },
        }
    }

    #[inline(always)]
    fn is_one(self: @UnitQuaternion<T>) -> bool {
        let q = *self.quaternion;
        q.i == R::zero() && q.j == R::zero() && q.k == R::zero() && q.w == R::one()
    }

    #[inline(always)]
    fn is_non_one(self: @UnitQuaternion<T>) -> bool {
        !Self::is_one(self)
    }
}

/// `q.into()`: the isometry of rotation `q` and zero translation. Upstream: `SubsetOf<Isometry3>
/// for UnitQuaternion` (`nalgebra::convert(q)`).
pub impl Isometry3FromUnitQuaternion<
    T, impl R: Real<T>, +Drop<T>,
> of Into<UnitQuaternion<T>, Isometry3<T>> {
    #[inline(always)]
    fn into(self: UnitQuaternion<T>) -> Isometry3<T> {
        Isometry3 {
            rotation: self,
            translation: Translation3 {
                vector: Vector3 { x: R::zero(), y: R::zero(), z: R::zero() },
            },
        }
    }
}

/// `q.into()`: the similarity of rotation `q`, zero translation and scaling 1. Upstream:
/// `SubsetOf<Similarity3> for UnitQuaternion` (`nalgebra::convert(q)`).
pub impl Similarity3FromUnitQuaternion<
    T, impl R: Real<T>, +Drop<T>,
> of Into<UnitQuaternion<T>, Similarity3<T>> {
    #[inline(always)]
    fn into(self: UnitQuaternion<T>) -> Similarity3<T> {
        Similarity3 {
            isometry: Isometry3 {
                rotation: self,
                translation: Translation3 {
                    vector: Vector3 { x: R::zero(), y: R::zero(), z: R::zero() },
                },
            },
            scaling: R::one(),
        }
    }
}
