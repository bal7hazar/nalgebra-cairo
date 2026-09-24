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

use simba::scalar::{Real, Transcendental};
use crate::base::matrix3::Matrix3;
use crate::base::matrix4::Matrix4;
use crate::base::point3::Point3;
use crate::base::unit::{Unit, UnitTrait};
use crate::base::vector3::{Vector3, Vector3Trait};
use super::quaternion::{Quaternion, QuaternionInternalTrait, QuaternionTrait};
use super::rotation3::{Rotation3, Rotation3Trait};

#[cfg(test)]
mod benches;
#[cfg(test)]
mod oracle;
#[cfg(test)]
mod tests;

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
            (
                Some(u), Some(v),
            ) => {
                let c = u.cross(v);
                if c.is_zero() {
                    if R::is_sign_negative(Vector3Trait::dot(u, v)) {
                        // A half turn about an undefined axis: not a simple rotation.
                        None
                    } else {
                        Some(Self::identity())
                    }
                } else {
                    let q = Quaternion {
                        i: c.x, j: c.y, k: c.z, w: R::one() + Vector3Trait::dot(u, v),
                    };
                    Some(Self::new_normalize(q))
                }
            },
            // A zero-length input has no direction: upstream returns the identity.
            _ => Some(Self::identity()),
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
            ) => {
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
            },
            _ => Some(UnitQuaternionTrait::identity()),
        }
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
