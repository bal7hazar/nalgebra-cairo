//! `UnitComplex`: a 2D rotation stored as a unit complex number (upstream
//! `nalgebra::UnitComplex`, which is `Unit<Complex<T>>`).
//!
//! - `UnitComplexTrait` / `UnitComplexImpl`: construction, accessors, composition, transforms,
//!   conversions and renormalization — everything that is algebraic, hence available for any
//!   `simba::scalar::Real` scalar;
//! - `UnitComplexAngleTrait` / `UnitComplexAngleImpl`: the operations that go through an angle
//!   (`new`, `angle`, `powf`, `slerp`, ...), which additionally need
//!   `simba::scalar::Transcendental`;
//! - `a * b` (composition of two rotations): its impl lives in this module, where the compiler
//!   finds it without any import.
//!
//! The raw pair `(re, im) = (cos θ, sin θ)` is stored directly instead of wrapping a
//! `Unit<Vector2<T>>`: a rotation is not a vector (it composes with `*`, not with `+`), and the
//! extra layer costs an indirection in every formula for nothing. Nothing enforces the invariant
//! `re² + im² = 1`: build with `new` / `rotation_between` / `from_rotation_matrix`, or with
//! `from_cos_sin_unchecked` when the pair is known to be normalized. Rotations built from an angle
//! have a norm of `1` within a few ulp, not exactly `1` (fixed point cannot do better);
//! `renormalize` / `renormalize_fast` bring a drifted pair back.
//!
//! Numeric contract (AGENTS.md): every sum of products goes through a fused `Real` kernel (one
//! floor rounding and one overflow check per output scalar); nothing wraps silently.

use core::num::traits::One;
use simba::scalar::{Real, Transcendental};
use crate::base::matrix2::Matrix2;
use crate::base::matrix3::Matrix3;
use crate::base::point2::Point2;
use crate::base::unit::Unit;
use crate::base::vector2::Vector2;
use super::isometry2::Isometry2;
use super::quaternion::ApproxEqTrait;
use super::rotation2::Rotation2;
use super::similarity2::Similarity2;
use super::translation2::Translation2;

#[cfg(test)]
mod benches;
#[cfg(test)]
mod benches_ext;
#[cfg(test)]
mod oracle;
#[cfg(test)]
mod oracle_ext;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_ext;

/// Upper bound of the iterations of `UnitComplex::from_matrix_eps` when `max_iter > 0`
/// (`max_iter = 0`, upstream's "until convergence", is the closed form). The 2D iteration
/// `θ ← θ + tan(φ - θ)` converges cubically once `|φ - θ| < π/2`; measured on the oracle
/// set (`test_from_matrix_eps_iterations_on_the_oracle_set`): from the identity with `eps` = 32
/// ulp, 23 of the 24 cases stop in 4 to 12 iterations, the last one oscillates at the rounding
/// noise until the bound, 3 ulp from the limit (with `eps` = 1 ulp the loop usually runs to the
/// bound).
pub const FROM_MATRIX_MAX_ITER: usize = 16;

/// A 2D rotation of angle `θ`, stored as the unit complex number
/// `re + i·im = cos θ + i·sin θ`.
///
/// The layout `(re, im)` is the one of the oracle vectors and of upstream's
/// `UnitComplex::from_cos_sin_unchecked(cos, sin)`.
#[derive(Copy, Drop, PartialEq, Serde, Debug, Hash)]
pub struct UnitComplex<T> {
    pub re: T,
    pub im: T,
}

/// Operations of `UnitComplex<T>` that need no trigonometry, over a `Real` scalar. By value,
/// unrolled, no loop.
pub trait UnitComplexTrait<T> {
    /// The identity rotation `1 + 0i` (angle `0`). Exact. Upstream: `UnitComplex::identity`.
    fn identity() -> UnitComplex<T>;
    /// The rotation `cos + i·sin` WITHOUT normalizing it: the caller guarantees
    /// `cos² + sin² = 1`. Upstream: `UnitComplex::from_cos_sin_unchecked`.
    fn from_cos_sin_unchecked(cos: T, sin: T) -> UnitComplex<T>;
    /// The rotation of the matrix `r`, i.e. its first column `(m11, m21) = (cos θ, sin θ)`.
    /// Exact (a copy of two components). Upstream: `UnitComplex::from_rotation_matrix`.
    fn from_rotation_matrix(r: Rotation2<T>) -> UnitComplex<T>;
    /// The rotation taking the direction of `a` to the direction of `b`, or the identity when
    /// either vector is zero (upstream returns the identity through `try_normalize`).
    ///
    /// Algebraic, no trigonometry: `(cos θ, sin θ) = (a·b, a×b) / (|a|·|b|)`, i.e. one fused
    /// `dot`, one fused `perp` and one normalization of that pair — `|(a·b, a×b)| = |a|·|b|`
    /// exactly, so the norm is taken on the pair itself and no product of norms is ever formed.
    /// Upstream computes `from_angle(perp.atan2(dot))` on the normalized inputs, which costs an
    /// `atan2` AND a `sin_cos`: measured 14 060 gas here against 52 620 through angles, 3.7x
    /// (`bench_unit_complex_rotation_between__alt_atan2`).
    ///
    /// Accuracy: `a·b` and `a×b` are floored once each, then divided by their (floored) norm, so
    /// the error is about `2 + 2/(|a|·|b|)` ulp — it grows as the vectors get shorter, since the
    /// products lose the bits the normalization then amplifies (`|a| = |b| = 2^-7` loses 14 bits,
    /// see `test_rotation_between_short_vectors_lose_precision`). Normalizing `a` and `b` first
    /// keeps about 8 ulp whatever the lengths, for 27 900 gas, 2.0x
    /// (`bench_unit_complex_rotation_between__alt_normalized_inputs`): use that form when the
    /// inputs may be much shorter than 1.
    ///
    /// Panics when `a·b` or `a×b` does not fit (`|a|·|b|` above about 2.1e9).
    fn rotation_between(a: Vector2<T>, b: Vector2<T>) -> UnitComplex<T>;
    /// The stored pair `(re, im) = (cos θ, sin θ)` as a vector. Upstream: `complex` (a reference
    /// to the wrapped `Complex<T>`).
    fn complex(self: UnitComplex<T>) -> Vector2<T>;
    /// The real part `cos θ`. Upstream: `complex().re`.
    fn re(self: UnitComplex<T>) -> T;
    /// The imaginary part `sin θ`. Upstream: `complex().im`.
    fn im(self: UnitComplex<T>) -> T;
    /// `cos θ`, without computing `θ`. Exact (a stored component). Upstream: `cos_angle`.
    fn cos_angle(self: UnitComplex<T>) -> T;
    /// `sin θ`, without computing `θ`. Exact (a stored component). Upstream: `sin_angle`.
    fn sin_angle(self: UnitComplex<T>) -> T;
    /// The complex conjugate `re - i·im`, i.e. the rotation of angle `-θ`. Exact; panics on
    /// overflow (`-MIN`). Upstream: `conjugate`.
    fn conjugate(self: UnitComplex<T>) -> UnitComplex<T>;
    /// The inverse rotation, which for a unit complex number is the conjugate (no division).
    /// Exact. Upstream: `inverse`.
    fn inverse(self: UnitComplex<T>) -> UnitComplex<T>;
    /// The rotation `r` such that `self * r = other`, i.e. `other * self⁻¹`, in two fused
    /// kernels with the conjugation folded in (4 000 gas, the price of one product: the inverse is
    /// never formed). The composition of unit complex numbers is commutative, so this is ALSO
    /// `self⁻¹ * other`, the `inv_mul` rapier calls on every relative pose
    /// (docs/research/01, §3.3). Upstream: `rotation_to`.
    fn rotation_to(self: UnitComplex<T>, other: UnitComplex<T>) -> UnitComplex<T>;
    /// `self * v`: `v` rotated by `θ`, `(re·x - im·y, im·x + re·y)`, two fused kernels (one
    /// floor rounding per output component, so the result is the exactly floored rotation of `v`).
    /// Panics on overflow. Upstream: `transform_vector` (`self * v`).
    fn transform_vector(self: UnitComplex<T>, v: Vector2<T>) -> Vector2<T>;
    /// `self * p`: `p` rotated around the origin by `θ`. Same kernels, same rounding and panics
    /// as `transform_vector`. Upstream: `transform_point` (`self * p`).
    fn transform_point(self: UnitComplex<T>, p: Point2<T>) -> Point2<T>;
    /// `self⁻¹ * v`: `v` rotated by `-θ`, `(re·x + im·y, re·y - im·x)`, two fused kernels.
    /// Never forms the inverse rotation. Panics on overflow. Upstream: `inverse_transform_vector`.
    fn inverse_transform_vector(self: UnitComplex<T>, v: Vector2<T>) -> Vector2<T>;
    /// `self⁻¹ * p`: `p` rotated around the origin by `-θ`. Upstream:
    /// `inverse_transform_point`.
    fn inverse_transform_point(self: UnitComplex<T>, p: Point2<T>) -> Point2<T>;
    /// The same rotation as a 2x2 matrix `[[re, -im], [im, re]]`. Exact (two copies and two
    /// negations); panics on overflow (`-MIN`). Upstream: `to_rotation_matrix`.
    fn to_rotation_matrix(self: UnitComplex<T>) -> Rotation2<T>;
    /// The same rotation as a 3x3 homogeneous matrix (the rotation block, then `(0, 0, 1)`).
    /// Exact. Upstream: `to_homogeneous`.
    fn to_homogeneous(self: UnitComplex<T>) -> Matrix3<T>;
    /// Renormalizes exactly, in place: divides `(re, im)` by their norm (one `norm2`, then one
    /// exactly correctly rounded division per component), so the result is unit within about
    /// `1 + 1/|c|` ulp whatever the drift, and returns the norm it had. Panics with
    /// `Fixed: division by zero` on the zero pair. Upstream: `Unit::renormalize` (`&mut self`,
    /// returns the previous norm).
    fn renormalize(ref self: UnitComplex<T>) -> T;
    /// Renormalizes, in place, a pair whose norm is already close to 1 (the usual case: rounding
    /// accumulated by repeated compositions): one Newton step for the inverse square root,
    /// `c * (3 - |c|²) / 2`, with the factor as ONE fused kernel (`mul_add(|c|², -1/2, 3/2)`,
    /// floored once) and one product per component. No square root, no division. Upstream:
    /// `Unit::renormalize_fast`.
    ///
    /// With `|c|² = 1 + e` the new squared norm is `1 - 3e²/4 + e³/4`: the error is squared at
    /// every step, so one step is exact to the last ulp for `|e| < 2^-16`, and the norm stays
    /// within about 2 ulp below 1 (each product floors). The zero pair stays zero (no panic).
    /// Panics on overflow when `|c|²` does not fit.
    ///
    /// **It buys almost nothing here**: 7 600 gas against 7 700 for the exact `renormalize`
    /// (1.3 %), because a Q32.32 `norm2` is a `u128` square root at 2 200 gas and a division
    /// costs 2 800 against 1 750 for a product — nothing like the float case where the square
    /// root dominates. Prefer `renormalize`, which converges from any norm, unless the last
    /// percent matters. The other formulations are benchmarked as
    /// `bench_unit_complex_renormalize_fast__alt_*` and give the same bits: upstream's literal
    /// `1/2 · (3 - |c|²)` 8 440, a `lerp` per component 8 000, `c + c·(1 - |c|²)/2` 8 000.
    fn renormalize_fast(ref self: UnitComplex<T>);
    /// `true` when `re` and `im` are both within `ulps` smallest units (raw units for fixed
    /// point) of `other`'s; cannot overflow. Note that `-c` is the same rotation as `c` turned by
    /// `2π`, and is NOT `abs_diff_eq` to it. Upstream: `approx::AbsDiffEq::abs_diff_eq`, the
    /// tolerance being counted in ulp instead of a float epsilon (DESIGN D3).
    fn abs_diff_eq(self: UnitComplex<T>, other: UnitComplex<T>, ulps: u64) -> bool;
    /// `true` when `re` and `im` are each within `epsilon` ulp of `other`'s, or of the same sign
    /// and within `max_relative` times the larger magnitude (see `QuaternionTrait::relative_eq`).
    /// Upstream: `approx::RelativeEq::relative_eq`, `epsilon` counted in ulp (DESIGN D3).
    fn relative_eq(
        self: UnitComplex<T>, other: UnitComplex<T>, epsilon: u64, max_relative: T,
    ) -> bool;
    /// `true` when `re` and `im` are each within `epsilon` ulp of `other`'s, or of the same sign
    /// and within `max_ulps` ulp (see `QuaternionTrait::ulps_eq`). Upstream:
    /// `approx::UlpsEq::ulps_eq`.
    fn ulps_eq(self: UnitComplex<T>, other: UnitComplex<T>, epsilon: u64, max_ulps: u32) -> bool;
    /// The same rotation with both components converted by `Into<T, U>` (the identity for the
    /// single scalar `Fixed`). Upstream: `cast` (and `SubsetOf<UnitComplex<U>>`).
    fn cast<U, +Into<T, U>, +Drop<U>>(self: UnitComplex<T>) -> UnitComplex<U>;
    /// The complex number `q = (re, im)` (a `Vector2`: there is no `Complex` type here, the same
    /// layout as `complex()`) divided by its norm: one `norm2`, two correctly rounded divisions.
    /// Panics with `Fixed: division by zero` on zero (upstream returns NaN). Upstream:
    /// `UnitComplex::from_complex`.
    fn from_complex(q: Vector2<T>) -> UnitComplex<T>;
    /// `(from_complex(q), |q|)`. Upstream: `UnitComplex::from_complex_and_get`.
    fn from_complex_and_get(q: Vector2<T>) -> (UnitComplex<T>, T);
    /// The rotation whose matrix has the columns `basis[0]`, `basis[1]`, WITHOUT checking them:
    /// `(re, im)` is the first column, like `from_rotation_matrix`. Exact. Upstream:
    /// `UnitComplex::from_basis_unchecked`.
    fn from_basis_unchecked(basis: [Vector2<T>; 2]) -> UnitComplex<T>;
    /// `rotation_between` of two UNIT vectors: the same algebraic kernel (`(a·b, a×b)`
    /// normalised). Upstream computes `from_angle(atan2(a×b, a·b))`; both agree to the oracle
    /// tolerance. Upstream: `UnitComplex::rotation_between_axis`.
    fn rotation_between_axis(a: Unit<Vector2<T>>, b: Unit<Vector2<T>>) -> UnitComplex<T>;
    /// `self * v` for a unit vector: `transform_vector` of its value, not renormalised. Upstream:
    /// `Mul<Unit<Vector2>> for UnitComplex`.
    fn transform_unit_vector(self: UnitComplex<T>, v: Unit<Vector2<T>>) -> Unit<Vector2<T>>;
    /// `self⁻¹ * v` for a unit vector, not renormalised. Upstream:
    /// `inverse_transform_unit_vector`.
    fn inverse_transform_unit_vector(self: UnitComplex<T>, v: Unit<Vector2<T>>) -> Unit<Vector2<T>>;
    /// `self * r` with a rotation matrix: `self * from_rotation_matrix(r)`, a unit complex number
    /// (two fused kernels). Upstream: `Mul<Rotation2> for UnitComplex`.
    fn mul_rotation(self: UnitComplex<T>, r: Rotation2<T>) -> UnitComplex<T>;
    /// `self / r = self * r⁻¹`, the conjugation folded into the two kernels. Upstream:
    /// `Div<Rotation2> for UnitComplex`.
    fn div_rotation(self: UnitComplex<T>, r: Rotation2<T>) -> UnitComplex<T>;
    /// `self * t`: rotation `self`, translation `self · t`. Upstream: `Mul<Translation2> for
    /// UnitComplex` (an `Isometry2`).
    fn mul_translation(self: UnitComplex<T>, t: Translation2<T>) -> Isometry2<T>;
    /// `self * iso`: rotation `self * iso.rotation`, translation `self · iso.translation`.
    /// Upstream: `Mul<Isometry2> for UnitComplex`.
    fn mul_isometry(self: UnitComplex<T>, iso: Isometry2<T>) -> Isometry2<T>;
    /// `self * sim`: the isometry part composed like `mul_isometry`, the scaling unchanged.
    /// Upstream: `Mul<Similarity2> for UnitComplex`.
    fn mul_similarity(self: UnitComplex<T>, sim: Similarity2<T>) -> Similarity2<T>;
}

/// Operations of `UnitComplex<T>` that go through an angle, hence their own trait: scalars may
/// implement `Real` only.
///
/// Every method here costs at least one transcendental (`sin_cos` 16 800, `atan2` 15 400 gas on
/// `Fixed`), one to two orders of magnitude above the algebraic operations of
/// `UnitComplexTrait`. Prefer `rotation_between`, `rotation_to` and `append_axisangle_linearized`
/// where an angle is not the actual input or output.
pub trait UnitComplexAngleTrait<T> {
    /// The rotation of angle `angle` (radians): `(cos angle, sin angle)` through one `sin_cos`.
    /// The pair is unit within about 2 ulp (each component is floored). Upstream:
    /// `UnitComplex::new`.
    fn new(angle: T) -> UnitComplex<T>;
    /// Alias of `new`. Upstream: `UnitComplex::from_angle`.
    fn from_angle(angle: T) -> UnitComplex<T>;
    /// The angle of the rotation, in `(-π, π]`: `atan2(im, re)`, accurate to about 12 ulp
    /// (DESIGN D6). `angle()` of the identity is exactly `0`. Upstream: `angle`.
    fn angle(self: UnitComplex<T>) -> T;
    /// The angle of the rotation taking `self` to `other`, in `(-π, π]`: the angle of
    /// `other * self⁻¹` (one composition and one `atan2`, no subtraction of angles, so the
    /// result never needs wrapping). Upstream: `angle_to`.
    fn angle_to(self: UnitComplex<T>, other: UnitComplex<T>) -> T;
    /// The rotation taking the direction of `a` to the direction of `b`, scaled by `s` (the
    /// rotation of angle `s·θ`), or the identity when either vector is zero. Upstream:
    /// `UnitComplex::scaled_rotation_between`.
    ///
    /// `atan2(a×b, a·b)` is scale invariant, so unlike upstream the inputs are NOT normalized
    /// first: one fused `dot`, one fused `perp`, one `atan2`, one product and one `sin_cos`.
    /// Panics when `a·b` or `a×b` does not fit (`|a|·|b|` above about 2.1e9).
    fn scaled_rotation_between(a: Vector2<T>, b: Vector2<T>, s: T) -> UnitComplex<T>;
    /// The rotation of angle `n·θ` (`self` applied `n` times, for a real `n`): one `atan2`, one
    /// product and one `sin_cos`, about 34 000 gas. The angle is taken in `(-π, π]` first, so
    /// `powf(2)` of a rotation by `3π/4` is a rotation by `-π/2`, like upstream. Upstream:
    /// `powf`.
    fn powf(self: UnitComplex<T>, n: T) -> UnitComplex<T>;
    /// Spherical interpolation: the rotation of angle `θ_self + t · angle_to(other)`, i.e.
    /// upstream's `self * UnitComplex::new(self.angle_to(other) * t)`. `t` is not clamped;
    /// `t = 0` gives `self` (exactly) and `t = 1` gives `other` within the rounding of the
    /// composition. The SHORTEST arc is taken, since `angle_to` lands in `(-π, π]`.
    ///
    /// 43 930 gas (one fused product, one `atan2`, one `sin_cos` and one composition).
    /// The physics engine never calls `slerp` (docs/research/01, §3.3): it exists for clients
    /// interpolating between two poses. Upstream: `slerp`.
    fn slerp(self: UnitComplex<T>, other: UnitComplex<T>, t: T) -> UnitComplex<T>;
    /// The rotation of angle `axisangle` (the single coordinate of upstream's `Vector1`: there is
    /// no `Vector1` here). Upstream: `UnitComplex::from_scaled_axis`.
    fn from_scaled_axis(axisangle: T) -> UnitComplex<T>;
    /// The angle in `(-π, π]` (the single coordinate of upstream's `Vector1`). Upstream:
    /// `scaled_axis`.
    fn scaled_axis(self: UnitComplex<T>) -> T;
    /// `(axis, angle)` with the axis `±1` (the coordinate of upstream's `Unit<Vector1>`) and the
    /// angle in `(0, π]`: `(1, θ)` for `θ > 0`, `(-1, -θ)` for `θ < 0`, `None` for `θ = 0`.
    /// Upstream: `axis_angle`.
    fn axis_angle(self: UnitComplex<T>) -> Option<(T, T)>;
    /// `scaled_rotation_between` of two UNIT vectors (the same kernel, which does not normalise
    /// anyway). Upstream: `UnitComplex::scaled_rotation_between_axis`.
    fn scaled_rotation_between_axis(
        a: Unit<Vector2<T>>, b: Unit<Vector2<T>>, s: T,
    ) -> UnitComplex<T>;
    /// `from_matrix_eps(m, default_epsilon, 0, identity)`: the closed form. Upstream:
    /// `UnitComplex::from_matrix`.
    fn from_matrix(m: Matrix2<T>) -> UnitComplex<T>;
    /// The rotation part of `m` (the rotation maximising `tr(Rᵀ m)`).
    /// - `max_iter = 0` (upstream: iterate until convergence): the LIMIT in closed form, trig-free:
    ///   `(m11 + m22, m21 - m12)` normalised (one `norm2`, two divisions; the identity when that
    ///   pair is zero), 11 550 gas; `eps` and `guess` unused. Iterating instead costs 160 970 on
    ///   the bench matrix (`bench_unit_complex_from_matrix__alt_iterate`).
    /// - `max_iter > 0`: upstream's 2D Müller iteration from `guess`,
    ///   `δ = (Σ_c r_c ⊥ m_c) / (|Σ_c r_c · m_c| + ε)`, `R ← R(δ) · R` until `|δ| <=
    ///   eps`. With `R = (re, im)` both sums are one fused kernel of two products on the exact `m21
    ///   - m12`
    ///   and `m11 + m22` (panics when those overflow, entries above about 1e9), then one division,
    ///   one `sin_cos` and one composition per iteration (about 40 000 gas). **Bounded:** at most
    ///   `min(max_iter, FROM_MATRIX_MAX_ITER)` iterations.
    ///
    /// Upstream: `UnitComplex::from_matrix_eps`.
    fn from_matrix_eps(
        m: Matrix2<T>, eps: T, max_iter: usize, guess: UnitComplex<T>,
    ) -> UnitComplex<T>;
}

pub impl UnitComplexImpl<
    T, impl R: Real<T>, +Add<T>, +Sub<T>, +Mul<T>, +Neg<T>, +PartialEq<T>, +Copy<T>, +Drop<T>,
> of UnitComplexTrait<T> {
    #[inline(always)]
    fn identity() -> UnitComplex<T> {
        UnitComplex { re: R::one(), im: R::zero() }
    }

    #[inline(always)]
    fn from_cos_sin_unchecked(cos: T, sin: T) -> UnitComplex<T> {
        UnitComplex { re: cos, im: sin }
    }

    #[inline(always)]
    fn from_rotation_matrix(r: Rotation2<T>) -> UnitComplex<T> {
        UnitComplex { re: r.matrix.m11, im: r.matrix.m21 }
    }

    fn rotation_between(a: Vector2<T>, b: Vector2<T>) -> UnitComplex<T> {
        let dot = R::sum_prod2(a.x, b.x, a.y, b.y);
        let perp = R::diff_prod(a.x, b.y, a.y, b.x);
        // |(a·b, a×b)| = |a|·|b|, so the pair carries its own normalizing factor.
        let n = R::norm2(dot, perp);
        if n == R::zero() {
            return UnitComplex { re: R::one(), im: R::zero() };
        }
        UnitComplex { re: R::div(dot, n), im: R::div(perp, n) }
    }

    #[inline(always)]
    fn complex(self: UnitComplex<T>) -> Vector2<T> {
        Vector2 { x: self.re, y: self.im }
    }

    #[inline(always)]
    fn re(self: UnitComplex<T>) -> T {
        self.re
    }

    #[inline(always)]
    fn im(self: UnitComplex<T>) -> T {
        self.im
    }

    #[inline(always)]
    fn cos_angle(self: UnitComplex<T>) -> T {
        self.re
    }

    #[inline(always)]
    fn sin_angle(self: UnitComplex<T>) -> T {
        self.im
    }

    #[inline(always)]
    fn conjugate(self: UnitComplex<T>) -> UnitComplex<T> {
        UnitComplex { re: self.re, im: -self.im }
    }

    #[inline(always)]
    fn inverse(self: UnitComplex<T>) -> UnitComplex<T> {
        UnitComplex { re: self.re, im: -self.im }
    }

    #[inline(always)]
    fn rotation_to(self: UnitComplex<T>, other: UnitComplex<T>) -> UnitComplex<T> {
        // other * self⁻¹, with the conjugation folded into the kernels.
        UnitComplex {
            re: R::sum_prod2(other.re, self.re, other.im, self.im),
            im: R::diff_prod(other.im, self.re, other.re, self.im),
        }
    }

    #[inline(always)]
    fn transform_vector(self: UnitComplex<T>, v: Vector2<T>) -> Vector2<T> {
        Vector2 {
            x: R::diff_prod(self.re, v.x, self.im, v.y),
            y: R::sum_prod2(self.im, v.x, self.re, v.y),
        }
    }

    #[inline(always)]
    fn transform_point(self: UnitComplex<T>, p: Point2<T>) -> Point2<T> {
        Point2 {
            x: R::diff_prod(self.re, p.x, self.im, p.y),
            y: R::sum_prod2(self.im, p.x, self.re, p.y),
        }
    }

    #[inline(always)]
    fn inverse_transform_vector(self: UnitComplex<T>, v: Vector2<T>) -> Vector2<T> {
        Vector2 {
            x: R::sum_prod2(self.re, v.x, self.im, v.y),
            y: R::diff_prod(self.re, v.y, self.im, v.x),
        }
    }

    #[inline(always)]
    fn inverse_transform_point(self: UnitComplex<T>, p: Point2<T>) -> Point2<T> {
        Point2 {
            x: R::sum_prod2(self.re, p.x, self.im, p.y),
            y: R::diff_prod(self.re, p.y, self.im, p.x),
        }
    }

    #[inline(always)]
    fn to_rotation_matrix(self: UnitComplex<T>) -> Rotation2<T> {
        Rotation2 { matrix: Matrix2 { m11: self.re, m21: self.im, m12: -self.im, m22: self.re } }
    }

    #[inline(always)]
    fn to_homogeneous(self: UnitComplex<T>) -> Matrix3<T> {
        Matrix3 {
            m11: self.re,
            m21: self.im,
            m31: R::zero(),
            m12: -self.im,
            m22: self.re,
            m32: R::zero(),
            m13: R::zero(),
            m23: R::zero(),
            m33: R::one(),
        }
    }

    #[inline(always)]
    fn renormalize(ref self: UnitComplex<T>) -> T {
        let n = R::norm2(self.re, self.im);
        self = UnitComplex { re: R::div(self.re, n), im: R::div(self.im, n) };
        n
    }

    #[inline(always)]
    fn renormalize_fast(ref self: UnitComplex<T>) {
        // (3 - |c|²) / 2 = floor(-|c|² * 1/2 + 3/2): one fused kernel.
        let f = R::mul_add(R::norm_squared2(self.re, self.im), -R::HALF, R::HALF + R::one());
        self = UnitComplex { re: self.re * f, im: self.im * f };
    }

    #[inline(always)]
    fn abs_diff_eq(self: UnitComplex<T>, other: UnitComplex<T>, ulps: u64) -> bool {
        R::abs_diff_eq(self.re, other.re, ulps) && R::abs_diff_eq(self.im, other.im, ulps)
    }

    #[inline(always)]
    fn relative_eq(
        self: UnitComplex<T>, other: UnitComplex<T>, epsilon: u64, max_relative: T,
    ) -> bool {
        ApproxEqTrait::relative_eq(self.re, other.re, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(self.im, other.im, epsilon, max_relative)
    }

    #[inline(always)]
    fn ulps_eq(self: UnitComplex<T>, other: UnitComplex<T>, epsilon: u64, max_ulps: u32) -> bool {
        ApproxEqTrait::ulps_eq(self.re, other.re, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(self.im, other.im, epsilon, max_ulps)
    }

    fn cast<U, +Into<T, U>, +Drop<U>>(self: UnitComplex<T>) -> UnitComplex<U> {
        UnitComplex { re: self.re.into(), im: self.im.into() }
    }

    #[inline(always)]
    fn from_complex(q: Vector2<T>) -> UnitComplex<T> {
        let n = R::norm2(q.x, q.y);
        UnitComplex { re: R::div(q.x, n), im: R::div(q.y, n) }
    }

    #[inline(always)]
    fn from_complex_and_get(q: Vector2<T>) -> (UnitComplex<T>, T) {
        let n = R::norm2(q.x, q.y);
        (UnitComplex { re: R::div(q.x, n), im: R::div(q.y, n) }, n)
    }

    #[inline(always)]
    fn from_basis_unchecked(basis: [Vector2<T>; 2]) -> UnitComplex<T> {
        let [x, _y] = basis;
        UnitComplex { re: x.x, im: x.y }
    }

    #[inline(always)]
    fn rotation_between_axis(a: Unit<Vector2<T>>, b: Unit<Vector2<T>>) -> UnitComplex<T> {
        Self::rotation_between(a.value, b.value)
    }

    #[inline(always)]
    fn transform_unit_vector(self: UnitComplex<T>, v: Unit<Vector2<T>>) -> Unit<Vector2<T>> {
        Unit { value: Self::transform_vector(self, v.value) }
    }

    #[inline(always)]
    fn inverse_transform_unit_vector(
        self: UnitComplex<T>, v: Unit<Vector2<T>>,
    ) -> Unit<Vector2<T>> {
        Unit { value: Self::inverse_transform_vector(self, v.value) }
    }

    #[inline(always)]
    fn mul_rotation(self: UnitComplex<T>, r: Rotation2<T>) -> UnitComplex<T> {
        let (re, im) = (r.matrix.m11, r.matrix.m21);
        UnitComplex {
            re: R::diff_prod(self.re, re, self.im, im), im: R::sum_prod2(self.re, im, self.im, re),
        }
    }

    #[inline(always)]
    fn div_rotation(self: UnitComplex<T>, r: Rotation2<T>) -> UnitComplex<T> {
        let (re, im) = (r.matrix.m11, r.matrix.m21);
        UnitComplex {
            re: R::sum_prod2(self.re, re, self.im, im), im: R::diff_prod(self.im, re, self.re, im),
        }
    }

    #[inline(always)]
    fn mul_translation(self: UnitComplex<T>, t: Translation2<T>) -> Isometry2<T> {
        Isometry2 {
            rotation: self,
            translation: Translation2 { vector: Self::transform_vector(self, t.vector) },
        }
    }

    #[inline(always)]
    fn mul_isometry(self: UnitComplex<T>, iso: Isometry2<T>) -> Isometry2<T> {
        let r = iso.rotation;
        Isometry2 {
            rotation: UnitComplex {
                re: R::diff_prod(self.re, r.re, self.im, r.im),
                im: R::sum_prod2(self.re, r.im, self.im, r.re),
            },
            translation: Translation2 {
                vector: Self::transform_vector(self, iso.translation.vector),
            },
        }
    }

    #[inline(always)]
    fn mul_similarity(self: UnitComplex<T>, sim: Similarity2<T>) -> Similarity2<T> {
        Similarity2 { isometry: Self::mul_isometry(self, sim.isometry), scaling: sim.scaling }
    }
}

/// Crate-internal by-value forms of the in-place `renormalize` / `renormalize_fast` (WP 8.0: the
/// public methods are upstream's `&mut self` ones), for the tests and the value-style call sites.
#[generate_trait]
pub(crate) impl UnitComplexInternalImpl<
    T, impl R: Real<T>, +Add<T>, +Sub<T>, +Mul<T>, +Neg<T>, +PartialEq<T>, +Copy<T>, +Drop<T>,
> of UnitComplexInternalTrait<T> {
    /// `self` renormalized exactly (`UnitComplexTrait::renormalize`), by value.
    #[inline(always)]
    fn renormalized(self: UnitComplex<T>) -> UnitComplex<T> {
        let mut r = self;
        let _ = UnitComplexTrait::renormalize(ref r);
        r
    }

    /// `self` renormalized by one Newton step (`UnitComplexTrait::renormalize_fast`), by value.
    #[inline(always)]
    fn renormalized_fast(self: UnitComplex<T>) -> UnitComplex<T> {
        let mut r = self;
        UnitComplexTrait::renormalize_fast(ref r);
        r
    }
}

pub impl UnitComplexAngleImpl<
    T,
    impl R: Real<T>,
    impl Tr: Transcendental<T>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +Copy<T>,
    +Drop<T>,
> of UnitComplexAngleTrait<T> {
    #[inline(always)]
    fn new(angle: T) -> UnitComplex<T> {
        let (sin, cos) = Tr::sin_cos(angle);
        UnitComplex { re: cos, im: sin }
    }

    #[inline(always)]
    fn from_angle(angle: T) -> UnitComplex<T> {
        let (sin, cos) = Tr::sin_cos(angle);
        UnitComplex { re: cos, im: sin }
    }

    #[inline(always)]
    fn angle(self: UnitComplex<T>) -> T {
        Tr::atan2(self.im, self.re)
    }

    #[inline(always)]
    fn angle_to(self: UnitComplex<T>, other: UnitComplex<T>) -> T {
        // The angle of `other * self⁻¹`, the conjugation folded into the kernels.
        Tr::atan2(
            R::diff_prod(other.im, self.re, other.re, self.im),
            R::sum_prod2(other.re, self.re, other.im, self.im),
        )
    }

    fn scaled_rotation_between(a: Vector2<T>, b: Vector2<T>, s: T) -> UnitComplex<T> {
        let dot = R::sum_prod2(a.x, b.x, a.y, b.y);
        let perp = R::diff_prod(a.x, b.y, a.y, b.x);
        if dot == R::zero() && perp == R::zero() {
            return UnitComplex { re: R::one(), im: R::zero() };
        }
        let (sin, cos) = Tr::sin_cos(Tr::atan2(perp, dot) * s);
        UnitComplex { re: cos, im: sin }
    }

    #[inline(always)]
    fn powf(self: UnitComplex<T>, n: T) -> UnitComplex<T> {
        let (sin, cos) = Tr::sin_cos(Tr::atan2(self.im, self.re) * n);
        UnitComplex { re: cos, im: sin }
    }

    fn slerp(self: UnitComplex<T>, other: UnitComplex<T>, t: T) -> UnitComplex<T> {
        let angle = Tr::atan2(
            R::diff_prod(other.im, self.re, other.re, self.im),
            R::sum_prod2(other.re, self.re, other.im, self.im),
        );
        let (sin, cos) = Tr::sin_cos(angle * t);
        UnitComplex {
            re: R::diff_prod(self.re, cos, self.im, sin),
            im: R::sum_prod2(self.im, cos, self.re, sin),
        }
    }

    #[inline(always)]
    fn from_scaled_axis(axisangle: T) -> UnitComplex<T> {
        let (sin, cos) = Tr::sin_cos(axisangle);
        UnitComplex { re: cos, im: sin }
    }

    #[inline(always)]
    fn scaled_axis(self: UnitComplex<T>) -> T {
        Tr::atan2(self.im, self.re)
    }

    fn axis_angle(self: UnitComplex<T>) -> Option<(T, T)> {
        let ang = Tr::atan2(self.im, self.re);
        if ang == R::zero() {
            None
        } else if R::is_sign_positive(ang) {
            Some((R::one(), ang))
        } else {
            Some((R::NEG_ONE, -ang))
        }
    }

    #[inline(always)]
    fn scaled_rotation_between_axis(
        a: Unit<Vector2<T>>, b: Unit<Vector2<T>>, s: T,
    ) -> UnitComplex<T> {
        Self::scaled_rotation_between(a.value, b.value, s)
    }

    #[inline(always)]
    fn from_matrix(m: Matrix2<T>) -> UnitComplex<T> {
        Self::from_matrix_eps(m, R::default_epsilon(), 0, UnitComplexTrait::identity())
    }

    fn from_matrix_eps(
        m: Matrix2<T>, eps: T, max_iter: usize, guess: UnitComplex<T>,
    ) -> UnitComplex<T> {
        if max_iter == 0 {
            // The limit in closed form: the angle maximising `tr(Rᵀ m)` is that of
            // `(m11 + m22, m21 - m12)`.
            let (re, im) = (m.m11 + m.m22, m.m21 - m.m12);
            let n = R::norm2(re, im);
            if n == R::zero() {
                return UnitComplexTrait::identity();
            }
            return UnitComplex { re: R::div(re, n), im: R::div(im, n) };
        }
        let (r, _) = UnitComplexAngleInternalTrait::from_matrix_eps_count(m, eps, max_iter, guess);
        r
    }
}

/// Crate-internal kernel of `UnitComplexAngleTrait::from_matrix_eps`.
#[generate_trait]
pub(crate) impl UnitComplexAngleInternalImpl<
    T,
    impl R: Real<T>,
    impl Tr: Transcendental<T>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +Copy<T>,
    +Drop<T>,
> of UnitComplexAngleInternalTrait<T> {
    /// `from_matrix_eps` and the number of iterations it ran (for the convergence tests).
    fn from_matrix_eps_count(
        m: Matrix2<T>, eps: T, max_iter: usize, guess: UnitComplex<T>,
    ) -> (UnitComplex<T>, usize) {
        let cap = if max_iter == 0 || max_iter > FROM_MATRIX_MAX_ITER {
            FROM_MATRIX_MAX_ITER
        } else {
            max_iter
        };
        let mut r = guess;
        let mut iter: usize = 0;
        while iter < cap {
            iter += 1;
            // With columns (re, im) and (-im, re), a = m21 - m12 and b = m11 + m22 (exact):
            // axis = re·a - im·b, denom = re·b + im·a, one fused kernel each.
            let (a, b) = (m.m21 - m.m12, m.m11 + m.m22);
            let axis = R::diff_prod(r.re, a, r.im, b);
            let denom = R::sum_prod2(r.re, b, r.im, a);
            let angle = R::div(axis, R::abs(denom) + R::default_epsilon());
            // `|angle| <= eps`, with `Real` comparisons only (the trait carries no `PartialOrd`).
            if R::max(R::abs(angle), eps) == eps {
                break;
            }
            let (sin, cos) = Tr::sin_cos(angle);
            r =
                UnitComplex {
                    re: R::diff_prod(cos, r.re, sin, r.im), im: R::sum_prod2(cos, r.im, sin, r.re),
                };
        }
        (r, iter)
    }
}

/// `a * b`: the composition of two rotations (turn by `b`, then by `a` — the product of complex
/// numbers is commutative, so the order does not matter), `(re_a·re_b - im_a·im_b,
/// re_a·im_b + im_a·re_b)`. Two fused kernels: each output component is the exact product floored
/// once, bit for bit what upstream's complex multiplication gives. The norm of the result drifts
/// by up to 2 ulp per product: renormalize after a long chain. Upstream: `Mul`.
pub impl UnitComplexMul<T, impl R: Real<T>, +Copy<T>, +Drop<T>> of Mul<UnitComplex<T>> {
    #[inline(always)]
    fn mul(lhs: UnitComplex<T>, rhs: UnitComplex<T>) -> UnitComplex<T> {
        UnitComplex {
            re: R::diff_prod(lhs.re, rhs.re, lhs.im, rhs.im),
            im: R::sum_prod2(lhs.re, rhs.im, lhs.im, rhs.re),
        }
    }
}

/// `a / b = a * b⁻¹`, the conjugation folded into the two kernels:
/// `(re_a·re_b + im_a·im_b, im_a·re_b - re_a·im_b)`, each exactly floored. Upstream: `Div for
/// UnitComplex`.
pub impl UnitComplexDiv<T, impl R: Real<T>, +Copy<T>, +Drop<T>> of Div<UnitComplex<T>> {
    #[inline(always)]
    fn div(lhs: UnitComplex<T>, rhs: UnitComplex<T>) -> UnitComplex<T> {
        UnitComplex {
            re: R::sum_prod2(lhs.re, rhs.re, lhs.im, rhs.im),
            im: R::diff_prod(lhs.im, rhs.re, lhs.re, rhs.im),
        }
    }
}

/// `Default::default()`: the identity rotation. Upstream: `Default for UnitComplex`.
pub impl UnitComplexDefault<T, impl R: Real<T>, +Drop<T>> of Default<UnitComplex<T>> {
    #[inline(always)]
    fn default() -> UnitComplex<T> {
        UnitComplex { re: R::one(), im: R::zero() }
    }
}

/// `One::one()`: the identity rotation; `is_one` compares with `(1, 0)` exactly. Upstream:
/// `num::One for UnitComplex`.
pub impl UnitComplexOne<
    T, impl R: Real<T>, +PartialEq<T>, +Copy<T>, +Drop<T>,
> of One<UnitComplex<T>> {
    #[inline(always)]
    fn one() -> UnitComplex<T> {
        UnitComplex { re: R::one(), im: R::zero() }
    }

    #[inline(always)]
    fn is_one(self: @UnitComplex<T>) -> bool {
        *self.re == R::one() && *self.im == R::zero()
    }

    #[inline(always)]
    fn is_non_one(self: @UnitComplex<T>) -> bool {
        !Self::is_one(self)
    }
}

/// `c.into()`: the isometry of rotation `c` and zero translation. Upstream: `SubsetOf<Isometry2>
/// for UnitComplex` (`nalgebra::convert(c)`).
pub impl Isometry2FromUnitComplex<
    T, impl R: Real<T>, +Drop<T>,
> of Into<UnitComplex<T>, Isometry2<T>> {
    #[inline(always)]
    fn into(self: UnitComplex<T>) -> Isometry2<T> {
        Isometry2 {
            rotation: self,
            translation: Translation2 { vector: Vector2 { x: R::zero(), y: R::zero() } },
        }
    }
}

/// `c.into()`: the similarity of rotation `c`, zero translation and scaling 1. Upstream:
/// `SubsetOf<Similarity2> for UnitComplex` (`nalgebra::convert(c)`).
pub impl Similarity2FromUnitComplex<
    T, impl R: Real<T>, +Drop<T>,
> of Into<UnitComplex<T>, Similarity2<T>> {
    #[inline(always)]
    fn into(self: UnitComplex<T>) -> Similarity2<T> {
        Similarity2 {
            isometry: Isometry2 {
                rotation: self,
                translation: Translation2 { vector: Vector2 { x: R::zero(), y: R::zero() } },
            },
            scaling: R::one(),
        }
    }
}
