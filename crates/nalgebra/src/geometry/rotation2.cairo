//! `Rotation2`: a 2D rotation stored as a 2x2 orthogonal matrix (upstream
//! `nalgebra::Rotation2`, which is `Rotation<T, 2>`).
//!
//! - `Rotation2Trait` / `Rotation2Impl`: construction, accessors, composition, transforms,
//!   conversions and renormalization — everything that is algebraic, hence available for any
//!   `simba::scalar::Real` scalar;
//! - `Rotation2AngleTrait` / `Rotation2AngleImpl`: the operations that go through an angle
//!   (`new`, `angle`, `powf`, `scaled_rotation_between`), which additionally need
//!   `simba::scalar::Transcendental`;
//! - `a * b` (composition of two rotations) and the conversions from / to `UnitComplex<T>`: their
//!   impls live in this module, where the compiler finds them without any import.
//!
//! `Rotation2` and `UnitComplex` hold the same information: the matrix is
//! `[[cos θ, -sin θ], [sin θ, cos θ]]`, whose first column IS the complex `(re, im)`. Measured
//! (Sierra gas, net of the group baseline), the matrix form costs **2.6x more per composition**
//! (10 260 against 4 000) and twice the storage, for transforms that are bit-identical and cost
//! exactly the same (4 000 either way, `bench_rotation2_transform_point__alt_unit_complex`). It
//! exists because upstream has it, because it composes with the `Matrix2` world without a
//! conversion, and because a `Rotation2` is what `to_homogeneous` and the matrix decompositions
//! consume. **Prefer `UnitComplex` in hot code**; converting costs 800 gas one way and 2 400 the
//! other, so even a single composition pays for the round trip.
//!
//! Numeric contract (AGENTS.md): every sum of products goes through a fused `Real` kernel (one
//! floor rounding and one overflow check per output scalar); nothing wraps silently.

use core::num::traits::One;
use core::ops::Index;
use simba::scalar::{Real, Transcendental};
use crate::base::matrix1::Matrix1;
use crate::base::matrix2::Matrix2;
use crate::base::matrix3::Matrix3;
use crate::base::point2::Point2;
use crate::base::unit::Unit;
use crate::base::vector2::Vector2;
use super::isometry2::Isometry2;
use super::quaternion::ApproxEqTrait;
use super::similarity2::Similarity2;
use super::translation2::Translation2;
use super::unit_complex::{UnitComplex, UnitComplexAngleTrait};

#[cfg(test)]
mod tests;

/// Panic messages of `Rotation2` (stable API).
pub mod errors {
    /// `r[(i, j)]` with `i > 1` or `j > 1`.
    pub const INDEX_OUT_OF_BOUNDS: felt252 = 'nalgebra: index out of bounds';
}

/// A 2D rotation of angle `θ`, stored as the orthogonal matrix
/// `[[cos θ, -sin θ], [sin θ, cos θ]]` (`matrix.m11 = cos θ`, `matrix.m21 = sin θ`).
///
/// Nothing enforces the invariant `Rᵀ R = I`: build with `new` / `from_matrix` /
/// `rotation_between` / a `UnitComplex` conversion, or with `from_matrix_unchecked` when the
/// matrix is known to be a rotation.
#[derive(Copy, Drop, PartialEq, Serde, Debug, Hash)]
pub struct Rotation2<T> {
    pub matrix: Matrix2<T>,
}

/// Operations of `Rotation2<T>` that need no trigonometry, over a `Real` scalar. By value,
/// unrolled, no loop.
pub trait Rotation2Trait<T> {
    /// The identity rotation (the identity matrix, angle `0`). Exact. Upstream:
    /// `Rotation2::identity`.
    fn identity() -> Rotation2<T>;
    /// Wraps `m` WITHOUT checking that it is a rotation: the caller guarantees `Rᵀ R = I` and
    /// `det R = 1`. Upstream: `Rotation2::from_matrix_unchecked`.
    fn from_matrix_unchecked(m: Matrix2<T>) -> Rotation2<T>;
    /// The rotation closest to `m` (maximising `tr(Rᵀ m)`): the closed-form LIMIT of upstream's
    /// iteration `from_matrix_eps(m, default_epsilon, 0, identity)`, i.e. the angle of
    /// `(m11 + m22, m21 - m12)` — that pair normalised (one `norm2`, two correctly rounded
    /// divisions, no trigonometry), the identity when it is zero. Bit for bit
    /// `UnitComplex::from_matrix(m).into()`. Panics on overflow of the two exact sums (entries
    /// above about 1e9). Upstream: `Rotation2::from_matrix`.
    fn from_matrix(m: Matrix2<T>) -> Rotation2<T>;
    /// The wrapped matrix (a copy: everything is by value here). Upstream: `matrix` (a
    /// reference) / `into_inner`.
    fn matrix(self: Rotation2<T>) -> Matrix2<T>;
    /// Alias of `matrix`. Upstream: `Rotation::into_inner`.
    fn into_inner(self: Rotation2<T>) -> Matrix2<T>;
    /// The inverse rotation, which for an orthogonal matrix is the transpose (no division).
    /// Exact. Upstream: `inverse`.
    fn inverse(self: Rotation2<T>) -> Rotation2<T>;
    /// The transposed matrix, wrapped as a rotation: the same thing as `inverse`. Exact.
    /// Upstream: `transpose`.
    fn transpose(self: Rotation2<T>) -> Rotation2<T>;
    /// The rotation taking the direction of `a` to the direction of `b`, or the identity when
    /// either vector is zero: `UnitComplexTrait::rotation_between` expanded into a matrix (see
    /// its accuracy notes). Upstream: `Rotation2::rotation_between`.
    fn rotation_between(a: Vector2<T>, b: Vector2<T>) -> Rotation2<T>;
    /// `self * v`: `v` rotated by `θ`, two fused kernels (one floor rounding per output
    /// component). Bit for bit what `UnitComplex::transform_vector` gives, for exactly the same
    /// 4 000 gas: the matrix stores `-sin θ` where the complex negates inside the kernel. Panics
    /// on overflow. Upstream: `transform_vector` (`self * v`).
    fn transform_vector(self: Rotation2<T>, v: Vector2<T>) -> Vector2<T>;
    /// `self * p`: `p` rotated around the origin by `θ`. Upstream: `transform_point`.
    fn transform_point(self: Rotation2<T>, p: Point2<T>) -> Point2<T>;
    /// `Rᵀ v`: `v` rotated by `-θ`, two fused kernels. Never forms the inverse rotation.
    /// Upstream: `inverse_transform_vector`.
    fn inverse_transform_vector(self: Rotation2<T>, v: Vector2<T>) -> Vector2<T>;
    /// `Rᵀ p`: `p` rotated around the origin by `-θ`. Upstream: `inverse_transform_point`.
    fn inverse_transform_point(self: Rotation2<T>, p: Point2<T>) -> Point2<T>;
    /// The same rotation as a 3x3 homogeneous matrix (the rotation block, then `(0, 0, 1)`).
    /// Exact. Upstream: `to_homogeneous`.
    fn to_homogeneous(self: Rotation2<T>) -> Matrix3<T>;
    /// Renormalizes exactly: normalizes the first column (one `norm2` and two exactly floored
    /// divisions), then rebuilds the matrix from it, so `Rᵀ R = I` within a few ulp again.
    /// Panics with `Fixed: division by zero` when that column is zero. Upstream:
    /// `Rotation2::renormalize` (which does the same through `UnitComplex`, in place).
    fn renormalize(ref self: Rotation2<T>);
    /// `true` when every component is within `ulps` smallest units (raw units for fixed point) of
    /// the matching component of `other`; cannot overflow. Upstream:
    /// `approx::AbsDiffEq::abs_diff_eq`, the tolerance being counted in ulp instead of a float
    /// epsilon (DESIGN D3).
    fn abs_diff_eq(self: Rotation2<T>, other: Rotation2<T>, ulps: u64) -> bool;
    /// Deprecated alias of `into_inner`. Upstream: `Rotation::unwrap`.
    fn unwrap(self: Rotation2<T>) -> Matrix2<T>;
    /// The rotation whose matrix has the columns `basis[0]` and `basis[1]`, WITHOUT checking that
    /// they are orthonormal. Exact. Upstream: `Rotation2::from_basis_unchecked`.
    fn from_basis_unchecked(basis: [Vector2<T>; 2]) -> Rotation2<T>;
    /// The rotation taking `self` to `other`: `other * self⁻¹`, the product with the transpose
    /// (four fused kernels, one rounding per entry). Upstream: `Rotation2::rotation_to`.
    fn rotation_to(self: Rotation2<T>, other: Rotation2<T>) -> Rotation2<T>;
    /// `self * v` for a unit vector: `transform_vector` of its value, re-wrapped WITHOUT
    /// renormalising. Upstream: `Mul<Unit<Vector2>> for Rotation2` (`r * v`).
    fn transform_unit_vector(self: Rotation2<T>, v: Unit<Vector2<T>>) -> Unit<Vector2<T>>;
    /// `self⁻¹ * v` for a unit vector, not renormalised. Upstream:
    /// `inverse_transform_unit_vector`.
    fn inverse_transform_unit_vector(self: Rotation2<T>, v: Unit<Vector2<T>>) -> Unit<Vector2<T>>;
    /// `self * c`: the unit complex `UnitComplex::from_rotation_matrix(self) * c` (the first
    /// column times `c`, two fused kernels). Upstream: `Mul<UnitComplex> for Rotation2`.
    fn mul_unit_complex(self: Rotation2<T>, c: UnitComplex<T>) -> UnitComplex<T>;
    /// `self / c = self * c⁻¹`: the first column times the conjugate of `c` (two fused
    /// kernels). Upstream: `Div<UnitComplex> for Rotation2`.
    fn div_unit_complex(self: Rotation2<T>, c: UnitComplex<T>) -> UnitComplex<T>;
    /// `true` when every entry is `relative_eq` to the matching entry of `other` (see
    /// `QuaternionTrait::relative_eq`). Upstream: `approx::RelativeEq::relative_eq` (DESIGN D3).
    fn relative_eq(self: Rotation2<T>, other: Rotation2<T>, epsilon: u64, max_relative: T) -> bool;
    /// `true` when every entry is `ulps_eq` to the matching entry of `other` (see
    /// `QuaternionTrait::ulps_eq`). Upstream: `approx::UlpsEq::ulps_eq` (DESIGN D3).
    fn ulps_eq(self: Rotation2<T>, other: Rotation2<T>, epsilon: u64, max_ulps: u32) -> bool;
    /// The same rotation with every entry converted by `Into<T, U>` (the identity for the single
    /// scalar `Fixed`). Upstream: `cast` (and `SubsetOf<Rotation2<U>>`).
    fn cast<U, +Into<T, U>, +Drop<U>>(self: Rotation2<T>) -> Rotation2<U>;
}

/// Operations of `Rotation2<T>` that go through an angle, hence their own trait: scalars may
/// implement `Real` only. Each costs at least one transcendental (`sin_cos` 16 800, `atan2`
/// 15 400 gas on `Fixed`).
pub trait Rotation2AngleTrait<T> {
    /// The rotation of angle `angle` (radians): one `sin_cos`, then the matrix
    /// `[[cos, -sin], [sin, cos]]`. Upstream: `Rotation2::new`.
    fn new(angle: T) -> Rotation2<T>;
    /// The angle of the rotation, in `(-π, π]`: `atan2(m21, m11)`, accurate to about 12 ulp
    /// (DESIGN D6). Exactly `0` for the identity. Upstream: `angle`.
    fn angle(self: Rotation2<T>) -> T;
    /// The angle of the rotation taking `self` to `other`, in `(-π, π]`: the angle of
    /// `other * self⁻¹` through the unit complex form (two fused kernels and one `atan2`),
    /// 19 300 gas against 25 760 for the full matrix product followed by `angle`
    /// (`bench_rotation2_angle_to__alt_product_then_angle`). Upstream: `angle_to`.
    fn angle_to(self: Rotation2<T>, other: Rotation2<T>) -> T;
    /// The rotation taking the direction of `a` to the direction of `b`, scaled by `s`, or the
    /// identity when either vector is zero. Upstream: `Rotation2::scaled_rotation_between`.
    fn scaled_rotation_between(a: Vector2<T>, b: Vector2<T>, s: T) -> Rotation2<T>;
    /// The rotation of angle `n·θ`: one `atan2`, one product and one `sin_cos`. Upstream:
    /// `powf`.
    fn powf(self: Rotation2<T>, n: T) -> Rotation2<T>;
    /// The rotation of angle `axisangle.x` (upstream's `Vector1` scaled axis): `new`. Upstream:
    /// `Rotation2::from_scaled_axis`.
    fn from_scaled_axis(axisangle: Matrix1<T>) -> Rotation2<T>;
    /// The angle as a `Vector1` (`angle()`, in `(-π, π]`). Upstream: `Rotation2::scaled_axis`.
    fn scaled_axis(self: Rotation2<T>) -> Matrix1<T>;
    /// Spherical interpolation, upstream's formula: the `slerp` of the two unit complex numbers
    /// (the first columns; one `atan2`, one `sin_cos`, one composition), expanded back into a
    /// matrix. The shortest arc; `t = 0` gives `self` exactly. Upstream: `Rotation2::slerp`.
    fn slerp(self: Rotation2<T>, other: Rotation2<T>, t: T) -> Rotation2<T>;
    /// The rotation part of `m`: `UnitComplex::from_matrix_eps(m, eps, max_iter, guess.into())`
    /// as a matrix — `max_iter = 0` is the closed-form limit (`from_matrix`), `max_iter > 0`
    /// runs upstream's 2D Müller iteration from `guess`, BOUNDED by `FROM_MATRIX_MAX_ITER` (see
    /// `UnitComplexAngleTrait::from_matrix_eps`). Upstream iterates on matrices; the complex form
    /// is the same iteration with a cheaper composition. Upstream: `Rotation2::from_matrix_eps`.
    fn from_matrix_eps(m: Matrix2<T>, eps: T, max_iter: usize, guess: Rotation2<T>) -> Rotation2<T>;
}

pub impl Rotation2Impl<
    T, impl R: Real<T>, +Add<T>, +Sub<T>, +Mul<T>, +Neg<T>, +PartialEq<T>, +Copy<T>, +Drop<T>,
> of Rotation2Trait<T> {
    #[inline(always)]
    fn identity() -> Rotation2<T> {
        Rotation2 {
            matrix: Matrix2 { m11: R::one(), m21: R::zero(), m12: R::zero(), m22: R::one() },
        }
    }

    #[inline(always)]
    fn from_matrix_unchecked(m: Matrix2<T>) -> Rotation2<T> {
        Rotation2 { matrix: m }
    }

    fn from_matrix(m: Matrix2<T>) -> Rotation2<T> {
        let (re, im) = (m.m11 + m.m22, m.m21 - m.m12);
        let n = R::norm2(re, im);
        if n == R::zero() {
            return Self::identity();
        }
        let (re, im) = (R::div(re, n), R::div(im, n));
        Rotation2 { matrix: Matrix2 { m11: re, m21: im, m12: -im, m22: re } }
    }

    #[inline(always)]
    fn matrix(self: Rotation2<T>) -> Matrix2<T> {
        self.matrix
    }

    #[inline(always)]
    fn into_inner(self: Rotation2<T>) -> Matrix2<T> {
        self.matrix
    }

    #[inline(always)]
    fn inverse(self: Rotation2<T>) -> Rotation2<T> {
        Rotation2 {
            matrix: Matrix2 {
                m11: self.matrix.m11,
                m21: self.matrix.m12,
                m12: self.matrix.m21,
                m22: self.matrix.m22,
            },
        }
    }

    #[inline(always)]
    fn transpose(self: Rotation2<T>) -> Rotation2<T> {
        Rotation2 {
            matrix: Matrix2 {
                m11: self.matrix.m11,
                m21: self.matrix.m12,
                m12: self.matrix.m21,
                m22: self.matrix.m22,
            },
        }
    }

    fn rotation_between(a: Vector2<T>, b: Vector2<T>) -> Rotation2<T> {
        let dot = R::sum_prod2(a.x, b.x, a.y, b.y);
        let perp = R::diff_prod(a.x, b.y, a.y, b.x);
        let n = R::norm2(dot, perp);
        if n == R::zero() {
            return Rotation2 {
                matrix: Matrix2 { m11: R::one(), m21: R::zero(), m12: R::zero(), m22: R::one() },
            };
        }
        let (re, im) = (R::div(dot, n), R::div(perp, n));
        Rotation2 { matrix: Matrix2 { m11: re, m21: im, m12: -im, m22: re } }
    }

    #[inline(always)]
    fn transform_vector(self: Rotation2<T>, v: Vector2<T>) -> Vector2<T> {
        Vector2 {
            x: R::sum_prod2(self.matrix.m11, v.x, self.matrix.m12, v.y),
            y: R::sum_prod2(self.matrix.m21, v.x, self.matrix.m22, v.y),
        }
    }

    #[inline(always)]
    fn transform_point(self: Rotation2<T>, p: Point2<T>) -> Point2<T> {
        Point2 {
            x: R::sum_prod2(self.matrix.m11, p.x, self.matrix.m12, p.y),
            y: R::sum_prod2(self.matrix.m21, p.x, self.matrix.m22, p.y),
        }
    }

    #[inline(always)]
    fn inverse_transform_vector(self: Rotation2<T>, v: Vector2<T>) -> Vector2<T> {
        Vector2 {
            x: R::sum_prod2(self.matrix.m11, v.x, self.matrix.m21, v.y),
            y: R::sum_prod2(self.matrix.m12, v.x, self.matrix.m22, v.y),
        }
    }

    #[inline(always)]
    fn inverse_transform_point(self: Rotation2<T>, p: Point2<T>) -> Point2<T> {
        Point2 {
            x: R::sum_prod2(self.matrix.m11, p.x, self.matrix.m21, p.y),
            y: R::sum_prod2(self.matrix.m12, p.x, self.matrix.m22, p.y),
        }
    }

    #[inline(always)]
    fn to_homogeneous(self: Rotation2<T>) -> Matrix3<T> {
        Matrix3 {
            m11: self.matrix.m11,
            m21: self.matrix.m21,
            m31: R::zero(),
            m12: self.matrix.m12,
            m22: self.matrix.m22,
            m32: R::zero(),
            m13: R::zero(),
            m23: R::zero(),
            m33: R::one(),
        }
    }

    #[inline(always)]
    fn renormalize(ref self: Rotation2<T>) {
        let n = R::norm2(self.matrix.m11, self.matrix.m21);
        let (re, im) = (R::div(self.matrix.m11, n), R::div(self.matrix.m21, n));
        self = Rotation2 { matrix: Matrix2 { m11: re, m21: im, m12: -im, m22: re } };
    }

    #[inline(always)]
    fn abs_diff_eq(self: Rotation2<T>, other: Rotation2<T>, ulps: u64) -> bool {
        R::abs_diff_eq(self.matrix.m11, other.matrix.m11, ulps)
            && R::abs_diff_eq(self.matrix.m21, other.matrix.m21, ulps)
            && R::abs_diff_eq(self.matrix.m12, other.matrix.m12, ulps)
            && R::abs_diff_eq(self.matrix.m22, other.matrix.m22, ulps)
    }

    #[inline(always)]
    fn unwrap(self: Rotation2<T>) -> Matrix2<T> {
        self.matrix
    }

    #[inline(always)]
    fn from_basis_unchecked(basis: [Vector2<T>; 2]) -> Rotation2<T> {
        let [a, b] = basis;
        Rotation2 { matrix: Matrix2 { m11: a.x, m21: a.y, m12: b.x, m22: b.y } }
    }

    #[inline(always)]
    fn rotation_to(self: Rotation2<T>, other: Rotation2<T>) -> Rotation2<T> {
        let (a, b) = (other.matrix, self.matrix);
        Rotation2 {
            matrix: Matrix2 {
                m11: R::sum_prod2(a.m11, b.m11, a.m12, b.m12),
                m21: R::sum_prod2(a.m21, b.m11, a.m22, b.m12),
                m12: R::sum_prod2(a.m11, b.m21, a.m12, b.m22),
                m22: R::sum_prod2(a.m21, b.m21, a.m22, b.m22),
            },
        }
    }

    #[inline(always)]
    fn transform_unit_vector(self: Rotation2<T>, v: Unit<Vector2<T>>) -> Unit<Vector2<T>> {
        Unit { value: Self::transform_vector(self, v.value) }
    }

    #[inline(always)]
    fn inverse_transform_unit_vector(self: Rotation2<T>, v: Unit<Vector2<T>>) -> Unit<Vector2<T>> {
        Unit { value: Self::inverse_transform_vector(self, v.value) }
    }

    #[inline(always)]
    fn mul_unit_complex(self: Rotation2<T>, c: UnitComplex<T>) -> UnitComplex<T> {
        UnitComplex { re: self.matrix.m11, im: self.matrix.m21 } * c
    }

    #[inline(always)]
    fn div_unit_complex(self: Rotation2<T>, c: UnitComplex<T>) -> UnitComplex<T> {
        UnitComplex { re: self.matrix.m11, im: self.matrix.m21 } / c
    }

    fn relative_eq(self: Rotation2<T>, other: Rotation2<T>, epsilon: u64, max_relative: T) -> bool {
        let (a, b) = (self.matrix, other.matrix);
        ApproxEqTrait::relative_eq(a.m11, b.m11, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(a.m21, b.m21, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(a.m12, b.m12, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(a.m22, b.m22, epsilon, max_relative)
    }

    fn ulps_eq(self: Rotation2<T>, other: Rotation2<T>, epsilon: u64, max_ulps: u32) -> bool {
        let (a, b) = (self.matrix, other.matrix);
        ApproxEqTrait::ulps_eq(a.m11, b.m11, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(a.m21, b.m21, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(a.m12, b.m12, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(a.m22, b.m22, epsilon, max_ulps)
    }

    fn cast<U, +Into<T, U>, +Drop<U>>(self: Rotation2<T>) -> Rotation2<U> {
        let m = self.matrix;
        Rotation2 {
            matrix: Matrix2 {
                m11: m.m11.into(), m21: m.m21.into(), m12: m.m12.into(), m22: m.m22.into(),
            },
        }
    }
}

/// Crate-internal by-value forms of the in-place `renormalize` (WP 8.0: the
/// public methods are upstream's `&mut self` ones), for the tests and the value-style call sites.
#[generate_trait]
pub(crate) impl Rotation2InternalImpl<
    T, impl R: Real<T>, +Add<T>, +Sub<T>, +Mul<T>, +Neg<T>, +PartialEq<T>, +Copy<T>, +Drop<T>,
> of Rotation2InternalTrait<T> {
    /// `self` renormalized exactly (`Rotation2Trait::renormalize`), by value.
    #[inline(always)]
    fn renormalized(self: Rotation2<T>) -> Rotation2<T> {
        let mut r = self;
        Rotation2Trait::renormalize(ref r);
        r
    }
}

pub impl Rotation2AngleImpl<
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
> of Rotation2AngleTrait<T> {
    #[inline(always)]
    fn new(angle: T) -> Rotation2<T> {
        let (sin, cos) = Tr::sin_cos(angle);
        Rotation2 { matrix: Matrix2 { m11: cos, m21: sin, m12: -sin, m22: cos } }
    }

    #[inline(always)]
    fn angle(self: Rotation2<T>) -> T {
        Tr::atan2(self.matrix.m21, self.matrix.m11)
    }

    #[inline(always)]
    fn angle_to(self: Rotation2<T>, other: Rotation2<T>) -> T {
        // The angle of `other * self⁻¹` on the first columns, the conjugation folded in.
        Tr::atan2(
            R::diff_prod(other.matrix.m21, self.matrix.m11, other.matrix.m11, self.matrix.m21),
            R::sum_prod2(other.matrix.m11, self.matrix.m11, other.matrix.m21, self.matrix.m21),
        )
    }

    fn scaled_rotation_between(a: Vector2<T>, b: Vector2<T>, s: T) -> Rotation2<T> {
        let dot = R::sum_prod2(a.x, b.x, a.y, b.y);
        let perp = R::diff_prod(a.x, b.y, a.y, b.x);
        if dot == R::zero() && perp == R::zero() {
            return Rotation2 {
                matrix: Matrix2 { m11: R::one(), m21: R::zero(), m12: R::zero(), m22: R::one() },
            };
        }
        let (sin, cos) = Tr::sin_cos(Tr::atan2(perp, dot) * s);
        Rotation2 { matrix: Matrix2 { m11: cos, m21: sin, m12: -sin, m22: cos } }
    }

    #[inline(always)]
    fn powf(self: Rotation2<T>, n: T) -> Rotation2<T> {
        let (sin, cos) = Tr::sin_cos(Tr::atan2(self.matrix.m21, self.matrix.m11) * n);
        Rotation2 { matrix: Matrix2 { m11: cos, m21: sin, m12: -sin, m22: cos } }
    }

    #[inline(always)]
    fn from_scaled_axis(axisangle: Matrix1<T>) -> Rotation2<T> {
        Self::new(axisangle.x)
    }

    #[inline(always)]
    fn scaled_axis(self: Rotation2<T>) -> Matrix1<T> {
        Matrix1 { x: Self::angle(self) }
    }

    fn slerp(self: Rotation2<T>, other: Rotation2<T>, t: T) -> Rotation2<T> {
        let c = UnitComplexAngleTrait::slerp(
            UnitComplex { re: self.matrix.m11, im: self.matrix.m21 },
            UnitComplex { re: other.matrix.m11, im: other.matrix.m21 },
            t,
        );
        Rotation2 { matrix: Matrix2 { m11: c.re, m21: c.im, m12: -c.im, m22: c.re } }
    }

    fn from_matrix_eps(
        m: Matrix2<T>, eps: T, max_iter: usize, guess: Rotation2<T>,
    ) -> Rotation2<T> {
        let c = UnitComplexAngleTrait::from_matrix_eps(
            m, eps, max_iter, UnitComplex { re: guess.matrix.m11, im: guess.matrix.m21 },
        );
        Rotation2 { matrix: Matrix2 { m11: c.re, m21: c.im, m12: -c.im, m22: c.re } }
    }
}

/// `a * b`: the composition of two rotations (turn by `b`, then by `a`), the product of their
/// matrices. FOUR fused kernels, each output component floored once, bit for bit what upstream's
/// matrix product gives.
///
/// The two extra products buy bit-exactness: composing through the complex form (two kernels,
/// 4 260 gas against 9 950, `bench_rotation2_mul__alt_complex`) gives `m12` as `-floor(sin)`
/// where the matrix product gives `floor(-sin)`, which differ by 1 ulp whenever the exact value
/// is not an integer — and the oracle (`rotation2_mul`, tolerance 0) expects upstream's floor.
/// Composing in `UnitComplex` form costs 4 000 gas and is exact there, which is what hot code
/// should do. Upstream: `Mul`.
pub impl Rotation2Mul<T, impl R: Real<T>, +Copy<T>, +Drop<T>> of Mul<Rotation2<T>> {
    fn mul(lhs: Rotation2<T>, rhs: Rotation2<T>) -> Rotation2<T> {
        let (a, b) = (lhs.matrix, rhs.matrix);
        Rotation2 {
            matrix: Matrix2 {
                m11: R::sum_prod2(a.m11, b.m11, a.m12, b.m21),
                m21: R::sum_prod2(a.m21, b.m11, a.m22, b.m21),
                m12: R::sum_prod2(a.m11, b.m12, a.m12, b.m22),
                m22: R::sum_prod2(a.m21, b.m12, a.m22, b.m22),
            },
        }
    }
}

/// `c.into()`: the rotation matrix of a unit complex number. Exact; panics on overflow (`-MIN`).
/// Upstream: `From<UnitComplex> for Rotation2` (`UnitComplex::to_rotation_matrix`).
pub impl Rotation2FromUnitComplex<
    T, +Neg<T>, +Copy<T>, +Drop<T>,
> of Into<UnitComplex<T>, Rotation2<T>> {
    #[inline(always)]
    fn into(self: UnitComplex<T>) -> Rotation2<T> {
        Rotation2 { matrix: Matrix2 { m11: self.re, m21: self.im, m12: -self.im, m22: self.re } }
    }
}

/// `r.into()`: the unit complex number of a rotation matrix, its first column. Exact. Upstream:
/// `From<Rotation2> for UnitComplex` (`UnitComplex::from_rotation_matrix`).
pub impl Rotation2IntoUnitComplex<T, +Copy<T>, +Drop<T>> of Into<Rotation2<T>, UnitComplex<T>> {
    #[inline(always)]
    fn into(self: Rotation2<T>) -> UnitComplex<T> {
        UnitComplex { re: self.matrix.m11, im: self.matrix.m21 }
    }
}

/// `a / b = a * b⁻¹`: the product with the transpose (four fused kernels, one rounding per
/// entry). Upstream: `Div<Rotation2>`.
pub impl Rotation2Div<T, impl R: Real<T>, +Copy<T>, +Drop<T>> of Div<Rotation2<T>> {
    fn div(lhs: Rotation2<T>, rhs: Rotation2<T>) -> Rotation2<T> {
        let (a, b) = (lhs.matrix, rhs.matrix);
        Rotation2 {
            matrix: Matrix2 {
                m11: R::sum_prod2(a.m11, b.m11, a.m12, b.m12),
                m21: R::sum_prod2(a.m21, b.m11, a.m22, b.m12),
                m12: R::sum_prod2(a.m11, b.m21, a.m12, b.m22),
                m22: R::sum_prod2(a.m21, b.m21, a.m22, b.m22),
            },
        }
    }
}

/// `Default::default()`: the identity rotation. Upstream: `Default for Rotation2`.
pub impl Rotation2Default<T, impl R: Real<T>, +Drop<T>> of Default<Rotation2<T>> {
    #[inline(always)]
    fn default() -> Rotation2<T> {
        Rotation2 {
            matrix: Matrix2 { m11: R::one(), m21: R::zero(), m12: R::zero(), m22: R::one() },
        }
    }
}

/// `One::one()`: the identity rotation; `is_one` compares with the identity matrix exactly.
/// Upstream: `num::One for Rotation2`.
pub impl Rotation2One<T, impl R: Real<T>, +PartialEq<T>, +Copy<T>, +Drop<T>> of One<Rotation2<T>> {
    #[inline(always)]
    fn one() -> Rotation2<T> {
        Rotation2Default::<T>::default()
    }

    #[inline(always)]
    fn is_one(self: @Rotation2<T>) -> bool {
        *self == Rotation2Default::<T>::default()
    }

    #[inline(always)]
    fn is_non_one(self: @Rotation2<T>) -> bool {
        !Self::is_one(self)
    }
}

/// `r[(i, j)]`: the entry of row `i` and column `j` of the matrix. Panics with
/// `nalgebra: index out of bounds` for `i > 1` or `j > 1`. Upstream: `Index<(usize, usize)> for
/// Rotation`.
pub impl Rotation2Index<T, +Copy<T>, +Drop<T>> of Index<Rotation2<T>, (usize, usize)> {
    type Target = T;

    fn index(ref self: Rotation2<T>, index: (usize, usize)) -> T {
        let m = self.matrix;
        match index {
            (0, 0) => m.m11,
            (0, 1) => m.m12,
            (1, 0) => m.m21,
            (1, 1) => m.m22,
            _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
        }
    }
}

/// `r.into()`: the isometry of rotation `r` (its first column as a unit complex) and zero
/// translation. Upstream: `SubsetOf<Isometry2> for Rotation2` (`nalgebra::convert(r)`).
pub impl Isometry2FromRotation2<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>,
> of Into<Rotation2<T>, Isometry2<T>> {
    #[inline(always)]
    fn into(self: Rotation2<T>) -> Isometry2<T> {
        Isometry2 {
            rotation: UnitComplex { re: self.matrix.m11, im: self.matrix.m21 },
            translation: Translation2 { vector: Vector2 { x: R::zero(), y: R::zero() } },
        }
    }
}

/// `r.into()`: the similarity of rotation `r`, zero translation and scaling 1. Upstream:
/// `SubsetOf<Similarity2> for Rotation2` (`nalgebra::convert(r)`).
pub impl Similarity2FromRotation2<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>,
> of Into<Rotation2<T>, Similarity2<T>> {
    #[inline(always)]
    fn into(self: Rotation2<T>) -> Similarity2<T> {
        Similarity2 { isometry: self.into(), scaling: R::one() }
    }
}
