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

use simba::scalar::{Real, Transcendental};
use crate::base::matrix2::Matrix2;
use crate::base::matrix3::Matrix3;
use crate::base::point2::Point2;
use crate::base::vector2::Vector2;
use super::unit_complex::UnitComplex;

#[cfg(test)]
mod benches;
#[cfg(test)]
mod oracle;
#[cfg(test)]
mod tests;

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
    /// The rotation closest to `m` in the sense of its FIRST COLUMN: `(m11, m21)` normalized,
    /// then expanded back into a rotation matrix. Panics with `simba: division by zero` when that
    /// column is zero.
    ///
    /// Deviates from upstream, which runs a Gauss-Newton optimization
    /// (`Rotation2::from_matrix` / `from_matrix_eps`, an iteration count on the input): a loop is
    /// forbidden here (AGENTS.md rule 1) and would be pointless in 2D, where the group is
    /// one-dimensional. On an exact rotation both agree; on a general matrix upstream splits the
    /// difference between the two columns while this keeps the first one.
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
    /// Alias of `transform_vector` (`self * v`), named like the heterogeneous products of the
    /// matrix types. Upstream: `Mul<Vector2>`.
    fn mul_vec(self: Rotation2<T>, v: Vector2<T>) -> Vector2<T>;
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
    /// The same rotation as a unit complex number: the first column of the matrix. Exact.
    /// Upstream: `UnitComplex::from_rotation_matrix` (there is no `Rotation2::to_unit_complex`;
    /// upstream goes through `From`, which is also available here as `r.into()`).
    fn to_unit_complex(self: Rotation2<T>) -> UnitComplex<T>;
    /// Renormalizes exactly: normalizes the first column (one `norm2` and two exactly floored
    /// divisions), then rebuilds the matrix from it, so `Rᵀ R = I` within a few ulp again.
    /// Panics with `simba: division by zero` when that column is zero. Upstream:
    /// `Rotation2::renormalize` (which does the same through `UnitComplex`, in place).
    fn renormalize(self: Rotation2<T>) -> Rotation2<T>;
    /// `true` when every component is within `ulps` smallest units (raw units for fixed point) of
    /// the matching component of `other`; cannot overflow. Upstream:
    /// `approx::AbsDiffEq::abs_diff_eq`, the tolerance being counted in ulp instead of a float
    /// epsilon (DESIGN D3).
    fn abs_diff_eq(self: Rotation2<T>, other: Rotation2<T>, ulps: u64) -> bool;
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
}

pub impl Rotation2Impl<
    T,
    impl R: Real<T>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Div<T>,
    +Neg<T>,
    +PartialEq<T>,
    +Copy<T>,
    +Drop<T>,
> of Rotation2Trait<T> {
    #[inline(always)]
    fn identity() -> Rotation2<T> {
        Rotation2 { matrix: Matrix2 { m11: R::ONE, m21: R::ZERO, m12: R::ZERO, m22: R::ONE } }
    }

    #[inline(always)]
    fn from_matrix_unchecked(m: Matrix2<T>) -> Rotation2<T> {
        Rotation2 { matrix: m }
    }

    #[inline(always)]
    fn from_matrix(m: Matrix2<T>) -> Rotation2<T> {
        let n = R::norm2(m.m11, m.m21);
        let (re, im) = (m.m11 / n, m.m21 / n);
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
        if n == R::ZERO {
            return Rotation2 {
                matrix: Matrix2 { m11: R::ONE, m21: R::ZERO, m12: R::ZERO, m22: R::ONE },
            };
        }
        let (re, im) = (dot / n, perp / n);
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
    fn mul_vec(self: Rotation2<T>, v: Vector2<T>) -> Vector2<T> {
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
            m31: R::ZERO,
            m12: self.matrix.m12,
            m22: self.matrix.m22,
            m32: R::ZERO,
            m13: R::ZERO,
            m23: R::ZERO,
            m33: R::ONE,
        }
    }

    #[inline(always)]
    fn to_unit_complex(self: Rotation2<T>) -> UnitComplex<T> {
        UnitComplex { re: self.matrix.m11, im: self.matrix.m21 }
    }

    #[inline(always)]
    fn renormalize(self: Rotation2<T>) -> Rotation2<T> {
        let n = R::norm2(self.matrix.m11, self.matrix.m21);
        let (re, im) = (self.matrix.m11 / n, self.matrix.m21 / n);
        Rotation2 { matrix: Matrix2 { m11: re, m21: im, m12: -im, m22: re } }
    }

    #[inline(always)]
    fn abs_diff_eq(self: Rotation2<T>, other: Rotation2<T>, ulps: u64) -> bool {
        R::abs_diff_eq(self.matrix.m11, other.matrix.m11, ulps)
            && R::abs_diff_eq(self.matrix.m21, other.matrix.m21, ulps)
            && R::abs_diff_eq(self.matrix.m12, other.matrix.m12, ulps)
            && R::abs_diff_eq(self.matrix.m22, other.matrix.m22, ulps)
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
        if dot == R::ZERO && perp == R::ZERO {
            return Rotation2 {
                matrix: Matrix2 { m11: R::ONE, m21: R::ZERO, m12: R::ZERO, m22: R::ONE },
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
}

/// `a * b`: the composition of two rotations (turn by `b`, then by `a`), the product of their
/// matrices. FOUR fused kernels, each output component floored once, bit for bit what upstream's
/// matrix product gives.
///
/// The two extra products buy bit-exactness: composing through the complex form (two kernels,
/// 4 600 gas against 10 260, `bench_rotation2_mul__alt_complex`) gives `m12` as `-floor(sin)`
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
