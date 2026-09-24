//! `Translation4`: a 4-dimensional translation (upstream `nalgebra::Translation4`, i.e.
//! `Translation<T, 4>`), WP 8.4-P09a.
//!
//! The same API as `Translation2` / `Translation3` with the WP 8.4-P09a completion, written from
//! one template for the sizes 1, 4, 5 and 6 (the vector is the matching shape `Vector4`). Every
//! operation is an exact addition, subtraction or negation: nothing rounds; overflow panics.

use core::num::traits::One;
use simba::scalar::Real;
use crate::base::matrix5::Matrix5;
use crate::base::vector4::Vector4;
use super::point4::Point4;
use super::quaternion::ApproxEqTrait;

/// A translation by `vector`. The field name is upstream's (`Translation { vector }`).
#[derive(Copy, Drop, PartialEq, Serde, Default, Debug, Hash)]
pub struct Translation4<T> {
    pub vector: Vector4<T>,
}

/// Operations of `Translation4<T>` over a `Real` scalar. By value, unrolled, no loop.
#[generate_trait]
pub impl Translation4Impl<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Sub<T>, +Mul<T>, +Neg<T>, +PartialEq<T>,
> of Translation4Trait<T> {
    /// The translation by `(x, y, z, w)`. Exact. Upstream: `Translation4::new`.
    #[inline(always)]
    fn new(x: T, y: T, z: T, w: T) -> Translation4<T> {
        Translation4 { vector: Vector4 { x, y, z, w } }
    }

    /// The identity translation (the zero vector). Exact. Upstream: `identity`.
    #[inline(always)]
    fn identity() -> Translation4<T> {
        Translation4 { vector: Vector4 { x: R::zero(), y: R::zero(), z: R::zero(), w: R::zero() } }
    }

    /// The translation by `v`. Exact. Upstream: `from_vector` (deprecated upstream for `From`),
    /// also `v.into()`.
    #[inline(always)]
    fn from_vector(v: Vector4<T>) -> Translation4<T> {
        Translation4 { vector: v }
    }

    /// The inverse translation, by `-vector`. Exact; panics on overflow (`-MIN`). Upstream:
    /// `inverse`.
    #[inline(always)]
    fn inverse(self: Translation4<T>) -> Translation4<T> {
        let v = self.vector;
        Translation4 { vector: Vector4 { x: -v.x, y: -v.y, z: -v.z, w: -v.w } }
    }

    /// `self * p`: the point translated by `vector`. Exact; panics on overflow. Upstream:
    /// `transform_point` (`t * p`).
    #[inline(always)]
    fn transform_point(self: Translation4<T>, p: Point4<T>) -> Point4<T> {
        let v = self.vector;
        Point4 { x: p.x + v.x, y: p.y + v.y, z: p.z + v.z, w: p.w + v.w }
    }

    /// `self⁻¹ * p`: the point translated by `-vector`, as one subtraction. Exact; panics on
    /// overflow. Upstream: `inverse_transform_point`.
    #[inline(always)]
    fn inverse_transform_point(self: Translation4<T>, p: Point4<T>) -> Point4<T> {
        let v = self.vector;
        Point4 { x: p.x - v.x, y: p.y - v.y, z: p.z - v.z, w: p.w - v.w }
    }

    /// The translation as a 5x5 homogeneous matrix: the identity with `vector` in the last
    /// column. Exact. Upstream: `to_homogeneous`.
    #[inline(always)]
    fn to_homogeneous(self: Translation4<T>) -> Matrix5<T> {
        Matrix5 {
            m11: R::one(),
            m21: R::zero(),
            m31: R::zero(),
            m41: R::zero(),
            m51: R::zero(),
            m12: R::zero(),
            m22: R::one(),
            m32: R::zero(),
            m42: R::zero(),
            m52: R::zero(),
            m13: R::zero(),
            m23: R::zero(),
            m33: R::one(),
            m43: R::zero(),
            m53: R::zero(),
            m14: R::zero(),
            m24: R::zero(),
            m34: R::zero(),
            m44: R::one(),
            m54: R::zero(),
            m15: self.vector.x,
            m25: self.vector.y,
            m35: self.vector.z,
            m45: self.vector.w,
            m55: R::one(),
        }
    }

    /// `true` when every component is within `ulps` smallest units (raw units for fixed point) of
    /// `other`'s; cannot overflow. Upstream: `approx::AbsDiffEq::abs_diff_eq` (DESIGN D3).
    #[inline(always)]
    fn abs_diff_eq(self: Translation4<T>, other: Translation4<T>, ulps: u64) -> bool {
        let (a, b) = (self.vector, other.vector);
        R::abs_diff_eq(a.x, b.x, ulps)
            && R::abs_diff_eq(a.y, b.y, ulps)
            && R::abs_diff_eq(a.z, b.z, ulps)
            && R::abs_diff_eq(a.w, b.w, ulps)
    }

    /// `true` when every component is `relative_eq` to the matching component of `other` (see
    /// `QuaternionTrait::relative_eq`). Upstream: `approx::RelativeEq::relative_eq` (DESIGN D3).
    fn relative_eq(
        self: Translation4<T>, other: Translation4<T>, epsilon: u64, max_relative: T,
    ) -> bool {
        let (a, b) = (self.vector, other.vector);
        ApproxEqTrait::relative_eq(a.x, b.x, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(a.y, b.y, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(a.z, b.z, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(a.w, b.w, epsilon, max_relative)
    }

    /// `true` when every component is `ulps_eq` to the matching component of `other` (see
    /// `QuaternionTrait::ulps_eq`). Upstream: `approx::UlpsEq::ulps_eq` (DESIGN D3).
    fn ulps_eq(self: Translation4<T>, other: Translation4<T>, epsilon: u64, max_ulps: u32) -> bool {
        let (a, b) = (self.vector, other.vector);
        ApproxEqTrait::ulps_eq(a.x, b.x, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(a.y, b.y, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(a.z, b.z, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(a.w, b.w, epsilon, max_ulps)
    }

    /// The same translation with every component converted by `Into<T, U>` (the identity for
    /// the single scalar `Fixed`). Upstream: `cast` (and `SubsetOf<Translation<U>>`).
    fn cast<U, +Into<T, U>, +Drop<U>>(self: Translation4<T>) -> Translation4<U> {
        let v = self.vector;
        Translation4 {
            vector: Vector4 { x: v.x.into(), y: v.y.into(), z: v.z.into(), w: v.w.into() },
        }
    }
}

/// `a * b`: the composition of two translations, the SUM of their vectors. Exact; panics on
/// overflow. Upstream: `Mul`.
pub impl Translation4Mul<T, +Add<T>, +Copy<T>, +Drop<T>> of Mul<Translation4<T>> {
    #[inline(always)]
    fn mul(lhs: Translation4<T>, rhs: Translation4<T>) -> Translation4<T> {
        let (a, b) = (lhs.vector, rhs.vector);
        Translation4 { vector: Vector4 { x: a.x + b.x, y: a.y + b.y, z: a.z + b.z, w: a.w + b.w } }
    }
}

/// `v.into()`: the translation by `v`. Upstream: `From<Vector4> for Translation4`.
pub impl Translation4FromVector<T> of Into<Vector4<T>, Translation4<T>> {
    #[inline(always)]
    fn into(self: Vector4<T>) -> Translation4<T> {
        Translation4 { vector: self }
    }
}

/// `a / b = a * b⁻¹`: the translation by `a.vector - b.vector`. Exact; panics on overflow.
/// Upstream: `Div<Translation>`.
pub impl Translation4Div<T, +Sub<T>, +Copy<T>, +Drop<T>> of Div<Translation4<T>> {
    #[inline(always)]
    fn div(lhs: Translation4<T>, rhs: Translation4<T>) -> Translation4<T> {
        let (a, b) = (lhs.vector, rhs.vector);
        Translation4 { vector: Vector4 { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z, w: a.w - b.w } }
    }
}

/// `One::one()`: the identity translation; `is_one` tests for the zero vector exactly.
/// Upstream: `num::One for Translation`.
pub impl Translation4One<
    T, impl R: Real<T>, +PartialEq<T>, +Copy<T>, +Drop<T>,
> of One<Translation4<T>> {
    #[inline(always)]
    fn one() -> Translation4<T> {
        Translation4 { vector: Vector4 { x: R::zero(), y: R::zero(), z: R::zero(), w: R::zero() } }
    }

    #[inline(always)]
    fn is_one(self: @Translation4<T>) -> bool {
        let v = *self.vector;
        v.x == R::zero() && v.y == R::zero() && v.z == R::zero() && v.w == R::zero()
    }

    #[inline(always)]
    fn is_non_one(self: @Translation4<T>) -> bool {
        !Self::is_one(self)
    }
}

/// `p.into()`: the translation by the position vector of `p`. Upstream: `From<Point4> for
/// Translation4`.
pub impl Translation4FromPoint<T> of Into<Point4<T>, Translation4<T>> {
    #[inline(always)]
    fn into(self: Point4<T>) -> Translation4<T> {
        let Point4 { x, y, z, w } = self;
        Translation4 { vector: Vector4 { x, y, z, w } }
    }
}

/// `[x, y, z, w].into()`: the translation by that vector. Upstream: `From<[T; 4]>`.
pub impl Translation4FromArray<T> of Into<[T; 4], Translation4<T>> {
    #[inline(always)]
    fn into(self: [T; 4]) -> Translation4<T> {
        let [x, y, z, w] = self;
        Translation4 { vector: Vector4 { x, y, z, w } }
    }
}

/// The components of the vector as an array. Upstream: `Into<[T; 4]>`.
pub impl Translation4IntoArray<T> of Into<Translation4<T>, [T; 4]> {
    #[inline(always)]
    fn into(self: Translation4<T>) -> [T; 4] {
        let Vector4 { x, y, z, w } = self.vector;
        [x, y, z, w]
    }
}
