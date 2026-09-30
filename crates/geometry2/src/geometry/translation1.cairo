//! `Translation1`: a 1-dimensional translation (upstream `nalgebra::Translation1`, i.e.
//! `Translation<T, 1>`), WP 8.4-P09a.
//!
//! The same API as `Translation2` / `Translation3` with the WP 8.4-P09a completion, written from
//! one template for the sizes 1, 4, 5 and 6 (the vector is the matching shape `Matrix1`). Every
//! operation is an exact addition, subtraction or negation: nothing rounds; overflow panics.

use nalgebra_core::base::matrix1::Matrix1;
use nalgebra_core::geometry::point1::Point1;
use nalgebra_core::geometry::translation1::Translation1;
use nalgebra_core::internal::geometry::quaternion::ApproxEqTrait;
use nalgebra_types2::base::matrix2::Matrix2;
use nalgebra_types2::internal::geometry::translation1::Matrix2FromTranslation1KernelTrait;
use simba::scalar::Real;

/// Operations of `Translation1<T>` over a `Real` scalar. By value, unrolled, no loop.
#[generate_trait]
pub impl Translation1Impl<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Sub<T>, +Mul<T>, +Neg<T>, +PartialEq<T>,
> of Translation1Trait<T> {
    /// The translation by `(x)`. Exact. Upstream: `Translation1::new`.
    #[inline(always)]
    fn new(x: T) -> Translation1<T> {
        Translation1 { vector: Matrix1 { x } }
    }

    /// The identity translation (the zero vector). Exact. Upstream: `identity`.
    #[inline(always)]
    fn identity() -> Translation1<T> {
        Translation1 { vector: Matrix1 { x: R::zero() } }
    }

    /// The translation by `v`. Exact. Upstream: `from_vector` (deprecated upstream for `From`),
    /// also `v.into()`.
    #[inline(always)]
    fn from_vector(v: Matrix1<T>) -> Translation1<T> {
        Translation1 { vector: v }
    }

    /// The inverse translation, by `-vector`. Exact; panics on overflow (`-MIN`). Upstream:
    /// `inverse`.
    #[inline(always)]
    fn inverse(self: Translation1<T>) -> Translation1<T> {
        let v = self.vector;
        Translation1 { vector: Matrix1 { x: -v.x } }
    }

    /// `self = self.inverse()` in place (`-vector`; the by-value form is the cheapest, so the bits
    /// are those of `inverse`). Exact; panics on overflow (`-MIN`). Upstream: `inverse_mut`.
    #[inline(always)]
    fn inverse_mut(ref self: Translation1<T>) {
        self = Self::inverse(self);
    }

    /// `self * p`: the point translated by `vector`. Exact; panics on overflow. Upstream:
    /// `transform_point` (`t * p`).
    #[inline(always)]
    fn transform_point(self: Translation1<T>, p: Point1<T>) -> Point1<T> {
        let v = self.vector;
        Point1 { x: p.x + v.x }
    }

    /// `self⁻¹ * p`: the point translated by `-vector`, as one subtraction. Exact; panics on
    /// overflow. Upstream: `inverse_transform_point`.
    #[inline(always)]
    fn inverse_transform_point(self: Translation1<T>, p: Point1<T>) -> Point1<T> {
        let v = self.vector;
        Point1 { x: p.x - v.x }
    }

    /// The translation as a 2x2 homogeneous matrix: the identity with `vector` in the last
    /// column. Exact. Upstream: `to_homogeneous`.
    #[inline(always)]
    fn to_homogeneous(self: Translation1<T>) -> Matrix2<T> {
        Matrix2FromTranslation1KernelTrait::to_homogeneous(self)
    }

    /// `true` when every component is within `ulps` smallest units (raw units for fixed point) of
    /// `other`'s; cannot overflow. Upstream: `approx::AbsDiffEq::abs_diff_eq` (DESIGN D3).
    #[inline(always)]
    fn abs_diff_eq(self: Translation1<T>, other: Translation1<T>, ulps: u64) -> bool {
        let (a, b) = (self.vector, other.vector);
        R::abs_diff_eq(a.x, b.x, ulps)
    }

    /// `true` when every component is `relative_eq` to the matching component of `other` (see
    /// `QuaternionTrait::relative_eq`). Upstream: `approx::RelativeEq::relative_eq` (DESIGN D3).
    fn relative_eq(
        self: Translation1<T>, other: Translation1<T>, epsilon: u64, max_relative: T,
    ) -> bool {
        let (a, b) = (self.vector, other.vector);
        ApproxEqTrait::relative_eq(a.x, b.x, epsilon, max_relative)
    }

    /// `true` when every component is `ulps_eq` to the matching component of `other` (see
    /// `QuaternionTrait::ulps_eq`). Upstream: `approx::UlpsEq::ulps_eq` (DESIGN D3).
    fn ulps_eq(self: Translation1<T>, other: Translation1<T>, epsilon: u64, max_ulps: u32) -> bool {
        let (a, b) = (self.vector, other.vector);
        ApproxEqTrait::ulps_eq(a.x, b.x, epsilon, max_ulps)
    }

    /// The same translation with every component converted by `Into<T, U>` (the identity for
    /// the single scalar `Fixed`). Upstream: `cast` (and `SubsetOf<Translation<U>>`).
    fn cast<U, +Into<T, U>, +Drop<U>>(self: Translation1<T>) -> Translation1<U> {
        let v = self.vector;
        Translation1 { vector: Matrix1 { x: v.x.into() } }
    }
}
