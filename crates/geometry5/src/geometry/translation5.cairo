//! `Translation5`: a 5-dimensional translation (upstream `nalgebra::Translation5`, i.e.
//! `Translation<T, 5>`), WP 8.4-P09a.
//!
//! The same API as `Translation2` / `Translation3` with the WP 8.4-P09a completion, written from
//! one template for the sizes 1, 4, 5 and 6 (the vector is the matching shape `Vector5`). Every
//! operation is an exact addition, subtraction or negation: nothing rounds; overflow panics.

use nalgebra_core::internal::geometry::quaternion::ApproxEqTrait;
use nalgebra_types5::base::vector5::Vector5;
use nalgebra_types5::geometry::point5::Point5;
use nalgebra_types5::geometry::translation5::Translation5;
use nalgebra_types6::base::matrix6::Matrix6;
use nalgebra_types6::internal::geometry::translation5::Matrix6FromTranslation5KernelTrait;
use simba::scalar::Real;

/// Operations of `Translation5<T>` over a `Real` scalar. By value, unrolled, no loop.
#[generate_trait]
pub impl Translation5Impl<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Sub<T>, +Mul<T>, +Neg<T>, +PartialEq<T>,
> of Translation5Trait<T> {
    /// The translation by `(x, y, z, w, a)`. Exact. Upstream: `Translation5::new`.
    #[inline(always)]
    fn new(x: T, y: T, z: T, w: T, a: T) -> Translation5<T> {
        Translation5 { vector: Vector5 { x, y, z, w, a } }
    }

    /// The identity translation (the zero vector). Exact. Upstream: `identity`.
    #[inline(always)]
    fn identity() -> Translation5<T> {
        Translation5 {
            vector: Vector5 {
                x: R::zero(), y: R::zero(), z: R::zero(), w: R::zero(), a: R::zero(),
            },
        }
    }

    /// The translation by `v`. Exact. Upstream: `from_vector` (deprecated upstream for `From`),
    /// also `v.into()`.
    #[inline(always)]
    fn from_vector(v: Vector5<T>) -> Translation5<T> {
        Translation5 { vector: v }
    }

    /// The inverse translation, by `-vector`. Exact; panics on overflow (`-MIN`). Upstream:
    /// `inverse`.
    #[inline(always)]
    fn inverse(self: Translation5<T>) -> Translation5<T> {
        let v = self.vector;
        Translation5 { vector: Vector5 { x: -v.x, y: -v.y, z: -v.z, w: -v.w, a: -v.a } }
    }

    /// `self = self.inverse()` in place (`-vector`; the by-value form is the cheapest, so the bits
    /// are those of `inverse`). Exact; panics on overflow (`-MIN`). Upstream: `inverse_mut`.
    #[inline(always)]
    fn inverse_mut(ref self: Translation5<T>) {
        self = Self::inverse(self);
    }

    /// `self * p`: the point translated by `vector`. Exact; panics on overflow. Upstream:
    /// `transform_point` (`t * p`).
    #[inline(always)]
    fn transform_point(self: Translation5<T>, p: Point5<T>) -> Point5<T> {
        let v = self.vector;
        Point5 { x: p.x + v.x, y: p.y + v.y, z: p.z + v.z, w: p.w + v.w, a: p.a + v.a }
    }

    /// `self⁻¹ * p`: the point translated by `-vector`, as one subtraction. Exact; panics on
    /// overflow. Upstream: `inverse_transform_point`.
    #[inline(always)]
    fn inverse_transform_point(self: Translation5<T>, p: Point5<T>) -> Point5<T> {
        let v = self.vector;
        Point5 { x: p.x - v.x, y: p.y - v.y, z: p.z - v.z, w: p.w - v.w, a: p.a - v.a }
    }

    /// The translation as a 6x6 homogeneous matrix: the identity with `vector` in the last
    /// column. Exact. Upstream: `to_homogeneous`.
    #[inline(always)]
    fn to_homogeneous(self: Translation5<T>) -> Matrix6<T> {
        Matrix6FromTranslation5KernelTrait::to_homogeneous(self)
    }

    /// `true` when every component is within `ulps` smallest units (raw units for fixed point) of
    /// `other`'s; cannot overflow. Upstream: `approx::AbsDiffEq::abs_diff_eq` (DESIGN D3).
    #[inline(always)]
    fn abs_diff_eq(self: Translation5<T>, other: Translation5<T>, ulps: u64) -> bool {
        let (a, b) = (self.vector, other.vector);
        R::abs_diff_eq(a.x, b.x, ulps)
            && R::abs_diff_eq(a.y, b.y, ulps)
            && R::abs_diff_eq(a.z, b.z, ulps)
            && R::abs_diff_eq(a.w, b.w, ulps)
            && R::abs_diff_eq(a.a, b.a, ulps)
    }

    /// `true` when every component is `relative_eq` to the matching component of `other` (see
    /// `QuaternionTrait::relative_eq`). Upstream: `approx::RelativeEq::relative_eq` (DESIGN D3).
    fn relative_eq(
        self: Translation5<T>, other: Translation5<T>, epsilon: u64, max_relative: T,
    ) -> bool {
        let (a, b) = (self.vector, other.vector);
        ApproxEqTrait::relative_eq(a.x, b.x, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(a.y, b.y, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(a.z, b.z, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(a.w, b.w, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(a.a, b.a, epsilon, max_relative)
    }

    /// `true` when every component is `ulps_eq` to the matching component of `other` (see
    /// `QuaternionTrait::ulps_eq`). Upstream: `approx::UlpsEq::ulps_eq` (DESIGN D3).
    fn ulps_eq(self: Translation5<T>, other: Translation5<T>, epsilon: u64, max_ulps: u32) -> bool {
        let (a, b) = (self.vector, other.vector);
        ApproxEqTrait::ulps_eq(a.x, b.x, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(a.y, b.y, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(a.z, b.z, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(a.w, b.w, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(a.a, b.a, epsilon, max_ulps)
    }

    /// The same translation with every component converted by `Into<T, U>` (the identity for
    /// the single scalar `Fixed`). Upstream: `cast` (and `SubsetOf<Translation<U>>`).
    fn cast<U, +Into<T, U>, +Drop<U>>(self: Translation5<T>) -> Translation5<U> {
        let v = self.vector;
        Translation5 {
            vector: Vector5 {
                x: v.x.into(), y: v.y.into(), z: v.z.into(), w: v.w.into(), a: v.a.into(),
            },
        }
    }
}
