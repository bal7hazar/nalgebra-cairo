//! `Projective3`: an invertible homogeneous matrix. `inverse` exists, and `transform_point` /
//! `transform_vector` divide by the homogeneous coordinate. Upstream: `nalgebra::Projective3`, i.e.
//! `Transform<T, TProjective, 3>`
//! (`geometry/transform*.rs`), WP 8.4-P11a. The design of the six transform types (one Cairo
//! struct per upstream alias, upstream's category rules) is in `transform.cairo`.
//!
//! - `Projective3Trait` / `Projective3Impl`: construction, accessors, inverse, transforms, the
//! products by the
//!   other geometry types (`mul_<rhs>` / `div_<rhs>`), comparisons;
//! - the operator / conversion impls: `*` / `/` with a `Projective3`, `Default`, `One`,
//! `Index<(usize, usize)>`,
//!   `Into` / `TryInto` (upstream's `SubsetOf` / `From`);
//! - the category-changing products are `TransformMul` / `TransformDiv`, `set_category` is
//!   `TransformSetCategory` (`transform.cairo`).
//!
//! Numeric contract (AGENTS.md): every sum of products is one fused `Real` kernel (one floor per
//! output scalar), every quotient correctly rounded; overflow panics.

use core::num::traits::One;
use core::ops::Index;
use simba::scalar::Real;
use crate::base::cg::Matrix4CgTrait;
use crate::base::matrix4::{Matrix4, Matrix4Trait};
use crate::base::point3::Point3;
use crate::base::vector3::Vector3;
use super::affine3::{Affine3, Affine3Trait};
use super::isometry3::{Isometry3, Isometry3Trait};
use super::isometry_matrix3::{IsometryMatrix3, IsometryMatrix3Trait};
use super::rotation3::{Rotation3, Rotation3Trait};
use super::scale3::{Scale3, Scale3Trait};
use super::similarity3::{Similarity3, Similarity3Trait};
use super::similarity_matrix3::{SimilarityMatrix3, SimilarityMatrix3Trait};
use super::transform::{TransformKernels, errors};
use super::transform3::{Transform3, Transform3Trait};
use super::translation3::{Translation3, Translation3Trait};
use super::unit_dual_quaternion::{UnitDualQuaternion, UnitDualQuaternionTrait};
use super::unit_quaternion::{UnitQuaternion, UnitQuaternionTrait};

/// A 3D transformation of category `TProjective`: the homogeneous 4x4 matrix `matrix`
/// (crate-private, like upstream's private field). `PartialEq` compares the matrices exactly,
/// `Serde` is the matrix's, `Hash` hashes the matrix (upstream's `Hash`). Upstream:
/// `Projective3<T>`.
#[derive(Copy, Drop, PartialEq, Serde, Debug, Hash)]
pub struct Projective3<T> {
    pub(crate) matrix: Matrix4<T>,
}

/// Operations of `Projective3<T>` over a `Real` scalar. By value, unrolled, no loop.
#[generate_trait]
pub impl Projective3Impl<
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
> of Projective3Trait<T> {
    /// The transform whose homogeneous matrix is `matrix`, NOT checked against the invariants of
    /// the category. Exact. Upstream: `Transform::from_matrix_unchecked`.
    #[inline(always)]
    fn from_matrix_unchecked(matrix: Matrix4<T>) -> Projective3<T> {
        Projective3 { matrix }
    }

    /// The identity transform. Exact. Upstream: `Transform::identity`.
    #[inline(always)]
    fn identity() -> Projective3<T> {
        Projective3 { matrix: Matrix4Trait::identity() }
    }

    /// The homogeneous matrix. Exact. Upstream: `into_inner`.
    #[inline(always)]
    fn into_inner(self: Projective3<T>) -> Matrix4<T> {
        self.matrix
    }

    /// The homogeneous matrix: `into_inner` (deprecated upstream, "use `.into_inner()`
    /// instead"). Exact. Upstream: `unwrap`.
    #[inline(always)]
    fn unwrap(self: Projective3<T>) -> Matrix4<T> {
        self.matrix
    }

    /// The homogeneous matrix (a copy: Cairo values are passed by value). Exact. Upstream:
    /// `matrix`.
    #[inline(always)]
    fn matrix(self: Projective3<T>) -> Matrix4<T> {
        self.matrix
    }

    /// The homogeneous matrix. Exact. Upstream: `to_homogeneous`.
    #[inline(always)]
    fn to_homogeneous(self: Projective3<T>) -> Matrix4<T> {
        self.matrix
    }

    /// The inverse, or `None` when the matrix is singular: `Matrix4::try_inverse` of the
    /// homogeneous matrix (its rounding and its singularity criterion: a determinant EXACTLY
    /// zero), upstream's formula. Upstream: `try_inverse`.
    #[inline(always)]
    fn try_inverse(self: Projective3<T>) -> Option<Projective3<T>> {
        match Matrix4Trait::try_inverse(self.matrix) {
            Option::Some(matrix) => Option::Some(Projective3 { matrix }),
            Option::None => Option::None,
        }
    }

    /// Inverts in place and returns `true`, or returns `false` and leaves `self` unchanged when
    /// it is not invertible (`try_inverse`, same rounding). Upstream: `try_inverse_mut`.
    #[inline(always)]
    fn try_inverse_mut(ref self: Projective3<T>) -> bool {
        match Self::try_inverse(self) {
            Option::Some(inv) => {
                self = inv;
                true
            },
            Option::None => false,
        }
    }

    /// The inverse (`try_inverse`, same rounding). Panics with `errors::NOT_INVERTIBLE` when the
    /// transform is singular (upstream's `unwrap` of `None`); a well-formed projective transform
    /// is invertible by definition. Upstream: `inverse` (projective and affine categories only).
    #[inline(always)]
    fn inverse(self: Projective3<T>) -> Projective3<T> {
        Self::try_inverse(self).expect(errors::NOT_INVERTIBLE)
    }

    /// Inverts in place (`try_inverse`, same rounding). Like upstream (`let _ =
    /// self.matrix.try_inverse_mut()`), a singular transform is left unchanged WITHOUT panicking.
    /// Upstream: `inverse_mut` (projective and affine categories only).
    #[inline(always)]
    fn inverse_mut(ref self: Projective3<T>) {
        let _ = Self::try_inverse_mut(ref self);
    }

    /// `pt` transformed, with the homogeneous division: `Matrix4::transform_point` (`q =
    /// m[:3, :3] * pt + m[:3, 3]`, `n = m[3, :3] . pt + m[3, 3]`, one fused sum of
    /// products each, then `q / n` correctly rounded if `n != 0`, else `q`: upstream's formula
    /// and branch). Upstream: `transform_point` (`Mul<Point>`).
    #[inline(always)]
    fn transform_point(self: Projective3<T>, pt: Point3<T>) -> Point3<T> {
        Matrix4CgTrait::transform_point(self.matrix, pt)
    }

    /// `v` transformed: `Matrix4::transform_vector` (with `n = m[3, :3] . v`, `m[:3, :3]
    /// * (v / n)` if `n != 0`, else `m[:3, :3] * v`: upstream's formula and branch).
    /// Upstream: `transform_vector` (`Mul<SVector>`).
    #[inline(always)]
    fn transform_vector(self: Projective3<T>, v: Vector3<T>) -> Vector3<T> {
        Matrix4CgTrait::transform_vector(self.matrix, v)
    }

    /// `self.inverse().transform_point(pt)` (upstream's formula: the inverse, then the
    /// transform). Panics like `inverse`. Upstream: `inverse_transform_point`.
    #[inline(always)]
    fn inverse_transform_point(self: Projective3<T>, pt: Point3<T>) -> Point3<T> {
        Self::transform_point(Self::inverse(self), pt)
    }

    /// `self.inverse().transform_vector(v)` (upstream's formula). Panics like `inverse`.
    /// Upstream: `inverse_transform_vector`.
    #[inline(always)]
    fn inverse_transform_vector(self: Projective3<T>, v: Vector3<T>) -> Vector3<T> {
        Self::transform_vector(Self::inverse(self), v)
    }

    /// `self * r`: `m * [[r, 0], [0, 1]]`, the first 3 columns one fused sum of products per
    /// entry, the last one copied (bit for bit the homogeneous product); a `Projective3`
    /// (upstream's `TCategoryMul<TAffine>` keeps the category). Upstream: `Mul<Rotation> for
    /// Transform`.
    #[inline(always)]
    fn mul_rotation(self: Projective3<T>, r: Rotation3<T>) -> Projective3<T> {
        Projective3 { matrix: TransformKernels::mul_rotation3(self.matrix, r.matrix) }
    }

    /// `self / r = self * r⁻¹` (the inverse rotation is the exact transpose), rounded like
    /// `mul_rotation`. Upstream: `Div<Rotation> for Transform`.
    #[inline(always)]
    fn div_rotation(self: Projective3<T>, r: Rotation3<T>) -> Projective3<T> {
        Projective3 { matrix: TransformKernels::mul_rotation3(self.matrix, r.inverse().matrix) }
    }

    /// `self * q`: the rotation matrix of `q` (upstream's `to_homogeneous` block,
    /// `UnitQuaternion::to_rotation_matrix`), then like `mul_rotation`. Upstream:
    /// `Mul<UnitQuaternion> for Transform`.
    #[inline(always)]
    fn mul_unit_quaternion(self: Projective3<T>, q: UnitQuaternion<T>) -> Projective3<T> {
        Projective3 {
            matrix: TransformKernels::mul_rotation3(self.matrix, q.to_rotation_matrix().matrix),
        }
    }

    /// `self / q = self * q⁻¹` (the conjugate, exact), rounded like `mul_unit_quaternion`.
    /// Upstream: `Div<UnitQuaternion> for Transform`.
    #[inline(always)]
    fn div_unit_quaternion(self: Projective3<T>, q: UnitQuaternion<T>) -> Projective3<T> {
        Projective3 {
            matrix: TransformKernels::mul_rotation3(
                self.matrix, q.inverse().to_rotation_matrix().matrix,
            ),
        }
    }

    /// `self * tr`: the translation `tr` followed by `self` (`Matrix4::prepend_translation`: ONE
    /// fused sum of products per entry of the last column, bit for bit the homogeneous product).
    /// Upstream: `Mul<Translation> for Transform`.
    #[inline(always)]
    fn mul_translation(self: Projective3<T>, tr: Translation3<T>) -> Projective3<T> {
        Projective3 { matrix: Matrix4CgTrait::prepend_translation(self.matrix, tr.vector) }
    }

    /// `self / tr = self * tr⁻¹` (the negated translation, exact), rounded like
    /// `mul_translation`. Upstream: `Div<Translation> for Transform`.
    #[inline(always)]
    fn div_translation(self: Projective3<T>, tr: Translation3<T>) -> Projective3<T> {
        Projective3 {
            matrix: Matrix4CgTrait::prepend_translation(self.matrix, tr.inverse().vector),
        }
    }

    /// `self * iso`: `m * iso.to_homogeneous()`, one fused sum of products per entry (the exact
    /// last row of the isometry skipped, bit for bit the homogeneous product). Upstream's generic
    /// `Mul<Isometry<T, R, D>>` also takes the rotation-matrix isometries: for an
    /// `IsometryMatrix3`, `self.mul_transform(Affine3::from(iso))` gives the same bits.
    /// Upstream: `Mul<Isometry> for Transform`.
    #[inline(always)]
    fn mul_isometry(self: Projective3<T>, iso: Isometry3<T>) -> Projective3<T> {
        Projective3 { matrix: TransformKernels::mul_affine3(self.matrix, iso.to_homogeneous()) }
    }

    /// `self * sim`: `m * sim.to_homogeneous()`, rounded like `mul_isometry` (for a
    /// `SimilarityMatrix3`: `self.mul_transform(Affine3::from(sim))`, same bits). Upstream:
    /// `Mul<Similarity> for Transform`.
    #[inline(always)]
    fn mul_similarity(self: Projective3<T>, sim: Similarity3<T>) -> Projective3<T> {
        Projective3 { matrix: TransformKernels::mul_affine3(self.matrix, sim.to_homogeneous()) }
    }

    /// `true` when every entry of the matrices is within `ulps` raw units. Upstream:
    /// `approx::AbsDiffEq::abs_diff_eq` (DESIGN D3).
    #[inline(always)]
    fn abs_diff_eq(self: Projective3<T>, other: Projective3<T>, ulps: u64) -> bool {
        Matrix4Trait::abs_diff_eq(self.matrix, other.matrix, ulps)
    }

    /// `relative_eq` of the matrices. Upstream: `approx::RelativeEq::relative_eq` (DESIGN D3).
    #[inline(always)]
    fn relative_eq(
        self: Projective3<T>, other: Projective3<T>, epsilon: u64, max_relative: T,
    ) -> bool {
        Matrix4Trait::relative_eq(self.matrix, other.matrix, epsilon, max_relative)
    }

    /// `ulps_eq` of the matrices. Upstream: `approx::UlpsEq::ulps_eq` (DESIGN D3).
    #[inline(always)]
    fn ulps_eq(self: Projective3<T>, other: Projective3<T>, epsilon: u64, max_ulps: u32) -> bool {
        Matrix4Trait::ulps_eq(self.matrix, other.matrix, epsilon, max_ulps)
    }
}

/// `a * b`: the full 4x4 product of the matrices (one fused sum of 4 products per entry),
/// a `Projective3` (upstream's `TCategoryMul<TProjective> for TProjective`). The other category
/// pairs are `TransformMul::mul_transform`. Upstream: `Mul<Transform> for Transform`.
pub impl Projective3Mul<
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
> of Mul<Projective3<T>> {
    #[inline(always)]
    fn mul(lhs: Projective3<T>, rhs: Projective3<T>) -> Projective3<T> {
        Projective3 { matrix: lhs.matrix * rhs.matrix }
    }
}

/// `a / b = a * b⁻¹`, a `Projective3`. Panics with `errors::NOT_INVERTIBLE` when `b` is
/// singular. The other category pairs are `TransformDiv::div_transform`. Upstream: `Div<Transform>
/// for Transform`.
pub impl Projective3Div<
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
> of Div<Projective3<T>> {
    #[inline(always)]
    fn div(lhs: Projective3<T>, rhs: Projective3<T>) -> Projective3<T> {
        Projective3 { matrix: lhs.matrix * rhs.inverse().matrix }
    }
}

/// `Default::default()`: the identity. Upstream: `Default for Transform`.
pub impl Projective3Default<
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
> of Default<Projective3<T>> {
    #[inline(always)]
    fn default() -> Projective3<T> {
        Projective3Trait::identity()
    }
}

/// `One::one()`: the identity; `is_one` compares with it exactly. Upstream: `num::One for
/// Transform`.
pub impl Projective3One<
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
> of One<Projective3<T>> {
    #[inline(always)]
    fn one() -> Projective3<T> {
        Projective3Trait::identity()
    }

    #[inline(always)]
    fn is_one(self: @Projective3<T>) -> bool {
        *self == Projective3Trait::identity()
    }

    #[inline(always)]
    fn is_non_one(self: @Projective3<T>) -> bool {
        !Self::is_one(self)
    }
}

/// `t[(i, j)]`: the entry of row `i` and column `j` of the homogeneous matrix. Panics with
/// `nalgebra: index out of bounds` for `i > 3` or `j > 3`. Upstream: `Index<(usize, usize)>
/// for Transform`.
pub impl Projective3Index<T, +Copy<T>, +Drop<T>> of Index<Projective3<T>, (usize, usize)> {
    type Target = T;

    #[inline(always)]
    fn index(ref self: Projective3<T>, index: (usize, usize)) -> T {
        let m = self.matrix;
        m[index]
    }
}

/// `t.into()`: the homogeneous matrix (`into_inner`). Exact. Upstream: `From<Transform> for
/// Matrix4` (`OMatrix`).
pub impl Matrix4FromProjective3<T> of Into<Projective3<T>, Matrix4<T>> {
    #[inline(always)]
    fn into(self: Projective3<T>) -> Matrix4<T> {
        self.matrix
    }
}

/// `x.try_into()`: `Some` when the matrix is invertible (`Matrix4::is_invertible`), the same
/// matrix; `None` otherwise (upstream's `is_in_subset`,
/// `TProjective::check_homogeneous_invariants`). Exact. Upstream: `SubsetOf<Matrix4> for
/// Projective3` (`nalgebra::try_convert`, `from_superset`).
pub impl Projective3TryFromMatrix4<
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
> of TryInto<Matrix4<T>, Projective3<T>> {
    fn try_into(self: Matrix4<T>) -> Option<Projective3<T>> {
        let m = self;
        if Matrix4Trait::is_invertible(m) {
            Option::Some(Projective3 { matrix: m })
        } else {
            Option::None
        }
    }
}

/// `x.try_into()`: `Some` when the matrix is invertible (`Matrix4::is_invertible`), the same
/// matrix; `None` otherwise (upstream's `is_in_subset`,
/// `TProjective::check_homogeneous_invariants`). Exact. Upstream: `SubsetOf<Transform3> for
/// Projective3` (`nalgebra::try_convert`, `from_superset`).
pub impl Projective3TryFromTransform3<
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
> of TryInto<Transform3<T>, Projective3<T>> {
    fn try_into(self: Transform3<T>) -> Option<Projective3<T>> {
        let m = self.into_inner();
        if Matrix4Trait::is_invertible(m) {
            Option::Some(Projective3 { matrix: m })
        } else {
            Option::None
        }
    }
}

/// `x.into()`: the same matrix as a `Projective3` (`TProjective` is a super-category of `TAffine`,
/// no check). Exact. Upstream: `SubsetOf<Projective3> for Affine3` (`nalgebra::convert`; also
/// `set_category`).
pub impl Projective3FromAffine3<
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
> of Into<Affine3<T>, Projective3<T>> {
    #[inline(always)]
    fn into(self: Affine3<T>) -> Projective3<T> {
        Projective3 { matrix: self.into_inner() }
    }
}

/// `g.into()`: the transform of homogeneous matrix `g.to_homogeneous()` (its rounding). Upstream:
/// `SubsetOf<Transform> for Rotation3` (`nalgebra::convert`, any category).
pub impl Projective3FromRotation3<
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
> of Into<Rotation3<T>, Projective3<T>> {
    #[inline(always)]
    fn into(self: Rotation3<T>) -> Projective3<T> {
        Projective3 { matrix: self.to_homogeneous() }
    }
}

/// `g.into()`: the transform of homogeneous matrix `g.to_homogeneous()` (its rounding). Upstream:
/// `SubsetOf<Transform> for UnitQuaternion` (`nalgebra::convert`, any category).
pub impl Projective3FromUnitQuaternion<
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
> of Into<UnitQuaternion<T>, Projective3<T>> {
    #[inline(always)]
    fn into(self: UnitQuaternion<T>) -> Projective3<T> {
        Projective3 { matrix: self.to_homogeneous() }
    }
}

/// `g.into()`: the transform of homogeneous matrix `g.to_homogeneous()` (its rounding). Upstream:
/// `SubsetOf<Transform> for Translation3` (`nalgebra::convert`, any category).
pub impl Projective3FromTranslation3<
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
> of Into<Translation3<T>, Projective3<T>> {
    #[inline(always)]
    fn into(self: Translation3<T>) -> Projective3<T> {
        Projective3 { matrix: self.to_homogeneous() }
    }
}

/// `g.into()`: the transform of homogeneous matrix `g.to_homogeneous()` (its rounding). Upstream:
/// `SubsetOf<Transform> for Isometry3` (`nalgebra::convert`, any category).
pub impl Projective3FromIsometry3<
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
> of Into<Isometry3<T>, Projective3<T>> {
    #[inline(always)]
    fn into(self: Isometry3<T>) -> Projective3<T> {
        Projective3 { matrix: self.to_homogeneous() }
    }
}

/// `g.into()`: the transform of homogeneous matrix `g.to_homogeneous()` (its rounding). Upstream:
/// `SubsetOf<Transform> for IsometryMatrix3` (`nalgebra::convert`, any category).
pub impl Projective3FromIsometryMatrix3<
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
> of Into<IsometryMatrix3<T>, Projective3<T>> {
    #[inline(always)]
    fn into(self: IsometryMatrix3<T>) -> Projective3<T> {
        Projective3 { matrix: self.to_homogeneous() }
    }
}

/// `g.into()`: the transform of homogeneous matrix `g.to_homogeneous()` (its rounding). Upstream:
/// `SubsetOf<Transform> for Similarity3` (`nalgebra::convert`, any category).
pub impl Projective3FromSimilarity3<
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
> of Into<Similarity3<T>, Projective3<T>> {
    #[inline(always)]
    fn into(self: Similarity3<T>) -> Projective3<T> {
        Projective3 { matrix: self.to_homogeneous() }
    }
}

/// `g.into()`: the transform of homogeneous matrix `g.to_homogeneous()` (its rounding). Upstream:
/// `SubsetOf<Transform> for SimilarityMatrix3` (`nalgebra::convert`, any category).
pub impl Projective3FromSimilarityMatrix3<
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
> of Into<SimilarityMatrix3<T>, Projective3<T>> {
    #[inline(always)]
    fn into(self: SimilarityMatrix3<T>) -> Projective3<T> {
        Projective3 { matrix: self.to_homogeneous() }
    }
}

/// `g.into()`: the transform of homogeneous matrix `g.to_homogeneous()` (its rounding). Upstream:
/// `SubsetOf<Transform> for Scale3` (`nalgebra::convert`, any category).
pub impl Projective3FromScale3<
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
> of Into<Scale3<T>, Projective3<T>> {
    #[inline(always)]
    fn into(self: Scale3<T>) -> Projective3<T> {
        Projective3 { matrix: self.to_homogeneous() }
    }
}

/// `g.into()`: the transform of homogeneous matrix `g.to_homogeneous()` (its rounding). Upstream:
/// `SubsetOf<Transform> for UnitDualQuaternion` (`nalgebra::convert`, any category).
pub impl Projective3FromUnitDualQuaternion<
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
> of Into<UnitDualQuaternion<T>, Projective3<T>> {
    #[inline(always)]
    fn into(self: UnitDualQuaternion<T>) -> Projective3<T> {
        Projective3 { matrix: self.to_homogeneous() }
    }
}
