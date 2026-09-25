//! `Transform2`: a general transformation: any homogeneous matrix, possibly singular. `try_inverse`
//! only (no `inverse`), and `transform_point` / `transform_vector` divide by the homogeneous
//! coordinate. Upstream: `nalgebra::Transform2`, i.e. `Transform<T, TGeneral, 2>`
//! (`geometry/transform*.rs`), WP 8.4-P11a. The design of the six transform types (one Cairo
//! struct per upstream alias, upstream's category rules) is in `transform.cairo`.
//!
//! - `Transform2Trait` / `Transform2Impl`: construction, accessors, inverse, transforms, the
//! products by the
//!   other geometry types (`mul_<rhs>` / `div_<rhs>`), comparisons;
//! - the operator / conversion impls: `*` with a `Transform2`, `Default`, `One`, `Index<(usize,
//! usize)>`,
//!   `Into` / `TryInto` (upstream's `SubsetOf` / `From`);
//! - the category-changing products are `TransformMul` / `TransformDiv`, `set_category` is
//!   `TransformSetCategory` (`transform.cairo`).
//!
//! Numeric contract (AGENTS.md): every sum of products is one fused `Real` kernel (one floor per
//! output scalar), every quotient correctly rounded; overflow panics.

use core::num::traits::One;
use core::ops::Index;
use simba::scalar::Real;
use crate::base::cg::Matrix3CgTrait;
use crate::base::matrix3::{Matrix3, Matrix3Trait};
use crate::base::point2::Point2;
use crate::base::vector2::Vector2;
use super::affine2::{Affine2, Affine2Trait};
use super::isometry2::{Isometry2, Isometry2Trait};
use super::isometry_matrix2::{IsometryMatrix2, IsometryMatrix2Trait};
use super::projective2::{Projective2, Projective2Trait};
use super::rotation2::{Rotation2, Rotation2Trait};
use super::scale2::{Scale2, Scale2Trait};
use super::similarity2::{Similarity2, Similarity2Trait};
use super::similarity_matrix2::{SimilarityMatrix2, SimilarityMatrix2Trait};
use super::transform::TransformKernels;
use super::translation2::{Translation2, Translation2Trait};
use super::unit_complex::{UnitComplex, UnitComplexTrait};

/// A 2D transformation of category `TGeneral`: the homogeneous 3x3 matrix `matrix`
/// (crate-private, like upstream's private field). `PartialEq` compares the matrices exactly,
/// `Serde` is the matrix's, `Hash` hashes the matrix (upstream's `Hash`). Upstream:
/// `Transform2<T>`.
#[derive(Copy, Drop, PartialEq, Serde, Debug, Hash)]
pub struct Transform2<T> {
    pub(crate) matrix: Matrix3<T>,
}

/// Operations of `Transform2<T>` over a `Real` scalar. By value, unrolled, no loop.
#[generate_trait]
pub impl Transform2Impl<
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
> of Transform2Trait<T> {
    /// The transform whose homogeneous matrix is `matrix`, NOT checked against the invariants of
    /// the category. Exact. Upstream: `Transform::from_matrix_unchecked`.
    #[inline(always)]
    fn from_matrix_unchecked(matrix: Matrix3<T>) -> Transform2<T> {
        Transform2 { matrix }
    }

    /// The identity transform. Exact. Upstream: `Transform::identity`.
    #[inline(always)]
    fn identity() -> Transform2<T> {
        Transform2 { matrix: Matrix3Trait::identity() }
    }

    /// The homogeneous matrix. Exact. Upstream: `into_inner`.
    #[inline(always)]
    fn into_inner(self: Transform2<T>) -> Matrix3<T> {
        self.matrix
    }

    /// The homogeneous matrix: `into_inner` (deprecated upstream, "use `.into_inner()`
    /// instead"). Exact. Upstream: `unwrap`.
    #[inline(always)]
    fn unwrap(self: Transform2<T>) -> Matrix3<T> {
        self.matrix
    }

    /// The homogeneous matrix (a copy: Cairo values are passed by value). Exact. Upstream:
    /// `matrix`.
    #[inline(always)]
    fn matrix(self: Transform2<T>) -> Matrix3<T> {
        self.matrix
    }

    /// The homogeneous matrix. Exact. Upstream: `to_homogeneous`.
    #[inline(always)]
    fn to_homogeneous(self: Transform2<T>) -> Matrix3<T> {
        self.matrix
    }

    /// The inverse, or `None` when the matrix is singular: `Matrix3::try_inverse` of the
    /// homogeneous matrix (its rounding and its singularity criterion: a determinant EXACTLY
    /// zero), upstream's formula. Upstream: `try_inverse`.
    #[inline(always)]
    fn try_inverse(self: Transform2<T>) -> Option<Transform2<T>> {
        match Matrix3Trait::try_inverse(self.matrix) {
            Option::Some(matrix) => Option::Some(Transform2 { matrix }),
            Option::None => Option::None,
        }
    }

    /// Inverts in place and returns `true`, or returns `false` and leaves `self` unchanged when
    /// it is not invertible (`try_inverse`, same rounding). Upstream: `try_inverse_mut`.
    #[inline(always)]
    fn try_inverse_mut(ref self: Transform2<T>) -> bool {
        match Self::try_inverse(self) {
            Option::Some(inv) => {
                self = inv;
                true
            },
            Option::None => false,
        }
    }

    /// `pt` transformed, with the homogeneous division: `Matrix3::transform_point` (`q =
    /// m[:2, :2] * pt + m[:2, 2]`, `n = m[2, :2] . pt + m[2, 2]`, one fused sum of
    /// products each, then `q / n` correctly rounded if `n != 0`, else `q`: upstream's formula
    /// and branch). Upstream: `transform_point` (`Mul<Point>`).
    #[inline(always)]
    fn transform_point(self: Transform2<T>, pt: Point2<T>) -> Point2<T> {
        Matrix3CgTrait::transform_point(self.matrix, pt)
    }

    /// `v` transformed: `Matrix3::transform_vector` (with `n = m[2, :2] . v`, `m[:2, :2]
    /// * (v / n)` if `n != 0`, else `m[:2, :2] * v`: upstream's formula and branch).
    /// Upstream: `transform_vector` (`Mul<SVector>`).
    #[inline(always)]
    fn transform_vector(self: Transform2<T>, v: Vector2<T>) -> Vector2<T> {
        Matrix3CgTrait::transform_vector(self.matrix, v)
    }

    /// `self * r`: `m * [[r, 0], [0, 1]]`, the first 2 columns one fused sum of products per
    /// entry, the last one copied (bit for bit the homogeneous product); a `Transform2` (upstream's
    /// `TCategoryMul<TAffine>` keeps the category). Upstream: `Mul<Rotation> for Transform`.
    #[inline(always)]
    fn mul_rotation(self: Transform2<T>, r: Rotation2<T>) -> Transform2<T> {
        Transform2 { matrix: TransformKernels::mul_rotation2(self.matrix, r.matrix) }
    }

    /// `self / r = self * r⁻¹` (the inverse rotation is the exact transpose), rounded like
    /// `mul_rotation`. Upstream: `Div<Rotation> for Transform`.
    #[inline(always)]
    fn div_rotation(self: Transform2<T>, r: Rotation2<T>) -> Transform2<T> {
        Transform2 { matrix: TransformKernels::mul_rotation2(self.matrix, r.inverse().matrix) }
    }

    /// `self * c`: the rotation matrix of `c` (exact), then like `mul_rotation`. Upstream:
    /// `Mul<UnitComplex> for Transform`.
    #[inline(always)]
    fn mul_unit_complex(self: Transform2<T>, c: UnitComplex<T>) -> Transform2<T> {
        Transform2 {
            matrix: TransformKernels::mul_rotation2(self.matrix, c.to_rotation_matrix().matrix),
        }
    }

    /// `self * tr`: the translation `tr` followed by `self` (`Matrix3::prepend_translation`: ONE
    /// fused sum of products per entry of the last column, bit for bit the homogeneous product).
    /// Upstream: `Mul<Translation> for Transform`.
    #[inline(always)]
    fn mul_translation(self: Transform2<T>, tr: Translation2<T>) -> Transform2<T> {
        Transform2 { matrix: Matrix3CgTrait::prepend_translation(self.matrix, tr.vector) }
    }

    /// `self / tr = self * tr⁻¹` (the negated translation, exact), rounded like
    /// `mul_translation`. Upstream: `Div<Translation> for Transform`.
    #[inline(always)]
    fn div_translation(self: Transform2<T>, tr: Translation2<T>) -> Transform2<T> {
        Transform2 { matrix: Matrix3CgTrait::prepend_translation(self.matrix, tr.inverse().vector) }
    }

    /// `self * iso`: `m * iso.to_homogeneous()`, one fused sum of products per entry (the exact
    /// last row of the isometry skipped, bit for bit the homogeneous product). Upstream's generic
    /// `Mul<Isometry<T, R, D>>` also takes the rotation-matrix isometries: for an
    /// `IsometryMatrix2`, `self.mul_transform(Affine2::from(iso))` gives the same bits.
    /// Upstream: `Mul<Isometry> for Transform`.
    #[inline(always)]
    fn mul_isometry(self: Transform2<T>, iso: Isometry2<T>) -> Transform2<T> {
        Transform2 { matrix: TransformKernels::mul_affine2(self.matrix, iso.to_homogeneous()) }
    }

    /// `self * sim`: `m * sim.to_homogeneous()`, rounded like `mul_isometry` (for a
    /// `SimilarityMatrix2`: `self.mul_transform(Affine2::from(sim))`, same bits). Upstream:
    /// `Mul<Similarity> for Transform`.
    #[inline(always)]
    fn mul_similarity(self: Transform2<T>, sim: Similarity2<T>) -> Transform2<T> {
        Transform2 { matrix: TransformKernels::mul_affine2(self.matrix, sim.to_homogeneous()) }
    }

    /// `true` when every entry of the matrices is within `ulps` raw units. Upstream:
    /// `approx::AbsDiffEq::abs_diff_eq` (DESIGN D3).
    #[inline(always)]
    fn abs_diff_eq(self: Transform2<T>, other: Transform2<T>, ulps: u64) -> bool {
        Matrix3Trait::abs_diff_eq(self.matrix, other.matrix, ulps)
    }

    /// `relative_eq` of the matrices. Upstream: `approx::RelativeEq::relative_eq` (DESIGN D3).
    #[inline(always)]
    fn relative_eq(
        self: Transform2<T>, other: Transform2<T>, epsilon: u64, max_relative: T,
    ) -> bool {
        Matrix3Trait::relative_eq(self.matrix, other.matrix, epsilon, max_relative)
    }

    /// `ulps_eq` of the matrices. Upstream: `approx::UlpsEq::ulps_eq` (DESIGN D3).
    #[inline(always)]
    fn ulps_eq(self: Transform2<T>, other: Transform2<T>, epsilon: u64, max_ulps: u32) -> bool {
        Matrix3Trait::ulps_eq(self.matrix, other.matrix, epsilon, max_ulps)
    }
}

/// `a * b`: the full 3x3 product of the matrices (one fused sum of 3 products per entry),
/// a `Transform2` (upstream's `TCategoryMul<TGeneral> for TGeneral`). The other category pairs are
/// `TransformMul::mul_transform`. Upstream: `Mul<Transform> for Transform`.
pub impl Transform2Mul<
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
> of Mul<Transform2<T>> {
    #[inline(always)]
    fn mul(lhs: Transform2<T>, rhs: Transform2<T>) -> Transform2<T> {
        Transform2 { matrix: lhs.matrix * rhs.matrix }
    }
}

/// `Default::default()`: the identity. Upstream: `Default for Transform`.
pub impl Transform2Default<
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
> of Default<Transform2<T>> {
    #[inline(always)]
    fn default() -> Transform2<T> {
        Transform2Trait::identity()
    }
}

/// `One::one()`: the identity; `is_one` compares with it exactly. Upstream: `num::One for
/// Transform`.
pub impl Transform2One<
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
> of One<Transform2<T>> {
    #[inline(always)]
    fn one() -> Transform2<T> {
        Transform2Trait::identity()
    }

    #[inline(always)]
    fn is_one(self: @Transform2<T>) -> bool {
        *self == Transform2Trait::identity()
    }

    #[inline(always)]
    fn is_non_one(self: @Transform2<T>) -> bool {
        !Self::is_one(self)
    }
}

/// `t[(i, j)]`: the entry of row `i` and column `j` of the homogeneous matrix. Panics with
/// `nalgebra: index out of bounds` for `i > 2` or `j > 2`. Upstream: `Index<(usize, usize)>
/// for Transform`.
pub impl Transform2Index<T, +Copy<T>, +Drop<T>> of Index<Transform2<T>, (usize, usize)> {
    type Target = T;

    #[inline(always)]
    fn index(ref self: Transform2<T>, index: (usize, usize)) -> T {
        let m = self.matrix;
        m[index]
    }
}

/// `t.into()`: the homogeneous matrix (`into_inner`). Exact. Upstream: `From<Transform> for
/// Matrix3` (`OMatrix`).
pub impl Matrix3FromTransform2<T> of Into<Transform2<T>, Matrix3<T>> {
    #[inline(always)]
    fn into(self: Transform2<T>) -> Matrix3<T> {
        self.matrix
    }
}

/// `x.try_into()`: `Some` always (`TGeneral` has no invariant), the same matrix; `None` otherwise
/// (upstream's `is_in_subset`, `TGeneral::check_homogeneous_invariants`). Exact. Upstream:
/// `SubsetOf<Matrix3> for Transform2` (`nalgebra::try_convert`, `from_superset`).
pub impl Transform2TryFromMatrix3<
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
> of TryInto<Matrix3<T>, Transform2<T>> {
    fn try_into(self: Matrix3<T>) -> Option<Transform2<T>> {
        let m = self;
        if true {
            Option::Some(Transform2 { matrix: m })
        } else {
            Option::None
        }
    }
}

/// `x.into()`: the same matrix as a `Transform2` (`TGeneral` is a super-category of `TProjective`,
/// no check). Exact. Upstream: `SubsetOf<Transform2> for Projective2` (`nalgebra::convert`; also
/// `set_category`).
pub impl Transform2FromProjective2<
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
> of Into<Projective2<T>, Transform2<T>> {
    #[inline(always)]
    fn into(self: Projective2<T>) -> Transform2<T> {
        Transform2 { matrix: self.into_inner() }
    }
}

/// `x.into()`: the same matrix as a `Transform2` (`TGeneral` is a super-category of `TAffine`,
/// no check). Exact. Upstream: `SubsetOf<Transform2> for Affine2` (`nalgebra::convert`; also
/// `set_category`).
pub impl Transform2FromAffine2<
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
> of Into<Affine2<T>, Transform2<T>> {
    #[inline(always)]
    fn into(self: Affine2<T>) -> Transform2<T> {
        Transform2 { matrix: self.into_inner() }
    }
}

/// `g.into()`: the transform of homogeneous matrix `g.to_homogeneous()` (its rounding). Upstream:
/// `SubsetOf<Transform> for Rotation2` (`nalgebra::convert`, any category).
pub impl Transform2FromRotation2<
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
> of Into<Rotation2<T>, Transform2<T>> {
    #[inline(always)]
    fn into(self: Rotation2<T>) -> Transform2<T> {
        Transform2 { matrix: self.to_homogeneous() }
    }
}

/// `g.into()`: the transform of homogeneous matrix `g.to_homogeneous()` (its rounding). Upstream:
/// `SubsetOf<Transform> for UnitComplex` (`nalgebra::convert`, any category).
pub impl Transform2FromUnitComplex<
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
> of Into<UnitComplex<T>, Transform2<T>> {
    #[inline(always)]
    fn into(self: UnitComplex<T>) -> Transform2<T> {
        Transform2 { matrix: self.to_homogeneous() }
    }
}

/// `g.into()`: the transform of homogeneous matrix `g.to_homogeneous()` (its rounding). Upstream:
/// `SubsetOf<Transform> for Translation2` (`nalgebra::convert`, any category).
pub impl Transform2FromTranslation2<
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
> of Into<Translation2<T>, Transform2<T>> {
    #[inline(always)]
    fn into(self: Translation2<T>) -> Transform2<T> {
        Transform2 { matrix: self.to_homogeneous() }
    }
}

/// `g.into()`: the transform of homogeneous matrix `g.to_homogeneous()` (its rounding). Upstream:
/// `SubsetOf<Transform> for Isometry2` (`nalgebra::convert`, any category).
pub impl Transform2FromIsometry2<
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
> of Into<Isometry2<T>, Transform2<T>> {
    #[inline(always)]
    fn into(self: Isometry2<T>) -> Transform2<T> {
        Transform2 { matrix: self.to_homogeneous() }
    }
}

/// `g.into()`: the transform of homogeneous matrix `g.to_homogeneous()` (its rounding). Upstream:
/// `SubsetOf<Transform> for IsometryMatrix2` (`nalgebra::convert`, any category).
pub impl Transform2FromIsometryMatrix2<
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
> of Into<IsometryMatrix2<T>, Transform2<T>> {
    #[inline(always)]
    fn into(self: IsometryMatrix2<T>) -> Transform2<T> {
        Transform2 { matrix: self.to_homogeneous() }
    }
}

/// `g.into()`: the transform of homogeneous matrix `g.to_homogeneous()` (its rounding). Upstream:
/// `SubsetOf<Transform> for Similarity2` (`nalgebra::convert`, any category).
pub impl Transform2FromSimilarity2<
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
> of Into<Similarity2<T>, Transform2<T>> {
    #[inline(always)]
    fn into(self: Similarity2<T>) -> Transform2<T> {
        Transform2 { matrix: self.to_homogeneous() }
    }
}

/// `g.into()`: the transform of homogeneous matrix `g.to_homogeneous()` (its rounding). Upstream:
/// `SubsetOf<Transform> for SimilarityMatrix2` (`nalgebra::convert`, any category).
pub impl Transform2FromSimilarityMatrix2<
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
> of Into<SimilarityMatrix2<T>, Transform2<T>> {
    #[inline(always)]
    fn into(self: SimilarityMatrix2<T>) -> Transform2<T> {
        Transform2 { matrix: self.to_homogeneous() }
    }
}

/// `g.into()`: the transform of homogeneous matrix `g.to_homogeneous()` (its rounding). Upstream:
/// `SubsetOf<Transform> for Scale2` (`nalgebra::convert`, any category).
pub impl Transform2FromScale2<
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
> of Into<Scale2<T>, Transform2<T>> {
    #[inline(always)]
    fn into(self: Scale2<T>) -> Transform2<T> {
        Transform2 { matrix: self.to_homogeneous() }
    }
}
