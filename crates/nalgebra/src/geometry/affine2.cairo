//! `Affine2`: an invertible homogeneous matrix whose last row is `(0, .., 0, 1)`. `inverse` exists
//! (by blocks), and `transform_point` / `transform_vector` ignore the last row (no normalizer).
//! Upstream: `nalgebra::Affine2`, i.e. `Transform<T, TAffine, 2>`
//! (`geometry/transform*.rs`), WP 8.4-P11a. The design of the six transform types (one Cairo
//! struct per upstream alias, upstream's category rules) is in `transform.cairo`.
//!
//! - `Affine2Trait` / `Affine2Impl`: construction, accessors, inverse, transforms, the products by
//! the
//!   other geometry types (`mul_<rhs>` / `div_<rhs>`), comparisons;
//! - the operator / conversion impls: `*` / `/` with a `Affine2`, `Default`, `One`, `Index<(usize,
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
use super::isometry2::{Isometry2, Isometry2Trait};
use super::isometry_matrix2::{IsometryMatrix2, IsometryMatrix2Trait};
use super::projective2::{Projective2, Projective2Trait};
use super::rotation2::{Rotation2, Rotation2Trait};
use super::scale2::{Scale2, Scale2Trait};
use super::similarity2::{Similarity2, Similarity2Trait};
use super::similarity_matrix2::{SimilarityMatrix2, SimilarityMatrix2Trait};
use super::transform::{TransformKernels, errors};
use super::transform2::{Transform2, Transform2Trait};
use super::translation2::{Translation2, Translation2Trait};
use super::unit_complex::{UnitComplex, UnitComplexTrait};

/// A 2D transformation of category `TAffine`: the homogeneous 3x3 matrix `matrix`
/// (crate-private, like upstream's private field). `PartialEq` compares the matrices exactly,
/// `Serde` is the matrix's, `Hash` hashes the matrix (upstream's `Hash`). Upstream: `Affine2<T>`.
#[derive(Copy, Drop, PartialEq, Serde, Debug, Hash)]
pub struct Affine2<T> {
    pub(crate) matrix: Matrix3<T>,
}

/// Operations of `Affine2<T>` over a `Real` scalar. By value, unrolled, no loop.
#[generate_trait]
pub impl Affine2Impl<
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
> of Affine2Trait<T> {
    /// The transform whose homogeneous matrix is `matrix`, NOT checked against the invariants of
    /// the category. Exact. Upstream: `Transform::from_matrix_unchecked`.
    #[inline(always)]
    fn from_matrix_unchecked(matrix: Matrix3<T>) -> Affine2<T> {
        Affine2 { matrix }
    }

    /// The identity transform. Exact. Upstream: `Transform::identity`.
    #[inline(always)]
    fn identity() -> Affine2<T> {
        Affine2 { matrix: Matrix3Trait::identity() }
    }

    /// The homogeneous matrix. Exact. Upstream: `into_inner`.
    #[inline(always)]
    fn into_inner(self: Affine2<T>) -> Matrix3<T> {
        self.matrix
    }

    /// The homogeneous matrix: `into_inner` (deprecated upstream, "use `.into_inner()`
    /// instead"). Exact. Upstream: `unwrap`.
    #[inline(always)]
    fn unwrap(self: Affine2<T>) -> Matrix3<T> {
        self.matrix
    }

    /// The homogeneous matrix (a copy: Cairo values are passed by value). Exact. Upstream:
    /// `matrix`.
    #[inline(always)]
    fn matrix(self: Affine2<T>) -> Matrix3<T> {
        self.matrix
    }

    /// The homogeneous matrix. Exact. Upstream: `to_homogeneous`.
    #[inline(always)]
    fn to_homogeneous(self: Affine2<T>) -> Matrix3<T> {
        self.matrix
    }

    /// The inverse, or `None` when the linear block is singular: by blocks, `l = m[:2, :2]⁻¹`
    /// (`Matrix2::try_inverse`, its singularity criterion: a determinant EXACTLY zero), the
    /// translation `l * (-t)` (ONE fused sum of products per coordinate) and the exact last row
    /// `(0, .., 0, 1)`, so the result is affine again. Upstream inverts the whole 3x3 matrix
    /// (`Matrix::try_inverse`, the same value in exact arithmetic): the block formula is cheaper,
    /// as accurate (oracle suite `transform`) and keeps the affine invariant exactly
    /// (`bench_affine2_try_inverse__alt_full_matrix`). Upstream: `try_inverse`.
    #[inline(always)]
    fn try_inverse(self: Affine2<T>) -> Option<Affine2<T>> {
        match TransformKernels::affine_inverse2(self.matrix) {
            Option::Some(matrix) => Option::Some(Affine2 { matrix }),
            Option::None => Option::None,
        }
    }

    /// Inverts in place and returns `true`, or returns `false` and leaves `self` unchanged when
    /// it is not invertible (`try_inverse`, same rounding). Upstream: `try_inverse_mut`.
    #[inline(always)]
    fn try_inverse_mut(ref self: Affine2<T>) -> bool {
        match Self::try_inverse(self) {
            Option::Some(inv) => {
                self = inv;
                true
            },
            Option::None => false,
        }
    }

    /// The inverse (`try_inverse`, same rounding). Panics with `errors::NOT_INVERTIBLE` when the
    /// transform is singular (upstream's `unwrap` of `None`); a well-formed affine transform
    /// is invertible by definition. Upstream: `inverse` (projective and affine categories only).
    #[inline(always)]
    fn inverse(self: Affine2<T>) -> Affine2<T> {
        Self::try_inverse(self).expect(errors::NOT_INVERTIBLE)
    }

    /// Inverts in place (`try_inverse`, same rounding). Like upstream (`let _ =
    /// self.matrix.try_inverse_mut()`), a singular transform is left unchanged WITHOUT panicking.
    /// Upstream: `inverse_mut` (projective and affine categories only).
    #[inline(always)]
    fn inverse_mut(ref self: Affine2<T>) {
        let _ = Self::try_inverse_mut(ref self);
    }

    /// `pt` transformed: `m[:2, :2] * pt + m[:2, 2]`, ONE fused sum of products per
    /// coordinate (the bits of upstream's product then sum). Like upstream (`TAffine` has no
    /// normalizer), the last row is not read. Upstream: `transform_point` (`Mul<Point>`).
    #[inline(always)]
    fn transform_point(self: Affine2<T>, pt: Point2<T>) -> Point2<T> {
        TransformKernels::affine_point2(self.matrix, pt)
    }

    /// `v` transformed: `m[:2, :2] * v`, ONE fused dot product per coordinate. Like upstream
    /// (`TAffine` has no normalizer), the last row is not read. Upstream: `transform_vector`
    /// (`Mul<SVector>`).
    #[inline(always)]
    fn transform_vector(self: Affine2<T>, v: Vector2<T>) -> Vector2<T> {
        TransformKernels::affine_vector2(self.matrix, v)
    }

    /// `self.inverse().transform_point(pt)` (upstream's formula: the inverse, then the
    /// transform). Panics like `inverse`. Upstream: `inverse_transform_point`.
    #[inline(always)]
    fn inverse_transform_point(self: Affine2<T>, pt: Point2<T>) -> Point2<T> {
        Self::transform_point(Self::inverse(self), pt)
    }

    /// `self.inverse().transform_vector(v)` (upstream's formula). Panics like `inverse`.
    /// Upstream: `inverse_transform_vector`.
    #[inline(always)]
    fn inverse_transform_vector(self: Affine2<T>, v: Vector2<T>) -> Vector2<T> {
        Self::transform_vector(Self::inverse(self), v)
    }

    /// `self * r`: `m * [[r, 0], [0, 1]]`, the first 2 columns one fused sum of products per
    /// entry, the last one copied (bit for bit the homogeneous product); a `Affine2` (upstream's
    /// `TCategoryMul<TAffine>` keeps the category). Upstream: `Mul<Rotation> for Transform`.
    #[inline(always)]
    fn mul_rotation(self: Affine2<T>, r: Rotation2<T>) -> Affine2<T> {
        Affine2 { matrix: TransformKernels::mul_rotation2(self.matrix, r.matrix) }
    }

    /// `self / r = self * r⁻¹` (the inverse rotation is the exact transpose), rounded like
    /// `mul_rotation`. Upstream: `Div<Rotation> for Transform`.
    #[inline(always)]
    fn div_rotation(self: Affine2<T>, r: Rotation2<T>) -> Affine2<T> {
        Affine2 { matrix: TransformKernels::mul_rotation2(self.matrix, r.inverse().matrix) }
    }

    /// `self * c`: the rotation matrix of `c` (exact), then like `mul_rotation`. Upstream:
    /// `Mul<UnitComplex> for Transform`.
    #[inline(always)]
    fn mul_unit_complex(self: Affine2<T>, c: UnitComplex<T>) -> Affine2<T> {
        Affine2 {
            matrix: TransformKernels::mul_rotation2(self.matrix, c.to_rotation_matrix().matrix),
        }
    }

    /// `self * tr`: the translation `tr` followed by `self` (`Matrix3::prepend_translation`: ONE
    /// fused sum of products per entry of the last column, bit for bit the homogeneous product).
    /// Upstream: `Mul<Translation> for Transform`.
    #[inline(always)]
    fn mul_translation(self: Affine2<T>, tr: Translation2<T>) -> Affine2<T> {
        Affine2 { matrix: Matrix3CgTrait::prepend_translation(self.matrix, tr.vector) }
    }

    /// `self / tr = self * tr⁻¹` (the negated translation, exact), rounded like
    /// `mul_translation`. Upstream: `Div<Translation> for Transform`.
    #[inline(always)]
    fn div_translation(self: Affine2<T>, tr: Translation2<T>) -> Affine2<T> {
        Affine2 { matrix: Matrix3CgTrait::prepend_translation(self.matrix, tr.inverse().vector) }
    }

    /// `self * iso`: `m * iso.to_homogeneous()`, one fused sum of products per entry (the exact
    /// last row of the isometry skipped, bit for bit the homogeneous product). Upstream's generic
    /// `Mul<Isometry<T, R, D>>` also takes the rotation-matrix isometries: for an
    /// `IsometryMatrix2`, `self.mul_transform(Affine2::from(iso))` gives the same bits.
    /// Upstream: `Mul<Isometry> for Transform`.
    #[inline(always)]
    fn mul_isometry(self: Affine2<T>, iso: Isometry2<T>) -> Affine2<T> {
        Affine2 { matrix: TransformKernels::mul_affine2(self.matrix, iso.to_homogeneous()) }
    }

    /// `self * sim`: `m * sim.to_homogeneous()`, rounded like `mul_isometry` (for a
    /// `SimilarityMatrix2`: `self.mul_transform(Affine2::from(sim))`, same bits). Upstream:
    /// `Mul<Similarity> for Transform`.
    #[inline(always)]
    fn mul_similarity(self: Affine2<T>, sim: Similarity2<T>) -> Affine2<T> {
        Affine2 { matrix: TransformKernels::mul_affine2(self.matrix, sim.to_homogeneous()) }
    }

    /// `true` when every entry of the matrices is within `ulps` raw units. Upstream:
    /// `approx::AbsDiffEq::abs_diff_eq` (DESIGN D3).
    #[inline(always)]
    fn abs_diff_eq(self: Affine2<T>, other: Affine2<T>, ulps: u64) -> bool {
        Matrix3Trait::abs_diff_eq(self.matrix, other.matrix, ulps)
    }

    /// `relative_eq` of the matrices. Upstream: `approx::RelativeEq::relative_eq` (DESIGN D3).
    #[inline(always)]
    fn relative_eq(self: Affine2<T>, other: Affine2<T>, epsilon: u64, max_relative: T) -> bool {
        Matrix3Trait::relative_eq(self.matrix, other.matrix, epsilon, max_relative)
    }

    /// `ulps_eq` of the matrices. Upstream: `approx::UlpsEq::ulps_eq` (DESIGN D3).
    #[inline(always)]
    fn ulps_eq(self: Affine2<T>, other: Affine2<T>, epsilon: u64, max_ulps: u32) -> bool {
        Matrix3Trait::ulps_eq(self.matrix, other.matrix, epsilon, max_ulps)
    }
}

/// `a * b`: the full 3x3 product of the matrices (one fused sum of 3 products per entry),
/// a `Affine2` (upstream's `TCategoryMul<TAffine> for TAffine`). The other category pairs are
/// `TransformMul::mul_transform`. Upstream: `Mul<Transform> for Transform`.
pub impl Affine2Mul<
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
> of Mul<Affine2<T>> {
    #[inline(always)]
    fn mul(lhs: Affine2<T>, rhs: Affine2<T>) -> Affine2<T> {
        Affine2 { matrix: lhs.matrix * rhs.matrix }
    }
}

/// `a / b = a * b⁻¹`, a `Affine2`. Panics with `errors::NOT_INVERTIBLE` when `b` is singular.
/// The other category pairs are `TransformDiv::div_transform`. Upstream: `Div<Transform> for
/// Transform`.
pub impl Affine2Div<
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
> of Div<Affine2<T>> {
    #[inline(always)]
    fn div(lhs: Affine2<T>, rhs: Affine2<T>) -> Affine2<T> {
        Affine2 { matrix: lhs.matrix * rhs.inverse().matrix }
    }
}

/// `Default::default()`: the identity. Upstream: `Default for Transform`.
pub impl Affine2Default<
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
> of Default<Affine2<T>> {
    #[inline(always)]
    fn default() -> Affine2<T> {
        Affine2Trait::identity()
    }
}

/// `One::one()`: the identity; `is_one` compares with it exactly. Upstream: `num::One for
/// Transform`.
pub impl Affine2One<
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
> of One<Affine2<T>> {
    #[inline(always)]
    fn one() -> Affine2<T> {
        Affine2Trait::identity()
    }

    #[inline(always)]
    fn is_one(self: @Affine2<T>) -> bool {
        *self == Affine2Trait::identity()
    }

    #[inline(always)]
    fn is_non_one(self: @Affine2<T>) -> bool {
        !Self::is_one(self)
    }
}

/// `t[(i, j)]`: the entry of row `i` and column `j` of the homogeneous matrix. Panics with
/// `nalgebra: index out of bounds` for `i > 2` or `j > 2`. Upstream: `Index<(usize, usize)>
/// for Transform`.
pub impl Affine2Index<T, +Copy<T>, +Drop<T>> of Index<Affine2<T>, (usize, usize)> {
    type Target = T;

    #[inline(always)]
    fn index(ref self: Affine2<T>, index: (usize, usize)) -> T {
        let m = self.matrix;
        m[index]
    }
}

/// `t.into()`: the homogeneous matrix (`into_inner`). Exact. Upstream: `From<Transform> for
/// Matrix3` (`OMatrix`).
pub impl Matrix3FromAffine2<T> of Into<Affine2<T>, Matrix3<T>> {
    #[inline(always)]
    fn into(self: Affine2<T>) -> Matrix3<T> {
        self.matrix
    }
}

/// `x.try_into()`: `Some` when the last row is EXACTLY `(0, .., 0, 1)` and the linear block is
/// invertible (`Matrix2::is_invertible`), the same matrix; `None` otherwise (upstream's
/// `is_in_subset`, `TAffine::check_homogeneous_invariants`). Exact. Upstream: `SubsetOf<Matrix3>
/// for Affine2` (`nalgebra::try_convert`, `from_superset`).
pub impl Affine2TryFromMatrix3<
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
> of TryInto<Matrix3<T>, Affine2<T>> {
    fn try_into(self: Matrix3<T>) -> Option<Affine2<T>> {
        let m = self;
        if TransformKernels::is_affine2(m) {
            Option::Some(Affine2 { matrix: m })
        } else {
            Option::None
        }
    }
}

/// `x.try_into()`: `Some` when the last row is EXACTLY `(0, .., 0, 1)` and the linear block is
/// invertible (`Matrix2::is_invertible`), the same matrix; `None` otherwise (upstream's
/// `is_in_subset`, `TAffine::check_homogeneous_invariants`). Exact. Upstream: `SubsetOf<Transform2>
/// for Affine2` (`nalgebra::try_convert`, `from_superset`).
pub impl Affine2TryFromTransform2<
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
> of TryInto<Transform2<T>, Affine2<T>> {
    fn try_into(self: Transform2<T>) -> Option<Affine2<T>> {
        let m = self.into_inner();
        if TransformKernels::is_affine2(m) {
            Option::Some(Affine2 { matrix: m })
        } else {
            Option::None
        }
    }
}

/// `x.try_into()`: `Some` when the last row is EXACTLY `(0, .., 0, 1)` and the linear block is
/// invertible (`Matrix2::is_invertible`), the same matrix; `None` otherwise (upstream's
/// `is_in_subset`, `TAffine::check_homogeneous_invariants`). Exact. Upstream:
/// `SubsetOf<Projective2> for Affine2` (`nalgebra::try_convert`, `from_superset`).
pub impl Affine2TryFromProjective2<
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
> of TryInto<Projective2<T>, Affine2<T>> {
    fn try_into(self: Projective2<T>) -> Option<Affine2<T>> {
        let m = self.into_inner();
        if TransformKernels::is_affine2(m) {
            Option::Some(Affine2 { matrix: m })
        } else {
            Option::None
        }
    }
}

/// `g.into()`: the transform of homogeneous matrix `g.to_homogeneous()` (its rounding). Upstream:
/// `SubsetOf<Transform> for Rotation2` (`nalgebra::convert`, any category).
pub impl Affine2FromRotation2<
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
> of Into<Rotation2<T>, Affine2<T>> {
    #[inline(always)]
    fn into(self: Rotation2<T>) -> Affine2<T> {
        Affine2 { matrix: self.to_homogeneous() }
    }
}

/// `g.into()`: the transform of homogeneous matrix `g.to_homogeneous()` (its rounding). Upstream:
/// `SubsetOf<Transform> for UnitComplex` (`nalgebra::convert`, any category).
pub impl Affine2FromUnitComplex<
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
> of Into<UnitComplex<T>, Affine2<T>> {
    #[inline(always)]
    fn into(self: UnitComplex<T>) -> Affine2<T> {
        Affine2 { matrix: self.to_homogeneous() }
    }
}

/// `g.into()`: the transform of homogeneous matrix `g.to_homogeneous()` (its rounding). Upstream:
/// `SubsetOf<Transform> for Translation2` (`nalgebra::convert`, any category).
pub impl Affine2FromTranslation2<
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
> of Into<Translation2<T>, Affine2<T>> {
    #[inline(always)]
    fn into(self: Translation2<T>) -> Affine2<T> {
        Affine2 { matrix: self.to_homogeneous() }
    }
}

/// `g.into()`: the transform of homogeneous matrix `g.to_homogeneous()` (its rounding). Upstream:
/// `SubsetOf<Transform> for Isometry2` (`nalgebra::convert`, any category).
pub impl Affine2FromIsometry2<
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
> of Into<Isometry2<T>, Affine2<T>> {
    #[inline(always)]
    fn into(self: Isometry2<T>) -> Affine2<T> {
        Affine2 { matrix: self.to_homogeneous() }
    }
}

/// `g.into()`: the transform of homogeneous matrix `g.to_homogeneous()` (its rounding). Upstream:
/// `SubsetOf<Transform> for IsometryMatrix2` (`nalgebra::convert`, any category).
pub impl Affine2FromIsometryMatrix2<
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
> of Into<IsometryMatrix2<T>, Affine2<T>> {
    #[inline(always)]
    fn into(self: IsometryMatrix2<T>) -> Affine2<T> {
        Affine2 { matrix: self.to_homogeneous() }
    }
}

/// `g.into()`: the transform of homogeneous matrix `g.to_homogeneous()` (its rounding). Upstream:
/// `SubsetOf<Transform> for Similarity2` (`nalgebra::convert`, any category).
pub impl Affine2FromSimilarity2<
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
> of Into<Similarity2<T>, Affine2<T>> {
    #[inline(always)]
    fn into(self: Similarity2<T>) -> Affine2<T> {
        Affine2 { matrix: self.to_homogeneous() }
    }
}

/// `g.into()`: the transform of homogeneous matrix `g.to_homogeneous()` (its rounding). Upstream:
/// `SubsetOf<Transform> for SimilarityMatrix2` (`nalgebra::convert`, any category).
pub impl Affine2FromSimilarityMatrix2<
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
> of Into<SimilarityMatrix2<T>, Affine2<T>> {
    #[inline(always)]
    fn into(self: SimilarityMatrix2<T>) -> Affine2<T> {
        Affine2 { matrix: self.to_homogeneous() }
    }
}

/// `g.into()`: the transform of homogeneous matrix `g.to_homogeneous()` (its rounding). Upstream:
/// `SubsetOf<Transform> for Scale2` (`nalgebra::convert`, any category).
pub impl Affine2FromScale2<
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
> of Into<Scale2<T>, Affine2<T>> {
    #[inline(always)]
    fn into(self: Scale2<T>) -> Affine2<T> {
        Affine2 { matrix: self.to_homogeneous() }
    }
}
