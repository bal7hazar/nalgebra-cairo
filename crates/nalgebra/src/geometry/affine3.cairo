//! `Affine3`: an invertible homogeneous matrix whose last row is `(0, .., 0, 1)`. `inverse` exists
//! (by blocks), and `transform_point` / `transform_vector` ignore the last row (no normalizer).
//! Upstream: `nalgebra::Affine3`, i.e. `Transform<T, TAffine, 3>`
//! (`geometry/transform*.rs`), WP 8.4-P11a. The design of the six transform types (one Cairo
//! struct per upstream alias, upstream's category rules) is in `transform.cairo`.
//!
//! - `Affine3Trait` / `Affine3Impl`: construction, accessors, inverse, transforms, the products by
//! the
//!   other geometry types (`mul_<rhs>` / `div_<rhs>`), comparisons;
//! - the operator / conversion impls: `*` / `/` with a `Affine3`, `Default`, `One`, `Index<(usize,
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
use crate::base::cg::Matrix4CgTrait;
use crate::base::matrix4::{Matrix4, Matrix4Trait};
use crate::base::point3::Point3;
use crate::base::vector3::Vector3;
use super::isometry3::{Isometry3, Isometry3Trait};
use super::isometry_matrix3::{IsometryMatrix3, IsometryMatrix3Trait};
use super::projective3::{Projective3, Projective3Trait};
use super::rotation3::{Rotation3, Rotation3Trait};
use super::scale3::{Scale3, Scale3Trait};
use super::similarity3::{Similarity3, Similarity3Trait};
use super::similarity_matrix3::{SimilarityMatrix3, SimilarityMatrix3Trait};
use super::transform::{TransformKernels, errors};
use super::transform3::{Transform3, Transform3Trait};
use super::translation3::{Translation3, Translation3Trait};
use super::unit_dual_quaternion::{UnitDualQuaternion, UnitDualQuaternionTrait};
use super::unit_quaternion::{UnitQuaternion, UnitQuaternionTrait};

/// A 3D transformation of category `TAffine`: the homogeneous 4x4 matrix `matrix`
/// (crate-private, like upstream's private field). `PartialEq` compares the matrices exactly,
/// `Serde` is the matrix's, `Hash` hashes the matrix (upstream's `Hash`). Upstream: `Affine3<T>`.
#[derive(Copy, Drop, PartialEq, Serde, Debug, Hash)]
pub struct Affine3<T> {
    pub(crate) matrix: Matrix4<T>,
}

/// Operations of `Affine3<T>` over a `Real` scalar. By value, unrolled, no loop.
#[generate_trait]
pub impl Affine3Impl<
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
> of Affine3Trait<T> {
    /// The transform whose homogeneous matrix is `matrix`, NOT checked against the invariants of
    /// the category. Exact. Upstream: `Transform::from_matrix_unchecked`.
    #[inline(always)]
    fn from_matrix_unchecked(matrix: Matrix4<T>) -> Affine3<T> {
        Affine3 { matrix }
    }

    /// The identity transform. Exact. Upstream: `Transform::identity`.
    #[inline(always)]
    fn identity() -> Affine3<T> {
        Affine3 { matrix: Matrix4Trait::identity() }
    }

    /// The homogeneous matrix. Exact. Upstream: `into_inner`.
    #[inline(always)]
    fn into_inner(self: Affine3<T>) -> Matrix4<T> {
        self.matrix
    }

    /// The homogeneous matrix: `into_inner` (deprecated upstream, "use `.into_inner()`
    /// instead"). Exact. Upstream: `unwrap`.
    #[inline(always)]
    fn unwrap(self: Affine3<T>) -> Matrix4<T> {
        self.matrix
    }

    /// The homogeneous matrix (a copy: Cairo values are passed by value). Exact. Upstream:
    /// `matrix`.
    #[inline(always)]
    fn matrix(self: Affine3<T>) -> Matrix4<T> {
        self.matrix
    }

    /// The homogeneous matrix. Exact. Upstream: `to_homogeneous`.
    #[inline(always)]
    fn to_homogeneous(self: Affine3<T>) -> Matrix4<T> {
        self.matrix
    }

    /// The inverse, or `None` when the linear block is singular: by blocks, `l = m[:3, :3]⁻¹`
    /// (`Matrix3::try_inverse`, its singularity criterion: a determinant EXACTLY zero), the
    /// translation `x = l * (-t)` refined once with the fused residual `r = -t - m[:3, :3] * x`
    /// (`x + l * r`), and the exact last row `(0, .., 0, 1)`, so the result is affine again.
    ///
    /// Upstream inverts the whole 4x4 matrix (`Matrix::try_inverse`, the same value in exact
    /// arithmetic). Measured candidates (`nalgebra_tests_geometry_transform`, `alt_*`): the
    /// whole-matrix inverse leaves `small` oracle cases outside their tolerance (the `1` of the
    /// last row keeps `Matrix4::try_inverse` from pre-scaling a small linear block, whose
    /// determinant then has few significant bits) and costs more; the plain block formula
    /// (without the refinement) multiplies the one-ulp roundings of `l` by `|t|` and leaves
    /// `medium` cases outside. The refined block formula passes every case. Upstream:
    /// `try_inverse`.
    #[inline(always)]
    fn try_inverse(self: Affine3<T>) -> Option<Affine3<T>> {
        match TransformKernels::affine_inverse3(self.matrix) {
            Option::Some(matrix) => Option::Some(Affine3 { matrix }),
            Option::None => Option::None,
        }
    }

    /// Inverts in place and returns `true`, or returns `false` and leaves `self` unchanged when
    /// it is not invertible (`try_inverse`, same rounding). Upstream: `try_inverse_mut`.
    #[inline(always)]
    fn try_inverse_mut(ref self: Affine3<T>) -> bool {
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
    fn inverse(self: Affine3<T>) -> Affine3<T> {
        Self::try_inverse(self).expect(errors::NOT_INVERTIBLE)
    }

    /// Inverts in place (`try_inverse`, same rounding). Like upstream (`let _ =
    /// self.matrix.try_inverse_mut()`), a singular transform is left unchanged WITHOUT panicking.
    /// Upstream: `inverse_mut` (projective and affine categories only).
    #[inline(always)]
    fn inverse_mut(ref self: Affine3<T>) {
        let _ = Self::try_inverse_mut(ref self);
    }

    /// `pt` transformed: `m[:3, :3] * pt + m[:3, 3]`, ONE fused sum of products per
    /// coordinate (the bits of upstream's product then sum). Like upstream (`TAffine` has no
    /// normalizer), the last row is not read. Upstream: `transform_point` (`Mul<Point>`).
    #[inline(always)]
    fn transform_point(self: Affine3<T>, pt: Point3<T>) -> Point3<T> {
        TransformKernels::affine_point3(self.matrix, pt)
    }

    /// `v` transformed: `m[:3, :3] * v`, ONE fused dot product per coordinate. Like upstream
    /// (`TAffine` has no normalizer), the last row is not read. Upstream: `transform_vector`
    /// (`Mul<SVector>`).
    #[inline(always)]
    fn transform_vector(self: Affine3<T>, v: Vector3<T>) -> Vector3<T> {
        TransformKernels::affine_vector3(self.matrix, v)
    }

    /// `self.inverse().transform_point(pt)` (upstream's formula: the inverse, then the
    /// transform). Panics like `inverse`. Upstream: `inverse_transform_point`.
    #[inline(always)]
    fn inverse_transform_point(self: Affine3<T>, pt: Point3<T>) -> Point3<T> {
        Self::transform_point(Self::inverse(self), pt)
    }

    /// `self.inverse().transform_vector(v)` (upstream's formula). Panics like `inverse`.
    /// Upstream: `inverse_transform_vector`.
    #[inline(always)]
    fn inverse_transform_vector(self: Affine3<T>, v: Vector3<T>) -> Vector3<T> {
        Self::transform_vector(Self::inverse(self), v)
    }

    /// `self * r`: `m * [[r, 0], [0, 1]]`, the first 3 columns one fused sum of products per
    /// entry, the last one copied (bit for bit the homogeneous product); a `Affine3` (upstream's
    /// `TCategoryMul<TAffine>` keeps the category). Upstream: `Mul<Rotation> for Transform`.
    #[inline(always)]
    fn mul_rotation(self: Affine3<T>, r: Rotation3<T>) -> Affine3<T> {
        Affine3 { matrix: TransformKernels::mul_rotation3(self.matrix, r.matrix) }
    }

    /// `self / r = self * r⁻¹` (the inverse rotation is the exact transpose), rounded like
    /// `mul_rotation`. Upstream: `Div<Rotation> for Transform`.
    #[inline(always)]
    fn div_rotation(self: Affine3<T>, r: Rotation3<T>) -> Affine3<T> {
        Affine3 { matrix: TransformKernels::mul_rotation3(self.matrix, r.inverse().matrix) }
    }

    /// `self * q`: the rotation matrix of `q` (upstream's `to_homogeneous` block,
    /// `UnitQuaternion::to_rotation_matrix`), then like `mul_rotation`. Upstream:
    /// `Mul<UnitQuaternion> for Transform`.
    #[inline(always)]
    fn mul_unit_quaternion(self: Affine3<T>, q: UnitQuaternion<T>) -> Affine3<T> {
        Affine3 {
            matrix: TransformKernels::mul_rotation3(self.matrix, q.to_rotation_matrix().matrix),
        }
    }

    /// `self / q = self * q⁻¹` (the conjugate, exact), rounded like `mul_unit_quaternion`.
    /// Upstream: `Div<UnitQuaternion> for Transform`.
    #[inline(always)]
    fn div_unit_quaternion(self: Affine3<T>, q: UnitQuaternion<T>) -> Affine3<T> {
        Affine3 {
            matrix: TransformKernels::mul_rotation3(
                self.matrix, q.inverse().to_rotation_matrix().matrix,
            ),
        }
    }

    /// `self * tr`: the translation `tr` followed by `self` (`Matrix4::prepend_translation`: ONE
    /// fused sum of products per entry of the last column, bit for bit the homogeneous product).
    /// Upstream: `Mul<Translation> for Transform`.
    #[inline(always)]
    fn mul_translation(self: Affine3<T>, tr: Translation3<T>) -> Affine3<T> {
        Affine3 { matrix: Matrix4CgTrait::prepend_translation(self.matrix, tr.vector) }
    }

    /// `self / tr = self * tr⁻¹` (the negated translation, exact), rounded like
    /// `mul_translation`. Upstream: `Div<Translation> for Transform`.
    #[inline(always)]
    fn div_translation(self: Affine3<T>, tr: Translation3<T>) -> Affine3<T> {
        Affine3 { matrix: Matrix4CgTrait::prepend_translation(self.matrix, tr.inverse().vector) }
    }

    /// `self * iso`: `m * iso.to_homogeneous()`, one fused sum of products per entry (the exact
    /// last row of the isometry skipped, bit for bit the homogeneous product). Upstream's generic
    /// `Mul<Isometry<T, R, D>>` also takes the rotation-matrix isometries: for an
    /// `IsometryMatrix3`, `self.mul_transform(Affine3::from(iso))` gives the same bits.
    /// Upstream: `Mul<Isometry> for Transform`.
    #[inline(always)]
    fn mul_isometry(self: Affine3<T>, iso: Isometry3<T>) -> Affine3<T> {
        Affine3 { matrix: TransformKernels::mul_affine3(self.matrix, iso.to_homogeneous()) }
    }

    /// `self * sim`: `m * sim.to_homogeneous()`, rounded like `mul_isometry` (for a
    /// `SimilarityMatrix3`: `self.mul_transform(Affine3::from(sim))`, same bits). Upstream:
    /// `Mul<Similarity> for Transform`.
    #[inline(always)]
    fn mul_similarity(self: Affine3<T>, sim: Similarity3<T>) -> Affine3<T> {
        Affine3 { matrix: TransformKernels::mul_affine3(self.matrix, sim.to_homogeneous()) }
    }

    /// `true` when every entry of the matrices is within `ulps` raw units. Upstream:
    /// `approx::AbsDiffEq::abs_diff_eq` (DESIGN D3).
    #[inline(always)]
    fn abs_diff_eq(self: Affine3<T>, other: Affine3<T>, ulps: u64) -> bool {
        Matrix4Trait::abs_diff_eq(self.matrix, other.matrix, ulps)
    }

    /// `relative_eq` of the matrices. Upstream: `approx::RelativeEq::relative_eq` (DESIGN D3).
    #[inline(always)]
    fn relative_eq(self: Affine3<T>, other: Affine3<T>, epsilon: u64, max_relative: T) -> bool {
        Matrix4Trait::relative_eq(self.matrix, other.matrix, epsilon, max_relative)
    }

    /// `ulps_eq` of the matrices. Upstream: `approx::UlpsEq::ulps_eq` (DESIGN D3).
    #[inline(always)]
    fn ulps_eq(self: Affine3<T>, other: Affine3<T>, epsilon: u64, max_ulps: u32) -> bool {
        Matrix4Trait::ulps_eq(self.matrix, other.matrix, epsilon, max_ulps)
    }
}

/// `a * b`: the full 4x4 product of the matrices (one fused sum of 4 products per entry),
/// a `Affine3` (upstream's `TCategoryMul<TAffine> for TAffine`). The other category pairs are
/// `TransformMul::mul_transform`. Upstream: `Mul<Transform> for Transform`.
pub impl Affine3Mul<
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
> of Mul<Affine3<T>> {
    #[inline(always)]
    fn mul(lhs: Affine3<T>, rhs: Affine3<T>) -> Affine3<T> {
        Affine3 { matrix: lhs.matrix * rhs.matrix }
    }
}

/// `a / b = a * b⁻¹`, a `Affine3`. Panics with `errors::NOT_INVERTIBLE` when `b` is singular.
/// The other category pairs are `TransformDiv::div_transform`. Upstream: `Div<Transform> for
/// Transform`.
pub impl Affine3Div<
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
> of Div<Affine3<T>> {
    #[inline(always)]
    fn div(lhs: Affine3<T>, rhs: Affine3<T>) -> Affine3<T> {
        Affine3 { matrix: lhs.matrix * rhs.inverse().matrix }
    }
}

/// `Default::default()`: the identity. Upstream: `Default for Transform`.
pub impl Affine3Default<
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
> of Default<Affine3<T>> {
    #[inline(always)]
    fn default() -> Affine3<T> {
        Affine3Trait::identity()
    }
}

/// `One::one()`: the identity; `is_one` compares with it exactly. Upstream: `num::One for
/// Transform`.
pub impl Affine3One<
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
> of One<Affine3<T>> {
    #[inline(always)]
    fn one() -> Affine3<T> {
        Affine3Trait::identity()
    }

    #[inline(always)]
    fn is_one(self: @Affine3<T>) -> bool {
        *self == Affine3Trait::identity()
    }

    #[inline(always)]
    fn is_non_one(self: @Affine3<T>) -> bool {
        !Self::is_one(self)
    }
}

/// `t[(i, j)]`: the entry of row `i` and column `j` of the homogeneous matrix. Panics with
/// `nalgebra: index out of bounds` for `i > 3` or `j > 3`. Upstream: `Index<(usize, usize)>
/// for Transform`.
pub impl Affine3Index<T, +Copy<T>, +Drop<T>> of Index<Affine3<T>, (usize, usize)> {
    type Target = T;

    #[inline(always)]
    fn index(ref self: Affine3<T>, index: (usize, usize)) -> T {
        let m = self.matrix;
        m[index]
    }
}

/// `t.into()`: the homogeneous matrix (`into_inner`). Exact. Upstream: `From<Transform> for
/// Matrix4` (`OMatrix`).
pub impl Matrix4FromAffine3<T> of Into<Affine3<T>, Matrix4<T>> {
    #[inline(always)]
    fn into(self: Affine3<T>) -> Matrix4<T> {
        self.matrix
    }
}

/// `x.try_into()`: `Some` when the last row is EXACTLY `(0, .., 0, 1)` and the linear block is
/// invertible (`Matrix3::is_invertible`), the same matrix; `None` otherwise (upstream's
/// `is_in_subset`, `TAffine::check_homogeneous_invariants`). Exact. Upstream: `SubsetOf<Matrix4>
/// for Affine3` (`nalgebra::try_convert`, `from_superset`).
pub impl Affine3TryFromMatrix4<
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
> of TryInto<Matrix4<T>, Affine3<T>> {
    fn try_into(self: Matrix4<T>) -> Option<Affine3<T>> {
        let m = self;
        if TransformKernels::is_affine3(m) {
            Option::Some(Affine3 { matrix: m })
        } else {
            Option::None
        }
    }
}

/// `x.try_into()`: `Some` when the last row is EXACTLY `(0, .., 0, 1)` and the linear block is
/// invertible (`Matrix3::is_invertible`), the same matrix; `None` otherwise (upstream's
/// `is_in_subset`, `TAffine::check_homogeneous_invariants`). Exact. Upstream: `SubsetOf<Transform3>
/// for Affine3` (`nalgebra::try_convert`, `from_superset`).
pub impl Affine3TryFromTransform3<
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
> of TryInto<Transform3<T>, Affine3<T>> {
    fn try_into(self: Transform3<T>) -> Option<Affine3<T>> {
        let m = self.into_inner();
        if TransformKernels::is_affine3(m) {
            Option::Some(Affine3 { matrix: m })
        } else {
            Option::None
        }
    }
}

/// `x.try_into()`: `Some` when the last row is EXACTLY `(0, .., 0, 1)` and the linear block is
/// invertible (`Matrix3::is_invertible`), the same matrix; `None` otherwise (upstream's
/// `is_in_subset`, `TAffine::check_homogeneous_invariants`). Exact. Upstream:
/// `SubsetOf<Projective3> for Affine3` (`nalgebra::try_convert`, `from_superset`).
pub impl Affine3TryFromProjective3<
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
> of TryInto<Projective3<T>, Affine3<T>> {
    fn try_into(self: Projective3<T>) -> Option<Affine3<T>> {
        let m = self.into_inner();
        if TransformKernels::is_affine3(m) {
            Option::Some(Affine3 { matrix: m })
        } else {
            Option::None
        }
    }
}

/// `g.into()`: the transform of homogeneous matrix `g.to_homogeneous()` (its rounding). Upstream:
/// `SubsetOf<Transform> for Rotation3` (`nalgebra::convert`, any category).
pub impl Affine3FromRotation3<
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
> of Into<Rotation3<T>, Affine3<T>> {
    #[inline(always)]
    fn into(self: Rotation3<T>) -> Affine3<T> {
        Affine3 { matrix: self.to_homogeneous() }
    }
}

/// `g.into()`: the transform of homogeneous matrix `g.to_homogeneous()` (its rounding). Upstream:
/// `SubsetOf<Transform> for UnitQuaternion` (`nalgebra::convert`, any category).
pub impl Affine3FromUnitQuaternion<
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
> of Into<UnitQuaternion<T>, Affine3<T>> {
    #[inline(always)]
    fn into(self: UnitQuaternion<T>) -> Affine3<T> {
        Affine3 { matrix: self.to_homogeneous() }
    }
}

/// `g.into()`: the transform of homogeneous matrix `g.to_homogeneous()` (its rounding). Upstream:
/// `SubsetOf<Transform> for Translation3` (`nalgebra::convert`, any category).
pub impl Affine3FromTranslation3<
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
> of Into<Translation3<T>, Affine3<T>> {
    #[inline(always)]
    fn into(self: Translation3<T>) -> Affine3<T> {
        Affine3 { matrix: self.to_homogeneous() }
    }
}

/// `g.into()`: the transform of homogeneous matrix `g.to_homogeneous()` (its rounding). Upstream:
/// `SubsetOf<Transform> for Isometry3` (`nalgebra::convert`, any category).
pub impl Affine3FromIsometry3<
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
> of Into<Isometry3<T>, Affine3<T>> {
    #[inline(always)]
    fn into(self: Isometry3<T>) -> Affine3<T> {
        Affine3 { matrix: self.to_homogeneous() }
    }
}

/// `g.into()`: the transform of homogeneous matrix `g.to_homogeneous()` (its rounding). Upstream:
/// `SubsetOf<Transform> for IsometryMatrix3` (`nalgebra::convert`, any category).
pub impl Affine3FromIsometryMatrix3<
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
> of Into<IsometryMatrix3<T>, Affine3<T>> {
    #[inline(always)]
    fn into(self: IsometryMatrix3<T>) -> Affine3<T> {
        Affine3 { matrix: self.to_homogeneous() }
    }
}

/// `g.into()`: the transform of homogeneous matrix `g.to_homogeneous()` (its rounding). Upstream:
/// `SubsetOf<Transform> for Similarity3` (`nalgebra::convert`, any category).
pub impl Affine3FromSimilarity3<
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
> of Into<Similarity3<T>, Affine3<T>> {
    #[inline(always)]
    fn into(self: Similarity3<T>) -> Affine3<T> {
        Affine3 { matrix: self.to_homogeneous() }
    }
}

/// `g.into()`: the transform of homogeneous matrix `g.to_homogeneous()` (its rounding). Upstream:
/// `SubsetOf<Transform> for SimilarityMatrix3` (`nalgebra::convert`, any category).
pub impl Affine3FromSimilarityMatrix3<
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
> of Into<SimilarityMatrix3<T>, Affine3<T>> {
    #[inline(always)]
    fn into(self: SimilarityMatrix3<T>) -> Affine3<T> {
        Affine3 { matrix: self.to_homogeneous() }
    }
}

/// `g.into()`: the transform of homogeneous matrix `g.to_homogeneous()` (its rounding). Upstream:
/// `SubsetOf<Transform> for Scale3` (`nalgebra::convert`, any category).
pub impl Affine3FromScale3<
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
> of Into<Scale3<T>, Affine3<T>> {
    #[inline(always)]
    fn into(self: Scale3<T>) -> Affine3<T> {
        Affine3 { matrix: self.to_homogeneous() }
    }
}

/// `g.into()`: the transform of homogeneous matrix `g.to_homogeneous()` (its rounding). Upstream:
/// `SubsetOf<Transform> for UnitDualQuaternion` (`nalgebra::convert`, any category).
pub impl Affine3FromUnitDualQuaternion<
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
> of Into<UnitDualQuaternion<T>, Affine3<T>> {
    #[inline(always)]
    fn into(self: UnitDualQuaternion<T>) -> Affine3<T> {
        Affine3 { matrix: self.to_homogeneous() }
    }
}
