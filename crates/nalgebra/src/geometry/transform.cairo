//! Upstream's `Transform<T, C: TCategory, D>` (`geometry/transform*.rs`), WP 8.4-P11a: the shared
//! part of the six transform types, i.e. the panic messages, the generic operator traits whose
//! output category depends on BOTH operands, and the crate-internal kernels.
//!
//! **Design.** Upstream has one struct generic over the dimension `D` (2 or 3 through its
//! aliases) and a phantom category `C` (`TGeneral`, `TProjective`, `TAffine`), and six aliases:
//! `Transform2/3` (`TGeneral`), `Projective2/3` (`TProjective`), `Affine2/3` (`TAffine`). Cairo has
//! no const generics and a phantom category would not select the methods of a category, so each
//! alias is a Cairo struct of its own (`transform2.cairo` ... `affine3.cairo`), wrapping the
//! homogeneous `Matrix3` / `Matrix4` in a private `matrix` field (upstream's field is private
//! too: read it with `matrix` / `into_inner`, build with `from_matrix_unchecked`). Every type has
//! upstream's method names and semantics, and upstream's category rules are kept exactly:
//! - `try_inverse` / `try_inverse_mut` everywhere; `inverse`, `inverse_mut`,
//!   `inverse_transform_point`, `inverse_transform_vector` and the division BY a transform only
//!   for the projective and affine categories (upstream's `C: SubTCategoryOf<TProjective>`);
//! - the category of a product follows upstream's `TCategoryMul` (`TGeneral` absorbs everything,
//!   then `TProjective`; `TAffine * TAffine` is affine). Same-category products are the `*` / `/`
//!   operators; every pair of categories is `TransformMul::mul_transform` /
//!   `TransformDiv::div_transform`, which also carry the products of the other geometry types by
//!   a transform (`Isometry3 * Affine3`...). The products by a rotation, a translation, an
//!   isometry or a similarity keep the category of the transform (upstream's
//!   `TCategoryMul<TAffine>`) and are the `mul_<rhs>` / `div_<rhs>` methods of each type;
//! - `set_category` (`TransformSetCategory`) and `Into` widen a category without check
//!   (upstream's `SuperTCategoryOf`); `TryInto` narrows it with upstream's
//!   `check_homogeneous_invariants` (`is_in_subset`), from a transform or from a raw matrix;
//! - the affine category has no normalizer: its `transform_point` / `transform_vector` ignore the
//!   last row, like upstream's `TAffine::has_normalizer() == false`; the general and projective
//!   ones divide by the homogeneous coordinate (`Matrix3/4::transform_point`, upstream's formula).
//!
//! **Numerics** (AGENTS.md). Every sum of products is ONE fused `Real` kernel (one floor per
//! output scalar). The products by a rotation, a translation, an isometry or a similarity skip
//! the exact zeros and ones of its homogeneous matrix, which gives bit for bit the full product
//! for less gas; the products of two transforms are full matrix products, like upstream. The
//! affine inverse is computed by blocks (the linear block's inverse, then the translation), which
//! keeps the affine last row exact; the other categories invert the whole matrix
//! (`Matrix3/4::try_inverse`). Overflow panics.

use simba::scalar::Real;
use crate::base::cg::{Matrix3CgTrait, Matrix4CgTrait};
use crate::base::matrix2::{Matrix2, Matrix2Trait};
use crate::base::matrix3::{Matrix3, Matrix3Trait};
use crate::base::matrix4::Matrix4;
use crate::base::point2::Point2;
use crate::base::point3::Point3;
use crate::base::vector2::Vector2;
use crate::base::vector3::Vector3;
use super::affine2::{Affine2, Affine2Trait};
use super::affine3::{Affine3, Affine3Trait};
use super::isometry2::{Isometry2, Isometry2Trait};
use super::isometry3::{Isometry3, Isometry3Trait};
use super::isometry_matrix2::{IsometryMatrix2, IsometryMatrix2Trait};
use super::isometry_matrix3::{IsometryMatrix3, IsometryMatrix3Trait};
use super::projective2::{Projective2, Projective2Trait};
use super::projective3::{Projective3, Projective3Trait};
use super::rotation2::{Rotation2, Rotation2Trait};
use super::rotation3::{Rotation3, Rotation3Trait};
use super::similarity2::{Similarity2, Similarity2Trait};
use super::similarity3::{Similarity3, Similarity3Trait};
use super::similarity_matrix2::{SimilarityMatrix2, SimilarityMatrix2Trait};
use super::similarity_matrix3::{SimilarityMatrix3, SimilarityMatrix3Trait};
use super::transform2::{Transform2, Transform2Trait};
use super::transform3::{Transform3, Transform3Trait};
use super::translation2::{Translation2, Translation2Trait};
use super::translation3::{Translation3, Translation3Trait};
use super::unit_complex::{UnitComplex, UnitComplexTrait};
use super::unit_quaternion::{UnitQuaternion, UnitQuaternionTrait};

/// Panic messages of the transform types (stable API).
pub mod errors {
    /// `inverse`, `inverse_transform_*` and the divisions by a transform, on a singular transform
    /// (upstream: `Option::unwrap` of `try_inverse`'s `None`).
    pub const NOT_INVERTIBLE: felt252 = 'nalgebra: not invertible';
}

/// `lhs * rhs` when the category of the product depends on both operands (upstream's `Mul` impls
/// with `TCategoryMul`): transform by transform, and isometry / similarity / rotation /
/// translation / unit complex / unit quaternion by transform. Upstream: `Mul<Transform>`.
pub trait TransformMul<Lhs, Rhs> {
    /// The transform type of the product.
    type Output;
    /// `self * rhs`.
    fn mul_transform(self: Lhs, rhs: Rhs) -> Self::Output;
}

/// `lhs / rhs` when the category of the quotient depends on both operands. Upstream:
/// `Div<Transform>`.
pub trait TransformDiv<Lhs, Rhs> {
    /// The transform type of the quotient.
    type Output;
    /// `self / rhs`.
    fn div_transform(self: Lhs, rhs: Rhs) -> Self::Output;
}

/// `t.set_category()`: the same matrix in a super-category (the target type is inferred, e.g.
/// `let p: Projective3<Fixed> = a.set_category();`). Upstream: `Transform::set_category::<CNew>`
/// with `CNew: SuperTCategoryOf<C>`.
pub trait TransformSetCategory<From, To> {
    /// The same matrix, unchecked. Exact.
    fn set_category(self: From) -> To;
}

/// Crate-internal kernels of the transform types (methods of a generic impl, AGENTS.md rule 6).
#[generate_trait]
pub(crate) impl TransformKernelsImpl<
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
> of TransformKernels<T> {
    /// `m * [[r, 0], [0, 1]]` (upstream's `matrix * rotation.to_homogeneous()`): the first 2
    /// columns are ONE fused sum of 2 products each, the last column is copied. Bit-identical to
    /// the full 3x3 product (the omitted terms are exact zeros and ones).
    fn mul_rotation2(m: Matrix3<T>, r: Matrix2<T>) -> Matrix3<T> {
        Matrix3 {
            m11: R::sum_prod2(m.m11, r.m11, m.m12, r.m21),
            m21: R::sum_prod2(m.m21, r.m11, m.m22, r.m21),
            m31: R::sum_prod2(m.m31, r.m11, m.m32, r.m21),
            m12: R::sum_prod2(m.m11, r.m12, m.m12, r.m22),
            m22: R::sum_prod2(m.m21, r.m12, m.m22, r.m22),
            m32: R::sum_prod2(m.m31, r.m12, m.m32, r.m22),
            m13: m.m13,
            m23: m.m23,
            m33: m.m33,
        }
    }

    /// `[[r, 0], [0, 1]] * m` (upstream's `rotation.to_homogeneous() * matrix`): the first 2
    /// rows are ONE fused sum of 2 products each, the last row is copied. Bit-identical to the
    /// full 3x3 product.
    fn rotation_mul2(r: Matrix2<T>, m: Matrix3<T>) -> Matrix3<T> {
        Matrix3 {
            m11: R::sum_prod2(r.m11, m.m11, r.m12, m.m21),
            m21: R::sum_prod2(r.m21, m.m11, r.m22, m.m21),
            m31: m.m31,
            m12: R::sum_prod2(r.m11, m.m12, r.m12, m.m22),
            m22: R::sum_prod2(r.m21, m.m12, r.m22, m.m22),
            m32: m.m32,
            m13: R::sum_prod2(r.m11, m.m13, r.m12, m.m23),
            m23: R::sum_prod2(r.m21, m.m13, r.m22, m.m23),
            m33: m.m33,
        }
    }

    /// `m * h` for a homogeneous matrix `h` whose last row is exactly `(0, .., 0, 1)` (the
    /// `to_homogeneous` of an isometry or a similarity): ONE fused sum of products per entry, the
    /// last column `m[i, :2] . h[:2, 2] + m[i, 2]` (the addend is exact). Bit-identical to the
    /// full 3x3 product.
    fn mul_affine2(m: Matrix3<T>, h: Matrix3<T>) -> Matrix3<T> {
        Matrix3 {
            m11: R::sum_prod2(m.m11, h.m11, m.m12, h.m21),
            m21: R::sum_prod2(m.m21, h.m11, m.m22, h.m21),
            m31: R::sum_prod2(m.m31, h.m11, m.m32, h.m21),
            m12: R::sum_prod2(m.m11, h.m12, m.m12, h.m22),
            m22: R::sum_prod2(m.m21, h.m12, m.m22, h.m22),
            m32: R::sum_prod2(m.m31, h.m12, m.m32, h.m22),
            m13: R::wide_rescale(
                R::wide_add(
                    R::wide_add_prod(R::wide_add_prod(R::wide_zero(), m.m11, h.m13), m.m12, h.m23),
                    m.m13,
                ),
            ),
            m23: R::wide_rescale(
                R::wide_add(
                    R::wide_add_prod(R::wide_add_prod(R::wide_zero(), m.m21, h.m13), m.m22, h.m23),
                    m.m23,
                ),
            ),
            m33: R::wide_rescale(
                R::wide_add(
                    R::wide_add_prod(R::wide_add_prod(R::wide_zero(), m.m31, h.m13), m.m32, h.m23),
                    m.m33,
                ),
            ),
        }
    }

    /// `h * m` for a homogeneous matrix `h` whose last row is exactly `(0, .., 0, 1)`: the first
    /// 2 rows are ONE fused sum of 3 products each, the last row is copied. Bit-identical to the
    /// full 3x3 product.
    fn affine_mul2(h: Matrix3<T>, m: Matrix3<T>) -> Matrix3<T> {
        Matrix3 {
            m11: R::sum_prod3(h.m11, m.m11, h.m12, m.m21, h.m13, m.m31),
            m21: R::sum_prod3(h.m21, m.m11, h.m22, m.m21, h.m23, m.m31),
            m31: m.m31,
            m12: R::sum_prod3(h.m11, m.m12, h.m12, m.m22, h.m13, m.m32),
            m22: R::sum_prod3(h.m21, m.m12, h.m22, m.m22, h.m23, m.m32),
            m32: m.m32,
            m13: R::sum_prod3(h.m11, m.m13, h.m12, m.m23, h.m13, m.m33),
            m23: R::sum_prod3(h.m21, m.m13, h.m22, m.m23, h.m23, m.m33),
            m33: m.m33,
        }
    }

    /// `m[:2, :2] * p + m[:2, 2]`, ONE fused sum of products per coordinate (the bits of
    /// upstream's product then sum): the affine `transform_point`, which ignores the last row.
    fn affine_point2(m: Matrix3<T>, p: Point2<T>) -> Point2<T> {
        Point2 {
            x: R::wide_rescale(
                R::wide_add(
                    R::wide_add_prod(R::wide_add_prod(R::wide_zero(), m.m11, p.x), m.m12, p.y),
                    m.m13,
                ),
            ),
            y: R::wide_rescale(
                R::wide_add(
                    R::wide_add_prod(R::wide_add_prod(R::wide_zero(), m.m21, p.x), m.m22, p.y),
                    m.m23,
                ),
            ),
        }
    }

    /// `m[:2, :2] * v`, ONE fused dot product per coordinate: the affine `transform_vector`,
    /// which ignores the last row.
    fn affine_vector2(m: Matrix3<T>, v: Vector2<T>) -> Vector2<T> {
        Vector2 { x: R::sum_prod2(m.m11, v.x, m.m12, v.y), y: R::sum_prod2(m.m21, v.x, m.m22, v.y) }
    }

    /// The linear block `m[:2, :2]`.
    #[inline(always)]
    fn linear2(m: Matrix3<T>) -> Matrix2<T> {
        Matrix2 { m11: m.m11, m21: m.m21, m12: m.m12, m22: m.m22 }
    }

    /// The inverse of an affine homogeneous matrix by blocks: `l = m[:2, :2]⁻¹`
    /// (`Matrix2::try_inverse`, `None` when it is singular), the translation `l * (-m[:2, 2])`
    /// (ONE fused sum of products per coordinate, the negation exact), then the exact last row
    /// `(0, .., 0, 1)`. The last row of `m` is trusted to be `(0, .., 0, 1)` (the affine
    /// invariant).
    fn affine_inverse2(m: Matrix3<T>) -> Option<Matrix3<T>> {
        match Matrix2Trait::try_inverse(Self::linear2(m)) {
            Option::Some(l) => Option::Some(
                Matrix3 {
                    m11: l.m11,
                    m21: l.m21,
                    m31: R::zero(),
                    m12: l.m12,
                    m22: l.m22,
                    m32: R::zero(),
                    m13: R::sum_prod2(l.m11, -m.m13, l.m12, -m.m23),
                    m23: R::sum_prod2(l.m21, -m.m13, l.m22, -m.m23),
                    m33: R::one(),
                },
            ),
            Option::None => Option::None,
        }
    }

    /// Upstream's `TAffine::check_homogeneous_invariants`: the last row is EXACTLY `(0, .., 0,
    /// 1)` and the matrix is invertible (here: its linear block, `Matrix2::is_invertible`,
    /// equivalent for such a last row and consistent with `affine_inverse2`).
    fn is_affine2(m: Matrix3<T>) -> bool {
        m.m31 == R::zero()
            && m.m32 == R::zero()
            && m.m33 == R::one()
            && Matrix2Trait::is_invertible(Self::linear2(m))
    }

    /// `m * [[r, 0], [0, 1]]` (upstream's `matrix * rotation.to_homogeneous()`): the first 3
    /// columns are ONE fused sum of 3 products each, the last column is copied. Bit-identical to
    /// the full 4x4 product (the omitted terms are exact zeros and ones).
    fn mul_rotation3(m: Matrix4<T>, r: Matrix3<T>) -> Matrix4<T> {
        Matrix4 {
            m11: R::sum_prod3(m.m11, r.m11, m.m12, r.m21, m.m13, r.m31),
            m21: R::sum_prod3(m.m21, r.m11, m.m22, r.m21, m.m23, r.m31),
            m31: R::sum_prod3(m.m31, r.m11, m.m32, r.m21, m.m33, r.m31),
            m41: R::sum_prod3(m.m41, r.m11, m.m42, r.m21, m.m43, r.m31),
            m12: R::sum_prod3(m.m11, r.m12, m.m12, r.m22, m.m13, r.m32),
            m22: R::sum_prod3(m.m21, r.m12, m.m22, r.m22, m.m23, r.m32),
            m32: R::sum_prod3(m.m31, r.m12, m.m32, r.m22, m.m33, r.m32),
            m42: R::sum_prod3(m.m41, r.m12, m.m42, r.m22, m.m43, r.m32),
            m13: R::sum_prod3(m.m11, r.m13, m.m12, r.m23, m.m13, r.m33),
            m23: R::sum_prod3(m.m21, r.m13, m.m22, r.m23, m.m23, r.m33),
            m33: R::sum_prod3(m.m31, r.m13, m.m32, r.m23, m.m33, r.m33),
            m43: R::sum_prod3(m.m41, r.m13, m.m42, r.m23, m.m43, r.m33),
            m14: m.m14,
            m24: m.m24,
            m34: m.m34,
            m44: m.m44,
        }
    }

    /// `[[r, 0], [0, 1]] * m` (upstream's `rotation.to_homogeneous() * matrix`): the first 3
    /// rows are ONE fused sum of 3 products each, the last row is copied. Bit-identical to the
    /// full 4x4 product.
    fn rotation_mul3(r: Matrix3<T>, m: Matrix4<T>) -> Matrix4<T> {
        Matrix4 {
            m11: R::sum_prod3(r.m11, m.m11, r.m12, m.m21, r.m13, m.m31),
            m21: R::sum_prod3(r.m21, m.m11, r.m22, m.m21, r.m23, m.m31),
            m31: R::sum_prod3(r.m31, m.m11, r.m32, m.m21, r.m33, m.m31),
            m41: m.m41,
            m12: R::sum_prod3(r.m11, m.m12, r.m12, m.m22, r.m13, m.m32),
            m22: R::sum_prod3(r.m21, m.m12, r.m22, m.m22, r.m23, m.m32),
            m32: R::sum_prod3(r.m31, m.m12, r.m32, m.m22, r.m33, m.m32),
            m42: m.m42,
            m13: R::sum_prod3(r.m11, m.m13, r.m12, m.m23, r.m13, m.m33),
            m23: R::sum_prod3(r.m21, m.m13, r.m22, m.m23, r.m23, m.m33),
            m33: R::sum_prod3(r.m31, m.m13, r.m32, m.m23, r.m33, m.m33),
            m43: m.m43,
            m14: R::sum_prod3(r.m11, m.m14, r.m12, m.m24, r.m13, m.m34),
            m24: R::sum_prod3(r.m21, m.m14, r.m22, m.m24, r.m23, m.m34),
            m34: R::sum_prod3(r.m31, m.m14, r.m32, m.m24, r.m33, m.m34),
            m44: m.m44,
        }
    }

    /// `m * h` for a homogeneous matrix `h` whose last row is exactly `(0, .., 0, 1)` (the
    /// `to_homogeneous` of an isometry or a similarity): ONE fused sum of products per entry, the
    /// last column `m[i, :3] . h[:3, 3] + m[i, 3]` (the addend is exact). Bit-identical to the
    /// full 4x4 product.
    fn mul_affine3(m: Matrix4<T>, h: Matrix4<T>) -> Matrix4<T> {
        Matrix4 {
            m11: R::sum_prod3(m.m11, h.m11, m.m12, h.m21, m.m13, h.m31),
            m21: R::sum_prod3(m.m21, h.m11, m.m22, h.m21, m.m23, h.m31),
            m31: R::sum_prod3(m.m31, h.m11, m.m32, h.m21, m.m33, h.m31),
            m41: R::sum_prod3(m.m41, h.m11, m.m42, h.m21, m.m43, h.m31),
            m12: R::sum_prod3(m.m11, h.m12, m.m12, h.m22, m.m13, h.m32),
            m22: R::sum_prod3(m.m21, h.m12, m.m22, h.m22, m.m23, h.m32),
            m32: R::sum_prod3(m.m31, h.m12, m.m32, h.m22, m.m33, h.m32),
            m42: R::sum_prod3(m.m41, h.m12, m.m42, h.m22, m.m43, h.m32),
            m13: R::sum_prod3(m.m11, h.m13, m.m12, h.m23, m.m13, h.m33),
            m23: R::sum_prod3(m.m21, h.m13, m.m22, h.m23, m.m23, h.m33),
            m33: R::sum_prod3(m.m31, h.m13, m.m32, h.m23, m.m33, h.m33),
            m43: R::sum_prod3(m.m41, h.m13, m.m42, h.m23, m.m43, h.m33),
            m14: R::wide_rescale(
                R::wide_add(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(R::wide_zero(), m.m11, h.m14), m.m12, h.m24,
                        ),
                        m.m13,
                        h.m34,
                    ),
                    m.m14,
                ),
            ),
            m24: R::wide_rescale(
                R::wide_add(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(R::wide_zero(), m.m21, h.m14), m.m22, h.m24,
                        ),
                        m.m23,
                        h.m34,
                    ),
                    m.m24,
                ),
            ),
            m34: R::wide_rescale(
                R::wide_add(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(R::wide_zero(), m.m31, h.m14), m.m32, h.m24,
                        ),
                        m.m33,
                        h.m34,
                    ),
                    m.m34,
                ),
            ),
            m44: R::wide_rescale(
                R::wide_add(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(R::wide_zero(), m.m41, h.m14), m.m42, h.m24,
                        ),
                        m.m43,
                        h.m34,
                    ),
                    m.m44,
                ),
            ),
        }
    }

    /// `h * m` for a homogeneous matrix `h` whose last row is exactly `(0, .., 0, 1)`: the first
    /// 3 rows are ONE fused sum of 4 products each, the last row is copied. Bit-identical to the
    /// full 4x4 product.
    fn affine_mul3(h: Matrix4<T>, m: Matrix4<T>) -> Matrix4<T> {
        Matrix4 {
            m11: R::sum_prod4(h.m11, m.m11, h.m12, m.m21, h.m13, m.m31, h.m14, m.m41),
            m21: R::sum_prod4(h.m21, m.m11, h.m22, m.m21, h.m23, m.m31, h.m24, m.m41),
            m31: R::sum_prod4(h.m31, m.m11, h.m32, m.m21, h.m33, m.m31, h.m34, m.m41),
            m41: m.m41,
            m12: R::sum_prod4(h.m11, m.m12, h.m12, m.m22, h.m13, m.m32, h.m14, m.m42),
            m22: R::sum_prod4(h.m21, m.m12, h.m22, m.m22, h.m23, m.m32, h.m24, m.m42),
            m32: R::sum_prod4(h.m31, m.m12, h.m32, m.m22, h.m33, m.m32, h.m34, m.m42),
            m42: m.m42,
            m13: R::sum_prod4(h.m11, m.m13, h.m12, m.m23, h.m13, m.m33, h.m14, m.m43),
            m23: R::sum_prod4(h.m21, m.m13, h.m22, m.m23, h.m23, m.m33, h.m24, m.m43),
            m33: R::sum_prod4(h.m31, m.m13, h.m32, m.m23, h.m33, m.m33, h.m34, m.m43),
            m43: m.m43,
            m14: R::sum_prod4(h.m11, m.m14, h.m12, m.m24, h.m13, m.m34, h.m14, m.m44),
            m24: R::sum_prod4(h.m21, m.m14, h.m22, m.m24, h.m23, m.m34, h.m24, m.m44),
            m34: R::sum_prod4(h.m31, m.m14, h.m32, m.m24, h.m33, m.m34, h.m34, m.m44),
            m44: m.m44,
        }
    }

    /// `m[:3, :3] * p + m[:3, 3]`, ONE fused sum of products per coordinate (the bits of
    /// upstream's product then sum): the affine `transform_point`, which ignores the last row.
    fn affine_point3(m: Matrix4<T>, p: Point3<T>) -> Point3<T> {
        Point3 {
            x: R::wide_rescale(
                R::wide_add(
                    R::wide_add_prod(
                        R::wide_add_prod(R::wide_add_prod(R::wide_zero(), m.m11, p.x), m.m12, p.y),
                        m.m13,
                        p.z,
                    ),
                    m.m14,
                ),
            ),
            y: R::wide_rescale(
                R::wide_add(
                    R::wide_add_prod(
                        R::wide_add_prod(R::wide_add_prod(R::wide_zero(), m.m21, p.x), m.m22, p.y),
                        m.m23,
                        p.z,
                    ),
                    m.m24,
                ),
            ),
            z: R::wide_rescale(
                R::wide_add(
                    R::wide_add_prod(
                        R::wide_add_prod(R::wide_add_prod(R::wide_zero(), m.m31, p.x), m.m32, p.y),
                        m.m33,
                        p.z,
                    ),
                    m.m34,
                ),
            ),
        }
    }

    /// `m[:3, :3] * v`, ONE fused dot product per coordinate: the affine `transform_vector`,
    /// which ignores the last row.
    fn affine_vector3(m: Matrix4<T>, v: Vector3<T>) -> Vector3<T> {
        Vector3 {
            x: R::sum_prod3(m.m11, v.x, m.m12, v.y, m.m13, v.z),
            y: R::sum_prod3(m.m21, v.x, m.m22, v.y, m.m23, v.z),
            z: R::sum_prod3(m.m31, v.x, m.m32, v.y, m.m33, v.z),
        }
    }

    /// The linear block `m[:3, :3]`.
    #[inline(always)]
    fn linear3(m: Matrix4<T>) -> Matrix3<T> {
        Matrix3 {
            m11: m.m11,
            m21: m.m21,
            m31: m.m31,
            m12: m.m12,
            m22: m.m22,
            m32: m.m32,
            m13: m.m13,
            m23: m.m23,
            m33: m.m33,
        }
    }

    /// The inverse of an affine homogeneous matrix by blocks: `l = m[:3, :3]⁻¹`
    /// (`Matrix3::try_inverse`, `None` when it is singular), the translation `l * (-m[:3, 3])`
    /// (ONE fused sum of products per coordinate, the negation exact), then the exact last row
    /// `(0, .., 0, 1)`. The last row of `m` is trusted to be `(0, .., 0, 1)` (the affine
    /// invariant).
    fn affine_inverse3(m: Matrix4<T>) -> Option<Matrix4<T>> {
        match Matrix3Trait::try_inverse(Self::linear3(m)) {
            Option::Some(l) => Option::Some(
                Matrix4 {
                    m11: l.m11,
                    m21: l.m21,
                    m31: l.m31,
                    m41: R::zero(),
                    m12: l.m12,
                    m22: l.m22,
                    m32: l.m32,
                    m42: R::zero(),
                    m13: l.m13,
                    m23: l.m23,
                    m33: l.m33,
                    m43: R::zero(),
                    m14: R::sum_prod3(l.m11, -m.m14, l.m12, -m.m24, l.m13, -m.m34),
                    m24: R::sum_prod3(l.m21, -m.m14, l.m22, -m.m24, l.m23, -m.m34),
                    m34: R::sum_prod3(l.m31, -m.m14, l.m32, -m.m24, l.m33, -m.m34),
                    m44: R::one(),
                },
            ),
            Option::None => Option::None,
        }
    }

    /// Upstream's `TAffine::check_homogeneous_invariants`: the last row is EXACTLY `(0, .., 0,
    /// 1)` and the matrix is invertible (here: its linear block, `Matrix3::is_invertible`,
    /// equivalent for such a last row and consistent with `affine_inverse3`).
    fn is_affine3(m: Matrix4<T>) -> bool {
        m.m41 == R::zero()
            && m.m42 == R::zero()
            && m.m43 == R::zero()
            && m.m44 == R::one()
            && Matrix3Trait::is_invertible(Self::linear3(m))
    }
}

/// `a.mul_transform(b)`: `Transform2 * Transform2` is a `Transform2` (upstream's `TCategoryMul`:
/// `TGeneral *
/// TGeneral => TGeneral`), the full 3x3 product of the matrices (one fused sum of 3
/// products per entry). Upstream: `Mul<Transform> for Transform`.
pub impl Transform2MulTransform2<
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
> of TransformMul<Transform2<T>, Transform2<T>> {
    type Output = Transform2<T>;
    #[inline(always)]
    fn mul_transform(self: Transform2<T>, rhs: Transform2<T>) -> Transform2<T> {
        Transform2Trait::from_matrix_unchecked(self.into_inner() * rhs.into_inner())
    }
}

/// `a.mul_transform(b)`: `Transform2 * Projective2` is a `Transform2` (upstream's `TCategoryMul`:
/// `TGeneral *
/// TProjective => TGeneral`), the full 3x3 product of the matrices (one fused sum of 3
/// products per entry). Upstream: `Mul<Transform> for Transform`.
pub impl Transform2MulProjective2<
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
> of TransformMul<Transform2<T>, Projective2<T>> {
    type Output = Transform2<T>;
    #[inline(always)]
    fn mul_transform(self: Transform2<T>, rhs: Projective2<T>) -> Transform2<T> {
        Transform2Trait::from_matrix_unchecked(self.into_inner() * rhs.into_inner())
    }
}

/// `a.mul_transform(b)`: `Transform2 * Affine2` is a `Transform2` (upstream's `TCategoryMul`:
/// `TGeneral *
/// TAffine => TGeneral`), the full 3x3 product of the matrices (one fused sum of 3
/// products per entry). Upstream: `Mul<Transform> for Transform`.
pub impl Transform2MulAffine2<
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
> of TransformMul<Transform2<T>, Affine2<T>> {
    type Output = Transform2<T>;
    #[inline(always)]
    fn mul_transform(self: Transform2<T>, rhs: Affine2<T>) -> Transform2<T> {
        Transform2Trait::from_matrix_unchecked(self.into_inner() * rhs.into_inner())
    }
}

/// `a.mul_transform(b)`: `Projective2 * Transform2` is a `Transform2` (upstream's `TCategoryMul`:
/// `TProjective *
/// TGeneral => TGeneral`), the full 3x3 product of the matrices (one fused sum of 3
/// products per entry). Upstream: `Mul<Transform> for Transform`.
pub impl Projective2MulTransform2<
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
> of TransformMul<Projective2<T>, Transform2<T>> {
    type Output = Transform2<T>;
    #[inline(always)]
    fn mul_transform(self: Projective2<T>, rhs: Transform2<T>) -> Transform2<T> {
        Transform2Trait::from_matrix_unchecked(self.into_inner() * rhs.into_inner())
    }
}

/// `a.mul_transform(b)`: `Projective2 * Projective2` is a `Projective2` (upstream's `TCategoryMul`:
/// `TProjective *
/// TProjective => TProjective`), the full 3x3 product of the matrices (one fused sum of 3
/// products per entry). Upstream: `Mul<Transform> for Transform`.
pub impl Projective2MulProjective2<
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
> of TransformMul<Projective2<T>, Projective2<T>> {
    type Output = Projective2<T>;
    #[inline(always)]
    fn mul_transform(self: Projective2<T>, rhs: Projective2<T>) -> Projective2<T> {
        Projective2Trait::from_matrix_unchecked(self.into_inner() * rhs.into_inner())
    }
}

/// `a.mul_transform(b)`: `Projective2 * Affine2` is a `Projective2` (upstream's `TCategoryMul`:
/// `TProjective *
/// TAffine => TProjective`), the full 3x3 product of the matrices (one fused sum of 3
/// products per entry). Upstream: `Mul<Transform> for Transform`.
pub impl Projective2MulAffine2<
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
> of TransformMul<Projective2<T>, Affine2<T>> {
    type Output = Projective2<T>;
    #[inline(always)]
    fn mul_transform(self: Projective2<T>, rhs: Affine2<T>) -> Projective2<T> {
        Projective2Trait::from_matrix_unchecked(self.into_inner() * rhs.into_inner())
    }
}

/// `a.mul_transform(b)`: `Affine2 * Transform2` is a `Transform2` (upstream's `TCategoryMul`:
/// `TAffine *
/// TGeneral => TGeneral`), the full 3x3 product of the matrices (one fused sum of 3
/// products per entry). Upstream: `Mul<Transform> for Transform`.
pub impl Affine2MulTransform2<
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
> of TransformMul<Affine2<T>, Transform2<T>> {
    type Output = Transform2<T>;
    #[inline(always)]
    fn mul_transform(self: Affine2<T>, rhs: Transform2<T>) -> Transform2<T> {
        Transform2Trait::from_matrix_unchecked(self.into_inner() * rhs.into_inner())
    }
}

/// `a.mul_transform(b)`: `Affine2 * Projective2` is a `Projective2` (upstream's `TCategoryMul`:
/// `TAffine *
/// TProjective => TProjective`), the full 3x3 product of the matrices (one fused sum of 3
/// products per entry). Upstream: `Mul<Transform> for Transform`.
pub impl Affine2MulProjective2<
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
> of TransformMul<Affine2<T>, Projective2<T>> {
    type Output = Projective2<T>;
    #[inline(always)]
    fn mul_transform(self: Affine2<T>, rhs: Projective2<T>) -> Projective2<T> {
        Projective2Trait::from_matrix_unchecked(self.into_inner() * rhs.into_inner())
    }
}

/// `a.mul_transform(b)`: `Affine2 * Affine2` is a `Affine2` (upstream's `TCategoryMul`: `TAffine *
/// TAffine => TAffine`), the full 3x3 product of the matrices (one fused sum of 3
/// products per entry). Upstream: `Mul<Transform> for Transform`.
pub impl Affine2MulAffine2<
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
> of TransformMul<Affine2<T>, Affine2<T>> {
    type Output = Affine2<T>;
    #[inline(always)]
    fn mul_transform(self: Affine2<T>, rhs: Affine2<T>) -> Affine2<T> {
        Affine2Trait::from_matrix_unchecked(self.into_inner() * rhs.into_inner())
    }
}

/// `a.div_transform(b)`: `Transform2 * Projective2⁻¹`, a `Transform2` (upstream's
/// `TCategoryMul`; the divisor must be projective or affine). Panics with `errors::NOT_INVERTIBLE`
/// when `b` is singular. Upstream: `Div<Transform> for Transform` (`self * rhs.inverse()`).
pub impl Transform2DivProjective2<
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
> of TransformDiv<Transform2<T>, Projective2<T>> {
    type Output = Transform2<T>;
    #[inline(always)]
    fn div_transform(self: Transform2<T>, rhs: Projective2<T>) -> Transform2<T> {
        Transform2Trait::from_matrix_unchecked(self.into_inner() * rhs.inverse().into_inner())
    }
}

/// `a.div_transform(b)`: `Transform2 * Affine2⁻¹`, a `Transform2` (upstream's `TCategoryMul`;
/// the divisor must be projective or affine). Panics with `errors::NOT_INVERTIBLE` when `b` is
/// singular. Upstream: `Div<Transform> for Transform` (`self * rhs.inverse()`).
pub impl Transform2DivAffine2<
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
> of TransformDiv<Transform2<T>, Affine2<T>> {
    type Output = Transform2<T>;
    #[inline(always)]
    fn div_transform(self: Transform2<T>, rhs: Affine2<T>) -> Transform2<T> {
        Transform2Trait::from_matrix_unchecked(self.into_inner() * rhs.inverse().into_inner())
    }
}

/// `a.div_transform(b)`: `Projective2 * Projective2⁻¹`, a `Projective2` (upstream's
/// `TCategoryMul`; the divisor must be projective or affine). Panics with `errors::NOT_INVERTIBLE`
/// when `b` is singular. Upstream: `Div<Transform> for Transform` (`self * rhs.inverse()`).
pub impl Projective2DivProjective2<
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
> of TransformDiv<Projective2<T>, Projective2<T>> {
    type Output = Projective2<T>;
    #[inline(always)]
    fn div_transform(self: Projective2<T>, rhs: Projective2<T>) -> Projective2<T> {
        Projective2Trait::from_matrix_unchecked(self.into_inner() * rhs.inverse().into_inner())
    }
}

/// `a.div_transform(b)`: `Projective2 * Affine2⁻¹`, a `Projective2` (upstream's `TCategoryMul`;
/// the divisor must be projective or affine). Panics with `errors::NOT_INVERTIBLE` when `b` is
/// singular. Upstream: `Div<Transform> for Transform` (`self * rhs.inverse()`).
pub impl Projective2DivAffine2<
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
> of TransformDiv<Projective2<T>, Affine2<T>> {
    type Output = Projective2<T>;
    #[inline(always)]
    fn div_transform(self: Projective2<T>, rhs: Affine2<T>) -> Projective2<T> {
        Projective2Trait::from_matrix_unchecked(self.into_inner() * rhs.inverse().into_inner())
    }
}

/// `a.div_transform(b)`: `Affine2 * Projective2⁻¹`, a `Projective2` (upstream's `TCategoryMul`;
/// the divisor must be projective or affine). Panics with `errors::NOT_INVERTIBLE` when `b` is
/// singular. Upstream: `Div<Transform> for Transform` (`self * rhs.inverse()`).
pub impl Affine2DivProjective2<
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
> of TransformDiv<Affine2<T>, Projective2<T>> {
    type Output = Projective2<T>;
    #[inline(always)]
    fn div_transform(self: Affine2<T>, rhs: Projective2<T>) -> Projective2<T> {
        Projective2Trait::from_matrix_unchecked(self.into_inner() * rhs.inverse().into_inner())
    }
}

/// `a.div_transform(b)`: `Affine2 * Affine2⁻¹`, a `Affine2` (upstream's `TCategoryMul`; the
/// divisor must be projective or affine). Panics with `errors::NOT_INVERTIBLE` when `b` is
/// singular. Upstream: `Div<Transform> for Transform` (`self * rhs.inverse()`).
pub impl Affine2DivAffine2<
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
> of TransformDiv<Affine2<T>, Affine2<T>> {
    type Output = Affine2<T>;
    #[inline(always)]
    fn div_transform(self: Affine2<T>, rhs: Affine2<T>) -> Affine2<T> {
        Affine2Trait::from_matrix_unchecked(self.into_inner() * rhs.inverse().into_inner())
    }
}

/// `g.mul_transform(t)`: `[[r, 0], [0, 1]] * m`: one fused sum of products per entry of the first
/// rows; a `Transform2` (upstream's `TGeneral`, the category of `t`).
/// Upstream: `Mul<Transform> for Rotation`.
pub impl Rotation2MulTransform2<
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
> of TransformMul<Rotation2<T>, Transform2<T>> {
    type Output = Transform2<T>;
    #[inline(always)]
    fn mul_transform(self: Rotation2<T>, rhs: Transform2<T>) -> Transform2<T> {
        let m = rhs.into_inner();
        Transform2Trait::from_matrix_unchecked(TransformKernels::rotation_mul2(self.matrix, m))
    }
}

/// `g.mul_transform(t)`: `[[r, 0], [0, 1]] * m`: one fused sum of products per entry of the first
/// rows; a `Projective2` (upstream's `TProjective`, the category of `t`).
/// Upstream: `Mul<Transform> for Rotation`.
pub impl Rotation2MulProjective2<
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
> of TransformMul<Rotation2<T>, Projective2<T>> {
    type Output = Projective2<T>;
    #[inline(always)]
    fn mul_transform(self: Rotation2<T>, rhs: Projective2<T>) -> Projective2<T> {
        let m = rhs.into_inner();
        Projective2Trait::from_matrix_unchecked(TransformKernels::rotation_mul2(self.matrix, m))
    }
}

/// `g.mul_transform(t)`: `[[r, 0], [0, 1]] * m`: one fused sum of products per entry of the first
/// rows; a `Affine2` (upstream's `TAffine`, the category of `t`).
/// Upstream: `Mul<Transform> for Rotation`.
pub impl Rotation2MulAffine2<
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
> of TransformMul<Rotation2<T>, Affine2<T>> {
    type Output = Affine2<T>;
    #[inline(always)]
    fn mul_transform(self: Rotation2<T>, rhs: Affine2<T>) -> Affine2<T> {
        let m = rhs.into_inner();
        Affine2Trait::from_matrix_unchecked(TransformKernels::rotation_mul2(self.matrix, m))
    }
}

/// `g.mul_transform(t)`: the rotation matrix of `self` (its `to_homogeneous` block), then `[[r, 0],
/// [0, 1]] * m`; a `Transform2` (upstream's `TGeneral`, the category of `t`).
/// Upstream: `Mul<Transform> for UnitComplex`.
pub impl UnitComplexMulTransform2<
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
> of TransformMul<UnitComplex<T>, Transform2<T>> {
    type Output = Transform2<T>;
    #[inline(always)]
    fn mul_transform(self: UnitComplex<T>, rhs: Transform2<T>) -> Transform2<T> {
        let m = rhs.into_inner();
        Transform2Trait::from_matrix_unchecked(
            TransformKernels::rotation_mul2(self.to_rotation_matrix().matrix, m),
        )
    }
}

/// `g.mul_transform(t)`: the rotation matrix of `self` (its `to_homogeneous` block), then `[[r, 0],
/// [0, 1]] * m`; a `Projective2` (upstream's `TProjective`, the category of `t`).
/// Upstream: `Mul<Transform> for UnitComplex`.
pub impl UnitComplexMulProjective2<
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
> of TransformMul<UnitComplex<T>, Projective2<T>> {
    type Output = Projective2<T>;
    #[inline(always)]
    fn mul_transform(self: UnitComplex<T>, rhs: Projective2<T>) -> Projective2<T> {
        let m = rhs.into_inner();
        Projective2Trait::from_matrix_unchecked(
            TransformKernels::rotation_mul2(self.to_rotation_matrix().matrix, m),
        )
    }
}

/// `g.mul_transform(t)`: the rotation matrix of `self` (its `to_homogeneous` block), then `[[r, 0],
/// [0, 1]] * m`; a `Affine2` (upstream's `TAffine`, the category of `t`).
/// Upstream: `Mul<Transform> for UnitComplex`.
pub impl UnitComplexMulAffine2<
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
> of TransformMul<UnitComplex<T>, Affine2<T>> {
    type Output = Affine2<T>;
    #[inline(always)]
    fn mul_transform(self: UnitComplex<T>, rhs: Affine2<T>) -> Affine2<T> {
        let m = rhs.into_inner();
        Affine2Trait::from_matrix_unchecked(
            TransformKernels::rotation_mul2(self.to_rotation_matrix().matrix, m),
        )
    }
}

/// `g.mul_transform(t)`: `m` followed by the translation (`Matrix{n}::append_translation`: one
/// fused `mul_add` per entry of the first rows, bit for bit the homogeneous product); a
/// `Transform2` (upstream's `TGeneral`, the category of `t`).
/// Upstream: `Mul<Transform> for Translation`.
pub impl Translation2MulTransform2<
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
> of TransformMul<Translation2<T>, Transform2<T>> {
    type Output = Transform2<T>;
    #[inline(always)]
    fn mul_transform(self: Translation2<T>, rhs: Transform2<T>) -> Transform2<T> {
        let m = rhs.into_inner();
        Transform2Trait::from_matrix_unchecked(Matrix3CgTrait::append_translation(m, self.vector))
    }
}

/// `g.mul_transform(t)`: `m` followed by the translation (`Matrix{n}::append_translation`: one
/// fused `mul_add` per entry of the first rows, bit for bit the homogeneous product); a
/// `Projective2` (upstream's `TProjective`, the category of `t`).
/// Upstream: `Mul<Transform> for Translation`.
pub impl Translation2MulProjective2<
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
> of TransformMul<Translation2<T>, Projective2<T>> {
    type Output = Projective2<T>;
    #[inline(always)]
    fn mul_transform(self: Translation2<T>, rhs: Projective2<T>) -> Projective2<T> {
        let m = rhs.into_inner();
        Projective2Trait::from_matrix_unchecked(Matrix3CgTrait::append_translation(m, self.vector))
    }
}

/// `g.mul_transform(t)`: `m` followed by the translation (`Matrix{n}::append_translation`: one
/// fused `mul_add` per entry of the first rows, bit for bit the homogeneous product); a `Affine2`
/// (upstream's `TAffine`, the category of `t`).
/// Upstream: `Mul<Transform> for Translation`.
pub impl Translation2MulAffine2<
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
> of TransformMul<Translation2<T>, Affine2<T>> {
    type Output = Affine2<T>;
    #[inline(always)]
    fn mul_transform(self: Translation2<T>, rhs: Affine2<T>) -> Affine2<T> {
        let m = rhs.into_inner();
        Affine2Trait::from_matrix_unchecked(Matrix3CgTrait::append_translation(m, self.vector))
    }
}

/// `g.mul_transform(t)`: `self.to_homogeneous() * m` (fused, the exact last row of the isometry
/// skipped); a `Transform2` (upstream's `TGeneral`, the category of `t`).
/// Upstream: `Mul<Transform> for Isometry`.
pub impl Isometry2MulTransform2<
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
> of TransformMul<Isometry2<T>, Transform2<T>> {
    type Output = Transform2<T>;
    #[inline(always)]
    fn mul_transform(self: Isometry2<T>, rhs: Transform2<T>) -> Transform2<T> {
        let m = rhs.into_inner();
        Transform2Trait::from_matrix_unchecked(
            TransformKernels::affine_mul2(self.to_homogeneous(), m),
        )
    }
}

/// `g.mul_transform(t)`: `self.to_homogeneous() * m` (fused, the exact last row of the isometry
/// skipped); a `Projective2` (upstream's `TProjective`, the category of `t`).
/// Upstream: `Mul<Transform> for Isometry`.
pub impl Isometry2MulProjective2<
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
> of TransformMul<Isometry2<T>, Projective2<T>> {
    type Output = Projective2<T>;
    #[inline(always)]
    fn mul_transform(self: Isometry2<T>, rhs: Projective2<T>) -> Projective2<T> {
        let m = rhs.into_inner();
        Projective2Trait::from_matrix_unchecked(
            TransformKernels::affine_mul2(self.to_homogeneous(), m),
        )
    }
}

/// `g.mul_transform(t)`: `self.to_homogeneous() * m` (fused, the exact last row of the isometry
/// skipped); a `Affine2` (upstream's `TAffine`, the category of `t`).
/// Upstream: `Mul<Transform> for Isometry`.
pub impl Isometry2MulAffine2<
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
> of TransformMul<Isometry2<T>, Affine2<T>> {
    type Output = Affine2<T>;
    #[inline(always)]
    fn mul_transform(self: Isometry2<T>, rhs: Affine2<T>) -> Affine2<T> {
        let m = rhs.into_inner();
        Affine2Trait::from_matrix_unchecked(TransformKernels::affine_mul2(self.to_homogeneous(), m))
    }
}

/// `g.mul_transform(t)`: `self.to_homogeneous() * m` (fused, the exact last row of the isometry
/// skipped); a `Transform2` (upstream's `TGeneral`, the category of `t`).
/// Upstream: `Mul<Transform> for Isometry`.
pub impl IsometryMatrix2MulTransform2<
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
> of TransformMul<IsometryMatrix2<T>, Transform2<T>> {
    type Output = Transform2<T>;
    #[inline(always)]
    fn mul_transform(self: IsometryMatrix2<T>, rhs: Transform2<T>) -> Transform2<T> {
        let m = rhs.into_inner();
        Transform2Trait::from_matrix_unchecked(
            TransformKernels::affine_mul2(self.to_homogeneous(), m),
        )
    }
}

/// `g.mul_transform(t)`: `self.to_homogeneous() * m` (fused, the exact last row of the isometry
/// skipped); a `Projective2` (upstream's `TProjective`, the category of `t`).
/// Upstream: `Mul<Transform> for Isometry`.
pub impl IsometryMatrix2MulProjective2<
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
> of TransformMul<IsometryMatrix2<T>, Projective2<T>> {
    type Output = Projective2<T>;
    #[inline(always)]
    fn mul_transform(self: IsometryMatrix2<T>, rhs: Projective2<T>) -> Projective2<T> {
        let m = rhs.into_inner();
        Projective2Trait::from_matrix_unchecked(
            TransformKernels::affine_mul2(self.to_homogeneous(), m),
        )
    }
}

/// `g.mul_transform(t)`: `self.to_homogeneous() * m` (fused, the exact last row of the isometry
/// skipped); a `Affine2` (upstream's `TAffine`, the category of `t`).
/// Upstream: `Mul<Transform> for Isometry`.
pub impl IsometryMatrix2MulAffine2<
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
> of TransformMul<IsometryMatrix2<T>, Affine2<T>> {
    type Output = Affine2<T>;
    #[inline(always)]
    fn mul_transform(self: IsometryMatrix2<T>, rhs: Affine2<T>) -> Affine2<T> {
        let m = rhs.into_inner();
        Affine2Trait::from_matrix_unchecked(TransformKernels::affine_mul2(self.to_homogeneous(), m))
    }
}

/// `g.mul_transform(t)`: `self.to_homogeneous() * m` (fused, the exact last row of the similarity
/// skipped); a `Transform2` (upstream's `TGeneral`, the category of `t`).
/// Upstream: `Mul<Transform> for Similarity`.
pub impl Similarity2MulTransform2<
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
> of TransformMul<Similarity2<T>, Transform2<T>> {
    type Output = Transform2<T>;
    #[inline(always)]
    fn mul_transform(self: Similarity2<T>, rhs: Transform2<T>) -> Transform2<T> {
        let m = rhs.into_inner();
        Transform2Trait::from_matrix_unchecked(
            TransformKernels::affine_mul2(self.to_homogeneous(), m),
        )
    }
}

/// `g.mul_transform(t)`: `self.to_homogeneous() * m` (fused, the exact last row of the similarity
/// skipped); a `Projective2` (upstream's `TProjective`, the category of `t`).
/// Upstream: `Mul<Transform> for Similarity`.
pub impl Similarity2MulProjective2<
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
> of TransformMul<Similarity2<T>, Projective2<T>> {
    type Output = Projective2<T>;
    #[inline(always)]
    fn mul_transform(self: Similarity2<T>, rhs: Projective2<T>) -> Projective2<T> {
        let m = rhs.into_inner();
        Projective2Trait::from_matrix_unchecked(
            TransformKernels::affine_mul2(self.to_homogeneous(), m),
        )
    }
}

/// `g.mul_transform(t)`: `self.to_homogeneous() * m` (fused, the exact last row of the similarity
/// skipped); a `Affine2` (upstream's `TAffine`, the category of `t`).
/// Upstream: `Mul<Transform> for Similarity`.
pub impl Similarity2MulAffine2<
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
> of TransformMul<Similarity2<T>, Affine2<T>> {
    type Output = Affine2<T>;
    #[inline(always)]
    fn mul_transform(self: Similarity2<T>, rhs: Affine2<T>) -> Affine2<T> {
        let m = rhs.into_inner();
        Affine2Trait::from_matrix_unchecked(TransformKernels::affine_mul2(self.to_homogeneous(), m))
    }
}

/// `g.mul_transform(t)`: `self.to_homogeneous() * m` (fused, the exact last row of the similarity
/// skipped); a `Transform2` (upstream's `TGeneral`, the category of `t`).
/// Upstream: `Mul<Transform> for Similarity`.
pub impl SimilarityMatrix2MulTransform2<
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
> of TransformMul<SimilarityMatrix2<T>, Transform2<T>> {
    type Output = Transform2<T>;
    #[inline(always)]
    fn mul_transform(self: SimilarityMatrix2<T>, rhs: Transform2<T>) -> Transform2<T> {
        let m = rhs.into_inner();
        Transform2Trait::from_matrix_unchecked(
            TransformKernels::affine_mul2(self.to_homogeneous(), m),
        )
    }
}

/// `g.mul_transform(t)`: `self.to_homogeneous() * m` (fused, the exact last row of the similarity
/// skipped); a `Projective2` (upstream's `TProjective`, the category of `t`).
/// Upstream: `Mul<Transform> for Similarity`.
pub impl SimilarityMatrix2MulProjective2<
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
> of TransformMul<SimilarityMatrix2<T>, Projective2<T>> {
    type Output = Projective2<T>;
    #[inline(always)]
    fn mul_transform(self: SimilarityMatrix2<T>, rhs: Projective2<T>) -> Projective2<T> {
        let m = rhs.into_inner();
        Projective2Trait::from_matrix_unchecked(
            TransformKernels::affine_mul2(self.to_homogeneous(), m),
        )
    }
}

/// `g.mul_transform(t)`: `self.to_homogeneous() * m` (fused, the exact last row of the similarity
/// skipped); a `Affine2` (upstream's `TAffine`, the category of `t`).
/// Upstream: `Mul<Transform> for Similarity`.
pub impl SimilarityMatrix2MulAffine2<
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
> of TransformMul<SimilarityMatrix2<T>, Affine2<T>> {
    type Output = Affine2<T>;
    #[inline(always)]
    fn mul_transform(self: SimilarityMatrix2<T>, rhs: Affine2<T>) -> Affine2<T> {
        let m = rhs.into_inner();
        Affine2Trait::from_matrix_unchecked(TransformKernels::affine_mul2(self.to_homogeneous(), m))
    }
}

/// `g.div_transform(t)`: upstream's `self.inverse() * rhs` (sic: the inverse of the LEFT operand,
/// exact, times `t`, whatever its category, not `g * t⁻¹`); a `Transform2`. Upstream:
/// `Div<Transform>
/// for Rotation`.
pub impl Rotation2DivTransform2<
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
> of TransformDiv<Rotation2<T>, Transform2<T>> {
    type Output = Transform2<T>;
    #[inline(always)]
    fn div_transform(self: Rotation2<T>, rhs: Transform2<T>) -> Transform2<T> {
        let m = rhs.into_inner();
        Transform2Trait::from_matrix_unchecked(
            TransformKernels::rotation_mul2(self.inverse().matrix, m),
        )
    }
}

/// `g.div_transform(t)`: upstream's `self.inverse() * rhs` (sic: the inverse of the LEFT operand,
/// exact, times `t`, whatever its category, not `g * t⁻¹`); a `Projective2`. Upstream:
/// `Div<Transform>
/// for Rotation`.
pub impl Rotation2DivProjective2<
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
> of TransformDiv<Rotation2<T>, Projective2<T>> {
    type Output = Projective2<T>;
    #[inline(always)]
    fn div_transform(self: Rotation2<T>, rhs: Projective2<T>) -> Projective2<T> {
        let m = rhs.into_inner();
        Projective2Trait::from_matrix_unchecked(
            TransformKernels::rotation_mul2(self.inverse().matrix, m),
        )
    }
}

/// `g.div_transform(t)`: upstream's `self.inverse() * rhs` (sic: the inverse of the LEFT operand,
/// exact, times `t`, whatever its category, not `g * t⁻¹`); a `Affine2`. Upstream:
/// `Div<Transform>
/// for Rotation`.
pub impl Rotation2DivAffine2<
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
> of TransformDiv<Rotation2<T>, Affine2<T>> {
    type Output = Affine2<T>;
    #[inline(always)]
    fn div_transform(self: Rotation2<T>, rhs: Affine2<T>) -> Affine2<T> {
        let m = rhs.into_inner();
        Affine2Trait::from_matrix_unchecked(
            TransformKernels::rotation_mul2(self.inverse().matrix, m),
        )
    }
}

/// `g.div_transform(t)`: upstream's `self.inverse() * rhs` (sic: the inverse of the LEFT operand,
/// exact, times `t`, whatever its category, not `g * t⁻¹`); a `Transform2`. Upstream:
/// `Div<Transform>
/// for Translation`.
pub impl Translation2DivTransform2<
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
> of TransformDiv<Translation2<T>, Transform2<T>> {
    type Output = Transform2<T>;
    #[inline(always)]
    fn div_transform(self: Translation2<T>, rhs: Transform2<T>) -> Transform2<T> {
        let m = rhs.into_inner();
        Transform2Trait::from_matrix_unchecked(
            Matrix3CgTrait::append_translation(m, self.inverse().vector),
        )
    }
}

/// `g.div_transform(t)`: upstream's `self.inverse() * rhs` (sic: the inverse of the LEFT operand,
/// exact, times `t`, whatever its category, not `g * t⁻¹`); a `Projective2`. Upstream:
/// `Div<Transform>
/// for Translation`.
pub impl Translation2DivProjective2<
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
> of TransformDiv<Translation2<T>, Projective2<T>> {
    type Output = Projective2<T>;
    #[inline(always)]
    fn div_transform(self: Translation2<T>, rhs: Projective2<T>) -> Projective2<T> {
        let m = rhs.into_inner();
        Projective2Trait::from_matrix_unchecked(
            Matrix3CgTrait::append_translation(m, self.inverse().vector),
        )
    }
}

/// `g.div_transform(t)`: upstream's `self.inverse() * rhs` (sic: the inverse of the LEFT operand,
/// exact, times `t`, whatever its category, not `g * t⁻¹`); a `Affine2`. Upstream:
/// `Div<Transform>
/// for Translation`.
pub impl Translation2DivAffine2<
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
> of TransformDiv<Translation2<T>, Affine2<T>> {
    type Output = Affine2<T>;
    #[inline(always)]
    fn div_transform(self: Translation2<T>, rhs: Affine2<T>) -> Affine2<T> {
        let m = rhs.into_inner();
        Affine2Trait::from_matrix_unchecked(
            Matrix3CgTrait::append_translation(m, self.inverse().vector),
        )
    }
}

/// `t.set_category()` into a `Transform2`: the same matrix, unchecked (`TGeneral` is a
/// super-category of `TGeneral`). Exact. Upstream: `Transform::set_category`.
pub impl Transform2SetCategoryTransform2<
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
> of TransformSetCategory<Transform2<T>, Transform2<T>> {
    #[inline(always)]
    fn set_category(self: Transform2<T>) -> Transform2<T> {
        Transform2Trait::from_matrix_unchecked(self.into_inner())
    }
}

/// `t.set_category()` into a `Transform2`: the same matrix, unchecked (`TGeneral` is a
/// super-category of `TProjective`). Exact. Upstream: `Transform::set_category`.
pub impl Projective2SetCategoryTransform2<
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
> of TransformSetCategory<Projective2<T>, Transform2<T>> {
    #[inline(always)]
    fn set_category(self: Projective2<T>) -> Transform2<T> {
        Transform2Trait::from_matrix_unchecked(self.into_inner())
    }
}

/// `t.set_category()` into a `Projective2`: the same matrix, unchecked (`TProjective` is a
/// super-category of `TProjective`). Exact. Upstream: `Transform::set_category`.
pub impl Projective2SetCategoryProjective2<
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
> of TransformSetCategory<Projective2<T>, Projective2<T>> {
    #[inline(always)]
    fn set_category(self: Projective2<T>) -> Projective2<T> {
        Projective2Trait::from_matrix_unchecked(self.into_inner())
    }
}

/// `t.set_category()` into a `Transform2`: the same matrix, unchecked (`TGeneral` is a
/// super-category of `TAffine`). Exact. Upstream: `Transform::set_category`.
pub impl Affine2SetCategoryTransform2<
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
> of TransformSetCategory<Affine2<T>, Transform2<T>> {
    #[inline(always)]
    fn set_category(self: Affine2<T>) -> Transform2<T> {
        Transform2Trait::from_matrix_unchecked(self.into_inner())
    }
}

/// `t.set_category()` into a `Projective2`: the same matrix, unchecked (`TProjective` is a
/// super-category of `TAffine`). Exact. Upstream: `Transform::set_category`.
pub impl Affine2SetCategoryProjective2<
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
> of TransformSetCategory<Affine2<T>, Projective2<T>> {
    #[inline(always)]
    fn set_category(self: Affine2<T>) -> Projective2<T> {
        Projective2Trait::from_matrix_unchecked(self.into_inner())
    }
}

/// `t.set_category()` into a `Affine2`: the same matrix, unchecked (`TAffine` is a
/// super-category of `TAffine`). Exact. Upstream: `Transform::set_category`.
pub impl Affine2SetCategoryAffine2<
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
> of TransformSetCategory<Affine2<T>, Affine2<T>> {
    #[inline(always)]
    fn set_category(self: Affine2<T>) -> Affine2<T> {
        Affine2Trait::from_matrix_unchecked(self.into_inner())
    }
}

/// `a.mul_transform(b)`: `Transform3 * Transform3` is a `Transform3` (upstream's `TCategoryMul`:
/// `TGeneral *
/// TGeneral => TGeneral`), the full 4x4 product of the matrices (one fused sum of 4
/// products per entry). Upstream: `Mul<Transform> for Transform`.
pub impl Transform3MulTransform3<
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
> of TransformMul<Transform3<T>, Transform3<T>> {
    type Output = Transform3<T>;
    #[inline(always)]
    fn mul_transform(self: Transform3<T>, rhs: Transform3<T>) -> Transform3<T> {
        Transform3Trait::from_matrix_unchecked(self.into_inner() * rhs.into_inner())
    }
}

/// `a.mul_transform(b)`: `Transform3 * Projective3` is a `Transform3` (upstream's `TCategoryMul`:
/// `TGeneral *
/// TProjective => TGeneral`), the full 4x4 product of the matrices (one fused sum of 4
/// products per entry). Upstream: `Mul<Transform> for Transform`.
pub impl Transform3MulProjective3<
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
> of TransformMul<Transform3<T>, Projective3<T>> {
    type Output = Transform3<T>;
    #[inline(always)]
    fn mul_transform(self: Transform3<T>, rhs: Projective3<T>) -> Transform3<T> {
        Transform3Trait::from_matrix_unchecked(self.into_inner() * rhs.into_inner())
    }
}

/// `a.mul_transform(b)`: `Transform3 * Affine3` is a `Transform3` (upstream's `TCategoryMul`:
/// `TGeneral *
/// TAffine => TGeneral`), the full 4x4 product of the matrices (one fused sum of 4
/// products per entry). Upstream: `Mul<Transform> for Transform`.
pub impl Transform3MulAffine3<
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
> of TransformMul<Transform3<T>, Affine3<T>> {
    type Output = Transform3<T>;
    #[inline(always)]
    fn mul_transform(self: Transform3<T>, rhs: Affine3<T>) -> Transform3<T> {
        Transform3Trait::from_matrix_unchecked(self.into_inner() * rhs.into_inner())
    }
}

/// `a.mul_transform(b)`: `Projective3 * Transform3` is a `Transform3` (upstream's `TCategoryMul`:
/// `TProjective *
/// TGeneral => TGeneral`), the full 4x4 product of the matrices (one fused sum of 4
/// products per entry). Upstream: `Mul<Transform> for Transform`.
pub impl Projective3MulTransform3<
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
> of TransformMul<Projective3<T>, Transform3<T>> {
    type Output = Transform3<T>;
    #[inline(always)]
    fn mul_transform(self: Projective3<T>, rhs: Transform3<T>) -> Transform3<T> {
        Transform3Trait::from_matrix_unchecked(self.into_inner() * rhs.into_inner())
    }
}

/// `a.mul_transform(b)`: `Projective3 * Projective3` is a `Projective3` (upstream's `TCategoryMul`:
/// `TProjective *
/// TProjective => TProjective`), the full 4x4 product of the matrices (one fused sum of 4
/// products per entry). Upstream: `Mul<Transform> for Transform`.
pub impl Projective3MulProjective3<
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
> of TransformMul<Projective3<T>, Projective3<T>> {
    type Output = Projective3<T>;
    #[inline(always)]
    fn mul_transform(self: Projective3<T>, rhs: Projective3<T>) -> Projective3<T> {
        Projective3Trait::from_matrix_unchecked(self.into_inner() * rhs.into_inner())
    }
}

/// `a.mul_transform(b)`: `Projective3 * Affine3` is a `Projective3` (upstream's `TCategoryMul`:
/// `TProjective *
/// TAffine => TProjective`), the full 4x4 product of the matrices (one fused sum of 4
/// products per entry). Upstream: `Mul<Transform> for Transform`.
pub impl Projective3MulAffine3<
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
> of TransformMul<Projective3<T>, Affine3<T>> {
    type Output = Projective3<T>;
    #[inline(always)]
    fn mul_transform(self: Projective3<T>, rhs: Affine3<T>) -> Projective3<T> {
        Projective3Trait::from_matrix_unchecked(self.into_inner() * rhs.into_inner())
    }
}

/// `a.mul_transform(b)`: `Affine3 * Transform3` is a `Transform3` (upstream's `TCategoryMul`:
/// `TAffine *
/// TGeneral => TGeneral`), the full 4x4 product of the matrices (one fused sum of 4
/// products per entry). Upstream: `Mul<Transform> for Transform`.
pub impl Affine3MulTransform3<
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
> of TransformMul<Affine3<T>, Transform3<T>> {
    type Output = Transform3<T>;
    #[inline(always)]
    fn mul_transform(self: Affine3<T>, rhs: Transform3<T>) -> Transform3<T> {
        Transform3Trait::from_matrix_unchecked(self.into_inner() * rhs.into_inner())
    }
}

/// `a.mul_transform(b)`: `Affine3 * Projective3` is a `Projective3` (upstream's `TCategoryMul`:
/// `TAffine *
/// TProjective => TProjective`), the full 4x4 product of the matrices (one fused sum of 4
/// products per entry). Upstream: `Mul<Transform> for Transform`.
pub impl Affine3MulProjective3<
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
> of TransformMul<Affine3<T>, Projective3<T>> {
    type Output = Projective3<T>;
    #[inline(always)]
    fn mul_transform(self: Affine3<T>, rhs: Projective3<T>) -> Projective3<T> {
        Projective3Trait::from_matrix_unchecked(self.into_inner() * rhs.into_inner())
    }
}

/// `a.mul_transform(b)`: `Affine3 * Affine3` is a `Affine3` (upstream's `TCategoryMul`: `TAffine *
/// TAffine => TAffine`), the full 4x4 product of the matrices (one fused sum of 4
/// products per entry). Upstream: `Mul<Transform> for Transform`.
pub impl Affine3MulAffine3<
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
> of TransformMul<Affine3<T>, Affine3<T>> {
    type Output = Affine3<T>;
    #[inline(always)]
    fn mul_transform(self: Affine3<T>, rhs: Affine3<T>) -> Affine3<T> {
        Affine3Trait::from_matrix_unchecked(self.into_inner() * rhs.into_inner())
    }
}

/// `a.div_transform(b)`: `Transform3 * Projective3⁻¹`, a `Transform3` (upstream's
/// `TCategoryMul`; the divisor must be projective or affine). Panics with `errors::NOT_INVERTIBLE`
/// when `b` is singular. Upstream: `Div<Transform> for Transform` (`self * rhs.inverse()`).
pub impl Transform3DivProjective3<
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
> of TransformDiv<Transform3<T>, Projective3<T>> {
    type Output = Transform3<T>;
    #[inline(always)]
    fn div_transform(self: Transform3<T>, rhs: Projective3<T>) -> Transform3<T> {
        Transform3Trait::from_matrix_unchecked(self.into_inner() * rhs.inverse().into_inner())
    }
}

/// `a.div_transform(b)`: `Transform3 * Affine3⁻¹`, a `Transform3` (upstream's `TCategoryMul`;
/// the divisor must be projective or affine). Panics with `errors::NOT_INVERTIBLE` when `b` is
/// singular. Upstream: `Div<Transform> for Transform` (`self * rhs.inverse()`).
pub impl Transform3DivAffine3<
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
> of TransformDiv<Transform3<T>, Affine3<T>> {
    type Output = Transform3<T>;
    #[inline(always)]
    fn div_transform(self: Transform3<T>, rhs: Affine3<T>) -> Transform3<T> {
        Transform3Trait::from_matrix_unchecked(self.into_inner() * rhs.inverse().into_inner())
    }
}

/// `a.div_transform(b)`: `Projective3 * Projective3⁻¹`, a `Projective3` (upstream's
/// `TCategoryMul`; the divisor must be projective or affine). Panics with `errors::NOT_INVERTIBLE`
/// when `b` is singular. Upstream: `Div<Transform> for Transform` (`self * rhs.inverse()`).
pub impl Projective3DivProjective3<
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
> of TransformDiv<Projective3<T>, Projective3<T>> {
    type Output = Projective3<T>;
    #[inline(always)]
    fn div_transform(self: Projective3<T>, rhs: Projective3<T>) -> Projective3<T> {
        Projective3Trait::from_matrix_unchecked(self.into_inner() * rhs.inverse().into_inner())
    }
}

/// `a.div_transform(b)`: `Projective3 * Affine3⁻¹`, a `Projective3` (upstream's `TCategoryMul`;
/// the divisor must be projective or affine). Panics with `errors::NOT_INVERTIBLE` when `b` is
/// singular. Upstream: `Div<Transform> for Transform` (`self * rhs.inverse()`).
pub impl Projective3DivAffine3<
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
> of TransformDiv<Projective3<T>, Affine3<T>> {
    type Output = Projective3<T>;
    #[inline(always)]
    fn div_transform(self: Projective3<T>, rhs: Affine3<T>) -> Projective3<T> {
        Projective3Trait::from_matrix_unchecked(self.into_inner() * rhs.inverse().into_inner())
    }
}

/// `a.div_transform(b)`: `Affine3 * Projective3⁻¹`, a `Projective3` (upstream's `TCategoryMul`;
/// the divisor must be projective or affine). Panics with `errors::NOT_INVERTIBLE` when `b` is
/// singular. Upstream: `Div<Transform> for Transform` (`self * rhs.inverse()`).
pub impl Affine3DivProjective3<
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
> of TransformDiv<Affine3<T>, Projective3<T>> {
    type Output = Projective3<T>;
    #[inline(always)]
    fn div_transform(self: Affine3<T>, rhs: Projective3<T>) -> Projective3<T> {
        Projective3Trait::from_matrix_unchecked(self.into_inner() * rhs.inverse().into_inner())
    }
}

/// `a.div_transform(b)`: `Affine3 * Affine3⁻¹`, a `Affine3` (upstream's `TCategoryMul`; the
/// divisor must be projective or affine). Panics with `errors::NOT_INVERTIBLE` when `b` is
/// singular. Upstream: `Div<Transform> for Transform` (`self * rhs.inverse()`).
pub impl Affine3DivAffine3<
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
> of TransformDiv<Affine3<T>, Affine3<T>> {
    type Output = Affine3<T>;
    #[inline(always)]
    fn div_transform(self: Affine3<T>, rhs: Affine3<T>) -> Affine3<T> {
        Affine3Trait::from_matrix_unchecked(self.into_inner() * rhs.inverse().into_inner())
    }
}

/// `g.mul_transform(t)`: `[[r, 0], [0, 1]] * m`: one fused sum of products per entry of the first
/// rows; a `Transform3` (upstream's `TGeneral`, the category of `t`).
/// Upstream: `Mul<Transform> for Rotation`.
pub impl Rotation3MulTransform3<
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
> of TransformMul<Rotation3<T>, Transform3<T>> {
    type Output = Transform3<T>;
    #[inline(always)]
    fn mul_transform(self: Rotation3<T>, rhs: Transform3<T>) -> Transform3<T> {
        let m = rhs.into_inner();
        Transform3Trait::from_matrix_unchecked(TransformKernels::rotation_mul3(self.matrix, m))
    }
}

/// `g.mul_transform(t)`: `[[r, 0], [0, 1]] * m`: one fused sum of products per entry of the first
/// rows; a `Projective3` (upstream's `TProjective`, the category of `t`).
/// Upstream: `Mul<Transform> for Rotation`.
pub impl Rotation3MulProjective3<
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
> of TransformMul<Rotation3<T>, Projective3<T>> {
    type Output = Projective3<T>;
    #[inline(always)]
    fn mul_transform(self: Rotation3<T>, rhs: Projective3<T>) -> Projective3<T> {
        let m = rhs.into_inner();
        Projective3Trait::from_matrix_unchecked(TransformKernels::rotation_mul3(self.matrix, m))
    }
}

/// `g.mul_transform(t)`: `[[r, 0], [0, 1]] * m`: one fused sum of products per entry of the first
/// rows; a `Affine3` (upstream's `TAffine`, the category of `t`).
/// Upstream: `Mul<Transform> for Rotation`.
pub impl Rotation3MulAffine3<
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
> of TransformMul<Rotation3<T>, Affine3<T>> {
    type Output = Affine3<T>;
    #[inline(always)]
    fn mul_transform(self: Rotation3<T>, rhs: Affine3<T>) -> Affine3<T> {
        let m = rhs.into_inner();
        Affine3Trait::from_matrix_unchecked(TransformKernels::rotation_mul3(self.matrix, m))
    }
}

/// `g.mul_transform(t)`: the rotation matrix of `self` (its `to_homogeneous` block), then `[[r, 0],
/// [0, 1]] * m`; a `Transform3` (upstream's `TGeneral`, the category of `t`).
/// Upstream: `Mul<Transform> for UnitQuaternion`.
pub impl UnitQuaternionMulTransform3<
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
> of TransformMul<UnitQuaternion<T>, Transform3<T>> {
    type Output = Transform3<T>;
    #[inline(always)]
    fn mul_transform(self: UnitQuaternion<T>, rhs: Transform3<T>) -> Transform3<T> {
        let m = rhs.into_inner();
        Transform3Trait::from_matrix_unchecked(
            TransformKernels::rotation_mul3(self.to_rotation_matrix().matrix, m),
        )
    }
}

/// `g.mul_transform(t)`: the rotation matrix of `self` (its `to_homogeneous` block), then `[[r, 0],
/// [0, 1]] * m`; a `Projective3` (upstream's `TProjective`, the category of `t`).
/// Upstream: `Mul<Transform> for UnitQuaternion`.
pub impl UnitQuaternionMulProjective3<
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
> of TransformMul<UnitQuaternion<T>, Projective3<T>> {
    type Output = Projective3<T>;
    #[inline(always)]
    fn mul_transform(self: UnitQuaternion<T>, rhs: Projective3<T>) -> Projective3<T> {
        let m = rhs.into_inner();
        Projective3Trait::from_matrix_unchecked(
            TransformKernels::rotation_mul3(self.to_rotation_matrix().matrix, m),
        )
    }
}

/// `g.mul_transform(t)`: the rotation matrix of `self` (its `to_homogeneous` block), then `[[r, 0],
/// [0, 1]] * m`; a `Affine3` (upstream's `TAffine`, the category of `t`).
/// Upstream: `Mul<Transform> for UnitQuaternion`.
pub impl UnitQuaternionMulAffine3<
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
> of TransformMul<UnitQuaternion<T>, Affine3<T>> {
    type Output = Affine3<T>;
    #[inline(always)]
    fn mul_transform(self: UnitQuaternion<T>, rhs: Affine3<T>) -> Affine3<T> {
        let m = rhs.into_inner();
        Affine3Trait::from_matrix_unchecked(
            TransformKernels::rotation_mul3(self.to_rotation_matrix().matrix, m),
        )
    }
}

/// `g.mul_transform(t)`: `m` followed by the translation (`Matrix{n}::append_translation`: one
/// fused `mul_add` per entry of the first rows, bit for bit the homogeneous product); a
/// `Transform3` (upstream's `TGeneral`, the category of `t`).
/// Upstream: `Mul<Transform> for Translation`.
pub impl Translation3MulTransform3<
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
> of TransformMul<Translation3<T>, Transform3<T>> {
    type Output = Transform3<T>;
    #[inline(always)]
    fn mul_transform(self: Translation3<T>, rhs: Transform3<T>) -> Transform3<T> {
        let m = rhs.into_inner();
        Transform3Trait::from_matrix_unchecked(Matrix4CgTrait::append_translation(m, self.vector))
    }
}

/// `g.mul_transform(t)`: `m` followed by the translation (`Matrix{n}::append_translation`: one
/// fused `mul_add` per entry of the first rows, bit for bit the homogeneous product); a
/// `Projective3` (upstream's `TProjective`, the category of `t`).
/// Upstream: `Mul<Transform> for Translation`.
pub impl Translation3MulProjective3<
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
> of TransformMul<Translation3<T>, Projective3<T>> {
    type Output = Projective3<T>;
    #[inline(always)]
    fn mul_transform(self: Translation3<T>, rhs: Projective3<T>) -> Projective3<T> {
        let m = rhs.into_inner();
        Projective3Trait::from_matrix_unchecked(Matrix4CgTrait::append_translation(m, self.vector))
    }
}

/// `g.mul_transform(t)`: `m` followed by the translation (`Matrix{n}::append_translation`: one
/// fused `mul_add` per entry of the first rows, bit for bit the homogeneous product); a `Affine3`
/// (upstream's `TAffine`, the category of `t`).
/// Upstream: `Mul<Transform> for Translation`.
pub impl Translation3MulAffine3<
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
> of TransformMul<Translation3<T>, Affine3<T>> {
    type Output = Affine3<T>;
    #[inline(always)]
    fn mul_transform(self: Translation3<T>, rhs: Affine3<T>) -> Affine3<T> {
        let m = rhs.into_inner();
        Affine3Trait::from_matrix_unchecked(Matrix4CgTrait::append_translation(m, self.vector))
    }
}

/// `g.mul_transform(t)`: `self.to_homogeneous() * m` (fused, the exact last row of the isometry
/// skipped); a `Transform3` (upstream's `TGeneral`, the category of `t`).
/// Upstream: `Mul<Transform> for Isometry`.
pub impl Isometry3MulTransform3<
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
> of TransformMul<Isometry3<T>, Transform3<T>> {
    type Output = Transform3<T>;
    #[inline(always)]
    fn mul_transform(self: Isometry3<T>, rhs: Transform3<T>) -> Transform3<T> {
        let m = rhs.into_inner();
        Transform3Trait::from_matrix_unchecked(
            TransformKernels::affine_mul3(self.to_homogeneous(), m),
        )
    }
}

/// `g.mul_transform(t)`: `self.to_homogeneous() * m` (fused, the exact last row of the isometry
/// skipped); a `Projective3` (upstream's `TProjective`, the category of `t`).
/// Upstream: `Mul<Transform> for Isometry`.
pub impl Isometry3MulProjective3<
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
> of TransformMul<Isometry3<T>, Projective3<T>> {
    type Output = Projective3<T>;
    #[inline(always)]
    fn mul_transform(self: Isometry3<T>, rhs: Projective3<T>) -> Projective3<T> {
        let m = rhs.into_inner();
        Projective3Trait::from_matrix_unchecked(
            TransformKernels::affine_mul3(self.to_homogeneous(), m),
        )
    }
}

/// `g.mul_transform(t)`: `self.to_homogeneous() * m` (fused, the exact last row of the isometry
/// skipped); a `Affine3` (upstream's `TAffine`, the category of `t`).
/// Upstream: `Mul<Transform> for Isometry`.
pub impl Isometry3MulAffine3<
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
> of TransformMul<Isometry3<T>, Affine3<T>> {
    type Output = Affine3<T>;
    #[inline(always)]
    fn mul_transform(self: Isometry3<T>, rhs: Affine3<T>) -> Affine3<T> {
        let m = rhs.into_inner();
        Affine3Trait::from_matrix_unchecked(TransformKernels::affine_mul3(self.to_homogeneous(), m))
    }
}

/// `g.mul_transform(t)`: `self.to_homogeneous() * m` (fused, the exact last row of the isometry
/// skipped); a `Transform3` (upstream's `TGeneral`, the category of `t`).
/// Upstream: `Mul<Transform> for Isometry`.
pub impl IsometryMatrix3MulTransform3<
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
> of TransformMul<IsometryMatrix3<T>, Transform3<T>> {
    type Output = Transform3<T>;
    #[inline(always)]
    fn mul_transform(self: IsometryMatrix3<T>, rhs: Transform3<T>) -> Transform3<T> {
        let m = rhs.into_inner();
        Transform3Trait::from_matrix_unchecked(
            TransformKernels::affine_mul3(self.to_homogeneous(), m),
        )
    }
}

/// `g.mul_transform(t)`: `self.to_homogeneous() * m` (fused, the exact last row of the isometry
/// skipped); a `Projective3` (upstream's `TProjective`, the category of `t`).
/// Upstream: `Mul<Transform> for Isometry`.
pub impl IsometryMatrix3MulProjective3<
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
> of TransformMul<IsometryMatrix3<T>, Projective3<T>> {
    type Output = Projective3<T>;
    #[inline(always)]
    fn mul_transform(self: IsometryMatrix3<T>, rhs: Projective3<T>) -> Projective3<T> {
        let m = rhs.into_inner();
        Projective3Trait::from_matrix_unchecked(
            TransformKernels::affine_mul3(self.to_homogeneous(), m),
        )
    }
}

/// `g.mul_transform(t)`: `self.to_homogeneous() * m` (fused, the exact last row of the isometry
/// skipped); a `Affine3` (upstream's `TAffine`, the category of `t`).
/// Upstream: `Mul<Transform> for Isometry`.
pub impl IsometryMatrix3MulAffine3<
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
> of TransformMul<IsometryMatrix3<T>, Affine3<T>> {
    type Output = Affine3<T>;
    #[inline(always)]
    fn mul_transform(self: IsometryMatrix3<T>, rhs: Affine3<T>) -> Affine3<T> {
        let m = rhs.into_inner();
        Affine3Trait::from_matrix_unchecked(TransformKernels::affine_mul3(self.to_homogeneous(), m))
    }
}

/// `g.mul_transform(t)`: `self.to_homogeneous() * m` (fused, the exact last row of the similarity
/// skipped); a `Transform3` (upstream's `TGeneral`, the category of `t`).
/// Upstream: `Mul<Transform> for Similarity`.
pub impl Similarity3MulTransform3<
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
> of TransformMul<Similarity3<T>, Transform3<T>> {
    type Output = Transform3<T>;
    #[inline(always)]
    fn mul_transform(self: Similarity3<T>, rhs: Transform3<T>) -> Transform3<T> {
        let m = rhs.into_inner();
        Transform3Trait::from_matrix_unchecked(
            TransformKernels::affine_mul3(self.to_homogeneous(), m),
        )
    }
}

/// `g.mul_transform(t)`: `self.to_homogeneous() * m` (fused, the exact last row of the similarity
/// skipped); a `Projective3` (upstream's `TProjective`, the category of `t`).
/// Upstream: `Mul<Transform> for Similarity`.
pub impl Similarity3MulProjective3<
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
> of TransformMul<Similarity3<T>, Projective3<T>> {
    type Output = Projective3<T>;
    #[inline(always)]
    fn mul_transform(self: Similarity3<T>, rhs: Projective3<T>) -> Projective3<T> {
        let m = rhs.into_inner();
        Projective3Trait::from_matrix_unchecked(
            TransformKernels::affine_mul3(self.to_homogeneous(), m),
        )
    }
}

/// `g.mul_transform(t)`: `self.to_homogeneous() * m` (fused, the exact last row of the similarity
/// skipped); a `Affine3` (upstream's `TAffine`, the category of `t`).
/// Upstream: `Mul<Transform> for Similarity`.
pub impl Similarity3MulAffine3<
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
> of TransformMul<Similarity3<T>, Affine3<T>> {
    type Output = Affine3<T>;
    #[inline(always)]
    fn mul_transform(self: Similarity3<T>, rhs: Affine3<T>) -> Affine3<T> {
        let m = rhs.into_inner();
        Affine3Trait::from_matrix_unchecked(TransformKernels::affine_mul3(self.to_homogeneous(), m))
    }
}

/// `g.mul_transform(t)`: `self.to_homogeneous() * m` (fused, the exact last row of the similarity
/// skipped); a `Transform3` (upstream's `TGeneral`, the category of `t`).
/// Upstream: `Mul<Transform> for Similarity`.
pub impl SimilarityMatrix3MulTransform3<
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
> of TransformMul<SimilarityMatrix3<T>, Transform3<T>> {
    type Output = Transform3<T>;
    #[inline(always)]
    fn mul_transform(self: SimilarityMatrix3<T>, rhs: Transform3<T>) -> Transform3<T> {
        let m = rhs.into_inner();
        Transform3Trait::from_matrix_unchecked(
            TransformKernels::affine_mul3(self.to_homogeneous(), m),
        )
    }
}

/// `g.mul_transform(t)`: `self.to_homogeneous() * m` (fused, the exact last row of the similarity
/// skipped); a `Projective3` (upstream's `TProjective`, the category of `t`).
/// Upstream: `Mul<Transform> for Similarity`.
pub impl SimilarityMatrix3MulProjective3<
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
> of TransformMul<SimilarityMatrix3<T>, Projective3<T>> {
    type Output = Projective3<T>;
    #[inline(always)]
    fn mul_transform(self: SimilarityMatrix3<T>, rhs: Projective3<T>) -> Projective3<T> {
        let m = rhs.into_inner();
        Projective3Trait::from_matrix_unchecked(
            TransformKernels::affine_mul3(self.to_homogeneous(), m),
        )
    }
}

/// `g.mul_transform(t)`: `self.to_homogeneous() * m` (fused, the exact last row of the similarity
/// skipped); a `Affine3` (upstream's `TAffine`, the category of `t`).
/// Upstream: `Mul<Transform> for Similarity`.
pub impl SimilarityMatrix3MulAffine3<
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
> of TransformMul<SimilarityMatrix3<T>, Affine3<T>> {
    type Output = Affine3<T>;
    #[inline(always)]
    fn mul_transform(self: SimilarityMatrix3<T>, rhs: Affine3<T>) -> Affine3<T> {
        let m = rhs.into_inner();
        Affine3Trait::from_matrix_unchecked(TransformKernels::affine_mul3(self.to_homogeneous(), m))
    }
}

/// `g.div_transform(t)`: upstream's `self.inverse() * rhs` (sic: the inverse of the LEFT operand,
/// exact, times `t`, whatever its category, not `g * t⁻¹`); a `Transform3`. Upstream:
/// `Div<Transform>
/// for Rotation`.
pub impl Rotation3DivTransform3<
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
> of TransformDiv<Rotation3<T>, Transform3<T>> {
    type Output = Transform3<T>;
    #[inline(always)]
    fn div_transform(self: Rotation3<T>, rhs: Transform3<T>) -> Transform3<T> {
        let m = rhs.into_inner();
        Transform3Trait::from_matrix_unchecked(
            TransformKernels::rotation_mul3(self.inverse().matrix, m),
        )
    }
}

/// `g.div_transform(t)`: upstream's `self.inverse() * rhs` (sic: the inverse of the LEFT operand,
/// exact, times `t`, whatever its category, not `g * t⁻¹`); a `Projective3`. Upstream:
/// `Div<Transform>
/// for Rotation`.
pub impl Rotation3DivProjective3<
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
> of TransformDiv<Rotation3<T>, Projective3<T>> {
    type Output = Projective3<T>;
    #[inline(always)]
    fn div_transform(self: Rotation3<T>, rhs: Projective3<T>) -> Projective3<T> {
        let m = rhs.into_inner();
        Projective3Trait::from_matrix_unchecked(
            TransformKernels::rotation_mul3(self.inverse().matrix, m),
        )
    }
}

/// `g.div_transform(t)`: upstream's `self.inverse() * rhs` (sic: the inverse of the LEFT operand,
/// exact, times `t`, whatever its category, not `g * t⁻¹`); a `Affine3`. Upstream:
/// `Div<Transform>
/// for Rotation`.
pub impl Rotation3DivAffine3<
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
> of TransformDiv<Rotation3<T>, Affine3<T>> {
    type Output = Affine3<T>;
    #[inline(always)]
    fn div_transform(self: Rotation3<T>, rhs: Affine3<T>) -> Affine3<T> {
        let m = rhs.into_inner();
        Affine3Trait::from_matrix_unchecked(
            TransformKernels::rotation_mul3(self.inverse().matrix, m),
        )
    }
}

/// `g.div_transform(t)`: upstream's `self.inverse() * rhs` (sic: the inverse of the LEFT operand,
/// exact, times `t`, whatever its category, not `g * t⁻¹`); a `Transform3`. Upstream:
/// `Div<Transform>
/// for UnitQuaternion`.
pub impl UnitQuaternionDivTransform3<
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
> of TransformDiv<UnitQuaternion<T>, Transform3<T>> {
    type Output = Transform3<T>;
    #[inline(always)]
    fn div_transform(self: UnitQuaternion<T>, rhs: Transform3<T>) -> Transform3<T> {
        let m = rhs.into_inner();
        Transform3Trait::from_matrix_unchecked(
            TransformKernels::rotation_mul3(self.inverse().to_rotation_matrix().matrix, m),
        )
    }
}

/// `g.div_transform(t)`: upstream's `self.inverse() * rhs` (sic: the inverse of the LEFT operand,
/// exact, times `t`, whatever its category, not `g * t⁻¹`); a `Projective3`. Upstream:
/// `Div<Transform>
/// for UnitQuaternion`.
pub impl UnitQuaternionDivProjective3<
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
> of TransformDiv<UnitQuaternion<T>, Projective3<T>> {
    type Output = Projective3<T>;
    #[inline(always)]
    fn div_transform(self: UnitQuaternion<T>, rhs: Projective3<T>) -> Projective3<T> {
        let m = rhs.into_inner();
        Projective3Trait::from_matrix_unchecked(
            TransformKernels::rotation_mul3(self.inverse().to_rotation_matrix().matrix, m),
        )
    }
}

/// `g.div_transform(t)`: upstream's `self.inverse() * rhs` (sic: the inverse of the LEFT operand,
/// exact, times `t`, whatever its category, not `g * t⁻¹`); a `Affine3`. Upstream:
/// `Div<Transform>
/// for UnitQuaternion`.
pub impl UnitQuaternionDivAffine3<
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
> of TransformDiv<UnitQuaternion<T>, Affine3<T>> {
    type Output = Affine3<T>;
    #[inline(always)]
    fn div_transform(self: UnitQuaternion<T>, rhs: Affine3<T>) -> Affine3<T> {
        let m = rhs.into_inner();
        Affine3Trait::from_matrix_unchecked(
            TransformKernels::rotation_mul3(self.inverse().to_rotation_matrix().matrix, m),
        )
    }
}

/// `g.div_transform(t)`: upstream's `self.inverse() * rhs` (sic: the inverse of the LEFT operand,
/// exact, times `t`, whatever its category, not `g * t⁻¹`); a `Transform3`. Upstream:
/// `Div<Transform>
/// for Translation`.
pub impl Translation3DivTransform3<
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
> of TransformDiv<Translation3<T>, Transform3<T>> {
    type Output = Transform3<T>;
    #[inline(always)]
    fn div_transform(self: Translation3<T>, rhs: Transform3<T>) -> Transform3<T> {
        let m = rhs.into_inner();
        Transform3Trait::from_matrix_unchecked(
            Matrix4CgTrait::append_translation(m, self.inverse().vector),
        )
    }
}

/// `g.div_transform(t)`: upstream's `self.inverse() * rhs` (sic: the inverse of the LEFT operand,
/// exact, times `t`, whatever its category, not `g * t⁻¹`); a `Projective3`. Upstream:
/// `Div<Transform>
/// for Translation`.
pub impl Translation3DivProjective3<
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
> of TransformDiv<Translation3<T>, Projective3<T>> {
    type Output = Projective3<T>;
    #[inline(always)]
    fn div_transform(self: Translation3<T>, rhs: Projective3<T>) -> Projective3<T> {
        let m = rhs.into_inner();
        Projective3Trait::from_matrix_unchecked(
            Matrix4CgTrait::append_translation(m, self.inverse().vector),
        )
    }
}

/// `g.div_transform(t)`: upstream's `self.inverse() * rhs` (sic: the inverse of the LEFT operand,
/// exact, times `t`, whatever its category, not `g * t⁻¹`); a `Affine3`. Upstream:
/// `Div<Transform>
/// for Translation`.
pub impl Translation3DivAffine3<
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
> of TransformDiv<Translation3<T>, Affine3<T>> {
    type Output = Affine3<T>;
    #[inline(always)]
    fn div_transform(self: Translation3<T>, rhs: Affine3<T>) -> Affine3<T> {
        let m = rhs.into_inner();
        Affine3Trait::from_matrix_unchecked(
            Matrix4CgTrait::append_translation(m, self.inverse().vector),
        )
    }
}

/// `t.set_category()` into a `Transform3`: the same matrix, unchecked (`TGeneral` is a
/// super-category of `TGeneral`). Exact. Upstream: `Transform::set_category`.
pub impl Transform3SetCategoryTransform3<
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
> of TransformSetCategory<Transform3<T>, Transform3<T>> {
    #[inline(always)]
    fn set_category(self: Transform3<T>) -> Transform3<T> {
        Transform3Trait::from_matrix_unchecked(self.into_inner())
    }
}

/// `t.set_category()` into a `Transform3`: the same matrix, unchecked (`TGeneral` is a
/// super-category of `TProjective`). Exact. Upstream: `Transform::set_category`.
pub impl Projective3SetCategoryTransform3<
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
> of TransformSetCategory<Projective3<T>, Transform3<T>> {
    #[inline(always)]
    fn set_category(self: Projective3<T>) -> Transform3<T> {
        Transform3Trait::from_matrix_unchecked(self.into_inner())
    }
}

/// `t.set_category()` into a `Projective3`: the same matrix, unchecked (`TProjective` is a
/// super-category of `TProjective`). Exact. Upstream: `Transform::set_category`.
pub impl Projective3SetCategoryProjective3<
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
> of TransformSetCategory<Projective3<T>, Projective3<T>> {
    #[inline(always)]
    fn set_category(self: Projective3<T>) -> Projective3<T> {
        Projective3Trait::from_matrix_unchecked(self.into_inner())
    }
}

/// `t.set_category()` into a `Transform3`: the same matrix, unchecked (`TGeneral` is a
/// super-category of `TAffine`). Exact. Upstream: `Transform::set_category`.
pub impl Affine3SetCategoryTransform3<
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
> of TransformSetCategory<Affine3<T>, Transform3<T>> {
    #[inline(always)]
    fn set_category(self: Affine3<T>) -> Transform3<T> {
        Transform3Trait::from_matrix_unchecked(self.into_inner())
    }
}

/// `t.set_category()` into a `Projective3`: the same matrix, unchecked (`TProjective` is a
/// super-category of `TAffine`). Exact. Upstream: `Transform::set_category`.
pub impl Affine3SetCategoryProjective3<
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
> of TransformSetCategory<Affine3<T>, Projective3<T>> {
    #[inline(always)]
    fn set_category(self: Affine3<T>) -> Projective3<T> {
        Projective3Trait::from_matrix_unchecked(self.into_inner())
    }
}

/// `t.set_category()` into a `Affine3`: the same matrix, unchecked (`TAffine` is a
/// super-category of `TAffine`). Exact. Upstream: `Transform::set_category`.
pub impl Affine3SetCategoryAffine3<
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
> of TransformSetCategory<Affine3<T>, Affine3<T>> {
    #[inline(always)]
    fn set_category(self: Affine3<T>) -> Affine3<T> {
        Affine3Trait::from_matrix_unchecked(self.into_inner())
    }
}
