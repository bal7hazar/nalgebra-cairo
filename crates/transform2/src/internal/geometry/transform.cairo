//! Internal, no stability promise: the crate-private items of `geometry::transform` that the
//! packages above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use nalgebra_static2::base::matrix2::Matrix2Trait;
use nalgebra_static3::base::matrix3::Matrix3Trait;
use nalgebra_types2::base::matrix2::Matrix2;
use nalgebra_types2::base::point2::Point2;
use nalgebra_types2::base::vector2::Vector2;
use nalgebra_types3::base::matrix3::Matrix3;
use nalgebra_types3::base::point3::Point3;
use nalgebra_types3::base::vector3::Vector3;
use nalgebra_types4::base::matrix4::Matrix4;
use simba::scalar::Real;

/// Crate-internal kernels of the transform types (methods of a generic impl, AGENTS.md rule 6).
#[generate_trait]
pub impl TransformKernelsImpl<
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

    /// The inverse of an affine homogeneous matrix by blocks, with one step of iterative
    /// refinement of the translation: `l = m[:2, :2]⁻¹` (`Matrix2::try_inverse`, `None` when
    /// it is singular), `x = l * (-t)`, the residual `r = -t - m[:2, :2] * x` (ONE fused sum of
    /// products per coordinate: the floor of the exact residual), then `x + l * r` (one fused sum
    /// per coordinate). The refinement removes the one-ulp roundings of `l` multiplied by `|t|`;
    /// the last row is the exact `(0, .., 0, 1)` (the affine invariant, trusted on input).
    fn affine_inverse2(m: Matrix3<T>) -> Option<Matrix3<T>> {
        let lin = Self::linear2(m);
        let l = Matrix2Trait::try_inverse(lin)?;
        let xx = R::sum_prod2(l.m11, -m.m13, l.m12, -m.m23);
        let xy = R::sum_prod2(l.m21, -m.m13, l.m22, -m.m23);
        let rx = R::wide_rescale(
            R::wide_sub(
                R::wide_sub_prod(R::wide_sub_prod(R::wide_zero(), lin.m11, xx), lin.m12, xy), m.m13,
            ),
        );
        let ry = R::wide_rescale(
            R::wide_sub(
                R::wide_sub_prod(R::wide_sub_prod(R::wide_zero(), lin.m21, xx), lin.m22, xy), m.m23,
            ),
        );
        Option::Some(
            Matrix3 {
                m11: l.m11,
                m21: l.m21,
                m31: R::zero(),
                m12: l.m12,
                m22: l.m22,
                m32: R::zero(),
                m13: R::wide_rescale(
                    R::wide_add(
                        R::wide_add_prod(R::wide_add_prod(R::wide_zero(), l.m11, rx), l.m12, ry),
                        xx,
                    ),
                ),
                m23: R::wide_rescale(
                    R::wide_add(
                        R::wide_add_prod(R::wide_add_prod(R::wide_zero(), l.m21, rx), l.m22, ry),
                        xy,
                    ),
                ),
                m33: R::one(),
            },
        )
    }

    /// Upstream's `TAffine::check_homogeneous_invariants`: the last row is EXACTLY `(0, .., 0,
    /// 1)` and the matrix is invertible (here: its linear block, `Matrix2::is_invertible`,
    /// equivalent for such a last row, and the singularity criterion of `affine_inverse2`).
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

    /// The inverse of an affine homogeneous matrix by blocks, with one step of iterative
    /// refinement of the translation: `l = m[:3, :3]⁻¹` (`Matrix3::try_inverse`, `None` when
    /// it is singular), `x = l * (-t)`, the residual `r = -t - m[:3, :3] * x` (ONE fused sum of
    /// products per coordinate: the floor of the exact residual), then `x + l * r` (one fused sum
    /// per coordinate). The refinement removes the one-ulp roundings of `l` multiplied by `|t|`;
    /// the last row is the exact `(0, .., 0, 1)` (the affine invariant, trusted on input).
    fn affine_inverse3(m: Matrix4<T>) -> Option<Matrix4<T>> {
        let lin = Self::linear3(m);
        let l = Matrix3Trait::try_inverse(lin)?;
        let xx = R::sum_prod3(l.m11, -m.m14, l.m12, -m.m24, l.m13, -m.m34);
        let xy = R::sum_prod3(l.m21, -m.m14, l.m22, -m.m24, l.m23, -m.m34);
        let xz = R::sum_prod3(l.m31, -m.m14, l.m32, -m.m24, l.m33, -m.m34);
        let rx = R::wide_rescale(
            R::wide_sub(
                R::wide_sub_prod(
                    R::wide_sub_prod(R::wide_sub_prod(R::wide_zero(), lin.m11, xx), lin.m12, xy),
                    lin.m13,
                    xz,
                ),
                m.m14,
            ),
        );
        let ry = R::wide_rescale(
            R::wide_sub(
                R::wide_sub_prod(
                    R::wide_sub_prod(R::wide_sub_prod(R::wide_zero(), lin.m21, xx), lin.m22, xy),
                    lin.m23,
                    xz,
                ),
                m.m24,
            ),
        );
        let rz = R::wide_rescale(
            R::wide_sub(
                R::wide_sub_prod(
                    R::wide_sub_prod(R::wide_sub_prod(R::wide_zero(), lin.m31, xx), lin.m32, xy),
                    lin.m33,
                    xz,
                ),
                m.m34,
            ),
        );
        Option::Some(
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
                m14: R::wide_rescale(
                    R::wide_add(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(R::wide_zero(), l.m11, rx), l.m12, ry,
                            ),
                            l.m13,
                            rz,
                        ),
                        xx,
                    ),
                ),
                m24: R::wide_rescale(
                    R::wide_add(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(R::wide_zero(), l.m21, rx), l.m22, ry,
                            ),
                            l.m23,
                            rz,
                        ),
                        xy,
                    ),
                ),
                m34: R::wide_rescale(
                    R::wide_add(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(R::wide_zero(), l.m31, rx), l.m32, ry,
                            ),
                            l.m33,
                            rz,
                        ),
                        xz,
                    ),
                ),
                m44: R::one(),
            },
        )
    }

    /// Upstream's `TAffine::check_homogeneous_invariants`: the last row is EXACTLY `(0, .., 0,
    /// 1)` and the matrix is invertible (here: its linear block, `Matrix3::is_invertible`,
    /// equivalent for such a last row, and the singularity criterion of `affine_inverse3`).
    fn is_affine3(m: Matrix4<T>) -> bool {
        m.m41 == R::zero()
            && m.m42 == R::zero()
            && m.m43 == R::zero()
            && m.m44 == R::one()
            && Matrix3Trait::is_invertible(Self::linear3(m))
    }
}
