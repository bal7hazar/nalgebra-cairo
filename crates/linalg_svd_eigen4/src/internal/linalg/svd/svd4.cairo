//! Internal, no stability promise: the crate-private items of `linalg::svd::svd4` that the packages
//! above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use core::internal::revoke_ap_tracking;
use nalgebra_core::base::matrix4::Matrix4;
use nalgebra_core::base::matrix_mul::MatrixMul;
use nalgebra_core::base::vector4::Vector4;
use simba::scalar::Real;
use crate::internal::linalg::svd::kernels::SvdRightImpl;
use crate::internal::linalg::symmetric_eigen4::Sym4;
use crate::linalg::svd::kernels::SvdComplete4Impl;
use crate::linalg::svd::svd4::{SortedSvd4, Svd4};

/// Crate-internal kernels of `Svd4<T>`: the Gram matrix, the right singular vectors, the sorted
/// columns `M v_i`, the decomposition.
#[generate_trait]
pub impl Svd4InternalImpl<
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
> of Svd4InternalTrait<T> {
    /// `m / max |m_ij|` (one prepared divisor per 6 entries), or `m` when it is zero: the input
    /// of the Gram matrix. The right singular vectors do not depend on the scale, but their
    /// fixed-point PRECISION does: the Gram matrix of a small matrix has tiny entries, whose
    /// absolute rounding is a large relative error on the eigenvectors (measured on `Svd4`:
    /// `pseudo_inverse` 1 749 ulp over the oracle tolerance on `small` inputs without it), and
    /// the Gram matrix of a large one overflows. Normalised, `MᵀM` has entries in `[0, 4]`.
    fn normalised(m: Matrix4<T>) -> Matrix4<T> {
        let mut a = m.m11.abs();
        if m.m21.abs() > a {
            a = m.m21.abs();
        }
        if m.m31.abs() > a {
            a = m.m31.abs();
        }
        if m.m41.abs() > a {
            a = m.m41.abs();
        }
        if m.m12.abs() > a {
            a = m.m12.abs();
        }
        if m.m22.abs() > a {
            a = m.m22.abs();
        }
        if m.m32.abs() > a {
            a = m.m32.abs();
        }
        if m.m42.abs() > a {
            a = m.m42.abs();
        }
        if m.m13.abs() > a {
            a = m.m13.abs();
        }
        if m.m23.abs() > a {
            a = m.m23.abs();
        }
        if m.m33.abs() > a {
            a = m.m33.abs();
        }
        if m.m43.abs() > a {
            a = m.m43.abs();
        }
        if m.m14.abs() > a {
            a = m.m14.abs();
        }
        if m.m24.abs() > a {
            a = m.m24.abs();
        }
        if m.m34.abs() > a {
            a = m.m34.abs();
        }
        if m.m44.abs() > a {
            a = m.m44.abs();
        }
        if a == R::zero() {
            return m;
        }
        let (q0, q1, q2, q3, q4, q5) = R::div6(m.m11, m.m21, m.m31, m.m41, m.m12, m.m22, a);
        let (q6, q7, q8, q9, q10, q11) = R::div6(m.m32, m.m42, m.m13, m.m23, m.m33, m.m43, a);
        let (q12, q13, q14, q15) = R::div4(m.m14, m.m24, m.m34, m.m44, a);
        Matrix4 {
            m11: q0,
            m21: q1,
            m31: q2,
            m41: q3,
            m12: q4,
            m22: q5,
            m32: q6,
            m42: q7,
            m13: q8,
            m23: q9,
            m33: q10,
            m43: q11,
            m14: q12,
            m24: q13,
            m34: q14,
            m44: q15,
        }
    }

    /// `MᵀM` (10 fused sums of 4 products, one rounding each).
    #[inline(always)]
    fn gram(m: Matrix4<T>) -> Sym4<T> {
        Sym4 {
            m11: R::sum_prod4(m.m11, m.m11, m.m21, m.m21, m.m31, m.m31, m.m41, m.m41),
            m12: R::sum_prod4(m.m11, m.m12, m.m21, m.m22, m.m31, m.m32, m.m41, m.m42),
            m13: R::sum_prod4(m.m11, m.m13, m.m21, m.m23, m.m31, m.m33, m.m41, m.m43),
            m14: R::sum_prod4(m.m11, m.m14, m.m21, m.m24, m.m31, m.m34, m.m41, m.m44),
            m22: R::sum_prod4(m.m12, m.m12, m.m22, m.m22, m.m32, m.m32, m.m42, m.m42),
            m23: R::sum_prod4(m.m12, m.m13, m.m22, m.m23, m.m32, m.m33, m.m42, m.m43),
            m24: R::sum_prod4(m.m12, m.m14, m.m22, m.m24, m.m32, m.m34, m.m42, m.m44),
            m33: R::sum_prod4(m.m13, m.m13, m.m23, m.m23, m.m33, m.m33, m.m43, m.m43),
            m34: R::sum_prod4(m.m13, m.m14, m.m23, m.m24, m.m33, m.m34, m.m43, m.m44),
            m44: R::sum_prod4(m.m14, m.m14, m.m24, m.m24, m.m34, m.m34, m.m44, m.m44),
        }
    }

    /// The right singular vectors (the columns of `v`, ascending eigenvalue order of `MᵀM`).
    #[inline(always)]
    fn right(m: Matrix4<T>) -> Matrix4<T> {
        SvdRightImpl::<T>::right4(Self::gram(Self::normalised(m)))
    }

    /// `right`, or `None` when the eigen decomposition of `MᵀM` did not converge within `eps`.
    #[inline(always)]
    fn try_right(m: Matrix4<T>, eps: T) -> Option<Matrix4<T>> {
        SvdRightImpl::<T>::try_right4(Self::gram(Self::normalised(m)), eps)
    }

    /// `w_i = M v_i` (`MatrixMul::mul_mat`: one fused sum per component) and `σ_i = |w_i|`
    /// (floored norm of the exact
    /// sum of squares) for the right vectors `v` (columns), the triples sorted DESCENDING by `σ`
    /// (stable odd-even transposition network, strict comparison: ties keep their order). The
    /// singular values alone stop here.
    fn sorted(m: Matrix4<T>, v: Matrix4<T>) -> SortedSvd4<T> {
        revoke_ap_tracking();
        let v0 = Vector4 { x: v.m11, y: v.m21, z: v.m31, w: v.m41 };
        let v1 = Vector4 { x: v.m12, y: v.m22, z: v.m32, w: v.m42 };
        let v2 = Vector4 { x: v.m13, y: v.m23, z: v.m33, w: v.m43 };
        let v3 = Vector4 { x: v.m14, y: v.m24, z: v.m34, w: v.m44 };
        let mv = m.mul_mat(v);
        let w0 = Vector4 { x: mv.m11, y: mv.m21, z: mv.m31, w: mv.m41 };
        let w1 = Vector4 { x: mv.m12, y: mv.m22, z: mv.m32, w: mv.m42 };
        let w2 = Vector4 { x: mv.m13, y: mv.m23, z: mv.m33, w: mv.m43 };
        let w3 = Vector4 { x: mv.m14, y: mv.m24, z: mv.m34, w: mv.m44 };
        let s0 = R::norm4(w0.x, w0.y, w0.z, w0.w);
        let s1 = R::norm4(w1.x, w1.y, w1.z, w1.w);
        let s2 = R::norm4(w2.x, w2.y, w2.z, w2.w);
        let s3 = R::norm4(w3.x, w3.y, w3.z, w3.w);
        let (mut s0, mut w0, mut v0) = (s0, w0, v0);
        let (mut s1, mut w1, mut v1) = (s1, w1, v1);
        let (mut s2, mut w2, mut v2) = (s2, w2, v2);
        let (mut s3, mut w3, mut v3) = (s3, w3, v3);
        if s1 > s0 {
            let tmp0 = s0;
            let tmp1 = w0;
            let tmp2 = v0;
            s0 = s1;
            w0 = w1;
            v0 = v1;
            s1 = tmp0;
            w1 = tmp1;
            v1 = tmp2;
        }
        if s3 > s2 {
            let tmp0 = s2;
            let tmp1 = w2;
            let tmp2 = v2;
            s2 = s3;
            w2 = w3;
            v2 = v3;
            s3 = tmp0;
            w3 = tmp1;
            v3 = tmp2;
        }
        if s2 > s1 {
            let tmp0 = s1;
            let tmp1 = w1;
            let tmp2 = v1;
            s1 = s2;
            w1 = w2;
            v1 = v2;
            s2 = tmp0;
            w2 = tmp1;
            v2 = tmp2;
        }
        if s1 > s0 {
            let tmp0 = s0;
            let tmp1 = w0;
            let tmp2 = v0;
            s0 = s1;
            w0 = w1;
            v0 = v1;
            s1 = tmp0;
            w1 = tmp1;
            v1 = tmp2;
        }
        if s3 > s2 {
            let tmp0 = s2;
            let tmp1 = w2;
            let tmp2 = v2;
            s2 = s3;
            w2 = w3;
            v2 = v3;
            s3 = tmp0;
            w3 = tmp1;
            v3 = tmp2;
        }
        if s2 > s1 {
            let tmp0 = s1;
            let tmp1 = w1;
            let tmp2 = v1;
            s1 = s2;
            w1 = w2;
            v1 = v2;
            s2 = tmp0;
            w2 = tmp1;
            v2 = tmp2;
        }
        SortedSvd4 { s0, s1, s2, s3, w0, w1, w2, w3, v0, v1, v2, v3 }
    }

    /// The decomposition from the right singular vectors `v` (columns): `sorted`, then, when
    /// `compute_u`, the left vectors by classical Gram-Schmidt run TWICE ("twice is enough",
    /// `SvdComplete4::gs*`): the second pass costs about as much as the first and keeps `U`
    /// orthonormal to the rounding of the residual even when `σ_k` is tiny (rank deficiency),
    /// where one pass leaves `u_k` as far from the others as `rounding / σ_k`. A column that
    /// vanishes EXACTLY (or `σ_1 = 0`) is completed by the axis least represented in the span of
    /// the previous ones. `compute_v` only decides whether `v_t` is kept (the right vectors are
    /// what the singular values are read off).
    fn from_right(m: Matrix4<T>, v: Matrix4<T>, compute_u: bool, compute_v: bool) -> Svd4<T> {
        let t = Self::sorted(m, v);
        let u = if compute_u {
            Some(Self::left(t))
        } else {
            None
        };
        let v_t = if compute_v {
            let v0 = t.v0;
            let v1 = t.v1;
            let v2 = t.v2;
            let v3 = t.v3;
            Some(
                Matrix4 {
                    m11: v0.x,
                    m21: v1.x,
                    m31: v2.x,
                    m41: v3.x,
                    m12: v0.y,
                    m22: v1.y,
                    m32: v2.y,
                    m42: v3.y,
                    m13: v0.z,
                    m23: v1.z,
                    m33: v2.z,
                    m43: v3.z,
                    m14: v0.w,
                    m24: v1.w,
                    m34: v2.w,
                    m44: v3.w,
                },
            )
        } else {
            None
        };
        Svd4 { u, singular_values: Vector4 { x: t.s0, y: t.s1, z: t.s2, w: t.s3 }, v_t }
    }

    /// The left singular vectors of `from_right`. Skipped (`compute_u = false`) they cost no
    /// Cairo step; the Sierra gas of the snapshots charges a branch at its costliest path
    /// whatever is executed (docs/BENCHMARK.md), a call boundary would only add its overhead.
    #[inline(always)]
    fn left(t: SortedSvd4<T>) -> Matrix4<T> {
        let (s0, w0) = (t.s0, t.w0);
        let w1 = t.w1;
        let w2 = t.w2;
        let w3 = t.w3;
        let u0 = SvdComplete4Impl::<T>::first(w0, s0);
        let u1 = SvdComplete4Impl::<T>::gs1(u0, w1);
        let u2 = SvdComplete4Impl::<T>::gs2(u0, u1, w2);
        let u3 = SvdComplete4Impl::<T>::gs3(u0, u1, u2, w3);
        Matrix4 {
            m11: u0.x,
            m21: u0.y,
            m31: u0.z,
            m41: u0.w,
            m12: u1.x,
            m22: u1.y,
            m32: u1.z,
            m42: u1.w,
            m13: u2.x,
            m23: u2.y,
            m33: u2.z,
            m43: u2.w,
            m14: u3.x,
            m24: u3.y,
            m34: u3.z,
            m44: u3.w,
        }
    }

    /// The singular values alone: `sorted`, without the left vectors.
    fn values_from_right(m: Matrix4<T>, v: Matrix4<T>) -> Vector4<T> {
        let t = Self::sorted(m, v);
        Vector4 { x: t.s0, y: t.s1, z: t.s2, w: t.s3 }
    }
}
