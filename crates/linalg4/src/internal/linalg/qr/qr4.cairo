//! Internal, no stability promise: the crate-private items of `linalg::qr::qr4` that the packages
//! above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use nalgebra_core::base::matrix4::Matrix4;
use nalgebra_core::base::matrix_tr_mul::MatrixTrMul;
use nalgebra_core::base::vector4::Vector4;
use nalgebra_core::internal::base::solve::SolveKernel;
use nalgebra_static4::base::matrix4::Matrix4Trait;
use nalgebra_static4::internal::base::matrix4::Matrix4InternalTrait;
use simba::scalar::Real;
use crate::linalg::qr::qr4::Qr4;

/// Crate-internal kernels of `Qr4<T>` (WP 8.0: the public API is strictly upstream's). The
/// Gram-Schmidt steps of `new` (`unit`, `project_out`), the back substitution shared by `solve` and
/// `try_inverse`, and `determinant` (upstream's `QR::determinant` is commented out), which the
/// tests use to check the factors.
#[generate_trait]
pub impl Qr4InternalImpl<
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
> of Qr4InternalTrait<T> {
    /// `v / n`, or the zero vector when `n` is exactly zero (the rank-deficient fallback of the
    /// module doc). One correctly rounded division per component. No upstream equivalent.
    #[inline(always)]
    fn unit(v: Vector4<T>, n: T) -> Vector4<T> {
        if n == R::zero() {
            Vector4 { x: R::zero(), y: R::zero(), z: R::zero(), w: R::zero() }
        } else {
            {
                let (x, y, z, w) = R::div4(v.x, v.y, v.z, v.w, n);
                Vector4 { x, y, z, w }
            }
        }
    }
    /// `v - c * q`, one fused `mul_add` per component (one rounding each). No upstream equivalent.
    #[inline(always)]
    fn project_out(v: Vector4<T>, q: Vector4<T>, c: T) -> Vector4<T> {
        Vector4 {
            x: R::mul_add(-c, q.x, v.x),
            y: R::mul_add(-c, q.y, v.y),
            z: R::mul_add(-c, q.z, v.z),
            w: R::mul_add(-c, q.w, v.w),
        }
    }
    /// `R^-1 * y` by back substitution, the shared body of `solve` and `try_inverse`. The caller
    /// guarantees a nonzero diagonal. Two roundings per component: the numerator, accumulated
    /// exactly in `Real::Wide`, then the correctly rounded division. No upstream equivalent.
    fn back_substitute(self: Qr4<T>, y: Vector4<T>) -> Vector4<T> {
        let x4 = R::div(y.w, self.r.m44);
        let x3 = R::div(R::mul_add(-self.r.m34, x4, y.z), self.r.m33);
        let w = R::wide_sub_prod(R::wide_add(R::wide_zero(), y.y), self.r.m23, x3);
        let x2 = R::div(R::wide_rescale(R::wide_sub_prod(w, self.r.m24, x4)), self.r.m22);
        let w = R::wide_sub_prod(R::wide_add(R::wide_zero(), y.x), self.r.m12, x2);
        let w = R::wide_sub_prod(w, self.r.m13, x3);
        let x1 = R::div(R::wide_rescale(R::wide_sub_prod(w, self.r.m14, x4)), self.r.m11);
        Vector4 { x: x1, y: x2, z: x3, w: x4 }
    }
    /// The determinant: `det(Q) * r11 * r22 * r33 * r44`, see `Qr3::determinant` for the sign
    /// convention and the rounding of the product chain. Upstream: `QR::determinant`.
    ///
    /// `det(Q)` costs a full `Matrix4::determinant` (the 4x4 has no cheaper sign test), which is
    /// most of the price of this method; PREFER `Matrix4::determinant` on the matrix itself when
    /// the factorisation is not needed for something else.
    fn determinant(self: Qr4<T>) -> T {
        let d = if self.q.determinant().is_sign_negative() {
            -self.r.m11
        } else {
            self.r.m11
        };
        d * self.r.m22 * self.r.m33 * self.r.m44
    }
}
