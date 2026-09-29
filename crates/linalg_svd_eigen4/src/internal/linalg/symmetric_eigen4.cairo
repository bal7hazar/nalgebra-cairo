//! Internal, no stability promise: the crate-private items of `linalg::symmetric_eigen4` that the
//! packages above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use core::internal::revoke_ap_tracking;
use nalgebra_core::base::vector4::Vector4;
use simba::scalar::Real;
use crate::linalg::symmetric_eigen4::{Jacobi4Impl, SymmetricEigen4};

/// The 10 independent components of a symmetric 4x4 matrix, `mIJ` with `I <= J`
/// (crate-internal: the input of the Jacobi kernel and the Gram matrix of the SVD).
#[derive(Copy, Drop)]
pub struct Sym4<T> {
    pub m11: T,
    pub m12: T,
    pub m13: T,
    pub m14: T,
    pub m22: T,
    pub m23: T,
    pub m24: T,
    pub m33: T,
    pub m34: T,
    pub m44: T,
}

/// Crate-internal kernels of `SymmetricEigen4<T>`, on the 10 independent components.
#[generate_trait]
pub impl SymmetricEigen4InternalImpl<
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
> of SymmetricEigen4InternalTrait<T> {
    /// The kernel of `new`.
    fn new_sym(s: Sym4<T>) -> SymmetricEigen4<T> {
        Jacobi4Impl::<T>::run(s).finish()
    }
    /// The kernel of `try_new`.
    fn try_new_sym(s: Sym4<T>, eps: T) -> Option<SymmetricEigen4<T>> {
        let j = Jacobi4Impl::<T>::run(s);
        if Jacobi4Impl::<T>::converged(j.s, eps) {
            Some(j.finish())
        } else {
            None
        }
    }
    /// The eigenvalues alone, ascending: the same 5 sweeps with the rotation dropped,
    /// bit-identical to `new_sym(s).eigenvalues`.
    fn eigenvalues(s: Sym4<T>) -> Vector4<T> {
        revoke_ap_tracking();
        let s = Jacobi4Impl::<T>::sweep_s(s);
        let s = Jacobi4Impl::<T>::sweep_s(s);
        let s = Jacobi4Impl::<T>::sweep_s(s);
        let s = Jacobi4Impl::<T>::sweep_s(s);
        let s = Jacobi4Impl::<T>::sweep_s(s);
        let mut l0 = s.m11;
        let mut l1 = s.m22;
        let mut l2 = s.m33;
        let mut l3 = s.m44;
        if l1 < l0 {
            let tmp0 = l0;
            l0 = l1;
            l1 = tmp0;
        }
        if l3 < l2 {
            let tmp0 = l2;
            l2 = l3;
            l3 = tmp0;
        }
        if l2 < l1 {
            let tmp0 = l1;
            l1 = l2;
            l2 = tmp0;
        }
        if l1 < l0 {
            let tmp0 = l0;
            l0 = l1;
            l1 = tmp0;
        }
        if l3 < l2 {
            let tmp0 = l2;
            l2 = l3;
            l3 = tmp0;
        }
        if l2 < l1 {
            let tmp0 = l1;
            l1 = l2;
            l2 = tmp0;
        }
        Vector4 { x: l0, y: l1, z: l2, w: l3 }
    }
    /// The upper triangle of `V * diag(eigenvalues) * Vᵀ`.
    fn recompose_sym(self: SymmetricEigen4<T>) -> Sym4<T> {
        revoke_ap_tracking();
        let e = self;
        let w00 = e.eigenvectors.m11 * e.eigenvalues.x;
        let w01 = e.eigenvectors.m12 * e.eigenvalues.y;
        let w02 = e.eigenvectors.m13 * e.eigenvalues.z;
        let w03 = e.eigenvectors.m14 * e.eigenvalues.w;
        let w10 = e.eigenvectors.m21 * e.eigenvalues.x;
        let w11 = e.eigenvectors.m22 * e.eigenvalues.y;
        let w12 = e.eigenvectors.m23 * e.eigenvalues.z;
        let w13 = e.eigenvectors.m24 * e.eigenvalues.w;
        let w20 = e.eigenvectors.m31 * e.eigenvalues.x;
        let w21 = e.eigenvectors.m32 * e.eigenvalues.y;
        let w22 = e.eigenvectors.m33 * e.eigenvalues.z;
        let w23 = e.eigenvectors.m34 * e.eigenvalues.w;
        let w30 = e.eigenvectors.m41 * e.eigenvalues.x;
        let w31 = e.eigenvectors.m42 * e.eigenvalues.y;
        let w32 = e.eigenvectors.m43 * e.eigenvalues.z;
        let w33 = e.eigenvectors.m44 * e.eigenvalues.w;
        Sym4 {
            m11: R::sum_prod4(
                e.eigenvectors.m11,
                w00,
                e.eigenvectors.m12,
                w01,
                e.eigenvectors.m13,
                w02,
                e.eigenvectors.m14,
                w03,
            ),
            m12: R::sum_prod4(
                e.eigenvectors.m11,
                w10,
                e.eigenvectors.m12,
                w11,
                e.eigenvectors.m13,
                w12,
                e.eigenvectors.m14,
                w13,
            ),
            m13: R::sum_prod4(
                e.eigenvectors.m11,
                w20,
                e.eigenvectors.m12,
                w21,
                e.eigenvectors.m13,
                w22,
                e.eigenvectors.m14,
                w23,
            ),
            m14: R::sum_prod4(
                e.eigenvectors.m11,
                w30,
                e.eigenvectors.m12,
                w31,
                e.eigenvectors.m13,
                w32,
                e.eigenvectors.m14,
                w33,
            ),
            m22: R::sum_prod4(
                e.eigenvectors.m21,
                w10,
                e.eigenvectors.m22,
                w11,
                e.eigenvectors.m23,
                w12,
                e.eigenvectors.m24,
                w13,
            ),
            m23: R::sum_prod4(
                e.eigenvectors.m21,
                w20,
                e.eigenvectors.m22,
                w21,
                e.eigenvectors.m23,
                w22,
                e.eigenvectors.m24,
                w23,
            ),
            m24: R::sum_prod4(
                e.eigenvectors.m21,
                w30,
                e.eigenvectors.m22,
                w31,
                e.eigenvectors.m23,
                w32,
                e.eigenvectors.m24,
                w33,
            ),
            m33: R::sum_prod4(
                e.eigenvectors.m31,
                w20,
                e.eigenvectors.m32,
                w21,
                e.eigenvectors.m33,
                w22,
                e.eigenvectors.m34,
                w23,
            ),
            m34: R::sum_prod4(
                e.eigenvectors.m31,
                w30,
                e.eigenvectors.m32,
                w31,
                e.eigenvectors.m33,
                w32,
                e.eigenvectors.m34,
                w33,
            ),
            m44: R::sum_prod4(
                e.eigenvectors.m41,
                w30,
                e.eigenvectors.m42,
                w31,
                e.eigenvectors.m43,
                w32,
                e.eigenvectors.m44,
                w33,
            ),
        }
    }
}
