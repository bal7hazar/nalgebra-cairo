//! `Vector::axpy_cs` (upstream `sparse/cs_matrix_ops.rs`): `y = alpha * x + beta * y` for a dense
//! vector `y` and a sparse one-column `x` (a `CsVector`), on the static column vectors
//! (`Matrix1`, `Vector2..6`) and on `DVector`.

use simba::scalar::Real;
use crate::base::dynamic::DVector;
use crate::base::errors;
use crate::base::matrix1::Matrix1;
use crate::base::vector2::Vector2;
use crate::base::vector3::Vector3;
use crate::base::vector4::Vector4;
use crate::base::vector5::Vector5;
use crate::base::vector6::Vector6;
use super::cs_matrix::CsMatrix;

/// `self = alpha * x + beta * self` for a sparse one-column `x` of `self`'s dimension (panics
/// with `nalgebra: dimension mismatch` otherwise). As upstream, `beta == 0` only writes the
/// components in the pattern of `x` (`self[k] = alpha * x[k]`, the others untouched); otherwise
/// `self[k] = alpha * x[k] + beta * self[k]` (ONE fused sum of products, floored once; upstream
/// rounds `self *= beta`, then the update) and `self[k] = beta * self[k]` off the pattern.
/// Upstream: `Vector::axpy_cs(&mut self, alpha, &x, beta)`.
pub trait AxpyCs<V, T> {
    fn axpy_cs(ref self: V, alpha: T, x: CsMatrix<T>, beta: T);
}

/// The kernel on the components of `y`.
fn axpy_cs_span<T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Mul<T>, +PartialEq<T>>(
    mut y: Span<T>, alpha: T, x: CsMatrix<T>, beta: T,
) -> Span<T> {
    if x.nrows != y.len() || x.ncols != 1 {
        core::panic_with_felt252(errors::DIMENSION_MISMATCH);
    }
    let beta_zero = beta == R::zero();
    let mut xi = x.i;
    let mut xv = x.vals;
    let mut out: Array<T> = array![];
    let mut k: usize = 0;
    while let Some(yk) = y.pop_front() {
        let yk = *yk;
        if xi.len() != 0 && *xi[0] == k {
            xi.pop_front().unwrap();
            let v = *xv.pop_front().unwrap();
            out.append(if beta_zero {
                alpha * v
            } else {
                R::sum_prod2(alpha, v, beta, yk)
            });
        } else {
            out.append(if beta_zero {
                yk
            } else {
                beta * yk
            });
        }
        k += 1;
    }
    out.span()
}

/// `axpy_cs` on a `DVector`.
pub impl DVectorAxpyCs<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Mul<T>, +PartialEq<T>,
> of AxpyCs<DVector<T>, T> {
    fn axpy_cs(ref self: DVector<T>, alpha: T, x: CsMatrix<T>, beta: T) {
        self = DVector { data: axpy_cs_span(self.data, alpha, x, beta) };
    }
}

/// `axpy_cs` on a `Matrix1`.
pub impl Matrix1AxpyCs<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Mul<T>, +PartialEq<T>,
> of AxpyCs<Matrix1<T>, T> {
    fn axpy_cs(ref self: Matrix1<T>, alpha: T, x: CsMatrix<T>, beta: T) {
        let y = axpy_cs_span(array![self.x].span(), alpha, x, beta);
        let boxed: @Box<[T; 1]> = y.try_into().unwrap();
        let [a0] = boxed.unbox();
        self = Matrix1 { x: a0 };
    }
}

/// `axpy_cs` on a `Vector2`.
pub impl Vector2AxpyCs<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Mul<T>, +PartialEq<T>,
> of AxpyCs<Vector2<T>, T> {
    fn axpy_cs(ref self: Vector2<T>, alpha: T, x: CsMatrix<T>, beta: T) {
        let y = axpy_cs_span(array![self.x, self.y].span(), alpha, x, beta);
        let boxed: @Box<[T; 2]> = y.try_into().unwrap();
        let [a0, a1] = boxed.unbox();
        self = Vector2 { x: a0, y: a1 };
    }
}

/// `axpy_cs` on a `Vector3`.
pub impl Vector3AxpyCs<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Mul<T>, +PartialEq<T>,
> of AxpyCs<Vector3<T>, T> {
    fn axpy_cs(ref self: Vector3<T>, alpha: T, x: CsMatrix<T>, beta: T) {
        let y = axpy_cs_span(array![self.x, self.y, self.z].span(), alpha, x, beta);
        let boxed: @Box<[T; 3]> = y.try_into().unwrap();
        let [a0, a1, a2] = boxed.unbox();
        self = Vector3 { x: a0, y: a1, z: a2 };
    }
}

/// `axpy_cs` on a `Vector4`.
pub impl Vector4AxpyCs<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Mul<T>, +PartialEq<T>,
> of AxpyCs<Vector4<T>, T> {
    fn axpy_cs(ref self: Vector4<T>, alpha: T, x: CsMatrix<T>, beta: T) {
        let y = axpy_cs_span(array![self.x, self.y, self.z, self.w].span(), alpha, x, beta);
        let boxed: @Box<[T; 4]> = y.try_into().unwrap();
        let [a0, a1, a2, a3] = boxed.unbox();
        self = Vector4 { x: a0, y: a1, z: a2, w: a3 };
    }
}

/// `axpy_cs` on a `Vector5`.
pub impl Vector5AxpyCs<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Mul<T>, +PartialEq<T>,
> of AxpyCs<Vector5<T>, T> {
    fn axpy_cs(ref self: Vector5<T>, alpha: T, x: CsMatrix<T>, beta: T) {
        let y = axpy_cs_span(array![self.x, self.y, self.z, self.w, self.a].span(), alpha, x, beta);
        let boxed: @Box<[T; 5]> = y.try_into().unwrap();
        let [a0, a1, a2, a3, a4] = boxed.unbox();
        self = Vector5 { x: a0, y: a1, z: a2, w: a3, a: a4 };
    }
}

/// `axpy_cs` on a `Vector6`.
pub impl Vector6AxpyCs<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Mul<T>, +PartialEq<T>,
> of AxpyCs<Vector6<T>, T> {
    fn axpy_cs(ref self: Vector6<T>, alpha: T, x: CsMatrix<T>, beta: T) {
        let y = axpy_cs_span(
            array![self.x, self.y, self.z, self.w, self.a, self.b].span(), alpha, x, beta,
        );
        let boxed: @Box<[T; 6]> = y.try_into().unwrap();
        let [a0, a1, a2, a3, a4, a5] = boxed.unbox();
        self = Vector6 { x: a0, y: a1, z: a2, w: a3, a: a4, b: a5 };
    }
}
