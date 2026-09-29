//! Internal, no stability promise: the crate-private items of `linalg::householder` that the
//! packages above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use nalgebra_core::base::matrix1::Matrix1;
use nalgebra_core::base::vector2::Vector2;
use nalgebra_core::base::vector3::Vector3;
use nalgebra_core::base::vector4::Vector4;
use simba::scalar::Real;

/// The Householder axis of one column vector shape (crate-private: the free function below is
/// upstream's interface).
pub trait HouseholderAxis<V, T> {
    fn reflection_axis_mut(ref column: V) -> (T, bool);
}

pub impl Matrix1HouseholderAxis<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of HouseholderAxis<Matrix1<T>, T> {
    fn reflection_axis_mut(ref column: Matrix1<T>) -> (T, bool) {
        let norm = R::wide_sqrt(R::wide_add_prod(R::wide_zero(), column.x, column.x));
        if norm == R::zero() {
            return (R::zero(), false);
        }
        let x0 = column.x;
        let signed = if x0 < R::zero() {
            -norm
        } else {
            norm
        };
        let y0 = x0 + signed;
        let d = R::wide_sqrt(R::wide_add_prod(R::wide_zero(), y0, y0));
        let u0 = R::div(y0, d);
        column = Matrix1 { x: u0 };
        (-signed, true)
    }
}

pub impl Vector2HouseholderAxis<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of HouseholderAxis<Vector2<T>, T> {
    fn reflection_axis_mut(ref column: Vector2<T>) -> (T, bool) {
        let norm = R::wide_sqrt(
            R::wide_add_prod(
                R::wide_add_prod(R::wide_zero(), column.x, column.x), column.y, column.y,
            ),
        );
        if norm == R::zero() {
            return (R::zero(), false);
        }
        let x0 = column.x;
        let signed = if x0 < R::zero() {
            -norm
        } else {
            norm
        };
        let y0 = x0 + signed;
        let d = R::wide_sqrt(
            R::wide_add_prod(R::wide_add_prod(R::wide_zero(), y0, y0), column.y, column.y),
        );
        let u0 = R::div(y0, d);
        let u1 = R::div(column.y, d);
        column = Vector2 { x: u0, y: u1 };
        (-signed, true)
    }
}

pub impl Vector3HouseholderAxis<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of HouseholderAxis<Vector3<T>, T> {
    fn reflection_axis_mut(ref column: Vector3<T>) -> (T, bool) {
        let norm = R::wide_sqrt(
            R::wide_add_prod(
                R::wide_add_prod(
                    R::wide_add_prod(R::wide_zero(), column.x, column.x), column.y, column.y,
                ),
                column.z,
                column.z,
            ),
        );
        if norm == R::zero() {
            return (R::zero(), false);
        }
        let x0 = column.x;
        let signed = if x0 < R::zero() {
            -norm
        } else {
            norm
        };
        let y0 = x0 + signed;
        let d = R::wide_sqrt(
            R::wide_add_prod(
                R::wide_add_prod(R::wide_add_prod(R::wide_zero(), y0, y0), column.y, column.y),
                column.z,
                column.z,
            ),
        );
        let (u0, u1, u2) = R::div3(y0, column.y, column.z, d);
        column = Vector3 { x: u0, y: u1, z: u2 };
        (-signed, true)
    }
}

pub impl Vector4HouseholderAxis<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of HouseholderAxis<Vector4<T>, T> {
    fn reflection_axis_mut(ref column: Vector4<T>) -> (T, bool) {
        let norm = R::wide_sqrt(
            R::wide_add_prod(
                R::wide_add_prod(
                    R::wide_add_prod(
                        R::wide_add_prod(R::wide_zero(), column.x, column.x), column.y, column.y,
                    ),
                    column.z,
                    column.z,
                ),
                column.w,
                column.w,
            ),
        );
        if norm == R::zero() {
            return (R::zero(), false);
        }
        let x0 = column.x;
        let signed = if x0 < R::zero() {
            -norm
        } else {
            norm
        };
        let y0 = x0 + signed;
        let d = R::wide_sqrt(
            R::wide_add_prod(
                R::wide_add_prod(
                    R::wide_add_prod(R::wide_add_prod(R::wide_zero(), y0, y0), column.y, column.y),
                    column.z,
                    column.z,
                ),
                column.w,
                column.w,
            ),
        );
        let (u0, u1, u2, u3) = R::div4(y0, column.y, column.z, column.w, d);
        column = Vector4 { x: u0, y: u1, z: u2, w: u3 };
        (-signed, true)
    }
}
