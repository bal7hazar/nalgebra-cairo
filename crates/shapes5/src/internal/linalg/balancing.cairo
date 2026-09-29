//! Internal, no stability promise: the crate-private items of `linalg::balancing` that the packages
//! above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use nalgebra_core::base::matrix1::Matrix1;
use nalgebra_core::base::matrix2::Matrix2;
use nalgebra_core::base::matrix3::Matrix3;
use nalgebra_core::base::matrix4::Matrix4;
use nalgebra_core::base::vector2::Vector2;
use nalgebra_core::base::vector3::Vector3;
use nalgebra_core::base::vector4::Vector4;
use simba::scalar::Real;

/// The balancing of one square shape `M` with its diagonal vector `V` (crate-private: the free
/// functions below are upstream's interface).
pub trait Balancing<M, V> {
    fn balance_parlett_reinsch(ref matrix: M) -> V;
    fn unbalance(ref m: M, d: V);
}

/// `balance_parlett_reinsch` / `unbalance` on `Matrix1` (see the free functions).
pub impl Matrix1Balancing<
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
> of Balancing<Matrix1<T>, Matrix1<T>> {
    fn balance_parlett_reinsch(ref matrix: Matrix1<T>) -> Matrix1<T> {
        let mut a00 = matrix.x;
        let mut d0 = R::one();
        let two = R::from_int(2);
        let half = R::from_ratio(1, 2);
        let tol = R::from_ratio(95, 100);
        let mut converged = false;
        while !converged {
            converged = true;
            {
                let c0 = R::abs(a00);
                let r0 = R::abs(a00);
                if c0 != R::zero() && r0 != R::zero() {
                    let big = if c0 > r0 {
                        c0
                    } else {
                        r0
                    };
                    let (mut n_col, mut n_row) = (R::div(c0, big), R::div(r0, big));
                    let s = R::sum_prod2(n_col, n_col, n_row, n_row);
                    let mut f = R::one();
                    let mut finv = R::one();
                    while n_col < n_row * half {
                        n_col = n_col * two;
                        n_row = n_row * half;
                        f = f * two;
                        finv = finv * half;
                    }
                    while n_col >= n_row * two {
                        n_col = n_col * half;
                        n_row = n_row * two;
                        f = f * half;
                        finv = finv * two;
                    }
                    if R::sum_prod2(n_col, n_col, n_row, n_row) < tol * s {
                        converged = false;
                        d0 = d0 * f;
                        a00 = a00 * f;
                        a00 = a00 * finv;
                    }
                }
            }
        }
        matrix = Matrix1 { x: a00 };
        Matrix1 { x: d0 }
    }

    fn unbalance(ref m: Matrix1<T>, d: Matrix1<T>) {
        let dinv0 = R::recip(d.x);
        m = Matrix1 { x: m.x * (d.x * dinv0) };
    }
}

/// `balance_parlett_reinsch` / `unbalance` on `Matrix2` (see the free functions).
pub impl Matrix2Balancing<
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
> of Balancing<Matrix2<T>, Vector2<T>> {
    fn balance_parlett_reinsch(ref matrix: Matrix2<T>) -> Vector2<T> {
        let mut a00 = matrix.m11;
        let mut a10 = matrix.m21;
        let mut a01 = matrix.m12;
        let mut a11 = matrix.m22;
        let mut d0 = R::one();
        let mut d1 = R::one();
        let two = R::from_int(2);
        let half = R::from_ratio(1, 2);
        let tol = R::from_ratio(95, 100);
        let mut converged = false;
        while !converged {
            converged = true;
            {
                let c0 = R::norm2(a00, a10);
                let r0 = R::norm2(a00, a01);
                if c0 != R::zero() && r0 != R::zero() {
                    let big = if c0 > r0 {
                        c0
                    } else {
                        r0
                    };
                    let (mut n_col, mut n_row) = (R::div(c0, big), R::div(r0, big));
                    let s = R::sum_prod2(n_col, n_col, n_row, n_row);
                    let mut f = R::one();
                    let mut finv = R::one();
                    while n_col < n_row * half {
                        n_col = n_col * two;
                        n_row = n_row * half;
                        f = f * two;
                        finv = finv * half;
                    }
                    while n_col >= n_row * two {
                        n_col = n_col * half;
                        n_row = n_row * two;
                        f = f * half;
                        finv = finv * two;
                    }
                    if R::sum_prod2(n_col, n_col, n_row, n_row) < tol * s {
                        converged = false;
                        d0 = d0 * f;
                        a00 = a00 * f;
                        a10 = a10 * f;
                        a00 = a00 * finv;
                        a01 = a01 * finv;
                    }
                }
            }
            {
                let c0 = R::norm2(a01, a11);
                let r0 = R::norm2(a10, a11);
                if c0 != R::zero() && r0 != R::zero() {
                    let big = if c0 > r0 {
                        c0
                    } else {
                        r0
                    };
                    let (mut n_col, mut n_row) = (R::div(c0, big), R::div(r0, big));
                    let s = R::sum_prod2(n_col, n_col, n_row, n_row);
                    let mut f = R::one();
                    let mut finv = R::one();
                    while n_col < n_row * half {
                        n_col = n_col * two;
                        n_row = n_row * half;
                        f = f * two;
                        finv = finv * half;
                    }
                    while n_col >= n_row * two {
                        n_col = n_col * half;
                        n_row = n_row * two;
                        f = f * half;
                        finv = finv * two;
                    }
                    if R::sum_prod2(n_col, n_col, n_row, n_row) < tol * s {
                        converged = false;
                        d1 = d1 * f;
                        a01 = a01 * f;
                        a11 = a11 * f;
                        a10 = a10 * finv;
                        a11 = a11 * finv;
                    }
                }
            }
        }
        matrix = Matrix2 { m11: a00, m21: a10, m12: a01, m22: a11 };
        Vector2 { x: d0, y: d1 }
    }

    fn unbalance(ref m: Matrix2<T>, d: Vector2<T>) {
        let dinv0 = R::recip(d.x);
        let dinv1 = R::recip(d.y);
        m =
            Matrix2 {
                m11: m.m11 * (d.x * dinv0),
                m21: m.m21 * (d.y * dinv0),
                m12: m.m12 * (d.x * dinv1),
                m22: m.m22 * (d.y * dinv1),
            };
    }
}

/// `balance_parlett_reinsch` / `unbalance` on `Matrix3` (see the free functions).
pub impl Matrix3Balancing<
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
> of Balancing<Matrix3<T>, Vector3<T>> {
    fn balance_parlett_reinsch(ref matrix: Matrix3<T>) -> Vector3<T> {
        let mut a00 = matrix.m11;
        let mut a10 = matrix.m21;
        let mut a20 = matrix.m31;
        let mut a01 = matrix.m12;
        let mut a11 = matrix.m22;
        let mut a21 = matrix.m32;
        let mut a02 = matrix.m13;
        let mut a12 = matrix.m23;
        let mut a22 = matrix.m33;
        let mut d0 = R::one();
        let mut d1 = R::one();
        let mut d2 = R::one();
        let two = R::from_int(2);
        let half = R::from_ratio(1, 2);
        let tol = R::from_ratio(95, 100);
        let mut converged = false;
        while !converged {
            converged = true;
            {
                let c0 = R::norm3(a00, a10, a20);
                let r0 = R::norm3(a00, a01, a02);
                if c0 != R::zero() && r0 != R::zero() {
                    let big = if c0 > r0 {
                        c0
                    } else {
                        r0
                    };
                    let (mut n_col, mut n_row) = (R::div(c0, big), R::div(r0, big));
                    let s = R::sum_prod2(n_col, n_col, n_row, n_row);
                    let mut f = R::one();
                    let mut finv = R::one();
                    while n_col < n_row * half {
                        n_col = n_col * two;
                        n_row = n_row * half;
                        f = f * two;
                        finv = finv * half;
                    }
                    while n_col >= n_row * two {
                        n_col = n_col * half;
                        n_row = n_row * two;
                        f = f * half;
                        finv = finv * two;
                    }
                    if R::sum_prod2(n_col, n_col, n_row, n_row) < tol * s {
                        converged = false;
                        d0 = d0 * f;
                        a00 = a00 * f;
                        a10 = a10 * f;
                        a20 = a20 * f;
                        a00 = a00 * finv;
                        a01 = a01 * finv;
                        a02 = a02 * finv;
                    }
                }
            }
            {
                let c0 = R::norm3(a01, a11, a21);
                let r0 = R::norm3(a10, a11, a12);
                if c0 != R::zero() && r0 != R::zero() {
                    let big = if c0 > r0 {
                        c0
                    } else {
                        r0
                    };
                    let (mut n_col, mut n_row) = (R::div(c0, big), R::div(r0, big));
                    let s = R::sum_prod2(n_col, n_col, n_row, n_row);
                    let mut f = R::one();
                    let mut finv = R::one();
                    while n_col < n_row * half {
                        n_col = n_col * two;
                        n_row = n_row * half;
                        f = f * two;
                        finv = finv * half;
                    }
                    while n_col >= n_row * two {
                        n_col = n_col * half;
                        n_row = n_row * two;
                        f = f * half;
                        finv = finv * two;
                    }
                    if R::sum_prod2(n_col, n_col, n_row, n_row) < tol * s {
                        converged = false;
                        d1 = d1 * f;
                        a01 = a01 * f;
                        a11 = a11 * f;
                        a21 = a21 * f;
                        a10 = a10 * finv;
                        a11 = a11 * finv;
                        a12 = a12 * finv;
                    }
                }
            }
            {
                let c0 = R::norm3(a02, a12, a22);
                let r0 = R::norm3(a20, a21, a22);
                if c0 != R::zero() && r0 != R::zero() {
                    let big = if c0 > r0 {
                        c0
                    } else {
                        r0
                    };
                    let (mut n_col, mut n_row) = (R::div(c0, big), R::div(r0, big));
                    let s = R::sum_prod2(n_col, n_col, n_row, n_row);
                    let mut f = R::one();
                    let mut finv = R::one();
                    while n_col < n_row * half {
                        n_col = n_col * two;
                        n_row = n_row * half;
                        f = f * two;
                        finv = finv * half;
                    }
                    while n_col >= n_row * two {
                        n_col = n_col * half;
                        n_row = n_row * two;
                        f = f * half;
                        finv = finv * two;
                    }
                    if R::sum_prod2(n_col, n_col, n_row, n_row) < tol * s {
                        converged = false;
                        d2 = d2 * f;
                        a02 = a02 * f;
                        a12 = a12 * f;
                        a22 = a22 * f;
                        a20 = a20 * finv;
                        a21 = a21 * finv;
                        a22 = a22 * finv;
                    }
                }
            }
        }
        matrix =
            Matrix3 {
                m11: a00,
                m21: a10,
                m31: a20,
                m12: a01,
                m22: a11,
                m32: a21,
                m13: a02,
                m23: a12,
                m33: a22,
            };
        Vector3 { x: d0, y: d1, z: d2 }
    }

    fn unbalance(ref m: Matrix3<T>, d: Vector3<T>) {
        let dinv0 = R::recip(d.x);
        let dinv1 = R::recip(d.y);
        let dinv2 = R::recip(d.z);
        m =
            Matrix3 {
                m11: m.m11 * (d.x * dinv0),
                m21: m.m21 * (d.y * dinv0),
                m31: m.m31 * (d.z * dinv0),
                m12: m.m12 * (d.x * dinv1),
                m22: m.m22 * (d.y * dinv1),
                m32: m.m32 * (d.z * dinv1),
                m13: m.m13 * (d.x * dinv2),
                m23: m.m23 * (d.y * dinv2),
                m33: m.m33 * (d.z * dinv2),
            };
    }
}

/// `balance_parlett_reinsch` / `unbalance` on `Matrix4` (see the free functions).
pub impl Matrix4Balancing<
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
> of Balancing<Matrix4<T>, Vector4<T>> {
    fn balance_parlett_reinsch(ref matrix: Matrix4<T>) -> Vector4<T> {
        let mut a00 = matrix.m11;
        let mut a10 = matrix.m21;
        let mut a20 = matrix.m31;
        let mut a30 = matrix.m41;
        let mut a01 = matrix.m12;
        let mut a11 = matrix.m22;
        let mut a21 = matrix.m32;
        let mut a31 = matrix.m42;
        let mut a02 = matrix.m13;
        let mut a12 = matrix.m23;
        let mut a22 = matrix.m33;
        let mut a32 = matrix.m43;
        let mut a03 = matrix.m14;
        let mut a13 = matrix.m24;
        let mut a23 = matrix.m34;
        let mut a33 = matrix.m44;
        let mut d0 = R::one();
        let mut d1 = R::one();
        let mut d2 = R::one();
        let mut d3 = R::one();
        let two = R::from_int(2);
        let half = R::from_ratio(1, 2);
        let tol = R::from_ratio(95, 100);
        let mut converged = false;
        while !converged {
            converged = true;
            {
                let c0 = R::norm4(a00, a10, a20, a30);
                let r0 = R::norm4(a00, a01, a02, a03);
                if c0 != R::zero() && r0 != R::zero() {
                    let big = if c0 > r0 {
                        c0
                    } else {
                        r0
                    };
                    let (mut n_col, mut n_row) = (R::div(c0, big), R::div(r0, big));
                    let s = R::sum_prod2(n_col, n_col, n_row, n_row);
                    let mut f = R::one();
                    let mut finv = R::one();
                    while n_col < n_row * half {
                        n_col = n_col * two;
                        n_row = n_row * half;
                        f = f * two;
                        finv = finv * half;
                    }
                    while n_col >= n_row * two {
                        n_col = n_col * half;
                        n_row = n_row * two;
                        f = f * half;
                        finv = finv * two;
                    }
                    if R::sum_prod2(n_col, n_col, n_row, n_row) < tol * s {
                        converged = false;
                        d0 = d0 * f;
                        a00 = a00 * f;
                        a10 = a10 * f;
                        a20 = a20 * f;
                        a30 = a30 * f;
                        a00 = a00 * finv;
                        a01 = a01 * finv;
                        a02 = a02 * finv;
                        a03 = a03 * finv;
                    }
                }
            }
            {
                let c0 = R::norm4(a01, a11, a21, a31);
                let r0 = R::norm4(a10, a11, a12, a13);
                if c0 != R::zero() && r0 != R::zero() {
                    let big = if c0 > r0 {
                        c0
                    } else {
                        r0
                    };
                    let (mut n_col, mut n_row) = (R::div(c0, big), R::div(r0, big));
                    let s = R::sum_prod2(n_col, n_col, n_row, n_row);
                    let mut f = R::one();
                    let mut finv = R::one();
                    while n_col < n_row * half {
                        n_col = n_col * two;
                        n_row = n_row * half;
                        f = f * two;
                        finv = finv * half;
                    }
                    while n_col >= n_row * two {
                        n_col = n_col * half;
                        n_row = n_row * two;
                        f = f * half;
                        finv = finv * two;
                    }
                    if R::sum_prod2(n_col, n_col, n_row, n_row) < tol * s {
                        converged = false;
                        d1 = d1 * f;
                        a01 = a01 * f;
                        a11 = a11 * f;
                        a21 = a21 * f;
                        a31 = a31 * f;
                        a10 = a10 * finv;
                        a11 = a11 * finv;
                        a12 = a12 * finv;
                        a13 = a13 * finv;
                    }
                }
            }
            {
                let c0 = R::norm4(a02, a12, a22, a32);
                let r0 = R::norm4(a20, a21, a22, a23);
                if c0 != R::zero() && r0 != R::zero() {
                    let big = if c0 > r0 {
                        c0
                    } else {
                        r0
                    };
                    let (mut n_col, mut n_row) = (R::div(c0, big), R::div(r0, big));
                    let s = R::sum_prod2(n_col, n_col, n_row, n_row);
                    let mut f = R::one();
                    let mut finv = R::one();
                    while n_col < n_row * half {
                        n_col = n_col * two;
                        n_row = n_row * half;
                        f = f * two;
                        finv = finv * half;
                    }
                    while n_col >= n_row * two {
                        n_col = n_col * half;
                        n_row = n_row * two;
                        f = f * half;
                        finv = finv * two;
                    }
                    if R::sum_prod2(n_col, n_col, n_row, n_row) < tol * s {
                        converged = false;
                        d2 = d2 * f;
                        a02 = a02 * f;
                        a12 = a12 * f;
                        a22 = a22 * f;
                        a32 = a32 * f;
                        a20 = a20 * finv;
                        a21 = a21 * finv;
                        a22 = a22 * finv;
                        a23 = a23 * finv;
                    }
                }
            }
            {
                let c0 = R::norm4(a03, a13, a23, a33);
                let r0 = R::norm4(a30, a31, a32, a33);
                if c0 != R::zero() && r0 != R::zero() {
                    let big = if c0 > r0 {
                        c0
                    } else {
                        r0
                    };
                    let (mut n_col, mut n_row) = (R::div(c0, big), R::div(r0, big));
                    let s = R::sum_prod2(n_col, n_col, n_row, n_row);
                    let mut f = R::one();
                    let mut finv = R::one();
                    while n_col < n_row * half {
                        n_col = n_col * two;
                        n_row = n_row * half;
                        f = f * two;
                        finv = finv * half;
                    }
                    while n_col >= n_row * two {
                        n_col = n_col * half;
                        n_row = n_row * two;
                        f = f * half;
                        finv = finv * two;
                    }
                    if R::sum_prod2(n_col, n_col, n_row, n_row) < tol * s {
                        converged = false;
                        d3 = d3 * f;
                        a03 = a03 * f;
                        a13 = a13 * f;
                        a23 = a23 * f;
                        a33 = a33 * f;
                        a30 = a30 * finv;
                        a31 = a31 * finv;
                        a32 = a32 * finv;
                        a33 = a33 * finv;
                    }
                }
            }
        }
        matrix =
            Matrix4 {
                m11: a00,
                m21: a10,
                m31: a20,
                m41: a30,
                m12: a01,
                m22: a11,
                m32: a21,
                m42: a31,
                m13: a02,
                m23: a12,
                m33: a22,
                m43: a32,
                m14: a03,
                m24: a13,
                m34: a23,
                m44: a33,
            };
        Vector4 { x: d0, y: d1, z: d2, w: d3 }
    }

    fn unbalance(ref m: Matrix4<T>, d: Vector4<T>) {
        let dinv0 = R::recip(d.x);
        let dinv1 = R::recip(d.y);
        let dinv2 = R::recip(d.z);
        let dinv3 = R::recip(d.w);
        m =
            Matrix4 {
                m11: m.m11 * (d.x * dinv0),
                m21: m.m21 * (d.y * dinv0),
                m31: m.m31 * (d.z * dinv0),
                m41: m.m41 * (d.w * dinv0),
                m12: m.m12 * (d.x * dinv1),
                m22: m.m22 * (d.y * dinv1),
                m32: m.m32 * (d.z * dinv1),
                m42: m.m42 * (d.w * dinv1),
                m13: m.m13 * (d.x * dinv2),
                m23: m.m23 * (d.y * dinv2),
                m33: m.m33 * (d.z * dinv2),
                m43: m.m43 * (d.w * dinv2),
                m14: m.m14 * (d.x * dinv3),
                m24: m.m24 * (d.y * dinv3),
                m34: m.m34 * (d.z * dinv3),
                m44: m.m44 * (d.w * dinv3),
            };
    }
}
