//! Internal, no stability promise: the crate-private items of `linalg::lu::lu3` that the packages
//! above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use nalgebra_types3::base::matrix3::Matrix3;
use nalgebra_types3::base::vector3::Vector3;
use simba::scalar::Real;
use crate::linalg::lu::lu3::Lu3;

/// Crate-internal kernels of `Lu3<T>` (WP 8.0: the public API is strictly upstream's): the row
/// permutation applied to a vector or to the rows of a matrix, upstream's
/// `lu.p().permute_rows(&mut m)` (to be exposed as `Perm3::permute_rows` by the
/// `PermutationSequence` completion of docs/API_PARITY.md P14).
#[generate_trait]
pub impl Lu3InternalImpl<
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
> of Lu3InternalTrait<T> {
    /// `P * v`: the transpositions applied to the components of `v`, in factorisation order. Exact:
    /// moves and comparisons only. Upstream: `LU::p().permute_rows(&mut v)`.
    fn permute(self: Lu3<T>, v: Vector3<T>) -> Vector3<T> {
        let mut x1 = v.x;
        let mut x2 = v.y;
        let mut x3 = v.z;
        if self.p.p1 == 2 {
            let t = x1;
            x1 = x2;
            x2 = t;
        } else if self.p.p1 == 3 {
            let t = x1;
            x1 = x3;
            x3 = t;
        }
        if self.p.p2 == 3 {
            let t = x2;
            x2 = x3;
            x3 = t;
        }
        Vector3 { x: x1, y: x2, z: x3 }
    }
    /// `P * m`: the same transpositions applied to the ROWS of `m`, so that `lu.permute_rows(a)` is
    /// `lu.l() * lu.u()` up to the rounding of the factorisation (this is how the tests check the
    /// decomposition). Exact: moves and comparisons only. Upstream: `LU::p().permute_rows(&mut m)`.
    fn permute_rows(self: Lu3<T>, m: Matrix3<T>) -> Matrix3<T> {
        let mut a11 = m.m11;
        let mut a12 = m.m12;
        let mut a13 = m.m13;
        let mut a21 = m.m21;
        let mut a22 = m.m22;
        let mut a23 = m.m23;
        let mut a31 = m.m31;
        let mut a32 = m.m32;
        let mut a33 = m.m33;
        if self.p.p1 == 2 {
            let t = a11;
            a11 = a21;
            a21 = t;
            let t = a12;
            a12 = a22;
            a22 = t;
            let t = a13;
            a13 = a23;
            a23 = t;
        } else if self.p.p1 == 3 {
            let t = a11;
            a11 = a31;
            a31 = t;
            let t = a12;
            a12 = a32;
            a32 = t;
            let t = a13;
            a13 = a33;
            a33 = t;
        }
        if self.p.p2 == 3 {
            let t = a21;
            a21 = a31;
            a31 = t;
            let t = a22;
            a22 = a32;
            a32 = t;
            let t = a23;
            a23 = a33;
            a33 = t;
        }
        Matrix3 {
            m11: a11,
            m21: a21,
            m31: a31,
            m12: a12,
            m22: a22,
            m32: a32,
            m13: a13,
            m23: a23,
            m33: a33,
        }
    }
}
