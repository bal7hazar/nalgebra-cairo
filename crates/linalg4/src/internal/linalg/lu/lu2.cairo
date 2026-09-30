//! Internal, no stability promise: the crate-private items of `linalg::lu::lu2` that the packages
//! above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use nalgebra_types2::base::matrix2::Matrix2;
use nalgebra_types2::base::vector2::Vector2;
use simba::scalar::Real;
use crate::linalg::lu::lu2::Lu2;

/// Crate-internal kernels of `Lu2<T>` (WP 8.0: the public API is strictly upstream's): the row
/// permutation applied to a vector or to the rows of a matrix, upstream's
/// `lu.p().permute_rows(&mut m)` (to be exposed as `Perm2::permute_rows` by the
/// `PermutationSequence` completion of docs/API_PARITY.md P14).
#[generate_trait]
pub impl Lu2InternalImpl<
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
> of Lu2InternalTrait<T> {
    /// `P * v`: the transpositions applied to the components of `v`, in factorisation order. Exact:
    /// moves and comparisons only. Upstream: `LU::p().permute_rows(&mut v)`.
    fn permute(self: Lu2<T>, v: Vector2<T>) -> Vector2<T> {
        let mut x1 = v.x;
        let mut x2 = v.y;
        if self.p.p1 == 2 {
            let t = x1;
            x1 = x2;
            x2 = t;
        }
        Vector2 { x: x1, y: x2 }
    }
    /// `P * m`: the same transpositions applied to the ROWS of `m`, so that `lu.permute_rows(a)` is
    /// `lu.l() * lu.u()` up to the rounding of the factorisation (this is how the tests check the
    /// decomposition). Exact: moves and comparisons only. Upstream: `LU::p().permute_rows(&mut m)`.
    fn permute_rows(self: Lu2<T>, m: Matrix2<T>) -> Matrix2<T> {
        let mut a11 = m.m11;
        let mut a12 = m.m12;
        let mut a21 = m.m21;
        let mut a22 = m.m22;
        if self.p.p1 == 2 {
            let t = a11;
            a11 = a21;
            a21 = t;
            let t = a12;
            a12 = a22;
            a22 = t;
        }
        Matrix2 { m11: a11, m21: a21, m12: a12, m22: a22 }
    }
}
