//! Internal, no stability promise: the crate-private items of `linalg::lu::lu6` that the packages
//! above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use nalgebra_shapes6::base::matrix6::Matrix6;
use nalgebra_shapes6::base::vector6::Vector6;
use simba::scalar::Real;
use crate::linalg::lu::lu6::Lu6;

/// Crate-internal kernels of `Lu6<T>` (WP 8.0: the public API is strictly upstream's): the row
/// permutation applied to a vector or to the rows of a matrix, upstream's
/// `lu.p().permute_rows(&mut m)` (to be exposed as `Perm6::permute_rows` by the
/// `PermutationSequence` completion of docs/API_PARITY.md P14).
#[generate_trait]
pub impl Lu6InternalImpl<
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
> of Lu6InternalTrait<T> {
    /// `P * v`: the transpositions applied to the components of `v`, in factorisation order. Exact:
    /// moves and comparisons only. Upstream: `LU::p().permute_rows(&mut v)`.
    fn permute(self: Lu6<T>, v: Vector6<T>) -> Vector6<T> {
        let mut x1 = v.x;
        let mut x2 = v.y;
        let mut x3 = v.z;
        let mut x4 = v.w;
        let mut x5 = v.a;
        let mut x6 = v.b;
        if self.p.p1 == 2 {
            let t = x1;
            x1 = x2;
            x2 = t;
        } else if self.p.p1 == 3 {
            let t = x1;
            x1 = x3;
            x3 = t;
        } else if self.p.p1 == 4 {
            let t = x1;
            x1 = x4;
            x4 = t;
        } else if self.p.p1 == 5 {
            let t = x1;
            x1 = x5;
            x5 = t;
        } else if self.p.p1 == 6 {
            let t = x1;
            x1 = x6;
            x6 = t;
        }
        if self.p.p2 == 3 {
            let t = x2;
            x2 = x3;
            x3 = t;
        } else if self.p.p2 == 4 {
            let t = x2;
            x2 = x4;
            x4 = t;
        } else if self.p.p2 == 5 {
            let t = x2;
            x2 = x5;
            x5 = t;
        } else if self.p.p2 == 6 {
            let t = x2;
            x2 = x6;
            x6 = t;
        }
        if self.p.p3 == 4 {
            let t = x3;
            x3 = x4;
            x4 = t;
        } else if self.p.p3 == 5 {
            let t = x3;
            x3 = x5;
            x5 = t;
        } else if self.p.p3 == 6 {
            let t = x3;
            x3 = x6;
            x6 = t;
        }
        if self.p.p4 == 5 {
            let t = x4;
            x4 = x5;
            x5 = t;
        } else if self.p.p4 == 6 {
            let t = x4;
            x4 = x6;
            x6 = t;
        }
        if self.p.p5 == 6 {
            let t = x5;
            x5 = x6;
            x6 = t;
        }
        Vector6 { x: x1, y: x2, z: x3, w: x4, a: x5, b: x6 }
    }
    /// `P * m`: the same transpositions applied to the ROWS of `m`, so that `lu.permute_rows(a)` is
    /// `lu.l() * lu.u()` up to the rounding of the factorisation (this is how the tests check the
    /// decomposition). Exact: moves and comparisons only. Upstream: `LU::p().permute_rows(&mut m)`.
    fn permute_rows(self: Lu6<T>, m: Matrix6<T>) -> Matrix6<T> {
        let mut a11 = m.m11;
        let mut a12 = m.m12;
        let mut a13 = m.m13;
        let mut a14 = m.m14;
        let mut a15 = m.m15;
        let mut a16 = m.m16;
        let mut a21 = m.m21;
        let mut a22 = m.m22;
        let mut a23 = m.m23;
        let mut a24 = m.m24;
        let mut a25 = m.m25;
        let mut a26 = m.m26;
        let mut a31 = m.m31;
        let mut a32 = m.m32;
        let mut a33 = m.m33;
        let mut a34 = m.m34;
        let mut a35 = m.m35;
        let mut a36 = m.m36;
        let mut a41 = m.m41;
        let mut a42 = m.m42;
        let mut a43 = m.m43;
        let mut a44 = m.m44;
        let mut a45 = m.m45;
        let mut a46 = m.m46;
        let mut a51 = m.m51;
        let mut a52 = m.m52;
        let mut a53 = m.m53;
        let mut a54 = m.m54;
        let mut a55 = m.m55;
        let mut a56 = m.m56;
        let mut a61 = m.m61;
        let mut a62 = m.m62;
        let mut a63 = m.m63;
        let mut a64 = m.m64;
        let mut a65 = m.m65;
        let mut a66 = m.m66;
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
            let t = a14;
            a14 = a24;
            a24 = t;
            let t = a15;
            a15 = a25;
            a25 = t;
            let t = a16;
            a16 = a26;
            a26 = t;
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
            let t = a14;
            a14 = a34;
            a34 = t;
            let t = a15;
            a15 = a35;
            a35 = t;
            let t = a16;
            a16 = a36;
            a36 = t;
        } else if self.p.p1 == 4 {
            let t = a11;
            a11 = a41;
            a41 = t;
            let t = a12;
            a12 = a42;
            a42 = t;
            let t = a13;
            a13 = a43;
            a43 = t;
            let t = a14;
            a14 = a44;
            a44 = t;
            let t = a15;
            a15 = a45;
            a45 = t;
            let t = a16;
            a16 = a46;
            a46 = t;
        } else if self.p.p1 == 5 {
            let t = a11;
            a11 = a51;
            a51 = t;
            let t = a12;
            a12 = a52;
            a52 = t;
            let t = a13;
            a13 = a53;
            a53 = t;
            let t = a14;
            a14 = a54;
            a54 = t;
            let t = a15;
            a15 = a55;
            a55 = t;
            let t = a16;
            a16 = a56;
            a56 = t;
        } else if self.p.p1 == 6 {
            let t = a11;
            a11 = a61;
            a61 = t;
            let t = a12;
            a12 = a62;
            a62 = t;
            let t = a13;
            a13 = a63;
            a63 = t;
            let t = a14;
            a14 = a64;
            a64 = t;
            let t = a15;
            a15 = a65;
            a65 = t;
            let t = a16;
            a16 = a66;
            a66 = t;
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
            let t = a24;
            a24 = a34;
            a34 = t;
            let t = a25;
            a25 = a35;
            a35 = t;
            let t = a26;
            a26 = a36;
            a36 = t;
        } else if self.p.p2 == 4 {
            let t = a21;
            a21 = a41;
            a41 = t;
            let t = a22;
            a22 = a42;
            a42 = t;
            let t = a23;
            a23 = a43;
            a43 = t;
            let t = a24;
            a24 = a44;
            a44 = t;
            let t = a25;
            a25 = a45;
            a45 = t;
            let t = a26;
            a26 = a46;
            a46 = t;
        } else if self.p.p2 == 5 {
            let t = a21;
            a21 = a51;
            a51 = t;
            let t = a22;
            a22 = a52;
            a52 = t;
            let t = a23;
            a23 = a53;
            a53 = t;
            let t = a24;
            a24 = a54;
            a54 = t;
            let t = a25;
            a25 = a55;
            a55 = t;
            let t = a26;
            a26 = a56;
            a56 = t;
        } else if self.p.p2 == 6 {
            let t = a21;
            a21 = a61;
            a61 = t;
            let t = a22;
            a22 = a62;
            a62 = t;
            let t = a23;
            a23 = a63;
            a63 = t;
            let t = a24;
            a24 = a64;
            a64 = t;
            let t = a25;
            a25 = a65;
            a65 = t;
            let t = a26;
            a26 = a66;
            a66 = t;
        }
        if self.p.p3 == 4 {
            let t = a31;
            a31 = a41;
            a41 = t;
            let t = a32;
            a32 = a42;
            a42 = t;
            let t = a33;
            a33 = a43;
            a43 = t;
            let t = a34;
            a34 = a44;
            a44 = t;
            let t = a35;
            a35 = a45;
            a45 = t;
            let t = a36;
            a36 = a46;
            a46 = t;
        } else if self.p.p3 == 5 {
            let t = a31;
            a31 = a51;
            a51 = t;
            let t = a32;
            a32 = a52;
            a52 = t;
            let t = a33;
            a33 = a53;
            a53 = t;
            let t = a34;
            a34 = a54;
            a54 = t;
            let t = a35;
            a35 = a55;
            a55 = t;
            let t = a36;
            a36 = a56;
            a56 = t;
        } else if self.p.p3 == 6 {
            let t = a31;
            a31 = a61;
            a61 = t;
            let t = a32;
            a32 = a62;
            a62 = t;
            let t = a33;
            a33 = a63;
            a63 = t;
            let t = a34;
            a34 = a64;
            a64 = t;
            let t = a35;
            a35 = a65;
            a65 = t;
            let t = a36;
            a36 = a66;
            a66 = t;
        }
        if self.p.p4 == 5 {
            let t = a41;
            a41 = a51;
            a51 = t;
            let t = a42;
            a42 = a52;
            a52 = t;
            let t = a43;
            a43 = a53;
            a53 = t;
            let t = a44;
            a44 = a54;
            a54 = t;
            let t = a45;
            a45 = a55;
            a55 = t;
            let t = a46;
            a46 = a56;
            a56 = t;
        } else if self.p.p4 == 6 {
            let t = a41;
            a41 = a61;
            a61 = t;
            let t = a42;
            a42 = a62;
            a62 = t;
            let t = a43;
            a43 = a63;
            a63 = t;
            let t = a44;
            a44 = a64;
            a64 = t;
            let t = a45;
            a45 = a65;
            a65 = t;
            let t = a46;
            a46 = a66;
            a66 = t;
        }
        if self.p.p5 == 6 {
            let t = a51;
            a51 = a61;
            a61 = t;
            let t = a52;
            a52 = a62;
            a62 = t;
            let t = a53;
            a53 = a63;
            a63 = t;
            let t = a54;
            a54 = a64;
            a64 = t;
            let t = a55;
            a55 = a65;
            a65 = t;
            let t = a56;
            a56 = a66;
            a66 = t;
        }
        Matrix6 {
            m11: a11,
            m21: a21,
            m31: a31,
            m12: a12,
            m22: a22,
            m32: a32,
            m13: a13,
            m23: a23,
            m33: a33,
            m41: a41,
            m51: a51,
            m61: a61,
            m42: a42,
            m52: a52,
            m62: a62,
            m43: a43,
            m53: a53,
            m63: a63,
            m14: a14,
            m24: a24,
            m34: a34,
            m15: a15,
            m25: a25,
            m35: a35,
            m16: a16,
            m26: a26,
            m36: a36,
            m44: a44,
            m54: a54,
            m64: a64,
            m45: a45,
            m55: a55,
            m65: a65,
            m46: a46,
            m56: a56,
            m66: a66,
        }
    }
}
