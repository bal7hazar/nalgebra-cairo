//! Internal, no stability promise: the crate-private items of `base::matrix4` that the packages
//! above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use nalgebra_types4::base::matrix4::Matrix4;
use nalgebra_types4::base::vector4::Vector4;
use simba::scalar::Real;

/// Internal kernels of `Matrix4`.
#[generate_trait]
pub impl Matrix4Kernels<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>,
> of Matrix4KernelsTrait<T> {
    /// `a*b - c*d + e*f`, accumulated exactly and rounded once.
    #[inline(always)]
    fn pmp(a: T, b: T, c: T, d: T, e: T, f: T) -> T {
        let w = R::wide_add_prod(R::wide_zero(), a, b);
        R::wide_rescale(R::wide_add_prod(R::wide_sub_prod(w, c, d), e, f))
    }

    /// `-a*b + c*d - e*f`, accumulated exactly and rounded once.
    #[inline(always)]
    fn mpm(a: T, b: T, c: T, d: T, e: T, f: T) -> T {
        let w = R::wide_sub_prod(R::wide_zero(), a, b);
        R::wide_rescale(R::wide_sub_prod(R::wide_add_prod(w, c, d), e, f))
    }

    /// `(adjugate, determinant)` from the twelve 2x2 minors of the two upper rows (`s0..s5`)
    /// and of the two lower rows (`c0..c5`): 12 `diff_prod`, then 16 three-term and one six-term
    /// exact accumulations (two roundings per output).
    #[inline(always)]
    fn adjugate_determinant(m: Matrix4<T>) -> (Matrix4<T>, T) {
        let s0 = R::diff_prod(m.m11, m.m22, m.m21, m.m12);
        let s1 = R::diff_prod(m.m11, m.m23, m.m21, m.m13);
        let s2 = R::diff_prod(m.m11, m.m24, m.m21, m.m14);
        let s3 = R::diff_prod(m.m12, m.m23, m.m22, m.m13);
        let s4 = R::diff_prod(m.m12, m.m24, m.m22, m.m14);
        let s5 = R::diff_prod(m.m13, m.m24, m.m23, m.m14);
        let c5 = R::diff_prod(m.m33, m.m44, m.m43, m.m34);
        let c4 = R::diff_prod(m.m32, m.m44, m.m42, m.m34);
        let c3 = R::diff_prod(m.m32, m.m43, m.m42, m.m33);
        let c2 = R::diff_prod(m.m31, m.m44, m.m41, m.m34);
        let c1 = R::diff_prod(m.m31, m.m43, m.m41, m.m33);
        let c0 = R::diff_prod(m.m31, m.m42, m.m41, m.m32);
        let w = R::wide_add_prod(R::wide_zero(), s0, c5);
        let w = R::wide_sub_prod(w, s1, c4);
        let w = R::wide_add_prod(w, s2, c3);
        let w = R::wide_add_prod(w, s3, c2);
        let w = R::wide_sub_prod(w, s4, c1);
        let det = R::wide_rescale(R::wide_add_prod(w, s5, c0));
        let adj = Matrix4 {
            m11: Self::pmp(m.m22, c5, m.m23, c4, m.m24, c3),
            m21: Self::mpm(m.m21, c5, m.m23, c2, m.m24, c1),
            m31: Self::pmp(m.m21, c4, m.m22, c2, m.m24, c0),
            m41: Self::mpm(m.m21, c3, m.m22, c1, m.m23, c0),
            m12: Self::mpm(m.m12, c5, m.m13, c4, m.m14, c3),
            m22: Self::pmp(m.m11, c5, m.m13, c2, m.m14, c1),
            m32: Self::mpm(m.m11, c4, m.m12, c2, m.m14, c0),
            m42: Self::pmp(m.m11, c3, m.m12, c1, m.m13, c0),
            m13: Self::pmp(m.m42, s5, m.m43, s4, m.m44, s3),
            m23: Self::mpm(m.m41, s5, m.m43, s2, m.m44, s1),
            m33: Self::pmp(m.m41, s4, m.m42, s2, m.m44, s0),
            m43: Self::mpm(m.m41, s3, m.m42, s1, m.m43, s0),
            m14: Self::mpm(m.m32, s5, m.m33, s4, m.m34, s3),
            m24: Self::pmp(m.m31, s5, m.m33, s2, m.m34, s1),
            m34: Self::mpm(m.m31, s4, m.m32, s2, m.m34, s0),
            m44: Self::pmp(m.m31, s3, m.m32, s1, m.m33, s0),
        };
        (adj, det)
    }
}

/// Crate-internal kernels of `Matrix4<T>` with no upstream method of that name or shape (WP 8.0:
/// the public API is strictly upstream's). The structured kernels of DESIGN D4 (`from_outer`,
/// `adjugate`) and the unrolled row / column accessors that stand for upstream `row(i)` /
/// `column(i)` views, used by the decompositions.
#[generate_trait]
pub impl Matrix4InternalImpl<
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
> of Matrix4InternalTrait<T> {
    /// Column 1. Upstream: `column(0)`.
    #[inline(always)]
    fn column1(self: Matrix4<T>) -> Vector4<T> {
        Vector4 { x: self.m11, y: self.m21, z: self.m31, w: self.m41 }
    }

    /// Column 2. Upstream: `column(1)`.
    #[inline(always)]
    fn column2(self: Matrix4<T>) -> Vector4<T> {
        Vector4 { x: self.m12, y: self.m22, z: self.m32, w: self.m42 }
    }

    /// Column 3. Upstream: `column(2)`.
    #[inline(always)]
    fn column3(self: Matrix4<T>) -> Vector4<T> {
        Vector4 { x: self.m13, y: self.m23, z: self.m33, w: self.m43 }
    }

    /// Column 4. Upstream: `column(3)`.
    #[inline(always)]
    fn column4(self: Matrix4<T>) -> Vector4<T> {
        Vector4 { x: self.m14, y: self.m24, z: self.m34, w: self.m44 }
    }

    /// Row 1, as a (column) vector. Upstream: `row(0).transpose()`.
    #[inline(always)]
    fn row1(self: Matrix4<T>) -> Vector4<T> {
        Vector4 { x: self.m11, y: self.m12, z: self.m13, w: self.m14 }
    }

    /// Row 2, as a (column) vector. Upstream: `row(1).transpose()`.
    #[inline(always)]
    fn row2(self: Matrix4<T>) -> Vector4<T> {
        Vector4 { x: self.m21, y: self.m22, z: self.m23, w: self.m24 }
    }

    /// Row 3, as a (column) vector. Upstream: `row(2).transpose()`.
    #[inline(always)]
    fn row3(self: Matrix4<T>) -> Vector4<T> {
        Vector4 { x: self.m31, y: self.m32, z: self.m33, w: self.m34 }
    }

    /// Row 4, as a (column) vector. Upstream: `row(3).transpose()`.
    #[inline(always)]
    fn row4(self: Matrix4<T>) -> Vector4<T> {
        Vector4 { x: self.m41, y: self.m42, z: self.m43, w: self.m44 }
    }

    /// The outer product `a * bᵀ`: each component is one floored product. Panics with the
    /// scalar's overflow error. Upstream: `a * b.transpose()`.
    #[inline(always)]
    fn from_outer(a: Vector4<T>, b: Vector4<T>) -> Matrix4<T> {
        Matrix4 {
            m11: a.x * b.x,
            m21: a.y * b.x,
            m31: a.z * b.x,
            m41: a.w * b.x,
            m12: a.x * b.y,
            m22: a.y * b.y,
            m32: a.z * b.y,
            m42: a.w * b.y,
            m13: a.x * b.z,
            m23: a.y * b.z,
            m33: a.z * b.z,
            m43: a.w * b.z,
            m14: a.x * b.w,
            m24: a.y * b.w,
            m34: a.z * b.w,
            m44: a.w * b.w,
        }
    }

    /// The adjugate (transposed cofactor matrix): `self * adjugate = determinant * I`. 12
    /// `diff_prod` (2x2 minors) then 16 exact three-term accumulations: two roundings per
    /// component. Panics on overflow.
    /// No upstream equivalent (upstream `adjoint` is the conjugate transpose).
    fn adjugate(self: Matrix4<T>) -> Matrix4<T> {
        let (adj, _) = Matrix4Kernels::adjugate_determinant(self);
        adj
    }
}
