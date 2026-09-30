//! In `nalgebra_linalg4`: the `struct` `Ldlt4`; the impl `Ldlt4Impl`. This module is split over
//! packages; the other parts are in `nalgebra_linalg2`, `nalgebra_linalg3`, `nalgebra_linalg6`.
//!
//! Internal, no stability promise: the crate-private items of `linalg::ldlt` that the packages
//! above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use nalgebra_types4::base::matrix4::Matrix4;
use nalgebra_types4::base::vector4::Vector4;
use simba::scalar::Real;

/// The `LDLᵀ` factorisation `a = l * diag(d) * lᵀ` of a symmetric 4x4 matrix: the 6
/// strictly lower components of the UNIT lower triangular `l` (its diagonal is an implicit 1 and
/// is not stored) and the diagonal `d`.
///
/// `lIJ` is the entry at row `I`, column `J` (1-based, `I > J`). Fields are declared column by
/// column, then `d`, so `Serde` writes `l21, .., l41, l32, .., d`. Build one with
/// `Ldlt4Trait::new`; the fields are public so that a factor computed elsewhere can be
/// re-assembled, and nothing checks that they form a valid factor.
#[derive(Copy, Drop, PartialEq, Serde, Default, Debug, Hash)]
pub struct Ldlt4<T> {
    /// Row 2, column 1 of the unit lower triangular factor.
    pub l21: T,
    /// Row 3, column 1 of the unit lower triangular factor.
    pub l31: T,
    /// Row 4, column 1 of the unit lower triangular factor.
    pub l41: T,
    /// Row 3, column 2 of the unit lower triangular factor.
    pub l32: T,
    /// Row 4, column 2 of the unit lower triangular factor.
    pub l42: T,
    /// Row 4, column 3 of the unit lower triangular factor.
    pub l43: T,
    /// The diagonal of `diag(d)`.
    pub d: Vector4<T>,
}

/// Methods of `Ldlt4<T>` for any `Real` scalar.
#[generate_trait]
pub impl Ldlt4Impl<
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
> of Ldlt4Trait<T> {
    /// The `LDLᵀ` factorisation `a = l * diag(d) * lᵀ` of the symmetric `a`, or `None` when a
    /// pivot is exactly zero. Upstream: `UDU::new` on the reversed matrix (see the module doc).
    ///
    /// Like upstream, only the LOWER triangle of `a` is read (the entries at row `i`,
    /// column `j` with `i >= j`): the strictly upper triangle is ignored and the symmetry
    /// of `a` is NOT checked. There is no `SymMatrix4` type, so a symmetric 4x4
    /// matrix travels as a plain `Matrix4`.
    ///
    /// Column by column (j = 1..4), in terms of the numerators `n_ij = a_ij - Σ_(k<j) l_ik·n_jk`
    /// for `i >= j`: each is ONE exact accumulation floored ONCE, the pivot is `d_j = n_jj`, and
    /// `l_ij = n_ij / d_j` is one correctly rounded division. 6 divisions, no square root
    /// and no other product.
    ///
    /// `n_jk` IS the column of `l·diag(d)`: `l_jk·d_k` differs from it only by the remainder of
    /// the division that produced `l_jk`. Recomputing `l_jk·d_k` explicitly (the `alt_products`
    /// candidate of `benches.cairo`) costs 6 extra products, makes `new` 30 to 32 %
    /// dearer, and moves the factors FARTHER from the true ones: `a_jj - l_jk·(l_jk·d_k)` is
    /// second order in the rounding error of `l_jk` where `a_jj - l_jk·n_jk` is only first order
    /// (16 ulp against 34 at size 2, 27 against 42 at size 4). The backward error
    /// `l·diag(d)·lᵀ - a` is identical for both, being dominated by `l_ij·d_j - a_ij`, which
    /// no choice here changes — so the cheaper and more accurate form is the one that ships.
    ///
    /// SINGULARITY CRITERION: `None` iff a pivot `d_j` is EXACTLY zero, i.e. iff the leading
    /// principal minor of order j is zero after flooring. Unlike `Cholesky::new`, a negative
    /// pivot is accepted: `LDLᵀ` factorises any symmetric matrix whose leading principal minors
    /// are non-zero, indefinite ones included, and `d` then has mixed signs. It is NOT a
    /// definiteness test, and it has no pivoting: a symmetric matrix with a zero leading minor
    /// (`[[0, 1], [1, 0]]`) is rejected although it is invertible.
    ///
    /// Panics on overflow of a pivot or a numerator; never wraps.
    fn new(a: Matrix4<T>) -> Option<Ldlt4<T>> {
        let d1 = a.m11;
        if d1 == R::zero() {
            return None;
        }
        let n21 = a.m21;
        let l21 = R::div(n21, d1);
        let n31 = a.m31;
        let l31 = R::div(n31, d1);
        let n41 = a.m41;
        let l41 = R::div(n41, d1);
        let w = R::wide_add(R::wide_zero(), a.m22);
        let w = R::wide_sub_prod(w, l21, n21);
        let d2 = R::wide_rescale(w);
        if d2 == R::zero() {
            return None;
        }
        let w = R::wide_add(R::wide_zero(), a.m32);
        let w = R::wide_sub_prod(w, l31, n21);
        let n32 = R::wide_rescale(w);
        let l32 = R::div(n32, d2);
        let w = R::wide_add(R::wide_zero(), a.m42);
        let w = R::wide_sub_prod(w, l41, n21);
        let n42 = R::wide_rescale(w);
        let l42 = R::div(n42, d2);
        let w = R::wide_add(R::wide_zero(), a.m33);
        let w = R::wide_sub_prod(w, l31, n31);
        let w = R::wide_sub_prod(w, l32, n32);
        let d3 = R::wide_rescale(w);
        if d3 == R::zero() {
            return None;
        }
        let w = R::wide_add(R::wide_zero(), a.m43);
        let w = R::wide_sub_prod(w, l41, n31);
        let w = R::wide_sub_prod(w, l42, n32);
        let n43 = R::wide_rescale(w);
        let l43 = R::div(n43, d3);
        let w = R::wide_add(R::wide_zero(), a.m44);
        let w = R::wide_sub_prod(w, l41, n41);
        let w = R::wide_sub_prod(w, l42, n42);
        let w = R::wide_sub_prod(w, l43, n43);
        let d4 = R::wide_rescale(w);
        if d4 == R::zero() {
            return None;
        }
        Some(Ldlt4 { l21, l31, l41, l32, l42, l43, d: Vector4 { x: d1, y: d2, z: d3, w: d4 } })
    }

    /// The unit lower triangular factor, with explicit ones on the diagonal and zeros above it.
    /// Upstream: the reverse-permuted `UDU::u`. Exact: nothing is recomputed.
    #[inline(always)]
    fn l(self: Ldlt4<T>) -> Matrix4<T> {
        Matrix4 {
            m11: R::one(),
            m21: self.l21,
            m31: self.l31,
            m41: self.l41,
            m12: R::zero(),
            m22: R::one(),
            m32: self.l32,
            m42: self.l42,
            m13: R::zero(),
            m23: R::zero(),
            m33: R::one(),
            m43: self.l43,
            m14: R::zero(),
            m24: R::zero(),
            m34: R::zero(),
            m44: R::one(),
        }
    }

    /// The diagonal of `diag(d)`. Upstream: `UDU::d`, reversed. Exact: the stored vector.
    #[inline(always)]
    fn d(self: Ldlt4<T>) -> Vector4<T> {
        self.d
    }

    /// The solution `x` of `a * x = b`: forward substitution on the unit `l`, division by `d`,
    /// back substitution on the unit `lᵀ`. Upstream: `UDU` has no `solve`; the oracle's
    /// `udu4_solve` vectors apply unchanged.
    ///
    /// The unit diagonal removes the divisions of the two substitutions, so this costs 4
    /// divisions where `Cholesky4::solve` costs 8: each of the 8 substitution
    /// intermediates is ONE exact accumulation floored once and nothing more, and only the
    /// diagonal solve `z_i = y_i / d_i` divides.
    ///
    /// Panics on overflow. A factor built by `new` has non-zero pivots, so no division by zero can
    /// occur; a hand-assembled factor with a zero pivot panics with the scalar's error.
    fn solve(self: Ldlt4<T>, b: Vector4<T>) -> Vector4<T> {
        let y1 = b.x;
        let w = R::wide_add(R::wide_zero(), b.y);
        let w = R::wide_sub_prod(w, self.l21, y1);
        let y2 = R::wide_rescale(w);
        let w = R::wide_add(R::wide_zero(), b.z);
        let w = R::wide_sub_prod(w, self.l31, y1);
        let w = R::wide_sub_prod(w, self.l32, y2);
        let y3 = R::wide_rescale(w);
        let w = R::wide_add(R::wide_zero(), b.w);
        let w = R::wide_sub_prod(w, self.l41, y1);
        let w = R::wide_sub_prod(w, self.l42, y2);
        let w = R::wide_sub_prod(w, self.l43, y3);
        let y4 = R::wide_rescale(w);
        let z1 = R::div(y1, self.d.x);
        let z2 = R::div(y2, self.d.y);
        let z3 = R::div(y3, self.d.z);
        let z4 = R::div(y4, self.d.w);
        let x4 = z4;
        let w = R::wide_add(R::wide_zero(), z3);
        let w = R::wide_sub_prod(w, self.l43, x4);
        let x3 = R::wide_rescale(w);
        let w = R::wide_add(R::wide_zero(), z2);
        let w = R::wide_sub_prod(w, self.l32, x3);
        let w = R::wide_sub_prod(w, self.l42, x4);
        let x2 = R::wide_rescale(w);
        let w = R::wide_add(R::wide_zero(), z1);
        let w = R::wide_sub_prod(w, self.l21, x2);
        let w = R::wide_sub_prod(w, self.l31, x3);
        let w = R::wide_sub_prod(w, self.l41, x4);
        let x1 = R::wide_rescale(w);
        Vector4 { x: x1, y: x2, z: x3, w: x4 }
    }

    /// `a⁻¹ = l⁻ᵀ · diag(d)⁻¹ · l⁻¹`, as a full (symmetric) `Matrix4`.
    ///
    /// `q = l⁻¹` is unit lower triangular and is built WITHOUT any division
    /// (`q_ij = -(l_ij + Σ_(k=j+1..i-1) l_ik·q_kj)`, one exact accumulation floored once); the
    /// 10 scaled entries `s_kj = q_kj / d_k` are the only divisions (4 of them are
    /// `recip(d_k)`, the `1 / d_k` rounded to nearest), and `a⁻¹_ij = Σ_k q_ki·s_kj` is one
    /// exact accumulation floored once per output component.
    ///
    /// The result is symmetric by construction, so only its upper triangle is computed — the sum
    /// is taken as `Σ_k q_ki·s_kj` with `i <= j`, which for rounded `s` differs from
    /// `Σ_k q_kj·s_ki` by at most a few ulp. Panics on overflow.
    fn inverse(self: Ldlt4<T>) -> Matrix4<T> {
        let q21 = -self.l21;
        let w = R::wide_sub(R::wide_zero(), self.l31);
        let w = R::wide_sub_prod(w, self.l32, q21);
        let q31 = R::wide_rescale(w);
        let w = R::wide_sub(R::wide_zero(), self.l41);
        let w = R::wide_sub_prod(w, self.l42, q21);
        let w = R::wide_sub_prod(w, self.l43, q31);
        let q41 = R::wide_rescale(w);
        let q32 = -self.l32;
        let w = R::wide_sub(R::wide_zero(), self.l42);
        let w = R::wide_sub_prod(w, self.l43, q32);
        let q42 = R::wide_rescale(w);
        let q43 = -self.l43;
        let s11 = R::recip(self.d.x);
        let s21 = R::div(q21, self.d.y);
        let s31 = R::div(q31, self.d.z);
        let (s41, s42, s43) = R::div3(q41, q42, q43, self.d.w);
        let s22 = R::recip(self.d.y);
        let s32 = R::div(q32, self.d.z);
        let s33 = R::recip(self.d.z);
        let s44 = R::recip(self.d.w);
        let w = R::wide_zero();
        let w = R::wide_add(w, s11);
        let w = R::wide_add_prod(w, q21, s21);
        let w = R::wide_add_prod(w, q31, s31);
        let w = R::wide_add_prod(w, q41, s41);
        let r11 = R::wide_rescale(w);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, q21, s22);
        let w = R::wide_add_prod(w, q31, s32);
        let w = R::wide_add_prod(w, q41, s42);
        let r12 = R::wide_rescale(w);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, q31, s33);
        let w = R::wide_add_prod(w, q41, s43);
        let r13 = R::wide_rescale(w);
        let r14 = q41 * s44;
        let w = R::wide_zero();
        let w = R::wide_add(w, s22);
        let w = R::wide_add_prod(w, q32, s32);
        let w = R::wide_add_prod(w, q42, s42);
        let r22 = R::wide_rescale(w);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, q32, s33);
        let w = R::wide_add_prod(w, q42, s43);
        let r23 = R::wide_rescale(w);
        let r24 = q42 * s44;
        let w = R::wide_zero();
        let w = R::wide_add(w, s33);
        let w = R::wide_add_prod(w, q43, s43);
        let r33 = R::wide_rescale(w);
        let r34 = q43 * s44;
        let r44 = s44;
        Matrix4 {
            m11: r11,
            m21: r12,
            m31: r13,
            m41: r14,
            m12: r12,
            m22: r22,
            m32: r23,
            m42: r24,
            m13: r13,
            m23: r23,
            m33: r33,
            m43: r34,
            m14: r14,
            m24: r24,
            m34: r34,
            m44: r44,
        }
    }

    /// `det(a) = Π d_j`, as a balanced product tree: 3 floored products, the
    /// shallowest chain of roundings for 4 factors. `det(l) = 1`, so no other term appears.
    ///
    /// Panics if the determinant does not fit the scalar. Underflow is silent (a determinant below
    /// 1 ulp floors to 0), which is the scalar's floor rounding and NOT a singularity report.
    #[inline(always)]
    fn determinant(self: Ldlt4<T>) -> T {
        (self.d.x * self.d.y) * (self.d.z * self.d.w)
    }
}
