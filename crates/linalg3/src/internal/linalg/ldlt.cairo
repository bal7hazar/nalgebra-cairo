//! In `nalgebra_linalg3`: the `struct` `Ldlt3`; the impl `Ldlt3Impl`. This module is split over
//! packages; the other parts are in `nalgebra_linalg2`, `nalgebra_linalg4`, `nalgebra_linalg6`.
//!
//! Internal, no stability promise: the crate-private items of `linalg::ldlt` that the packages
//! above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use nalgebra_types3::base::matrix3::Matrix3;
use nalgebra_types3::base::vector3::Vector3;
use nalgebra_types3::internal::base::sym_matrix3::SymMatrix3;
use simba::scalar::Real;

/// The `LDLᵀ` factorisation `a = l * diag(d) * lᵀ` of a symmetric 3x3 matrix: the 3
/// strictly lower components of the UNIT lower triangular `l` (its diagonal is an implicit 1 and
/// is not stored) and the diagonal `d`.
///
/// `lIJ` is the entry at row `I`, column `J` (1-based, `I > J`). Fields are declared column by
/// column, then `d`, so `Serde` writes `l21, .., l31, l32, .., d`. Build one with
/// `Ldlt3Trait::new`; the fields are public so that a factor computed elsewhere can be
/// re-assembled, and nothing checks that they form a valid factor.
#[derive(Copy, Drop, PartialEq, Serde, Default, Debug, Hash)]
pub struct Ldlt3<T> {
    /// Row 2, column 1 of the unit lower triangular factor.
    pub l21: T,
    /// Row 3, column 1 of the unit lower triangular factor.
    pub l31: T,
    /// Row 3, column 2 of the unit lower triangular factor.
    pub l32: T,
    /// The diagonal of `diag(d)`.
    pub d: Vector3<T>,
}

/// Methods of `Ldlt3<T>` for any `Real` scalar.
#[generate_trait]
pub impl Ldlt3Impl<
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
> of Ldlt3Trait<T> {
    /// The `LDLᵀ` factorisation `a = l * diag(d) * lᵀ` of the symmetric `a`, or `None` when a
    /// pivot is exactly zero. Upstream: `UDU::new` on the reversed matrix (see the module doc).
    ///
    /// `a` is a `SymMatrix3`, so there is no triangle to choose: its 6 independent
    /// components ARE the matrix.
    ///
    /// Column by column (j = 1..3), in terms of the numerators `n_ij = a_ij - Σ_(k<j) l_ik·n_jk`
    /// for `i >= j`: each is ONE exact accumulation floored ONCE, the pivot is `d_j = n_jj`, and
    /// `l_ij = n_ij / d_j` is one correctly rounded division. 3 divisions, no square root
    /// and no other product.
    ///
    /// `n_jk` IS the column of `l·diag(d)`: `l_jk·d_k` differs from it only by the remainder of
    /// the division that produced `l_jk`. Recomputing `l_jk·d_k` explicitly (the `alt_products`
    /// candidate of `benches.cairo`) costs 3 extra products, makes `new` 30 to 32 %
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
    fn new(a: SymMatrix3<T>) -> Option<Ldlt3<T>> {
        let d1 = a.m11;
        if d1 == R::zero() {
            return None;
        }
        let n21 = a.m12;
        let l21 = R::div(n21, d1);
        let n31 = a.m13;
        let l31 = R::div(n31, d1);
        let w = R::wide_add(R::wide_zero(), a.m22);
        let w = R::wide_sub_prod(w, l21, n21);
        let d2 = R::wide_rescale(w);
        if d2 == R::zero() {
            return None;
        }
        let w = R::wide_add(R::wide_zero(), a.m23);
        let w = R::wide_sub_prod(w, l31, n21);
        let n32 = R::wide_rescale(w);
        let l32 = R::div(n32, d2);
        let w = R::wide_add(R::wide_zero(), a.m33);
        let w = R::wide_sub_prod(w, l31, n31);
        let w = R::wide_sub_prod(w, l32, n32);
        let d3 = R::wide_rescale(w);
        if d3 == R::zero() {
            return None;
        }
        Some(Ldlt3 { l21, l31, l32, d: Vector3 { x: d1, y: d2, z: d3 } })
    }

    /// The unit lower triangular factor, with explicit ones on the diagonal and zeros above it.
    /// Upstream: the reverse-permuted `UDU::u`. Exact: nothing is recomputed.
    #[inline(always)]
    fn l(self: Ldlt3<T>) -> Matrix3<T> {
        Matrix3 {
            m11: R::one(),
            m21: self.l21,
            m31: self.l31,
            m12: R::zero(),
            m22: R::one(),
            m32: self.l32,
            m13: R::zero(),
            m23: R::zero(),
            m33: R::one(),
        }
    }

    /// The diagonal of `diag(d)`. Upstream: `UDU::d`, reversed. Exact: the stored vector.
    #[inline(always)]
    fn d(self: Ldlt3<T>) -> Vector3<T> {
        self.d
    }

    /// The solution `x` of `a * x = b`: forward substitution on the unit `l`, division by `d`,
    /// back substitution on the unit `lᵀ`. Upstream: `UDU` has no `solve`; the oracle's
    /// `udu3_solve` vectors apply unchanged.
    ///
    /// The unit diagonal removes the divisions of the two substitutions, so this costs 3
    /// divisions where `Cholesky3::solve` costs 6: each of the 6 substitution
    /// intermediates is ONE exact accumulation floored once and nothing more, and only the
    /// diagonal solve `z_i = y_i / d_i` divides.
    ///
    /// Panics on overflow. A factor built by `new` has non-zero pivots, so no division by zero can
    /// occur; a hand-assembled factor with a zero pivot panics with the scalar's error.
    fn solve(self: Ldlt3<T>, b: Vector3<T>) -> Vector3<T> {
        let y1 = b.x;
        let w = R::wide_add(R::wide_zero(), b.y);
        let w = R::wide_sub_prod(w, self.l21, y1);
        let y2 = R::wide_rescale(w);
        let w = R::wide_add(R::wide_zero(), b.z);
        let w = R::wide_sub_prod(w, self.l31, y1);
        let w = R::wide_sub_prod(w, self.l32, y2);
        let y3 = R::wide_rescale(w);
        let z1 = R::div(y1, self.d.x);
        let z2 = R::div(y2, self.d.y);
        let z3 = R::div(y3, self.d.z);
        let x3 = z3;
        let w = R::wide_add(R::wide_zero(), z2);
        let w = R::wide_sub_prod(w, self.l32, x3);
        let x2 = R::wide_rescale(w);
        let w = R::wide_add(R::wide_zero(), z1);
        let w = R::wide_sub_prod(w, self.l21, x2);
        let w = R::wide_sub_prod(w, self.l31, x3);
        let x1 = R::wide_rescale(w);
        Vector3 { x: x1, y: x2, z: x3 }
    }

    /// `a⁻¹ = l⁻ᵀ · diag(d)⁻¹ · l⁻¹`, as its 6 independent components.
    ///
    /// `q = l⁻¹` is unit lower triangular and is built WITHOUT any division
    /// (`q_ij = -(l_ij + Σ_(k=j+1..i-1) l_ik·q_kj)`, one exact accumulation floored once); the
    /// 6 scaled entries `s_kj = q_kj / d_k` are the only divisions (3 of them are
    /// `recip(d_k)`, the `1 / d_k` rounded to nearest), and `a⁻¹_ij = Σ_k q_ki·s_kj` is one
    /// exact accumulation floored once per output component.
    ///
    /// The result is symmetric by construction, so only its upper triangle is computed — the sum
    /// is taken as `Σ_k q_ki·s_kj` with `i <= j`, which for rounded `s` differs from
    /// `Σ_k q_kj·s_ki` by at most a few ulp. Panics on overflow.
    fn inverse(self: Ldlt3<T>) -> SymMatrix3<T> {
        let q21 = -self.l21;
        let w = R::wide_sub(R::wide_zero(), self.l31);
        let w = R::wide_sub_prod(w, self.l32, q21);
        let q31 = R::wide_rescale(w);
        let q32 = -self.l32;
        let s11 = R::recip(self.d.x);
        let s21 = R::div(q21, self.d.y);
        let s31 = R::div(q31, self.d.z);
        let s22 = R::recip(self.d.y);
        let s32 = R::div(q32, self.d.z);
        let s33 = R::recip(self.d.z);
        let w = R::wide_zero();
        let w = R::wide_add(w, s11);
        let w = R::wide_add_prod(w, q21, s21);
        let w = R::wide_add_prod(w, q31, s31);
        let r11 = R::wide_rescale(w);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, q21, s22);
        let w = R::wide_add_prod(w, q31, s32);
        let r12 = R::wide_rescale(w);
        let r13 = q31 * s33;
        let w = R::wide_zero();
        let w = R::wide_add(w, s22);
        let w = R::wide_add_prod(w, q32, s32);
        let r22 = R::wide_rescale(w);
        let r23 = q32 * s33;
        let r33 = s33;
        SymMatrix3 { m11: r11, m12: r12, m13: r13, m22: r22, m23: r23, m33: r33 }
    }

    /// `det(a) = Π d_j`, as a balanced product tree: 2 floored products, the
    /// shallowest chain of roundings for 3 factors. `det(l) = 1`, so no other term appears.
    ///
    /// Panics if the determinant does not fit the scalar. Underflow is silent (a determinant below
    /// 1 ulp floors to 0), which is the scalar's floor rounding and NOT a singularity report.
    #[inline(always)]
    fn determinant(self: Ldlt3<T>) -> T {
        self.d.x * (self.d.y * self.d.z)
    }
}
