//! `Qr3`: the QR factorisation of a `Matrix3` (upstream `nalgebra::linalg::QR` on a 3x3 matrix).
//!
//! `A = Q * R` with `Q` orthonormal and `R` upper triangular with a non-negative diagonal — the
//! convention of upstream's unpacked `q()` / `r()`, see the module doc of `linalg::qr`.

// the in-crate tests reach the internal items of the module (and the other modules' tests
// through `crate::linalg::...`) here (WP 9-NS9)
#[cfg(test)]
pub(crate) use nalgebra_linalg3::internal::linalg::qr::qr3::Qr3InternalTrait;
pub use nalgebra_linalg3::linalg::qr::qr3::*;

/// Test-only field-wise equality (upstream `Qr3` has no `PartialEq`): the tests and the
/// benchmarks compare factors through it.
#[cfg(test)]
impl Qr3PartialEq<T, +PartialEq<T>> of PartialEq<Qr3<T>> {
    fn eq(lhs: @Qr3<T>, rhs: @Qr3<T>) -> bool {
        lhs.q == rhs.q && lhs.r == rhs.r
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests of `Qr3`, and the two alternative orthogonalisations that lost, kept as evidence
    //! together with the tests that show why (AGENTS.md rule 8):
    //!
    //! - `alt_householder`: upstream's algorithm, two reflections plus the sign normalisation.
    //! It lands FURTHER from the oracle factors than the shipped modified Gram-Schmidt (402 ulp
    //! against 121) and costs more gas: in fixed point each reflector pays a normalisation of its
    //! own axis and then two rounded reflections per column, where Gram-Schmidt rounds once per
    //! entry. Its `Q` is slightly more orthonormal (29 ulp against 36), which is the one thing
    //! the textbook promises and the only thing it wins here.
    //!
    //! - `alt_classical_gram_schmidt`: the projections of column 3 both taken against the ORIGINAL
    //! column. Same arithmetic, statement for statement, and the same orthonormality on these
    //! well-conditioned inputs; it measures 9 % cheaper (70 880 against 77 890 on `fixed` 0.3.0)
    //! only because it divides UNGUARDED — it panics with a division by zero on a matrix with a
    //! zero column, where the shipped `new` returns `r_ii = 0` — so its figure also prices the
    //! three `r_ii == 0` guards the shipped `new` carries. Worse in theory, see `new`; re-ranked
    //! in WP 7.2: kept as the loser.
    //!
    //! - `alt_completed_basis`: the rank-deficient fallback that completes `Q` to an orthonormal
    //! basis instead of leaving a zero column. Strictly dearer, on every call.
    //!
    //! In-crate part (WP 8.1c): the tests of crate-internal items only; the tests of the public API
    //! are in `crates/tests_linalg/src/qr/qr3/tests.cairo`.

    use fixed::Fixed;
    use nalgebra_static3::internal::base::matrix3::Matrix3InternalTrait;
    use nalgebra_static3::internal::base::vector3::Vector3InternalTrait;
    use simba::scalar::Real;
    use crate::base::matrix3::{Matrix3, Matrix3Trait};
    use crate::base::matrix_test_utils::{
        amax_m3, int, m3, max_ulp_diff3, orthonormality_error_m3, ulp_diff, v3t,
    };
    use crate::base::vector3::{Vector3, Vector3Trait};
    use crate::linalg::qr::oracle_qr3 as oracle;
    use crate::testing::black_box;
    use super::{Qr3, Qr3InternalTrait, Qr3Trait};

    /// The oracle's first 3x3 case: the benchmark input.
    fn a_bench() -> Matrix3<Fixed> {
        m3(
            [
                [-930291762, 206204041, 164059380], [-1057407058, -1146357637, -192379840],
                [-394072183, -440329712, 775368183],
            ],
        )
    }

    /// A right-hand side, from `qr3_solve`.
    fn b_bench() -> Vector3<Fixed> {
        v3t((-1811584373, 4204441265, -4234532070))
    }

    /// `a_bench()` already factored, so the benchmarks of the derived operations do not pay for
    /// `new`.
    fn f_bench() -> Qr3<Fixed> {
        Qr3Trait::new(a_bench())
    }

    /// A matrix whose QR is exact in fixed point: three mutually orthogonal columns whose norms
    /// are powers of two, so every normalisation divides exactly.
    fn a_exact() -> Matrix3<Fixed> {
        Matrix3Trait::new(int(2), int(0), int(0), int(0), int(0), int(-8), int(0), int(4), int(0))
    }

    /// A rank-2 matrix: the third column is the sum of the first two, so `r33` is exactly zero.
    fn a_rank2() -> Matrix3<Fixed> {
        Matrix3Trait::new(int(1), int(0), int(1), int(0), int(2), int(2), int(0), int(0), int(0))
    }

    // --- the losing candidates -----------------------------------------------------------------

    /// CLASSICAL Gram-Schmidt: `r23` is taken against the ORIGINAL third column instead of the
    /// one already stripped of its `q1` component. Identical in exact arithmetic, identical in
    /// gas, and measurably less orthogonal under rounding
    /// (`test_classical_gram_schmidt_candidate_is_less_orthogonal`).
    fn new_classical(matrix: Matrix3<Fixed>) -> Qr3<Fixed> {
        let r11 = Real::norm3(matrix.m11, matrix.m21, matrix.m31);
        let (q11, q21, q31) = (matrix.m11 / r11, matrix.m21 / r11, matrix.m31 / r11);
        let r12 = Real::sum_prod3(q11, matrix.m12, q21, matrix.m22, q31, matrix.m32);
        let r13 = Real::sum_prod3(q11, matrix.m13, q21, matrix.m23, q31, matrix.m33);
        let b21 = Real::mul_add(-r12, q11, matrix.m12);
        let b22 = Real::mul_add(-r12, q21, matrix.m22);
        let b23 = Real::mul_add(-r12, q31, matrix.m32);
        let r22 = Real::norm3(b21, b22, b23);
        let (q12, q22, q32) = (b21 / r22, b22 / r22, b23 / r22);
        // The classical difference: against `matrix`'s third column, not against `b3`.
        let r23 = Real::sum_prod3(q12, matrix.m13, q22, matrix.m23, q32, matrix.m33);
        let w = Real::wide_sub_prod(
            Real::wide_add(Real::<Fixed>::wide_zero(), matrix.m13), r13, q11,
        );
        let c31 = Real::wide_rescale(Real::wide_sub_prod(w, r23, q12));
        let w = Real::wide_sub_prod(
            Real::wide_add(Real::<Fixed>::wide_zero(), matrix.m23), r13, q21,
        );
        let c32 = Real::wide_rescale(Real::wide_sub_prod(w, r23, q22));
        let w = Real::wide_sub_prod(
            Real::wide_add(Real::<Fixed>::wide_zero(), matrix.m33), r13, q31,
        );
        let c33 = Real::wide_rescale(Real::wide_sub_prod(w, r23, q32));
        let r33 = Real::norm3(c31, c32, c33);
        let (q13, q23, q33) = (c31 / r33, c32 / r33, c33 / r33);
        Qr3 {
            q: Matrix3Trait::from_columns(
                Vector3 { x: q11, y: q21, z: q31 },
                Vector3 { x: q12, y: q22, z: q32 },
                Vector3 { x: q13, y: q23, z: q33 },
            ),
            r: Matrix3Trait::new(
                r11, r12, r13, Real::zero(), r22, r23, Real::zero(), Real::zero(), r33,
            ),
        }
    }

    /// The Householder axis of the full column `x`: `v = (x1 + sgn(x1) |x|, x2, x3)` normalised,
    /// or `None` when `x` is zero. `|v|² = 2(|x|² + |x1| |x|)` is upstream's `factor`, so this is
    /// upstream's `householder::reflection_axis_mut` with the normalisation written as a `norm3`.
    fn householder_axis(x: Vector3<Fixed>) -> Option<Vector3<Fixed>> {
        let n = Real::norm3(x.x, x.y, x.z);
        if n == Real::zero() {
            return None;
        }
        let head = if x.x.is_sign_negative() {
            x.x - n
        } else {
            x.x + n
        };
        let f = Real::norm3(head, x.y, x.z);
        Some(Vector3 { x: head / f, y: x.y / f, z: x.z / f })
    }

    /// The Householder axis of the SUB-column `(y, z)` — upstream applies
    /// `reflection_axis_mut` to `matrix.rows_range(i.., i)`, so the head of the reflector is the
    /// entry on the diagonal and the first coordinate of the axis is exactly zero. Getting this
    /// wrong (reflecting the full column at step 2) destroys the zero the first reflection
    /// created, which is why the sub-column has its own helper.
    fn householder_axis_sub(y: Fixed, z: Fixed) -> Option<Vector3<Fixed>> {
        let n = Real::norm2(y, z);
        if n == Real::zero() {
            return None;
        }
        let head = if y.is_sign_negative() {
            y - n
        } else {
            y + n
        };
        let f = Real::norm2(head, z);
        Some(Vector3 { x: Real::zero(), y: head / f, z: z / f })
    }

    /// `c - 2 <v, c> v`, the reflection of `c` in the hyperplane orthogonal to the unit `v`.
    /// Doubling is exact in fixed point, so the factor `2` costs no rounding.
    fn reflect(v: Vector3<Fixed>, c: Vector3<Fixed>) -> Vector3<Fixed> {
        let d = v.dot(c);
        let t = d + d;
        Vector3 {
            x: Real::mul_add(-t, v.x, c.x),
            y: Real::mul_add(-t, v.y, c.y),
            z: Real::mul_add(-t, v.z, c.z),
        }
    }

    /// UPSTREAM's algorithm: two Householder reflections (the third acts on a 1x1 block and is
    /// exactly the sign normalisation below), `Q` accumulated by reflecting the columns of the
    /// identity. Kept as evidence: same accuracy as the shipped modified Gram-Schmidt on the
    /// oracle (`test_householder_candidate_accuracy`), roughly twice the gas
    /// (`bench_qr3_new__alt_householder`).
    fn new_householder(matrix: Matrix3<Fixed>) -> Qr3<Fixed> {
        let (mut a1, mut a2, mut a3) = (matrix.column1(), matrix.column2(), matrix.column3());
        let (mut e1, mut e2, mut e3) = (
            Vector3 { x: Real::one(), y: Real::zero(), z: Real::zero() },
            Vector3 { x: Real::zero(), y: Real::one(), z: Real::zero() },
            Vector3 { x: Real::zero(), y: Real::zero(), z: Real::one() },
        );
        if let Some(v) = householder_axis(a1) {
            a1 = reflect(v, a1);
            a2 = reflect(v, a2);
            a3 = reflect(v, a3);
            e1 = reflect(v, e1);
            e2 = reflect(v, e2);
            e3 = reflect(v, e3);
        }
        // Second reflection: the sub-column (a2.y, a2.z), rows 2 and 3 of column 2.
        if let Some(v) = householder_axis_sub(a2.y, a2.z) {
            a2 = reflect(v, a2);
            a3 = reflect(v, a3);
            e1 = reflect(v, e1);
            e2 = reflect(v, e2);
            e3 = reflect(v, e3);
        }
        // `Q` is the transpose of the reflected identity (the reflections were applied on the
        // left), and the diagonal of `R` is signed: move each sign into the matching column of Q.
        let q = Matrix3Trait::from_columns(e1, e2, e3).transpose();
        let (mut q1, mut q2, mut q3) = (q.column1(), q.column2(), q.column3());
        let (mut rr1, mut rr2, mut rr3) = (
            Vector3 { x: a1.x, y: Real::zero(), z: Real::zero() },
            Vector3 { x: a2.x, y: a2.y, z: Real::zero() },
            a3,
        );
        if rr1.x.is_sign_negative() {
            rr1 = Vector3 { x: -rr1.x, y: rr1.y, z: rr1.z };
            rr2 = Vector3 { x: -rr2.x, y: rr2.y, z: rr2.z };
            rr3 = Vector3 { x: -rr3.x, y: rr3.y, z: rr3.z };
            q1 = -q1;
        }
        if rr2.y.is_sign_negative() {
            rr2 = Vector3 { x: rr2.x, y: -rr2.y, z: rr2.z };
            rr3 = Vector3 { x: rr3.x, y: -rr3.y, z: rr3.z };
            q2 = -q2;
        }
        if rr3.z.is_sign_negative() {
            rr3 = Vector3 { x: rr3.x, y: rr3.y, z: -rr3.z };
            q3 = -q3;
        }
        Qr3 {
            q: Matrix3Trait::from_columns(q1, q2, q3), r: Matrix3Trait::from_columns(rr1, rr2, rr3),
        }
    }

    /// The shipped `new` with the rank-deficient fallback COMPLETING the basis: `q1 = e1` when the
    /// first column vanishes, `q2` from `q1.orthonormal_basis()`, `q3 = q1 x q2`. Kept as
    /// evidence: it makes `Q` orthonormal on every input, and Sierra prices the worst-case branch,
    /// so every caller pays for it (`bench_qr3_new__alt_completed_basis`).
    fn new_completed(matrix: Matrix3<Fixed>) -> Qr3<Fixed> {
        let f = Qr3Trait::new(matrix);
        let (mut c1, mut c2, mut c3) = (f.q.column1(), f.q.column2(), f.q.column3());
        if f.r.m11 == Real::zero() {
            c1 = Vector3 { x: Real::one(), y: Real::zero(), z: Real::zero() };
        }
        if f.r.m22 == Real::zero() {
            let (u, _) = c1.orthonormal_basis();
            c2 = u;
        }
        if f.r.m33 == Real::zero() {
            c3 = c1.cross(c2);
        }
        Qr3 { q: Matrix3Trait::from_columns(c1, c2, c3), r: f.r }
    }

    // --- exact cases ---------------------------------------------------------------------------

    #[test]
    fn test_new_identity_and_diagonal_are_exact() {
        let f = Qr3Trait::new(Matrix3Trait::<Fixed>::identity());
        assert!(f.q() == Matrix3Trait::identity());
        assert!(f.r() == Matrix3Trait::identity());
        assert!(f.determinant() == int(1));
        let d = Matrix3Trait::from_diagonal(Vector3 { x: int(2), y: int(-5), z: int(3) });
        let f = Qr3Trait::new(d);
        assert!(f.q() == Matrix3Trait::from_diagonal(Vector3 { x: int(1), y: int(-1), z: int(1) }));
        assert!(f.r() == Matrix3Trait::from_diagonal(Vector3 { x: int(2), y: int(5), z: int(3) }));
        assert!(f.determinant() == int(-30));
    }

    #[test]
    fn test_new_permutation_is_exact() {
        // A cyclic permutation: Q is itself, R is the identity, det = +1.
        let p = Matrix3Trait::new(
            int(0), int(1), int(0), int(0), int(0), int(1), int(1), int(0), int(0),
        );
        let f = Qr3Trait::new(p);
        assert!(f.q() == p);
        assert!(f.r() == Matrix3Trait::identity());
        assert!(f.determinant() == int(1));
    }

    #[test]
    fn test_new_orthogonal_columns_are_exact() {
        let f = Qr3Trait::new(a_exact());
        assert!(f.r() == Matrix3Trait::from_diagonal(Vector3 { x: int(2), y: int(4), z: int(8) }));
        assert!(f.q() * f.r() == a_exact());
        assert!(orthonormality_error_m3(f.q()) == 0);
        assert!(f.determinant() == a_exact().determinant());
    }

    #[test]
    fn test_new_rank_deficient_leaves_a_zero_column() {
        let f = Qr3Trait::new(a_rank2());
        assert!(f.r().m33 == Real::zero());
        assert!(f.q().column3() == Vector3Trait::zeros());
        // `Q R = A` still holds exactly: row 3 of R is zero.
        assert!(f.q() * f.r() == a_rank2());
        assert!(!f.is_invertible());
        assert!(f.solve(b_bench()).is_none());
        assert!(f.try_inverse().is_none());
        assert!(f.determinant() == int(0));
        // The zero matrix: every column vanishes, nothing divides by zero.
        let z = Qr3Trait::new(Matrix3Trait::<Fixed>::zeros());
        assert!(z.q() == Matrix3Trait::zeros() && z.r() == Matrix3Trait::zeros());
        assert!(!z.is_invertible());
    }

    #[test]
    fn test_new_reconstruction_and_orthonormality_oracle() {
        let mut cases = oracle::qr3_q_r_cases();
        let (mut worst_rec, mut worst_orth) = (0, 0);
        while let Some(case) = cases.pop_front() {
            let (a, _, _, _) = *case;
            let f = Qr3Trait::new(m3(a));
            let rec = max_ulp_diff3(f.q() * f.r(), m3(a)) / amax_m3(m3(a));
            let orth = orthonormality_error_m3(f.q());
            worst_rec = core::cmp::max(worst_rec, rec);
            worst_orth = core::cmp::max(worst_orth, orth);
        }
        // Measured: `|A - Q R| <= worst_rec ulp * max(1, max |a_ij|)`, `|QᵀQ - I| <= worst_orth`.
        assert!(worst_rec == 2 && worst_orth == 34, "regressed: {worst_rec} {worst_orth}");
    }

    #[test]
    fn test_householder_candidate_accuracy() {
        // Upstream's own algorithm, on the same inputs.
        let mut cases = oracle::qr3_q_r_cases();
        let (mut worst_gap, mut worst_orth) = (0, 0);
        while let Some(case) = cases.pop_front() {
            let (a, q, r, _) = *case;
            let h = new_householder(m3(a));
            worst_gap = core::cmp::max(worst_gap, max_ulp_diff3(h.q, m3(q)));
            worst_gap = core::cmp::max(worst_gap, max_ulp_diff3(h.r, m3(r)));
            worst_orth = core::cmp::max(worst_orth, orthonormality_error_m3(h.q));
        }
        // Householder: 402 ulp from the oracle factors and 29 ulp of orthonormality, against 121
        // and 36 for the shipped modified Gram-Schmidt. More orthonormal, further from the
        // factors, and dearer (`bench_qr3_new__alt_householder`), which is why MGS ships.
        assert!(worst_gap == 142 && worst_orth == 28, "regressed: {worst_gap} {worst_orth}");
    }

    #[test]
    fn test_classical_gram_schmidt_candidates_agree_on_well_conditioned_inputs() {
        let mut cases = oracle::qr3_q_r_cases();
        let (mut worst_mgs, mut worst_cgs) = (0, 0);
        while let Some(case) = cases.pop_front() {
            let (a, _, _, _) = *case;
            worst_mgs = core::cmp::max(worst_mgs, orthonormality_error_m3(Qr3Trait::new(m3(a)).q));
            worst_cgs = core::cmp::max(worst_cgs, orthonormality_error_m3(new_classical(m3(a)).q));
        }
        // Identical on the oracle (condition number <= 8): the classical form's quadratic loss
        // of orthogonality only separates from the modified form's linear one beyond these inputs.
        assert!(worst_mgs == 34 && worst_cgs == 34, "regressed: {worst_mgs} {worst_cgs}");
    }

    #[test]
    fn test_completed_basis_candidate_is_orthonormal_on_a_deficient_input() {
        // What the extra gas buys: an orthonormal `Q` even at rank 2. `Q R = A` holds either way.
        let f = new_completed(a_rank2());
        assert!(orthonormality_error_m3(f.q) <= 4);
        assert!(max_ulp_diff3(f.q * f.r, a_rank2()) == 0);
        assert!(orthonormality_error_m3(Qr3Trait::new(a_rank2()).q) == 0x100000000);
    }

    #[test]
    fn test_determinant_versus_matrix3_cofactors() {
        // The closed form sums exact minors; this one multiplies three rounded norms.
        let mut cases = oracle::qr3_q_r_cases();
        let mut worst = 0;
        while let Some(case) = cases.pop_front() {
            let (a, _, _, _) = *case;
            let err = ulp_diff(Qr3Trait::new(m3(a)).determinant(), m3(a).determinant());
            worst = core::cmp::max(worst, err);
        }
        assert!(worst == 72181, "regressed: {worst}");
    }

    #[test]
    #[inline(never)]
    fn bench_qr3_new__alt_householder() {
        let a = black_box(a_bench());
        let e = black_box(true);
        let f = new_householder(a);
        assert!((f.r.m11 > Real::zero() && f.r.m33 > Real::zero()) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_qr3_new__alt_completed_basis() {
        let a = black_box(a_bench());
        let e = black_box(true);
        let f = new_completed(a);
        assert!((f.r.m11 > Real::zero() && f.r.m33 > Real::zero()) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_qr3_determinant__diagonal_product() {
        let f = black_box(f_bench());
        let e = black_box(true);
        assert!((f.determinant() != Real::zero()) == e);
    }
}
