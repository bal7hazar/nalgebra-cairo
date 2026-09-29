//! `Svd3`: the singular value decomposition `M = U · Σ · Vᵀ` of a `Matrix3`, and the polar
//! decomposition built on it (upstream `nalgebra::linalg::SVD`, DESIGN D6).
//!
//! Upstream bidiagonalises and then runs an implicitly shifted QR until the off-diagonal entries
//! fall below `eps`, bounded only by `max_niter`: an unbounded loop with a tolerance parameter,
//! which a proof system cannot price. DESIGN D6 goes through the **symmetric eigen decomposition
//! of `MᵀM`** instead — here the fixed-sweep cyclic Jacobi of `SymmetricEigen3`, whose cost is
//! a constant of the type.
//!
//! Two departures from the letter of D6, both measured (see `new` and the test module):
//!
//! 1. the singular values are `σ_i = |M v_i|`, a floored `Real::norm3` on an unscaled sum of
//!    squares, NOT `sqrt(λ_i)`: the eigenvalues come out of four Jacobi sweeps and carry their
//!    rounding, and `sqrt` halves what is left — measured at 47 ulp against 75 on the oracle
//!    vectors (`test_singular_values_candidates`). `sqrt(λ)` also panics if a rank-deficient
//!    `λ` rounds below zero, which `|M v|` cannot;
//! 2. only the FIRST left singular vector is read off `M v_1 / σ_1`; the second is
//!    re-orthogonalised against it and the third is their cross product, so `U` is orthonormal by
//!    construction instead of by accident. `u_3` is signed by `<u_1 x u_2, M v_3>` so that it
//!    still matches `M V` and not only its first two columns.
//!
//! A direct **one-sided Jacobi on `M`** (rotating the columns of `M` until they are orthogonal,
//! which never forms `MᵀM` and is the textbook answer for ill-conditioned inputs) was
//! implemented and measured against this. It trades one property for another: its `U Σ = M V`
//! holds by construction, so its `recompose` is five times tighter (12 ulp per unit against 64),
//! but its singular values are worse (67 against 47) and its `U` is five times further from
//! orthonormal (324 ulp against 67), because the columns it normalises are never
//! re-orthogonalised. It also costs more, since each of its 12 rotations recomputes three inner
//! products of the working columns where the Jacobi of `SymmetricEigen3` updates six scalars.
//! Orthonormal factors are what `to_polar`, `solve` and `pseudo_inverse` rest on, so the
//! `MᵀM` route ships and the candidate is kept as evidence in the test module
//! (`new_one_sided_jacobi`, `bench_svd3_new__alt_one_sided_jacobi`,
//! `test_one_sided_jacobi_candidate`).

// the in-crate tests reach the internal items of the module (and the other modules' tests
// through `crate::linalg::...`) here (WP 9-NS9)
#[cfg(test)]
pub(crate) use nalgebra_linalg_svd_eigen4::internal::linalg::svd3::Svd3InternalTrait;
pub use nalgebra_linalg_svd_eigen4::linalg::svd3::*;

/// Test-only field-wise equality (upstream `Svd3` has no `PartialEq`): the tests and the
/// benchmarks compare factors through it.
#[cfg(test)]
impl Svd3PartialEq<T, +PartialEq<T>> of PartialEq<Svd3<T>> {
    fn eq(lhs: @Svd3<T>, rhs: @Svd3<T>) -> bool {
        lhs.u == rhs.u && lhs.singular_values == rhs.singular_values && lhs.v_t == rhs.v_t
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests of `Svd3`: the exact cases (identity, diagonal, permutation, rank 1, rank 2,
    //! zero), the identities (`M = U Σ Vᵀ`, orthonormality, `M = P U`), the oracle vectors of
    //! `tools/oracle`, and the three candidates that lost, kept as evidence (AGENTS.md rule 8):
    //!
    //! - `alt_sqrt_eigenvalues`: `σ = sqrt(λ)` from the eigenvalues of `MᵀM`, the formulation
    //! DESIGN D6 describes. Cheaper, and two to three orders of magnitude less accurate.
    //!
    //! - `alt_normalised_columns`: every `u_i = M v_i / σ_i`, no re-orthogonalisation. Dearer (9
    //! divisions against 6 plus a cross product) and measurably less orthonormal.
    //!
    //! - `alt_one_sided_jacobi`: the decomposition computed directly on `M`, by rotating pairs of
    //! COLUMNS until they are orthogonal (`M J = U Σ`, `V = J`), never forming `MᵀM`. The
    //! textbook answer for ill-conditioned inputs. Same four sweeps, no better on the oracle, and
    //! dearer: every rotation recomputes three inner products of the working columns where the
    //! Jacobi on `MᵀM` updates six scalars.
    //!
    //! In-crate part (WP 8.1c): the tests of crate-internal items only; the tests of the public API
    //! are in `crates/tests_linalg/src/svd3/tests.cairo`.

    use fixed::Fixed;
    use nalgebra_static3::internal::base::matrix3::Matrix3InternalTrait;
    use simba::scalar::Real;
    use crate::base::MatrixMul;
    use crate::base::matrix3::{Matrix3, Matrix3Trait};
    use crate::base::matrix_test_utils::{
        amax_m3, int, m3, max_ulp_diff3, max_ulp_diff_v3, orthonormality_error_m3, v3t,
    };
    use crate::base::vector3::{Vector3, Vector3Trait};
    use crate::linalg::oracle_svd;
    use crate::linalg::symmetric_eigen3::{SymmetricEigen3, SymmetricEigen3InternalTrait};
    use crate::testing::black_box;
    use super::{Svd3, Svd3InternalTrait, Svd3Trait};

    /// An oracle `unit` 3x3 case: the benchmark input.
    fn a_bench() -> Matrix3<Fixed> {
        m3(
            [
                [-930291762, 206204041, 164059380], [-1057407058, -1146357637, -192379840],
                [-394072183, -440329712, 775368183],
            ],
        )
    }

    /// A rank-1 matrix whose decomposition is exact: one nonzero column, axis aligned.
    fn a_rank1() -> Matrix3<Fixed> {
        Matrix3Trait::new(int(0), int(0), int(2), int(0), int(0), int(0), int(0), int(0), int(0))
    }

    /// A rank-2 matrix whose decomposition is exact.
    fn a_rank2() -> Matrix3<Fixed> {
        Matrix3Trait::new(int(2), int(0), int(0), int(0), int(4), int(0), int(0), int(0), int(0))
    }

    // --- the losing candidates -----------------------------------------------------------------

    /// `σ = sqrt(λ)` from the eigenvalues of `MᵀM` (DESIGN D6's formulation), descending.
    fn singular_values_from_sqrt(m: Matrix3<Fixed>) -> Vector3<Fixed> {
        let e = SymmetricEigen3InternalTrait::eigenvalues(Svd3InternalTrait::gram(m));
        Vector3 { x: e.z.sqrt(), y: e.y.sqrt(), z: e.x.sqrt() }
    }

    /// Every left singular vector normalised from `M v_i`, with no re-orthogonalisation.
    fn u_from_normalised_columns(m: Matrix3<Fixed>) -> Matrix3<Fixed> {
        let f = Svd3Trait::new(m, true, true);
        let v = f.v_t.unwrap().transpose();
        let s = f.singular_values;
        let c1 = normalised_or_axis(m.mul_mat(v.column1()), s.x, 0);
        let c2 = normalised_or_axis(m.mul_mat(v.column2()), s.y, 1);
        let c3 = normalised_or_axis(m.mul_mat(v.column3()), s.z, 2);
        Matrix3Trait::from_columns(c1, c2, c3)
    }

    /// `w / s`, or the `i`-th axis when `s` is zero: the naive fallback of the candidate above.
    fn normalised_or_axis(w: Vector3<Fixed>, s: Fixed, i: u8) -> Vector3<Fixed> {
        if s == Real::zero() {
            if i == 0 {
                Vector3 { x: Real::one(), y: Real::zero(), z: Real::zero() }
            } else if i == 1 {
                Vector3 { x: Real::zero(), y: Real::one(), z: Real::zero() }
            } else {
                Vector3 { x: Real::zero(), y: Real::zero(), z: Real::one() }
            }
        } else {
            w.unscale(s)
        }
    }

    /// The state of the one-sided Jacobi candidate: the working columns of `M` and the
    /// accumulated right rotation.
    #[derive(Copy, Drop)]
    struct OneSided {
        a1: Vector3<Fixed>,
        a2: Vector3<Fixed>,
        a3: Vector3<Fixed>,
        v1: Vector3<Fixed>,
        v2: Vector3<Fixed>,
        v3: Vector3<Fixed>,
    }

    /// `(c, s)` of the plane rotation that makes the two columns of squared norms `app`, `aqq`
    /// and inner product `g` orthogonal — the same closed form as `SymmetricEigen3`'s
    /// `Jacobi3::rotation`, applied to the Gram entries recomputed from the columns.
    fn one_sided_rotation(app: Fixed, g: Fixed, aqq: Fixed) -> (Fixed, Fixed) {
        let h = Real::diff_prod(aqq, Real::HALF, app, Real::HALF);
        let num = if h.is_sign_negative() {
            -g
        } else {
            g
        };
        let t = num / (h.abs() + Real::norm2(h, g));
        let c = Real::recip(Real::sqrt(Real::mul_add(t, t, Real::one())));
        (c, t * c)
    }

    /// `(c p - s q, s p + c q)`, the rotation applied to a pair of columns.
    fn one_sided_apply(
        c: Fixed, s: Fixed, p: Vector3<Fixed>, q: Vector3<Fixed>,
    ) -> (Vector3<Fixed>, Vector3<Fixed>) {
        (
            Vector3 {
                x: Real::diff_prod(c, p.x, s, q.x),
                y: Real::diff_prod(c, p.y, s, q.y),
                z: Real::diff_prod(c, p.z, s, q.z),
            },
            Vector3 {
                x: Real::sum_prod2(s, p.x, c, q.x),
                y: Real::sum_prod2(s, p.y, c, q.y),
                z: Real::sum_prod2(s, p.z, c, q.z),
            },
        )
    }

    /// One cyclic sweep of the one-sided Jacobi: the rotations orthogonalising `(a1, a2)`, then
    /// `(a1, a3)`, then `(a2, a3)`.
    fn one_sided_sweep(j: OneSided) -> OneSided {
        let mut j = j;
        let g = j.a1.dot(j.a2);
        if g != Real::zero() {
            let (c, s) = one_sided_rotation(j.a1.norm_squared(), g, j.a2.norm_squared());
            let (a1, a2) = one_sided_apply(c, s, j.a1, j.a2);
            let (v1, v2) = one_sided_apply(c, s, j.v1, j.v2);
            j = OneSided { a1, a2, a3: j.a3, v1, v2, v3: j.v3 };
        }
        let g = j.a1.dot(j.a3);
        if g != Real::zero() {
            let (c, s) = one_sided_rotation(j.a1.norm_squared(), g, j.a3.norm_squared());
            let (a1, a3) = one_sided_apply(c, s, j.a1, j.a3);
            let (v1, v3) = one_sided_apply(c, s, j.v1, j.v3);
            j = OneSided { a1, a2: j.a2, a3, v1, v2: j.v2, v3 };
        }
        let g = j.a2.dot(j.a3);
        if g != Real::zero() {
            let (c, s) = one_sided_rotation(j.a2.norm_squared(), g, j.a3.norm_squared());
            let (a2, a3) = one_sided_apply(c, s, j.a2, j.a3);
            let (v2, v3) = one_sided_apply(c, s, j.v2, j.v3);
            j = OneSided { a1: j.a1, a2, a3, v1: j.v1, v2, v3 };
        }
        j
    }

    /// The SVD by **one-sided Jacobi on `M`**: four cyclic sweeps of column rotations, then
    /// `σ_i = |a_i|` and `u_i = a_i / σ_i`, sorted descending. Never forms `MᵀM`. Kept as
    /// evidence, see the module doc and `test_one_sided_jacobi_candidate`.
    fn new_one_sided_jacobi(m: Matrix3<Fixed>) -> Svd3<Fixed> {
        let j = OneSided {
            a1: m.column1(),
            a2: m.column2(),
            a3: m.column3(),
            v1: Vector3 { x: Real::one(), y: Real::zero(), z: Real::zero() },
            v2: Vector3 { x: Real::zero(), y: Real::one(), z: Real::zero() },
            v3: Vector3 { x: Real::zero(), y: Real::zero(), z: Real::one() },
        };
        let j = one_sided_sweep(one_sided_sweep(one_sided_sweep(one_sided_sweep(j))));
        let (mut s1, mut s2, mut s3) = (j.a1.norm(), j.a2.norm(), j.a3.norm());
        let (mut a1, mut a2, mut a3) = (j.a1, j.a2, j.a3);
        let (mut v1, mut v2, mut v3) = (j.v1, j.v2, j.v3);
        if s2 > s1 {
            let (ts, ta, tv) = (s1, a1, v1);
            s1 = s2;
            a1 = a2;
            v1 = v2;
            s2 = ts;
            a2 = ta;
            v2 = tv;
        }
        if s3 > s1 {
            let (ts, ta, tv) = (s1, a1, v1);
            s1 = s3;
            a1 = a3;
            v1 = v3;
            s3 = ts;
            a3 = ta;
            v3 = tv;
        }
        if s3 > s2 {
            let (ts, ta, tv) = (s2, a2, v2);
            s2 = s3;
            a2 = a3;
            v2 = v3;
            s3 = ts;
            a3 = ta;
            v3 = tv;
        }
        let u1 = normalised_or_axis(a1, s1, 0);
        let u2 = normalised_or_axis(a2, s2, 1);
        let u3 = normalised_or_axis(a3, s3, 2);
        Svd3 {
            u: Some(Matrix3Trait::from_columns(u1, u2, u3)),
            singular_values: Vector3 { x: s1, y: s2, z: s3 },
            v_t: Some(Matrix3Trait::from_rows(v1, v2, v3)),
        }
    }

    // --- exact cases ---------------------------------------------------------------------------

    #[test]
    fn test_new_identity_is_exact() {
        let f = Svd3Trait::new(Matrix3Trait::<Fixed>::identity(), true, true);
        assert!(f.singular_values == Vector3 { x: int(1), y: int(1), z: int(1) });
        assert!(f.recompose().unwrap() == Matrix3Trait::identity());
        assert!(f.rank(Real::zero()) == 3);
        assert!(orthonormality_error_m3(f.u.unwrap()) == 0);
        assert!(orthonormality_error_m3(f.v_t.unwrap()) == 0);
    }

    #[test]
    fn test_new_diagonal_is_exact() {
        let d = Matrix3Trait::from_diagonal(Vector3 { x: int(2), y: int(-5), z: int(3) });
        let f = Svd3Trait::new(d, true, true);
        assert!(f.singular_values == Vector3 { x: int(5), y: int(3), z: int(2) });
        assert!(f.recompose().unwrap() == d);
        assert!(orthonormality_error_m3(f.u.unwrap()) == 0);
    }

    #[test]
    fn test_new_rank_deficient_and_zero() {
        let f = Svd3Trait::new(a_rank1(), true, true);
        assert!(f.rank(Real::zero()) == 1);
        assert!(f.singular_values.x == int(2));
        assert!(orthonormality_error_m3(f.u.unwrap()) <= 4);
        assert!(max_ulp_diff3(f.recompose().unwrap(), a_rank1()) <= 4);
        let f = Svd3Trait::new(a_rank2(), true, true);
        assert!(f.rank(Real::zero()) == 2);
        assert!(f.singular_values == Vector3 { x: int(4), y: int(2), z: int(0) });
        assert!(orthonormality_error_m3(f.u.unwrap()) <= 4);
        assert!(max_ulp_diff3(f.recompose().unwrap(), a_rank2()) <= 4);
        // The zero matrix: every singular value vanishes and nothing divides by zero.
        let z = Svd3Trait::new(Matrix3Trait::<Fixed>::zeros(), true, true);
        assert!(z.singular_values == Vector3Trait::zeros());
        assert!(z.rank(Real::zero()) == 0);
        assert!(z.recompose().unwrap() == Matrix3Trait::zeros());
        assert!(orthonormality_error_m3(z.u.unwrap()) == 0);
    }

    #[test]
    fn test_new_recompose_and_orthonormality_oracle() {
        let mut cases = oracle_svd::svd3_singular_values_cases();
        let (mut worst_rec, mut worst_orth) = (0, 0);
        while let Some(case) = cases.pop_front() {
            let (a, _, _) = *case;
            let f = Svd3Trait::new(m3(a), true, true);
            let rec = max_ulp_diff3(f.recompose().unwrap(), m3(a)) / amax_m3(m3(a));
            let orth = core::cmp::max(
                orthonormality_error_m3(f.u.unwrap()), orthonormality_error_m3(f.v_t.unwrap()),
            );
            worst_rec = core::cmp::max(worst_rec, rec);
            worst_orth = core::cmp::max(worst_orth, orth);
        }
        assert!(worst_rec == 64 && worst_orth == 67, "regressed: {worst_rec} {worst_orth}");
    }

    #[test]
    fn test_singular_values_candidates() {
        // DESIGN D6's `sqrt(lambda)` against the shipped `|M v|`, on the oracle expectations.
        let mut cases = oracle_svd::svd3_singular_values_cases();
        let (mut worst_norm, mut worst_sqrt) = (0, 0);
        while let Some(case) = cases.pop_front() {
            let (a, expected, _) = *case;
            let e = v3t(expected);
            worst_norm =
                core::cmp::max(
                    worst_norm,
                    max_ulp_diff_v3(Svd3Trait::new(m3(a), true, true).singular_values, e),
                );
            worst_sqrt =
                core::cmp::max(worst_sqrt, max_ulp_diff_v3(singular_values_from_sqrt(m3(a)), e));
        }
        assert!(worst_norm == 71 && worst_sqrt == 95, "regressed: {worst_norm} {worst_sqrt}");
    }

    #[test]
    fn test_left_vectors_candidates() {
        // The re-orthogonalised `U` against the three independently normalised columns.
        let mut cases = oracle_svd::svd3_singular_values_cases();
        let (mut worst_orth, mut worst_naive) = (0, 0);
        while let Some(case) = cases.pop_front() {
            let (a, _, _) = *case;
            worst_orth =
                core::cmp::max(
                    worst_orth,
                    orthonormality_error_m3(Svd3Trait::new(m3(a), true, true).u.unwrap()),
                );
            worst_naive =
                core::cmp::max(
                    worst_naive, orthonormality_error_m3(u_from_normalised_columns(m3(a))),
                );
        }
        assert!(worst_orth == 67 && worst_naive == 2739, "regressed: {worst_orth} {worst_naive}");
    }

    #[test]
    fn test_one_sided_jacobi_candidate() {
        // The alternative DESIGN D6 does not take: rotate the COLUMNS of `M` instead of
        // diagonalising `MᵀM`. Measured on the same vectors, it is no more accurate.
        let mut cases = oracle_svd::svd3_singular_values_cases();
        let (mut worst_sv, mut worst_rec, mut worst_orth) = (0, 0, 0);
        while let Some(case) = cases.pop_front() {
            let (a, expected, _) = *case;
            let f = new_one_sided_jacobi(m3(a));
            worst_sv = core::cmp::max(worst_sv, max_ulp_diff_v3(f.singular_values, v3t(expected)));
            worst_rec =
                core::cmp::max(
                    worst_rec, max_ulp_diff3(f.recompose().unwrap(), m3(a)) / amax_m3(m3(a)),
                );
            worst_orth = core::cmp::max(worst_orth, orthonormality_error_m3(f.u.unwrap()));
        }
        assert!(
            worst_sv == 297 && worst_rec == 13 && worst_orth == 308,
            "regressed: {worst_sv} {worst_rec} {worst_orth}",
        );
    }

    #[test]
    fn test_to_polar_oracle() {
        let mut cases = oracle_svd::svd3_singular_values_cases();
        let (mut worst, mut worst_orth) = (0, 0);
        while let Some(case) = cases.pop_front() {
            let (a, _, _) = *case;
            let (p, u) = Svd3Trait::new(m3(a), true, true).to_polar().unwrap();
            // `P` is symmetric by construction and positive semi-definite: its determinant is the
            // product of the singular values.
            assert!(p == p.transpose(), "P is not symmetric");
            assert!(!p.determinant().is_sign_negative(), "P is not positive semi-definite");
            worst_orth = core::cmp::max(worst_orth, orthonormality_error_m3(u));
            worst = core::cmp::max(worst, max_ulp_diff3(p * u, m3(a)) / amax_m3(m3(a)));
        }
        // Measured: `|M - P U| <= worst ulp * max(1, max |m_ij|)`, `|UᵀU - I| <= worst_orth ulp`.
        assert!((worst, worst_orth) == (65, 65), "regressed: {worst} {worst_orth}");
    }

    /// `sqrt` of the eigenvalues of `MᵀM`, which skips the eigenvector accumulation entirely:
    /// twice cheaper, and not what ships (see the module doc — less accurate here, and it can
    /// take the square root of a negative number on a rank-deficient input).
    #[test]
    #[inline(never)]
    fn bench_svd3_singular_values__alt_sqrt_eigenvalues() {
        let a = black_box(a_bench());
        let e = black_box(true);
        let s = singular_values_from_sqrt(a);
        assert!((s.x >= s.z) == e);
    }

    /// `new` PLUS a second, naive construction of `U`: it calls `new` itself, so it can only be
    /// dearer than the shipped path. The decisive comparison is the accuracy one
    /// (`test_left_vectors_candidates`).
    #[test]
    #[inline(never)]
    fn bench_svd3_new__alt_normalised_columns() {
        let a = black_box(a_bench());
        let e = black_box(true);
        let u = u_from_normalised_columns(a);
        assert!((u.m11 != Real::zero()) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_svd3_new__alt_one_sided_jacobi() {
        let a = black_box(a_bench());
        let e = black_box(true);
        let f = new_one_sided_jacobi(a);
        assert!((f.singular_values.x >= f.singular_values.z) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_svd3_gram__fused() {
        let a = black_box(a_bench());
        let e = black_box(true);
        assert!((Svd3InternalTrait::gram(a).m11 != Real::zero()) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_svd3_gram__transpose_mul_transpose() {
        let a = black_box(a_bench());
        let e = black_box(true);
        assert!((a.transpose().mul_transpose().m11 != Real::zero()) == e);
    }
}
