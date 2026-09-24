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

use simba::scalar::Real;
use crate::base::matrix3::{Matrix3, Matrix3InternalTrait, Matrix3Trait};
use crate::base::sym_matrix3::{SymMatrix3, SymMatrix3Trait};
use crate::base::vector3::{Vector3, Vector3InternalTrait, Vector3Trait};
use crate::linalg::symmetric_eigen3::SymmetricEigen3InternalTrait;

/// The singular value decomposition `M = U · diag(singular_values) · v_t` of a `Matrix3<T>`.
///
/// `singular_values` are sorted **descending** and non-negative (like `tools/oracle` and upstream's
/// `singular_values`), `u` and `v_t` are orthonormal. Unlike upstream, where `u` and `v_t` are
/// `Option`s selected by the `compute_u` / `compute_v` flags, both are always present: the left
/// vectors are what the singular values are read off here, so skipping them would save nothing.
///
/// Sign and order convention (the decomposition is only defined up to a sign per column and up to
/// the order of equal singular values): the columns of `V` are the eigenvectors that
/// `SymmetricEigen3` returns for `MᵀM`, reordered DESCENDING by singular value (ties keep the
/// eigen order, so the decomposition of the identity is the identity); `u_1 = M v_1 / σ_1`,
/// `u_2` is `M v_2` re-orthogonalised against `u_1` and normalised, and `u_3 = ±(u_1 x u_2)` with
/// the sign of `<u_1 x u_2, M v_3>`. The decomposition of a given matrix is therefore a
/// deterministic function of its raw components, as AGENTS.md requires.
///
/// Upstream: `SVD { u: Option<OMatrix>, v_t: Option<OMatrix>, singular_values: OVector }`.
#[derive(Copy, Drop, Serde, Debug)]
pub struct Svd3<T> {
    /// The left singular vectors, as columns. Orthonormal.
    pub u: Matrix3<T>,
    /// The three singular values, descending, non-negative.
    pub singular_values: Vector3<T>,
    /// The TRANSPOSE of the right singular vectors, i.e. `v_i` is ROW `i`. Orthonormal.
    pub v_t: Matrix3<T>,
}

/// Test-only field-wise equality (upstream `Svd3` has no `PartialEq`): the tests and the
/// benchmarks compare factors through it.
#[cfg(test)]
impl Svd3PartialEq<T, +PartialEq<T>> of PartialEq<Svd3<T>> {
    fn eq(lhs: @Svd3<T>, rhs: @Svd3<T>) -> bool {
        lhs.u == rhs.u && lhs.singular_values == rhs.singular_values && lhs.v_t == rhs.v_t
    }
}

/// Methods of `Svd3<T>` for any `Real` scalar.
#[generate_trait]
pub impl Svd3Impl<
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
> of Svd3Trait<T> {
    /// The singular value decomposition of `matrix`. Upstream: `matrix.svd(true, true)` /
    /// `SVD::new(matrix, true, true)`.
    ///
    /// ```text
    /// S   = MᵀM                    (6 fused kernels)
    /// V   = eigenvectors of S      (SymmetricEigen3, four cyclic Jacobi sweeps), columns ORDERED
    ///                               so that the singular values come out descending
    /// w_i = M v_i                  (3 fused sum_prod3 each)
    /// σ_i = |w_i|                  (floored norm3 on the unscaled sum of squares), then sorted
    /// u_1 = w_1 / σ_1
    /// u_2 = normalize(w_2 - <u_1, w_2> u_1)
    /// u_3 = ±(u_1 x u_2),  sign of <u_1 x u_2, w_3>
    /// ```
    ///
    /// **Rank deficiency.** A floored `norm3` of a nonzero vector is at least 1 raw unit, so
    /// `σ_i == 0` exactly characterises `M v_i == 0`: the threshold of the fallbacks is EXACT
    /// ZERO, no epsilon, in the spirit of `Matrix3::try_inverse` and `LU::is_invertible`. The
    /// fallbacks keep `U` orthonormal: `σ_1 = 0` (`M = 0`) gives `u_1 = e_1`; a `w_2` that
    /// vanishes after the re-orthogonalisation gives `u_2` from `u_1.orthonormal_basis()`; and
    /// `u_3` is the cross product, which needs no division at all. A merely TINY singular value is
    /// NOT treated as zero — the direction it produces is as accurate as the input allows — and
    /// callers that need a rank decision have `rank(eps)`, `pseudo_inverse(eps)` and
    /// `solve(b, eps)`.
    ///
    /// **Measured** on the 30 vectors of the `svd3_singular_values` oracle suite
    /// (well-conditioned matrices, `small` / `unit` / `medium`): singular values within **47 ulp**
    /// of the floored exact result, every case INSIDE the oracle's own tolerance; `U` and `V`
    /// orthonormal within **67 ulp**; `|M - U Σ Vᵀ| <= 64 ulp * max(1, max |m_ij|)`; and
    /// the left polar factors within `|M - P U| <= 65 ulp * max(1, max |m_ij|)` with
    /// `|UᵀU - I| <= 65 ulp`.
    ///
    /// Cost: constant — four Jacobi sweeps whatever the input, no convergence test, no iteration
    /// count. Panics with the scalar's overflow error if a component of `MᵀM` does not fit (the
    /// squares of the entries must be representable, unlike `norm3`).
    fn new(matrix: Matrix3<T>) -> Svd3<T> {
        let eigen = SymmetricEigen3InternalTrait::new_sym(Svd3InternalTrait::gram(matrix));
        // RENORMALISE the eigenvectors. `SymmetricEigen3` divides each accumulated column by its
        // own FLOORED norm, and on a `small` matrix those columns are tiny in raw units, so the
        // floor costs `1 / |u|_raw` RELATIVE — up to 1.4e-7 on the oracle vectors. That bias
        // lands directly on `σ = |M v|`, on the orthonormality of `V` and on `recompose`. Here the
        // columns are of magnitude 2^32, so the same floor costs only 2.3e-10. Column 3 is the
        // cross product of the first two, exactly as `SymmetricEigen3` builds it.
        let ev = eigen.eigenvectors;
        let (c1, c2) = (ev.column1(), ev.column2());
        let n1 = R::norm3(c1.x, c1.y, c1.z);
        let n2 = R::norm3(c2.x, c2.y, c2.z);
        let v1 = {
            let (x, y, z) = R::div3(c1.x, c1.y, c1.z, n1);
            Vector3 { x, y, z }
        };
        let v2 = {
            let (x, y, z) = R::div3(c2.x, c2.y, c2.z, n2);
            Vector3 { x, y, z }
        };
        let v3 = v1.cross(v2);
        let (w1, w2, w3) = (matrix.mul_vec(v1), matrix.mul_vec(v2), matrix.mul_vec(v3));
        let s1 = R::norm3(w1.x, w1.y, w1.z);
        let s2 = R::norm3(w2.x, w2.y, w2.z);
        let s3 = R::norm3(w3.x, w3.y, w3.z);
        // `SymmetricEigen3` sorts the eigenvalues ASCENDING and the singular values are DESCENDING,
        // so the columns generally come out reversed — but reversing them unconditionally would
        // also reorder EQUAL singular values, and the decomposition of the identity would not be
        // the identity. The sorting network of 3 elements, run on the computed norms with a STRICT
        // comparison, reverses exactly when the order asks for it and leaves ties alone. Branches
        // and moves only.
        let (mut s1, mut s2, mut s3) = (s1, s2, s3);
        let (mut w1, mut w2, mut w3) = (w1, w2, w3);
        let (mut v1, mut v2, mut v3) = (v1, v2, v3);
        if s2 > s1 {
            let (ts, tw, tv) = (s1, w1, v1);
            s1 = s2;
            w1 = w2;
            v1 = v2;
            s2 = ts;
            w2 = tw;
            v2 = tv;
        }
        if s3 > s1 {
            let (ts, tw, tv) = (s1, w1, v1);
            s1 = s3;
            w1 = w3;
            v1 = v3;
            s3 = ts;
            w3 = tw;
            v3 = tv;
        }
        if s3 > s2 {
            let (ts, tw, tv) = (s2, w2, v2);
            s2 = s3;
            w2 = w3;
            v2 = v3;
            s3 = ts;
            w3 = tw;
            v3 = tv;
        }
        let u1 = if s1 == R::ZERO {
            Vector3 { x: R::ONE, y: R::ZERO, z: R::ZERO }
        } else {
            {
                let (x, y, z) = R::div3(w1.x, w1.y, w1.z, s1);
                Vector3 { x, y, z }
            }
        };
        // `w2` stripped of its `u1` component: one fused dot and one fused `mul_add` per
        // component, so `u2` is orthogonal to `u1` to within the final normalisation alone.
        let p = R::sum_prod3(u1.x, w2.x, u1.y, w2.y, u1.z, w2.z);
        let g = Vector3 {
            x: R::mul_add(-p, u1.x, w2.x),
            y: R::mul_add(-p, u1.y, w2.y),
            z: R::mul_add(-p, u1.z, w2.z),
        };
        let n = R::norm3(g.x, g.y, g.z);
        let u2 = if n == R::ZERO {
            let (basis, _) = u1.orthonormal_basis();
            basis
        } else {
            {
                let (x, y, z) = R::div3(g.x, g.y, g.z, n);
                Vector3 { x, y, z }
            }
        };
        let c = u1.cross(u2);
        let along = R::sum_prod3(c.x, w3.x, c.y, w3.y, c.z, w3.z);
        let u3 = if along.is_negative() {
            Vector3 { x: -c.x, y: -c.y, z: -c.z }
        } else {
            c
        };
        Svd3 {
            u: Matrix3Trait::from_columns(u1, u2, u3),
            singular_values: Vector3 { x: s1, y: s2, z: s3 },
            v_t: Matrix3Trait::from_rows(v1, v2, v3),
        }
    }

    /// The number of singular values strictly greater than `eps`. Upstream: `SVD::rank`.
    #[inline(always)]
    fn rank(self: Svd3<T>, eps: T) -> u32 {
        let mut n = 0_u32;
        if self.singular_values.x > eps {
            n += 1;
        }
        if self.singular_values.y > eps {
            n += 1;
        }
        if self.singular_values.z > eps {
            n += 1;
        }
        n
    }

    /// `U · diag(singular_values) · v_t`, the matrix the decomposition came from, up to its
    /// rounding (measured: **64 ulp per unit of `max |m_ij|`** on the oracle vectors).
    ///
    /// Two roundings per entry: the columns of `U` are scaled by the singular values (9 floored
    /// products), then the product with `v_t` is 9 fused `sum_prod3`. Panics on overflow.
    /// Upstream: `SVD::recompose`, which returns a `Result` because `u` / `v_t` may be missing;
    /// here they never are.
    fn recompose(self: Svd3<T>) -> Matrix3<T> {
        Svd3InternalTrait::scale_columns(self.u, self.singular_values) * self.v_t
    }

    /// The Moore-Penrose pseudo-inverse `V · diag(σ⁺) · Uᵀ`, where `σ⁺_i = 1 / σ_i` when
    /// `σ_i > eps` and `0` otherwise, or `None` when `eps` is negative.
    ///
    /// Upstream: `SVD::pseudo_inverse`, which returns `Err` on a negative `eps` and otherwise
    /// recomposes with the inverted singular values and takes the adjoint — the same expression.
    /// Upstream's test is `> eps` too, so `eps = 0` keeps every nonzero singular value, including
    /// one of 1 raw unit whose reciprocal overflows: pass an `eps` matched to the scale of the
    /// problem, not zero, unless an overflow panic is the wanted answer.
    ///
    /// Three roundings per entry (the reciprocal, the scaling, the product). Panics with the
    /// scalar's overflow error if a reciprocal or an entry does not fit.
    fn pseudo_inverse(self: Svd3<T>, eps: T) -> Option<Matrix3<T>> {
        if eps.is_negative() {
            return None;
        }
        let r = Vector3 {
            x: Svd3InternalTrait::inverted(self.singular_values.x, eps),
            y: Svd3InternalTrait::inverted(self.singular_values.y, eps),
            z: Svd3InternalTrait::inverted(self.singular_values.z, eps),
        };
        Some(Svd3InternalTrait::scale_columns(self.v_t.transpose(), r) * self.u.transpose())
    }

    /// The least-squares solution of `M x = b`, `V · (Uᵀ b / σ)` with the components whose
    /// singular value is `<= eps` zeroed, or `None` when `eps` is negative. Upstream: `SVD::solve`.
    ///
    /// Unlike `pseudo_inverse` this divides instead of multiplying by a reciprocal — one rounding
    /// per component instead of two, and no overflow on a tiny singular value unless the solution
    /// itself does not fit. Panics with the scalar's overflow error in that case.
    fn solve(self: Svd3<T>, b: Vector3<T>, eps: T) -> Option<Vector3<T>> {
        if eps.is_negative() {
            return None;
        }
        let y = self.u.tr_mul_vec(b);
        let z = Vector3 {
            x: Svd3InternalTrait::divided(y.x, self.singular_values.x, eps),
            y: Svd3InternalTrait::divided(y.y, self.singular_values.y, eps),
            z: Svd3InternalTrait::divided(y.z, self.singular_values.z, eps),
        };
        Some(self.v_t.tr_mul_vec(z))
    }

    /// The LEFT polar decomposition `M = P · U`, as `Some((P, U))`: `P = u · diag(σ) · uᵀ` is
    /// symmetric positive semi-definite and `U = u · v_t` is orthonormal (a rotation when
    /// `det(M) > 0`). Always `Some`: upstream returns `None` only when `u` or `v_t` was not
    /// computed, and both always are here.
    ///
    /// `U` costs one 3x3 product and `P` one structured quadratic form (only its 6 independent
    /// components are computed, then mirrored); both are two roundings per entry. Panics on
    /// overflow. Upstream: `SVD::to_polar`.
    #[inline(always)]
    fn to_polar(self: Svd3<T>) -> Option<(Matrix3<T>, Matrix3<T>)> {
        Some(
            (
                SymMatrix3Trait::quadform(self.u, self.singular_values).to_matrix(),
                self.u * self.v_t,
            ),
        )
    }
}

/// Crate-internal kernels of `Svd3<T>` (WP 8.0: the public API is strictly upstream's): the Gram
/// matrix `MᵀM` as a `SymMatrix3` (the input of the eigen decomposition), and the column scaling
/// and the guarded reciprocal / quotient shared by `recompose`, `pseudo_inverse` and `solve`.
#[generate_trait]
pub(crate) impl Svd3InternalImpl<
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
> of Svd3InternalTrait<T> {
    /// `MᵀM` as a symmetric matrix: 6 fused kernels instead of 27 products, bit-identical to the
    /// upper triangle of `m.transpose() * m` and to `m.transpose().mul_transpose()`. The latter
    /// would reuse `base` instead of repeating the kernel, and it is NOT free: the transpose
    /// costs 2 760 gas of moves that Sierra does not elide (13 500 against 16 260, measured,
    /// `bench_svd3_gram__fused` against `bench_svd3_gram__transpose_mul_transpose`). Panics on
    /// overflow. Upstream: `m.tr_mul(&m)`.
    #[inline(always)]
    fn gram(m: Matrix3<T>) -> SymMatrix3<T> {
        SymMatrix3 {
            m11: R::norm_squared3(m.m11, m.m21, m.m31),
            m12: R::sum_prod3(m.m11, m.m12, m.m21, m.m22, m.m31, m.m32),
            m13: R::sum_prod3(m.m11, m.m13, m.m21, m.m23, m.m31, m.m33),
            m22: R::norm_squared3(m.m12, m.m22, m.m32),
            m23: R::sum_prod3(m.m12, m.m13, m.m22, m.m23, m.m32, m.m33),
            m33: R::norm_squared3(m.m13, m.m23, m.m33),
        }
    }
    /// `m * diag(d)`: each column of `m` scaled by the matching component of `d`, 9 floored
    /// products. No upstream equivalent (upstream materialises `Matrix::from_diagonal`).
    #[inline(always)]
    fn scale_columns(m: Matrix3<T>, d: Vector3<T>) -> Matrix3<T> {
        Matrix3 {
            m11: m.m11 * d.x,
            m21: m.m21 * d.x,
            m31: m.m31 * d.x,
            m12: m.m12 * d.y,
            m22: m.m22 * d.y,
            m32: m.m32 * d.y,
            m13: m.m13 * d.z,
            m23: m.m23 * d.z,
            m33: m.m33 * d.z,
        }
    }
    /// `1 / s` when `s > eps`, `0` otherwise: upstream's `pseudo_inverse` filter.
    #[inline(always)]
    fn inverted(s: T, eps: T) -> T {
        if s > eps {
            s.recip()
        } else {
            R::ZERO
        }
    }
    /// `y / s` when `s > eps`, `0` otherwise: upstream's `solve` filter.
    #[inline(always)]
    fn divided(y: T, s: T, eps: T) -> T {
        if s > eps {
            R::div(y, s)
        } else {
            R::ZERO
        }
    }
}

/// `Matrix3` methods that go through the SVD; upstream carries them on the matrix itself.
/// Import `Matrix3SvdTrait` to use them.
#[generate_trait]
pub impl Matrix3SvdImpl<
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
> of Matrix3SvdTrait<T> {
    /// The singular value decomposition. Upstream: `Matrix3::svd(true, true)`.
    #[inline(always)]
    fn svd(self: Matrix3<T>) -> Svd3<T> {
        Svd3Trait::new(self)
    }

    /// The singular values alone, descending. Upstream: `Matrix3::singular_values`.
    ///
    /// Not cheaper than the full decomposition here (the left vectors are what the norms are read
    /// off), unlike upstream, where skipping `U` and `V` saves the whole accumulation.
    #[inline(always)]
    fn singular_values(self: Matrix3<T>) -> Vector3<T> {
        Svd3Trait::new(self).singular_values
    }

    /// The Moore-Penrose pseudo-inverse, see `Svd3::pseudo_inverse`. Upstream:
    /// `Matrix3::pseudo_inverse`.
    #[inline(always)]
    fn pseudo_inverse(self: Matrix3<T>, eps: T) -> Option<Matrix3<T>> {
        Svd3Trait::new(self).pseudo_inverse(eps)
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

    use fixed::Fixed;
    use nalgebra_testing::black_box;
    use simba::scalar::Real;
    use crate::base::matrix3::{Matrix3, Matrix3InternalTrait, Matrix3Trait};
    use crate::base::matrix_test_utils::{
        amax_m3, excess, int, m3, max_abs_v3, max_ulp_diff3, max_ulp_diff_v3, oracle_tol,
        orthonormality_error_m3, v3t,
    };
    use crate::base::vector3::{Vector3, Vector3Trait};
    use crate::linalg::oracle_svd;
    use crate::linalg::symmetric_eigen3::SymmetricEigen3InternalTrait;
    use super::{Matrix3SvdTrait, Svd3, Svd3InternalTrait, Svd3Trait};

    /// An oracle `unit` 3x3 case: the benchmark input.
    fn a_bench() -> Matrix3<Fixed> {
        m3(
            [
                [-930291762, 206204041, 164059380], [-1057407058, -1146357637, -192379840],
                [-394072183, -440329712, 775368183],
            ],
        )
    }

    /// `a_bench()` already decomposed.
    fn f_bench() -> Svd3<Fixed> {
        Svd3Trait::new(a_bench())
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
        let f = Svd3Trait::new(m);
        let v = f.v_t.transpose();
        let s = f.singular_values;
        let c1 = normalised_or_axis(m.mul_vec(v.column1()), s.x, 0);
        let c2 = normalised_or_axis(m.mul_vec(v.column2()), s.y, 1);
        let c3 = normalised_or_axis(m.mul_vec(v.column3()), s.z, 2);
        Matrix3Trait::from_columns(c1, c2, c3)
    }

    /// `w / s`, or the `i`-th axis when `s` is zero: the naive fallback of the candidate above.
    fn normalised_or_axis(w: Vector3<Fixed>, s: Fixed, i: u8) -> Vector3<Fixed> {
        if s == Real::ZERO {
            if i == 0 {
                Vector3 { x: Real::ONE, y: Real::ZERO, z: Real::ZERO }
            } else if i == 1 {
                Vector3 { x: Real::ZERO, y: Real::ONE, z: Real::ZERO }
            } else {
                Vector3 { x: Real::ZERO, y: Real::ZERO, z: Real::ONE }
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
        let num = if h.is_negative() {
            -g
        } else {
            g
        };
        let t = num / (h.abs() + Real::norm2(h, g));
        let c = Real::recip(Real::sqrt(Real::mul_add(t, t, Real::ONE)));
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
        if g != Real::ZERO {
            let (c, s) = one_sided_rotation(j.a1.norm_squared(), g, j.a2.norm_squared());
            let (a1, a2) = one_sided_apply(c, s, j.a1, j.a2);
            let (v1, v2) = one_sided_apply(c, s, j.v1, j.v2);
            j = OneSided { a1, a2, a3: j.a3, v1, v2, v3: j.v3 };
        }
        let g = j.a1.dot(j.a3);
        if g != Real::ZERO {
            let (c, s) = one_sided_rotation(j.a1.norm_squared(), g, j.a3.norm_squared());
            let (a1, a3) = one_sided_apply(c, s, j.a1, j.a3);
            let (v1, v3) = one_sided_apply(c, s, j.v1, j.v3);
            j = OneSided { a1, a2: j.a2, a3, v1, v2: j.v2, v3 };
        }
        let g = j.a2.dot(j.a3);
        if g != Real::ZERO {
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
            v1: Vector3 { x: Real::ONE, y: Real::ZERO, z: Real::ZERO },
            v2: Vector3 { x: Real::ZERO, y: Real::ONE, z: Real::ZERO },
            v3: Vector3 { x: Real::ZERO, y: Real::ZERO, z: Real::ONE },
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
            u: Matrix3Trait::from_columns(u1, u2, u3),
            singular_values: Vector3 { x: s1, y: s2, z: s3 },
            v_t: Matrix3Trait::from_rows(v1, v2, v3),
        }
    }

    // --- exact cases ---------------------------------------------------------------------------

    #[test]
    fn test_new_identity_is_exact() {
        let f = Svd3Trait::new(Matrix3Trait::<Fixed>::identity());
        assert!(f.singular_values == Vector3 { x: int(1), y: int(1), z: int(1) });
        assert!(f.recompose() == Matrix3Trait::identity());
        assert!(f.rank(Real::ZERO) == 3);
        assert!(orthonormality_error_m3(f.u) == 0);
        assert!(orthonormality_error_m3(f.v_t) == 0);
    }

    #[test]
    fn test_new_diagonal_is_exact() {
        let d = Matrix3Trait::from_diagonal(Vector3 { x: int(2), y: int(-5), z: int(3) });
        let f = Svd3Trait::new(d);
        assert!(f.singular_values == Vector3 { x: int(5), y: int(3), z: int(2) });
        assert!(f.recompose() == d);
        assert!(orthonormality_error_m3(f.u) == 0);
    }

    #[test]
    fn test_new_permutation_is_exact() {
        let p = Matrix3Trait::new(
            int(0), int(1), int(0), int(0), int(0), int(1), int(1), int(0), int(0),
        );
        let f = Svd3Trait::new(p);
        assert!(f.singular_values == Vector3 { x: int(1), y: int(1), z: int(1) });
        assert!(f.recompose() == p);
    }

    #[test]
    fn test_new_rank_deficient_and_zero() {
        let f = Svd3Trait::new(a_rank1());
        assert!(f.rank(Real::ZERO) == 1);
        assert!(f.singular_values.x == int(2));
        assert!(orthonormality_error_m3(f.u) <= 4);
        assert!(max_ulp_diff3(f.recompose(), a_rank1()) <= 4);
        let f = Svd3Trait::new(a_rank2());
        assert!(f.rank(Real::ZERO) == 2);
        assert!(f.singular_values == Vector3 { x: int(4), y: int(2), z: int(0) });
        assert!(orthonormality_error_m3(f.u) <= 4);
        assert!(max_ulp_diff3(f.recompose(), a_rank2()) <= 4);
        // The zero matrix: every singular value vanishes and nothing divides by zero.
        let z = Svd3Trait::new(Matrix3Trait::<Fixed>::zeros());
        assert!(z.singular_values == Vector3Trait::zeros());
        assert!(z.rank(Real::ZERO) == 0);
        assert!(z.recompose() == Matrix3Trait::zeros());
        assert!(orthonormality_error_m3(z.u) == 0);
    }

    #[test]
    fn test_accessors() {
        let f = f_bench();
        assert!(f.singular_values == f.singular_values);
        let g = a_bench().svd();
        assert!(g.u == f.u && g.singular_values == f.singular_values && g.v_t == f.v_t);
        assert!(a_bench().singular_values() == f.singular_values);
    }

    // --- oracle --------------------------------------------------------------------------------

    #[test]
    fn test_new_singular_values_oracle() {
        let mut cases = oracle_svd::svd3_singular_values_cases();
        let (mut worst, mut worst_ex) = (0, 0);
        while let Some(case) = cases.pop_front() {
            let (a, expected, tol) = *case;
            let got = Svd3Trait::new(m3(a)).singular_values;
            let e = v3t(expected);
            let err = max_ulp_diff_v3(got, e);
            worst_ex = core::cmp::max(worst_ex, excess(err, oracle_tol(max_abs_v3(e), tol)));
            assert!(got.x >= got.y && got.y >= got.z && got.z >= Real::ZERO, "not descending");
            worst = core::cmp::max(worst, err);
        }
        assert!((worst, worst_ex) == (71, 0), "regressed: {worst} {worst_ex}");
    }

    #[test]
    fn test_new_recompose_and_orthonormality_oracle() {
        let mut cases = oracle_svd::svd3_singular_values_cases();
        let (mut worst_rec, mut worst_orth) = (0, 0);
        while let Some(case) = cases.pop_front() {
            let (a, _, _) = *case;
            let f = Svd3Trait::new(m3(a));
            let rec = max_ulp_diff3(f.recompose(), m3(a)) / amax_m3(m3(a));
            let orth = core::cmp::max(orthonormality_error_m3(f.u), orthonormality_error_m3(f.v_t));
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
                    worst_norm, max_ulp_diff_v3(Svd3Trait::new(m3(a)).singular_values, e),
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
                core::cmp::max(worst_orth, orthonormality_error_m3(Svd3Trait::new(m3(a)).u));
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
                core::cmp::max(worst_rec, max_ulp_diff3(f.recompose(), m3(a)) / amax_m3(m3(a)));
            worst_orth = core::cmp::max(worst_orth, orthonormality_error_m3(f.u));
        }
        assert!(
            worst_sv == 297 && worst_rec == 13 && worst_orth == 308,
            "regressed: {worst_sv} {worst_rec} {worst_orth}",
        );
    }

    #[test]
    fn test_solve_matches_the_inverse_oracle() {
        let mut cases = oracle_svd::svd3_singular_values_cases();
        let mut worst = 0;
        let b = v3t((-1811584373, 4204441265, -4234532070));
        while let Some(case) = cases.pop_front() {
            let (a, _, _) = *case;
            let x = Svd3Trait::new(m3(a)).solve(b, Real::EPSILON).unwrap();
            let e = m3(a).try_inverse().unwrap().mul_vec(b);
            worst = core::cmp::max(worst, max_ulp_diff_v3(x, e));
        }
        // Measured gap to `Matrix3::try_inverse` * b over the 30 well-conditioned vectors.
        assert!(worst == 4119, "regressed: {worst}");
    }

    #[test]
    fn test_pseudo_inverse_oracle() {
        let mut cases = oracle_svd::svd3_singular_values_cases();
        let mut worst = 0;
        while let Some(case) = cases.pop_front() {
            let (a, _, _) = *case;
            let p = Svd3Trait::new(m3(a)).pseudo_inverse(Real::EPSILON).unwrap();
            worst = core::cmp::max(worst, max_ulp_diff3(p, m3(a).try_inverse().unwrap()));
        }
        assert!(worst == 77683, "regressed: {worst}");
    }

    #[test]
    fn test_pseudo_inverse_of_a_rank_deficient_matrix() {
        // The pseudo-inverse drops the null directions: `A A⁺ A = A`.
        let p = Svd3Trait::new(a_rank2()).pseudo_inverse(Real::EPSILON).unwrap();
        assert!(max_ulp_diff3(a_rank2() * p * a_rank2(), a_rank2()) <= 64);
        assert!(Svd3Trait::new(a_rank2()).pseudo_inverse(Real::NEG_ONE).is_none());
        assert!(Svd3Trait::new(a_rank2()).solve(Vector3Trait::zeros(), Real::NEG_ONE).is_none());
    }

    #[test]
    fn test_to_polar_oracle() {
        let mut cases = oracle_svd::svd3_singular_values_cases();
        let (mut worst, mut worst_orth) = (0, 0);
        while let Some(case) = cases.pop_front() {
            let (a, _, _) = *case;
            let (p, u) = Svd3Trait::new(m3(a)).to_polar().unwrap();
            // `P` is symmetric by construction and positive semi-definite: its determinant is the
            // product of the singular values.
            assert!(p == p.transpose(), "P is not symmetric");
            assert!(!p.determinant().is_negative(), "P is not positive semi-definite");
            worst_orth = core::cmp::max(worst_orth, orthonormality_error_m3(u));
            worst = core::cmp::max(worst, max_ulp_diff3(p * u, m3(a)) / amax_m3(m3(a)));
        }
        // Measured: `|M - P U| <= worst ulp * max(1, max |m_ij|)`, `|UᵀU - I| <= worst_orth ulp`.
        assert!((worst, worst_orth) == (65, 65), "regressed: {worst} {worst_orth}");
    }

    #[test]
    fn test_to_polar_of_a_rotation_is_the_rotation() {
        let r = Matrix3Trait::new(
            int(0), int(0), int(1), int(1), int(0), int(0), int(0), int(1), int(0),
        );
        let (p, q) = r.svd().to_polar().unwrap();
        assert!(max_ulp_diff3(q, r) <= 4);
        assert!(max_ulp_diff3(p, Matrix3Trait::identity()) <= 4);
    }

    #[test]
    #[should_panic(expected: 'Fixed: overflow')]
    fn test_new_overflow_panics() {
        // `MᵀM` does not fit: the squares of the entries must be representable.
        let m = black_box(Matrix3Trait::from_diagonal_element(Real::<Fixed>::MAX));
        Svd3Trait::new(m);
    }

    // --- gas benchmarks --------------------------------------------------------

    #[test]
    #[inline(never)]
    fn bench_svd3_new__baseline() {
        let _a = black_box(a_bench());
        let e = black_box(true);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_svd3_new__eigen_of_gram() {
        let a = black_box(a_bench());
        let e = black_box(true);
        let f = Svd3Trait::new(a);
        assert!((f.singular_values.x >= f.singular_values.z) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_svd3_singular_values__baseline() {
        let _a = black_box(a_bench());
        let e = black_box(true);
        assert!(e == e);
    }

    /// The singular values as the decomposition produces them, `|M v_i|`.
    #[test]
    #[inline(never)]
    fn bench_svd3_singular_values__from_left_vectors() {
        let a = black_box(a_bench());
        let e = black_box(true);
        let s = Svd3Trait::new(a).singular_values;
        assert!((s.x >= s.z) == e);
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
        assert!((u.m11 != Real::ZERO) == e);
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
    fn bench_svd3_gram__baseline() {
        let _a = black_box(a_bench());
        let e = black_box(true);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_svd3_gram__fused() {
        let a = black_box(a_bench());
        let e = black_box(true);
        assert!((Svd3InternalTrait::gram(a).m11 != Real::ZERO) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_svd3_gram__transpose_mul_transpose() {
        let a = black_box(a_bench());
        let e = black_box(true);
        assert!((a.transpose().mul_transpose().m11 != Real::ZERO) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_svd3_recompose__baseline() {
        let _f = black_box(f_bench());
        let e = black_box(true);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_svd3_recompose__scaled_product() {
        let f = black_box(f_bench());
        let e = black_box(true);
        assert!((f.recompose().m11 != Real::ZERO) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_svd3_solve__baseline() {
        let _f = black_box(f_bench());
        let _b = black_box(v3t((1, 2, 3)));
        let e = black_box(true);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_svd3_solve__divisions() {
        let f = black_box(f_bench());
        let b = black_box(v3t((0x100000000, 0x200000000, 0x300000000)));
        let e = black_box(true);
        assert!(f.solve(b, Real::EPSILON).is_some() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_svd3_pseudo_inverse__baseline() {
        let _f = black_box(f_bench());
        let e = black_box(true);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_svd3_pseudo_inverse__reciprocals() {
        let f = black_box(f_bench());
        let e = black_box(true);
        assert!(f.pseudo_inverse(Real::EPSILON).is_some() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_svd3_rank__baseline() {
        let _f = black_box(f_bench());
        let e = black_box(true);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_svd3_rank__comparisons() {
        let f = black_box(f_bench());
        let e = black_box(true);
        assert!((f.rank(Real::EPSILON) == 3) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_svd3_to_polar__baseline() {
        let _f = black_box(f_bench());
        let e = black_box(true);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_svd3_to_polar__quadform() {
        let f = black_box(f_bench());
        let e = black_box(true);
        let (p, u) = f.to_polar().unwrap();
        assert!((u.m11 != Real::ZERO && p.m11 != Real::ZERO) == e);
    }
}
