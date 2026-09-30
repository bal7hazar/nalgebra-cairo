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

use nalgebra_core::base::matrix_tr_mul::MatrixTrMul;
use nalgebra_static3::base::matrix3::Matrix3Trait;
use nalgebra_static3::internal::base::matrix3::Matrix3InternalTrait;
use nalgebra_static3::internal::base::sym_matrix3::SymMatrix3Trait;
use nalgebra_types3::base::matrix3::Matrix3;
use nalgebra_types3::base::vector3::Vector3;
use simba::scalar::Real;
use crate::internal::linalg::svd3::Svd3InternalTrait;
use crate::internal::linalg::symmetric_eigen3::SymmetricEigen3InternalTrait;

/// The singular value decomposition `M = U · diag(singular_values) · v_t` of a `Matrix3<T>`.
///
/// `singular_values` are sorted **descending** and non-negative (like `tools/oracle` and upstream's
/// `singular_values`), `u` and `v_t` are orthonormal, and `None` when the decomposition was built
/// without them (the `compute_u` / `compute_v` flags, like upstream).
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
    /// The left singular vectors, as columns, when computed. Orthonormal.
    pub u: Option<Matrix3<T>>,
    /// The three singular values, descending, non-negative.
    pub singular_values: Vector3<T>,
    /// The TRANSPOSE of the right singular vectors, i.e. `v_i` is ROW `i`, when computed.
    /// Orthonormal.
    pub v_t: Option<Matrix3<T>>,
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
    /// The singular value decomposition of `matrix`, with `u` when `compute_u` and `v_t` when
    /// `compute_v` (`None` otherwise; skipping `u` skips the last two lines below, `v_t` is free:
    /// the right vectors are what the singular values are read off). Upstream:
    /// `matrix.svd(compute_u, compute_v)` / `SVD::new(matrix, compute_u, compute_v)`.
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
    fn new(matrix: Matrix3<T>, compute_u: bool, compute_v: bool) -> Svd3<T> {
        Svd3InternalTrait::from_eigen(
            matrix,
            SymmetricEigen3InternalTrait::new_sym(Svd3InternalTrait::gram(matrix)),
            compute_u,
            compute_v,
        )
    }

    /// `new`: its sorting network already orders the singular values, so the unordered form
    /// costs the same (upstream's 3x3 path, `svd_ordered3`, is always ordered too). Upstream:
    /// `SVD::new_unordered`.
    #[inline(always)]
    fn new_unordered(matrix: Matrix3<T>, compute_u: bool, compute_v: bool) -> Svd3<T> {
        Self::new(matrix, compute_u, compute_v)
    }

    /// `new`, or `None` when the four Jacobi sweeps of `SymmetricEigen3` on `MᵀM` did not
    /// reach upstream's convergence criterion (every off-diagonal entry within `eps * (|s_ii| +
    /// |s_jj|)`, see `SymmetricEigen3Trait::try_new`). `max_niter` is accepted for signature
    /// parity and ignored: the iteration budget is a constant of the type. Bit-identical to
    /// `new` when `Some`. Upstream: `SVD::try_new(matrix, compute_u, compute_v, eps,
    /// max_niter)`.
    fn try_new(
        matrix: Matrix3<T>, compute_u: bool, compute_v: bool, eps: T, max_niter: usize,
    ) -> Option<Svd3<T>> {
        let _ = max_niter;
        match SymmetricEigen3InternalTrait::try_new_sym(Svd3InternalTrait::gram(matrix), eps) {
            Some(eigen) => Some(Svd3InternalTrait::from_eigen(matrix, eigen, compute_u, compute_v)),
            None => None,
        }
    }

    /// `try_new`, see `new_unordered`. Upstream: `SVD::try_new_unordered`.
    #[inline(always)]
    fn try_new_unordered(
        matrix: Matrix3<T>, compute_u: bool, compute_v: bool, eps: T, max_niter: usize,
    ) -> Option<Svd3<T>> {
        Self::try_new(matrix, compute_u, compute_v, eps, max_niter)
    }

    /// Sorts the singular values DESCENDING, permuting the columns of `u` and the rows of `v_t`
    /// (those that were computed) with them (three conditional swaps, strict comparison: equal
    /// values keep their order).
    /// `new` already returns them sorted, so this only matters after the fields were edited.
    /// Upstream: `SVD::sort_by_singular_values`.
    fn sort_by_singular_values(ref self: Svd3<T>) {
        let (mut s1, mut s2, mut s3) = (
            self.singular_values.x, self.singular_values.y, self.singular_values.z,
        );
        let (has_u, has_v) = (self.u.is_some(), self.v_t.is_some());
        let u = self.u.unwrap_or_else(|| Matrix3Trait::zeros());
        let v_t = self.v_t.unwrap_or_else(|| Matrix3Trait::zeros());
        let (mut u1, mut u2, mut u3) = (u.column1(), u.column2(), u.column3());
        let (mut v1, mut v2, mut v3) = (v_t.row1(), v_t.row2(), v_t.row3());
        if s2 > s1 {
            let (ts, tu, tv) = (s1, u1, v1);
            s1 = s2;
            u1 = u2;
            v1 = v2;
            s2 = ts;
            u2 = tu;
            v2 = tv;
        }
        if s3 > s2 {
            let (ts, tu, tv) = (s2, u2, v2);
            s2 = s3;
            u2 = u3;
            v2 = v3;
            s3 = ts;
            u3 = tu;
            v3 = tv;
        }
        if s2 > s1 {
            let (ts, tu, tv) = (s1, u1, v1);
            s1 = s2;
            u1 = u2;
            v1 = v2;
            s2 = ts;
            u2 = tu;
            v2 = tv;
        }
        self =
            Svd3 {
                u: if has_u {
                    Some(Matrix3Trait::from_columns(u1, u2, u3))
                } else {
                    None
                },
                singular_values: Vector3 { x: s1, y: s2, z: s3 },
                v_t: if has_v {
                    Some(Matrix3Trait::from_rows(v1, v2, v3))
                } else {
                    None
                },
            };
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
    /// rounding (measured: **64 ulp per unit of `max |m_ij|`** on the oracle vectors), or `None`
    /// when `u` or `v_t` was not computed.
    ///
    /// Two roundings per entry: the columns of `U` are scaled by the singular values (9 floored
    /// products), then the product with `v_t` is 9 fused `sum_prod3`. Panics on overflow.
    /// Upstream: `SVD::recompose` (`Err` when a factor is missing; `None` here).
    fn recompose(self: Svd3<T>) -> Option<Matrix3<T>> {
        let u = self.u?;
        let v_t = self.v_t?;
        Some(Svd3InternalTrait::scale_columns(u, self.singular_values) * v_t)
    }

    /// The Moore-Penrose pseudo-inverse `V · diag(σ⁺) · Uᵀ`, where `σ⁺_i = 1 / σ_i` when
    /// `σ_i > eps` and `0` otherwise, or `None` when `eps` is negative or when `u` or `v_t` was
    /// not computed.
    ///
    /// Upstream: `SVD::pseudo_inverse`, which returns `Err` in those cases and otherwise
    /// recomposes with the inverted singular values and takes the adjoint — the same expression.
    /// Upstream's test is `> eps` too, so `eps = 0` keeps every nonzero singular value, including
    /// one of 1 raw unit whose reciprocal overflows: pass an `eps` matched to the scale of the
    /// problem, not zero, unless an overflow panic is the wanted answer.
    ///
    /// Three roundings per entry (the reciprocal, the scaling, the product). Panics with the
    /// scalar's overflow error if a reciprocal or an entry does not fit.
    fn pseudo_inverse(self: Svd3<T>, eps: T) -> Option<Matrix3<T>> {
        if eps.is_sign_negative() {
            return None;
        }
        let u = self.u?;
        let v_t = self.v_t?;
        let r = Vector3 {
            x: Svd3InternalTrait::inverted(self.singular_values.x, eps),
            y: Svd3InternalTrait::inverted(self.singular_values.y, eps),
            z: Svd3InternalTrait::inverted(self.singular_values.z, eps),
        };
        Some(Svd3InternalTrait::scale_columns(v_t.transpose(), r) * u.transpose())
    }

    /// The least-squares solution of `M x = b`, `V · (Uᵀ b / σ)` with the components whose
    /// singular value is `<= eps` zeroed, or `None` when `eps` is negative or when `u` or `v_t`
    /// was not computed. Upstream: `SVD::solve` (`Err` in those cases).
    ///
    /// Unlike `pseudo_inverse` this divides instead of multiplying by a reciprocal — one rounding
    /// per component instead of two, and no overflow on a tiny singular value unless the solution
    /// itself does not fit. Panics with the scalar's overflow error in that case.
    fn solve(self: Svd3<T>, b: Vector3<T>, eps: T) -> Option<Vector3<T>> {
        if eps.is_sign_negative() {
            return None;
        }
        let u = self.u?;
        let v_t = self.v_t?;
        let y = u.tr_mul(b);
        let z = Vector3 {
            x: Svd3InternalTrait::divided(y.x, self.singular_values.x, eps),
            y: Svd3InternalTrait::divided(y.y, self.singular_values.y, eps),
            z: Svd3InternalTrait::divided(y.z, self.singular_values.z, eps),
        };
        Some(v_t.tr_mul(z))
    }

    /// The LEFT polar decomposition `M = P · U`, as `Some((P, U))`: `P = u · diag(σ) · uᵀ` is
    /// symmetric positive semi-definite and `U = u · v_t` is orthonormal (a rotation when
    /// `det(M) > 0`), or `None` when `u` or `v_t` was not computed (as upstream).
    ///
    /// `U` costs one 3x3 product and `P` one structured quadratic form (only its 6 independent
    /// components are computed, then mirrored); both are two roundings per entry. Panics on
    /// overflow. Upstream: `SVD::to_polar`.
    #[inline(always)]
    fn to_polar(self: Svd3<T>) -> Option<(Matrix3<T>, Matrix3<T>)> {
        let u = self.u?;
        let v_t = self.v_t?;
        Some((SymMatrix3Trait::quadform(u, self.singular_values).to_matrix(), u * v_t))
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
    /// The singular value decomposition, see `Svd3Trait::new`. Upstream: `Matrix3::svd`.
    #[inline(always)]
    fn svd(self: Matrix3<T>, compute_u: bool, compute_v: bool) -> Svd3<T> {
        Svd3Trait::new(self, compute_u, compute_v)
    }

    /// The singular values alone, descending: `svd(false, false)` (no left vectors). Upstream:
    /// `Matrix3::singular_values`.
    #[inline(always)]
    fn singular_values(self: Matrix3<T>) -> Vector3<T> {
        Svd3Trait::new(self, false, false).singular_values
    }

    /// The Moore-Penrose pseudo-inverse, see `Svd3::pseudo_inverse`. Upstream:
    /// `Matrix3::pseudo_inverse`.
    #[inline(always)]
    fn pseudo_inverse(self: Matrix3<T>, eps: T) -> Option<Matrix3<T>> {
        Svd3Trait::new(self, true, true).pseudo_inverse(eps)
    }

    /// `svd`: the decomposition is always sorted (see `Svd3Trait::new_unordered`). Upstream:
    /// `Matrix::svd_unordered`.
    #[inline(always)]
    fn svd_unordered(self: Matrix3<T>, compute_u: bool, compute_v: bool) -> Svd3<T> {
        Svd3Trait::new(self, compute_u, compute_v)
    }

    /// See `Svd3Trait::try_new`. Upstream: `Matrix::try_svd`.
    #[inline(always)]
    fn try_svd(
        self: Matrix3<T>, compute_u: bool, compute_v: bool, eps: T, max_niter: usize,
    ) -> Option<Svd3<T>> {
        Svd3Trait::try_new(self, compute_u, compute_v, eps, max_niter)
    }

    /// See `Svd3Trait::try_new_unordered`. Upstream: `Matrix::try_svd_unordered`.
    #[inline(always)]
    fn try_svd_unordered(
        self: Matrix3<T>, compute_u: bool, compute_v: bool, eps: T, max_niter: usize,
    ) -> Option<Svd3<T>> {
        Svd3Trait::try_new(self, compute_u, compute_v, eps, max_niter)
    }

    /// `singular_values` (always sorted). Upstream: `Matrix::singular_values_unordered`.
    #[inline(always)]
    fn singular_values_unordered(self: Matrix3<T>) -> Vector3<T> {
        Svd3Trait::new(self, false, false).singular_values
    }

    /// The number of singular values strictly greater than `eps`. Upstream: `Matrix::rank`
    /// (which asserts `eps >= 0`; a negative `eps` counts every value here, like `Svd3::rank`).
    fn rank(self: Matrix3<T>, eps: T) -> usize {
        let s = Svd3Trait::new(self, false, false).singular_values;
        let mut n: usize = 0;
        if s.x > eps {
            n += 1;
        }
        if s.y > eps {
            n += 1;
        }
        if s.z > eps {
            n += 1;
        }
        n
    }

    /// The left polar decomposition `M = P · U`, see `Svd3Trait::to_polar`. Upstream:
    /// `Matrix::polar`.
    fn polar(self: Matrix3<T>) -> (Matrix3<T>, Matrix3<T>) {
        Svd3Trait::new(self, true, true).to_polar().unwrap()
    }

    /// `polar`, or `None` when the decomposition did not converge within `eps`, see
    /// `Svd3Trait::try_new`. Upstream: `Matrix::try_polar`.
    fn try_polar(self: Matrix3<T>, eps: T, max_niter: usize) -> Option<(Matrix3<T>, Matrix3<T>)> {
        match Svd3Trait::try_new(self, true, true, eps, max_niter) {
            Some(d) => d.to_polar(),
            None => None,
        }
    }
}

/// The ordered SVD of a `Matrix3`: `Svd3Trait::try_new(m, compute_u, compute_v, eps, niter)`,
/// `None` when the Jacobi
/// sweeps did not converge within `eps`. Upstream: `nalgebra::linalg::svd_ordered3` (McAdams et
/// al.: the symmetric eigen decomposition of `MᵀM`, then a QR of `M V` — the route `Svd3` takes
/// too). `niter` is accepted for signature parity and ignored (constant budget).
pub fn svd_ordered3<
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
>(
    m: Matrix3<T>, compute_u: bool, compute_v: bool, eps: T, niter: usize,
) -> Option<Svd3<T>> {
    Svd3Trait::try_new(m, compute_u, compute_v, eps, niter)
}
