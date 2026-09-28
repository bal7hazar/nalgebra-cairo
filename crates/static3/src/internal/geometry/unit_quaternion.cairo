//! Internal, no stability promise: the crate-private items of `geometry::unit_quaternion` that the
//! packages above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use nalgebra_core::base::matrix3::Matrix3;
use simba::scalar::{Real, Transcendental};
use crate::geometry::quaternion::{Quaternion, QuaternionTrait};
use crate::geometry::unit_quaternion::{
    FROM_MATRIX_MAX_ITER, FROM_MATRIX_MAX_PERTURBATIONS, Sym4, UnitQuaternion, UnitQuaternionTrait,
};
use crate::internal::geometry::quaternion::QuaternionInternalTrait;

/// Crate-internal kernels of `UnitQuaternion<T>` (WP 8.0: the public API is strictly upstream's):
/// the fused `conj_mul` of `Isometry3::inv_mul` (upstream writes `self.inverse() * other`), and the
/// by-value forms of the in-place `renormalize` / `renormalize_fast` for the tests and the
/// value-style call sites.
#[generate_trait]
pub impl UnitQuaternionInternalImpl<
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
> of UnitQuaternionInternalTrait<T> {
    /// `self` renormalized exactly (`UnitQuaternionTrait::renormalize`), by value.
    #[inline(always)]
    fn renormalized(self: UnitQuaternion<T>) -> UnitQuaternion<T> {
        let mut r = self;
        let _ = UnitQuaternionTrait::renormalize(ref r);
        r
    }

    /// `self` renormalized by one Newton step (`UnitQuaternionTrait::renormalize_fast`), by value.
    #[inline(always)]
    fn renormalized_fast(self: UnitQuaternion<T>) -> UnitQuaternion<T> {
        let mut r = self;
        UnitQuaternionTrait::renormalize_fast(ref r);
        r
    }

    /// `self⁻¹ · other` (= `self.conjugate() * other`): the rotation `other` expressed in the
    /// frame of `self`, as ONE fused Hamilton product with the conjugate's signs folded in
    /// (`QuaternionTrait::conj_mul`) — bit-identical to `self.inverse() * other`, three negations
    /// cheaper, and a component equal to the scalar's `MIN` no longer panics. Upstream has no
    /// direct equivalent: it replaces `self.inverse() * other` (as in `Isometry3::inv_mul`).
    #[inline(always)]
    fn conj_mul(self: UnitQuaternion<T>, other: UnitQuaternion<T>) -> UnitQuaternion<T> {
        UnitQuaternion { quaternion: self.quaternion.conj_mul(other.quaternion) }
    }

    /// `Σ q qᵀ / n` over the span, in `(w, i, j, k)` order: ten exact wide sums (one loop over
    /// the dynamic input), each scaled by the rounded `1 / n` with ONE rounding
    /// (`Real::wide_mul_scalar`). `n` must be the length of the span, at least 1.
    fn outer_sum(unit_quaternions: Span<UnitQuaternion<T>>, n: usize) -> Sym4<T> {
        let mut span = unit_quaternions;
        let (mut ww, mut wi, mut wj, mut wk) = (
            R::wide_zero(), R::wide_zero(), R::wide_zero(), R::wide_zero(),
        );
        let (mut ii, mut ij, mut ik) = (R::wide_zero(), R::wide_zero(), R::wide_zero());
        let (mut jj, mut jk, mut kk) = (R::wide_zero(), R::wide_zero(), R::wide_zero());
        while let Some(q) = span.pop_front() {
            let Quaternion { i, j, k, w } = (*q).quaternion;
            ww = R::wide_add_prod(ww, w, w);
            wi = R::wide_add_prod(wi, w, i);
            wj = R::wide_add_prod(wj, w, j);
            wk = R::wide_add_prod(wk, w, k);
            ii = R::wide_add_prod(ii, i, i);
            ij = R::wide_add_prod(ij, i, j);
            ik = R::wide_add_prod(ik, i, k);
            jj = R::wide_add_prod(jj, j, j);
            jk = R::wide_add_prod(jk, j, k);
            kk = R::wide_add_prod(kk, k, k);
        }
        let r = R::from_ratio(1, n.into());
        Sym4 {
            ww: R::wide_mul_scalar(ww, r),
            wi: R::wide_mul_scalar(wi, r),
            wj: R::wide_mul_scalar(wj, r),
            wk: R::wide_mul_scalar(wk, r),
            ii: R::wide_mul_scalar(ii, r),
            ij: R::wide_mul_scalar(ij, r),
            ik: R::wide_mul_scalar(ik, r),
            jj: R::wide_mul_scalar(jj, r),
            jk: R::wide_mul_scalar(jk, r),
            kk: R::wide_mul_scalar(kk, r),
        }
    }

    /// One normalised squaring `S² / tr(S²)` of a symmetric positive semi-definite matrix of
    /// trace 1: `tr(S²) = ‖S‖²_F` (16 products, one wide sum) is inverted once (it is in
    /// `[1/4, 1]`, so its reciprocal is in `[1, 4]`), then every entry of `S²` is one wide sum of
    /// four products scaled by that reciprocal with ONE rounding. The result has trace 1 again
    /// (to rounding). Out of line: `mean_of` calls it `MEAN_OF_SQUARINGS` times.
    fn normalized_square(s: Sym4<T>) -> Sym4<T> {
        let Sym4 { ww, wi, wj, wk, ii, ij, ik, jj, jk, kk } = s;
        let t = R::wide_add_prod(R::wide_add_prod(R::wide_zero(), ww, ww), ii, ii);
        let t = R::wide_add_prod(R::wide_add_prod(t, jj, jj), kk, kk);
        let t = R::wide_add_prod(R::wide_add_prod(t, wi, wi), wi, wi);
        let t = R::wide_add_prod(R::wide_add_prod(t, wj, wj), wj, wj);
        let t = R::wide_add_prod(R::wide_add_prod(t, wk, wk), wk, wk);
        let t = R::wide_add_prod(R::wide_add_prod(t, ij, ij), ij, ij);
        let t = R::wide_add_prod(R::wide_add_prod(t, ik, ik), ik, ik);
        let t = R::wide_add_prod(R::wide_add_prod(t, jk, jk), jk, jk);
        let r = R::recip(R::wide_rescale(t));
        Sym4 {
            ww: Self::row_col(ww, ww, wi, wi, wj, wj, wk, wk, r),
            wi: Self::row_col(ww, wi, wi, ii, wj, ij, wk, ik, r),
            wj: Self::row_col(ww, wj, wi, ij, wj, jj, wk, jk, r),
            wk: Self::row_col(ww, wk, wi, ik, wj, jk, wk, kk, r),
            ii: Self::row_col(wi, wi, ii, ii, ij, ij, ik, ik, r),
            ij: Self::row_col(wi, wj, ii, ij, ij, jj, ik, jk, r),
            ik: Self::row_col(wi, wk, ii, ik, ij, jk, ik, kk, r),
            jj: Self::row_col(wj, wj, ij, ij, jj, jj, jk, jk, r),
            jk: Self::row_col(wj, wk, ij, ik, jj, jk, jk, kk, r),
            kk: Self::row_col(wk, wk, ik, ik, jk, jk, kk, kk, r),
        }
    }

    /// `(a0·b0 + a1·b1 + a2·b2 + a3·b3) · r`, the exact sum scaled with ONE rounding.
    #[inline(always)]
    fn row_col(a0: T, b0: T, a1: T, b1: T, a2: T, b2: T, a3: T, b3: T, r: T) -> T {
        let acc = R::wide_add_prod(R::wide_add_prod(R::wide_zero(), a0, b0), a1, b1);
        R::wide_mul_scalar(R::wide_add_prod(R::wide_add_prod(acc, a2, b2), a3, b3), r)
    }

    /// `MEAN_OF_SQUARINGS` (12) normalised squarings, unrolled: `S^4096` normalised, the
    /// projector on the dominant eigenvector to within `(λ₂ / λ₁)^4096`.
    fn dominant_projector(s: Sym4<T>) -> Sym4<T> {
        let s = Self::normalized_square(Self::normalized_square(Self::normalized_square(s)));
        let s = Self::normalized_square(Self::normalized_square(Self::normalized_square(s)));
        let s = Self::normalized_square(Self::normalized_square(Self::normalized_square(s)));
        Self::normalized_square(Self::normalized_square(Self::normalized_square(s)))
    }

    /// The rotation maximising `tr(Rᵀ m)` (closest to `m` in Frobenius norm): the unit
    /// eigenvector of the largest eigenvalue of Horn's symmetric matrix `K(m)`, for which
    /// `tr(R(q)ᵀ m) = qᵀ K q` (`(w, i, j, k)` order: `K_ww = tr m`, `K_wi = m32 - m23`, ...,
    /// `K_ii = m11 - m22 - m33`, `K_ij = m12 + m21`, ...; every entry an exact sum of entries of
    /// `m`). `K` has trace 0 and spectral radius at most `‖K‖_F = 2‖m‖_F`, so `K +
    /// 2‖m‖_F·I` is positive semi-definite with the same dominant eigenvector; scaled by `1 /
    /// (8‖m‖_F)` it has trace 1, and `dominant_projector` / `dominant_column` extract the
    /// eigenvector (the eigenvalue ratio is at most `(λ₂ + c) / (λ₁ + c)` with `λ₁ - λ₂
    /// = 2(σ₂ + σ₃)`, e.g. 0.85 for a condition number of 8). The identity for the zero
    /// matrix. Upstream's sign convention is applied by the caller.
    fn closest_rotation(m: Matrix3<T>) -> UnitQuaternion<T> {
        let acc = R::wide_add_prod(R::wide_add_prod(R::wide_zero(), m.m11, m.m11), m.m21, m.m21);
        let acc = R::wide_add_prod(R::wide_add_prod(acc, m.m31, m.m31), m.m12, m.m12);
        let acc = R::wide_add_prod(R::wide_add_prod(acc, m.m22, m.m22), m.m32, m.m32);
        let acc = R::wide_add_prod(R::wide_add_prod(acc, m.m13, m.m13), m.m23, m.m23);
        let f = R::wide_sqrt(R::wide_add_prod(acc, m.m33, m.m33));
        if f == R::zero() {
            return UnitQuaternionTrait::identity();
        }
        let c = f + f;
        let r = R::recip(c + c + c + c);
        let s = Sym4 {
            ww: (m.m11 + m.m22 + m.m33 + c) * r,
            wi: (m.m32 - m.m23) * r,
            wj: (m.m13 - m.m31) * r,
            wk: (m.m21 - m.m12) * r,
            ii: (m.m11 - m.m22 - m.m33 + c) * r,
            ij: (m.m12 + m.m21) * r,
            ik: (m.m13 + m.m31) * r,
            jj: (m.m22 - m.m11 - m.m33 + c) * r,
            jk: (m.m23 + m.m32) * r,
            kk: (m.m33 - m.m11 - m.m22 + c) * r,
        };
        Self::dominant_column(Self::dominant_projector(s))
    }

    /// `±q` with upstream's `from_rotation_matrix` (Shepperd) sign: `w > 0` when the trace of the
    /// rotation `3w² - |v|²` is positive, otherwise the component of largest magnitude among
    /// `i`, `j`, `k` (first in that order on ties, as Shepperd's branches) is made positive.
    /// Upstream's `from_matrix_eps` ends with that conversion, so its result carries that sign.
    fn shepperd_sign(q: UnitQuaternion<T>) -> UnitQuaternion<T> {
        let Quaternion { i, j, k, w } = q.quaternion;
        let tr = R::wide_add_prod(R::wide_add_prod(R::wide_zero(), w, w), w, w);
        let tr = R::wide_sub_prod(R::wide_sub_prod(R::wide_add_prod(tr, w, w), i, i), j, j);
        let tr = R::wide_rescale(R::wide_sub_prod(tr, k, k));
        let (ai, aj, ak) = (R::abs(i), R::abs(j), R::abs(k));
        let positive = if tr > R::zero() {
            R::is_sign_positive(w)
        } else if ai > aj && ai > ak {
            R::is_sign_positive(i)
        } else if aj > ak {
            R::is_sign_positive(j)
        } else {
            R::is_sign_positive(k)
        };
        if positive {
            q
        } else {
            UnitQuaternion { quaternion: -q.quaternion }
        }
    }

    /// The normalised column of largest diagonal entry of a rank-one projector `v vᵀ`: `±v`, the
    /// sign making its largest component positive.
    fn dominant_column(s: Sym4<T>) -> UnitQuaternion<T> {
        let Sym4 { ww, wi, wj, wk, ii, ij, ik, jj, jk, kk } = s;
        let q = if ww >= ii && ww >= jj && ww >= kk {
            Quaternion { i: wi, j: wj, k: wk, w: ww }
        } else if ii >= jj && ii >= kk {
            Quaternion { i: ii, j: ij, k: ik, w: wi }
        } else if jj >= kk {
            Quaternion { i: ij, j: jj, k: jk, w: wj }
        } else {
            Quaternion { i: ik, j: jk, k: kk, w: wk }
        };
        UnitQuaternion { quaternion: q.normalize() }
    }
}

/// Crate-internal kernels of `UnitQuaternionAngleTrait::from_matrix_eps`.
#[generate_trait]
pub impl UnitQuaternionAngleInternalImpl<
    T,
    impl R: Real<T>,
    impl Tr: Transcendental<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of UnitQuaternionAngleInternalTrait<T> {
    /// Müller's iteration of `from_matrix_eps` (`max_iter > 0`, or 0 for the cap alone) and the
    /// number of iterations it ran (for the convergence tests). No sign convention applied.
    fn from_matrix_eps_count(
        m: Matrix3<T>, eps: T, max_iter: usize, guess: UnitQuaternion<T>,
    ) -> (UnitQuaternion<T>, usize) {
        let cap = if max_iter == 0 || max_iter > FROM_MATRIX_MAX_ITER {
            FROM_MATRIX_MAX_ITER
        } else {
            max_iter
        };
        let eps_dist = R::max(R::sqrt(eps), eps * eps);
        let mut q = guess.quaternion;
        // Perturbation axis: x, then z, then y (upstream's `yzx` swizzle of the x axis, cycled).
        let mut axis: u8 = 0;
        let mut iter: usize = 0;
        while iter < cap {
            iter += 1;
            let r = UnitQuaternionTrait::to_rotation_matrix(UnitQuaternion { quaternion: q })
                .matrix;
            let (x, y, z) = Self::muller_axis(r, m);
            let n = R::norm3(x, y, z);
            if n > eps {
                let (s, c) = Tr::sin_cos(n * R::HALF);
                let f = R::div(s, n);
                q = Quaternion { i: x * f, j: y * f, k: z * f, w: c } * q;
                continue;
            }
            // A stationary point: perturb to tell a maximum of `tr(Rᵀ m)` from a saddle.
            let d0 = Self::distance_squared(m, r);
            let (ps, pc) = Tr::sin_cos(eps_dist * R::HALF);
            let e = if axis == 0 {
                Quaternion { i: ps, j: R::zero(), k: R::zero(), w: pc }
            } else if axis == 1 {
                Quaternion { i: R::zero(), j: R::zero(), k: ps, w: pc }
            } else {
                Quaternion { i: R::zero(), j: ps, k: R::zero(), w: pc }
            };
            let mut p = q;
            let mut d1 = d0;
            let mut moved = false;
            let mut tries: usize = 0;
            while tries < FROM_MATRIX_MAX_PERTURBATIONS {
                tries += 1;
                p = p * e;
                d1 =
                    Self::distance_squared(
                        m,
                        UnitQuaternionTrait::to_rotation_matrix(UnitQuaternion { quaternion: p })
                            .matrix,
                    );
                if !R::abs_diff_eq(d0, d1, 1) {
                    moved = true;
                    break;
                }
            }
            if !moved || d0 < d1 {
                // The distance grows in the perturbed direction: a minimum, done.
                break;
            }
            axis = if axis == 2 {
                0
            } else {
                axis + 1
            };
            q = p;
        }
        (UnitQuaternionTrait::new_normalize(q), iter)
    }

    /// Müller's rotation vector `Σ_c r_c × m_c / (|Σ_c r_c · m_c| + ε)` (columns `c`, `ε` =
    /// `default_epsilon`): each component one fused kernel of six products and the denominator
    /// one of nine (exactly floored), then three correctly rounded divisions.
    fn muller_axis(r: Matrix3<T>, m: Matrix3<T>) -> (T, T, T) {
        let x = R::wide_sub_prod(R::wide_add_prod(R::wide_zero(), r.m21, m.m31), r.m31, m.m21);
        let x = R::wide_sub_prod(R::wide_add_prod(x, r.m22, m.m32), r.m32, m.m22);
        let x = R::wide_rescale(R::wide_sub_prod(R::wide_add_prod(x, r.m23, m.m33), r.m33, m.m23));
        let y = R::wide_sub_prod(R::wide_add_prod(R::wide_zero(), r.m31, m.m11), r.m11, m.m31);
        let y = R::wide_sub_prod(R::wide_add_prod(y, r.m32, m.m12), r.m12, m.m32);
        let y = R::wide_rescale(R::wide_sub_prod(R::wide_add_prod(y, r.m33, m.m13), r.m13, m.m33));
        let z = R::wide_sub_prod(R::wide_add_prod(R::wide_zero(), r.m11, m.m21), r.m21, m.m11);
        let z = R::wide_sub_prod(R::wide_add_prod(z, r.m12, m.m22), r.m22, m.m12);
        let z = R::wide_rescale(R::wide_sub_prod(R::wide_add_prod(z, r.m13, m.m23), r.m23, m.m13));
        let d = R::wide_add_prod(R::wide_add_prod(R::wide_zero(), r.m11, m.m11), r.m21, m.m21);
        let d = R::wide_add_prod(R::wide_add_prod(d, r.m31, m.m31), r.m12, m.m12);
        let d = R::wide_add_prod(R::wide_add_prod(d, r.m22, m.m22), r.m32, m.m32);
        let d = R::wide_add_prod(R::wide_add_prod(d, r.m13, m.m13), r.m23, m.m23);
        let d = R::wide_rescale(R::wide_add_prod(d, r.m33, m.m33));
        R::div3(x, y, z, R::abs(d) + R::default_epsilon())
    }

    /// `‖m - r‖²_F`: nine exact differences, one wide sum of squares, floored once.
    fn distance_squared(m: Matrix3<T>, r: Matrix3<T>) -> T {
        let (a, b, c) = (m.m11 - r.m11, m.m21 - r.m21, m.m31 - r.m31);
        let (d, e, f) = (m.m12 - r.m12, m.m22 - r.m22, m.m32 - r.m32);
        let (g, h, k) = (m.m13 - r.m13, m.m23 - r.m23, m.m33 - r.m33);
        let acc = R::wide_add_prod(R::wide_add_prod(R::wide_zero(), a, a), b, b);
        let acc = R::wide_add_prod(R::wide_add_prod(acc, c, c), d, d);
        let acc = R::wide_add_prod(R::wide_add_prod(acc, e, e), f, f);
        let acc = R::wide_add_prod(R::wide_add_prod(acc, g, g), h, h);
        R::wide_rescale(R::wide_add_prod(acc, k, k))
    }
}
