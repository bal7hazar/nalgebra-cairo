//! `SymMatrix2`: a symmetric 2x2 matrix stored as its 3 independent components.
//!
//! No upstream nalgebra equivalent; the model is rapier / parry's `SdpMatrix2` (2D effective
//! masses, 2x2 constraint blocks). Structured kernels (`quadform`, `quadform_sym`,
//! `from_outer_self`, `Matrix2::mul_transpose`) compute only the 3 independent components, and are
//! bit-identical to the upper triangle of the generic `Matrix2` expression they replace.

use core::ops::{AddAssign, SubAssign};
use simba::scalar::Real;
use super::matrix2::Matrix2;
use super::vector2::Vector2;

/// A symmetric 2x2 matrix `[[m11, m12], [m12, m22]]`.
#[derive(Copy, Drop, PartialEq, Serde, Default, Debug, Hash)]
pub struct SymMatrix2<T> {
    pub m11: T,
    pub m12: T,
    pub m22: T,
}

/// Methods of `SymMatrix2<T>` for any `Real` scalar.
#[generate_trait]
pub impl SymMatrix2Impl<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Div<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of SymMatrix2Trait<T> {
    // --- constructors --------------------------------------------------------------------------

    /// The symmetric matrix `[[m11, m12], [m12, m22]]`. parry: `SdpMatrix2::new`.
    #[inline(always)]
    fn new(m11: T, m12: T, m22: T) -> SymMatrix2<T> {
        SymMatrix2 { m11, m12, m22 }
    }

    /// The zero matrix. parry: `SdpMatrix2::zero`.
    #[inline(always)]
    fn zeros() -> SymMatrix2<T> {
        SymMatrix2 { m11: R::ZERO, m12: R::ZERO, m22: R::ZERO }
    }

    /// The identity matrix.
    #[inline(always)]
    fn identity() -> SymMatrix2<T> {
        SymMatrix2 { m11: R::ONE, m12: R::ZERO, m22: R::ONE }
    }

    /// The diagonal matrix `diag(d.x, d.y)`.
    #[inline(always)]
    fn from_diagonal(d: Vector2<T>) -> SymMatrix2<T> {
        SymMatrix2 { m11: d.x, m12: R::ZERO, m22: d.y }
    }

    /// The matrix `e * I`. parry: `SdpMatrix2::diagonal`.
    #[inline(always)]
    fn from_diagonal_element(e: T) -> SymMatrix2<T> {
        SymMatrix2 { m11: e, m12: R::ZERO, m22: e }
    }

    /// The UPPER triangle of `m` (`m11`, `m12`, `m22`); `m21` is ignored, the symmetry of `m` is
    /// not checked. parry: `SdpMatrix2::from_sdp_matrix`.
    #[inline(always)]
    fn from_matrix_unchecked(m: Matrix2<T>) -> SymMatrix2<T> {
        SymMatrix2 { m11: m.m11, m12: m.m12, m22: m.m22 }
    }

    /// The outer product `v * vᵀ`: 3 floored products. Panics on overflow.
    /// Upstream: `v * v.transpose()`.
    #[inline(always)]
    fn from_outer_self(v: Vector2<T>) -> SymMatrix2<T> {
        SymMatrix2 { m11: v.x * v.x, m12: v.x * v.y, m22: v.y * v.y }
    }

    /// `r * diag(d) * rᵀ` without materialising the diagonal matrix: 4 products `r_ik * d_k`
    /// then 3 `sum_prod2` (two roundings per component, like `(r * diag(d)) * rᵀ`, to which it
    /// is bit-identical). Panics on overflow. Upstream: `quadform_tr` with a diagonal `mid`.
    fn quadform(r: Matrix2<T>, d: Vector2<T>) -> SymMatrix2<T> {
        let (t11, t12) = (r.m11 * d.x, r.m12 * d.y);
        let (t21, t22) = (r.m21 * d.x, r.m22 * d.y);
        SymMatrix2 {
            m11: R::sum_prod2(t11, r.m11, t12, r.m12),
            m12: R::sum_prod2(t11, r.m21, t12, r.m22),
            m22: R::sum_prod2(t21, r.m21, t22, r.m22),
        }
    }

    // --- accessors and conversions -------------------------------------------------------------

    /// The full matrix. parry: `SdpMatrix2::into_matrix`.
    #[inline(always)]
    fn to_matrix(self: SymMatrix2<T>) -> Matrix2<T> {
        Matrix2 { m11: self.m11, m21: self.m12, m12: self.m12, m22: self.m22 }
    }

    /// The diagonal `(m11, m22)`.
    #[inline(always)]
    fn diagonal(self: SymMatrix2<T>) -> Vector2<T> {
        Vector2 { x: self.m11, y: self.m22 }
    }

    /// `m11 + m22`. Exact; panics on overflow.
    #[inline(always)]
    fn trace(self: SymMatrix2<T>) -> T {
        self.m11 + self.m22
    }

    /// `self + e * I`. Exact; panics on overflow. parry: `SdpMatrix2::add_diagonal`.
    #[inline(always)]
    fn add_diagonal(self: SymMatrix2<T>, e: T) -> SymMatrix2<T> {
        SymMatrix2 { m11: self.m11 + e, m12: self.m12, m22: self.m22 + e }
    }

    // --- products ------------------------------------------------------------------------------

    /// `self * k`: 3 floored products. Panics on overflow. parry: `SdpMatrix2 * Real`.
    #[inline(always)]
    fn scale(self: SymMatrix2<T>, k: T) -> SymMatrix2<T> {
        SymMatrix2 { m11: self.m11 * k, m12: self.m12 * k, m22: self.m22 * k }
    }

    /// `self * v`: one `sum_prod2` per component. Panics on overflow. parry: `mul_vec`.
    #[inline(always)]
    fn mul_vec(self: SymMatrix2<T>, v: Vector2<T>) -> Vector2<T> {
        Vector2 {
            x: R::sum_prod2(self.m11, v.x, self.m12, v.y),
            y: R::sum_prod2(self.m12, v.x, self.m22, v.y),
        }
    }

    /// `self * m` (a general matrix): 4 `sum_prod2`. Panics on overflow.
    fn mul_matrix(self: SymMatrix2<T>, m: Matrix2<T>) -> Matrix2<T> {
        Matrix2 {
            m11: R::sum_prod2(self.m11, m.m11, self.m12, m.m21),
            m21: R::sum_prod2(self.m12, m.m11, self.m22, m.m21),
            m12: R::sum_prod2(self.m11, m.m12, self.m12, m.m22),
            m22: R::sum_prod2(self.m12, m.m12, self.m22, m.m22),
        }
    }

    /// `r * self * rᵀ`: 7 `sum_prod2` instead of 8, bit-identical to the upper triangle of
    /// `(r * self.to_matrix()) * r.transpose()` (two roundings per component). Panics on
    /// overflow. Upstream: `quadform_tr`; parry's `quadform(m)` is `quadform_sym(m.transpose())`.
    fn quadform_sym(self: SymMatrix2<T>, r: Matrix2<T>) -> SymMatrix2<T> {
        let t11 = R::sum_prod2(r.m11, self.m11, r.m12, self.m12);
        let t12 = R::sum_prod2(r.m11, self.m12, r.m12, self.m22);
        let t21 = R::sum_prod2(r.m21, self.m11, r.m22, self.m12);
        let t22 = R::sum_prod2(r.m21, self.m12, r.m22, self.m22);
        SymMatrix2 {
            m11: R::sum_prod2(t11, r.m11, t12, r.m12),
            m12: R::sum_prod2(t11, r.m21, t12, r.m22),
            m22: R::sum_prod2(t21, r.m21, t22, r.m22),
        }
    }

    // --- norms ---------------------------------------------------------------------------------

    /// Frobenius norm of the full matrix (the off-diagonal square counted twice): square root of
    /// the exact, unscaled sum of squares, one rounding. Bit-identical to
    /// `self.to_matrix().norm()`.
    #[inline(always)]
    fn norm(self: SymMatrix2<T>) -> T {
        R::norm4(self.m11, self.m12, self.m12, self.m22)
    }

    // --- determinant and inverse ---------------------------------------------------------------

    /// `m11 * m22 - m12^2` with a single rounding: the exact floor of the true determinant.
    /// Panics on overflow.
    #[inline(always)]
    fn determinant(self: SymMatrix2<T>) -> T {
        R::diff_prod(self.m11, self.m22, self.m12, self.m12)
    }

    /// The inverse, or `None` when the computed determinant is exactly zero. Same algorithm,
    /// criterion, rounding and panics as `Matrix2::try_inverse`, to which it is bit-identical
    /// (`s.try_inverse() == s.to_matrix().try_inverse()` on the upper triangle).
    fn try_inverse(self: SymMatrix2<T>) -> Option<SymMatrix2<T>> {
        let det = R::diff_prod(self.m11, self.m22, self.m12, self.m12);
        if det < R::HALF && det > -R::HALF {
            let f = R::norm4(self.m11, self.m12, self.m12, self.m22);
            if f == R::ZERO {
                return None;
            }
            let k = R::floor(R::TWO / f);
            if k >= R::TWO {
                let (b11, b12, b22) = (self.m11 * k, self.m12 * k, self.m22 * k);
                let det_b = R::diff_prod(b11, b22, b12, b12);
                if det_b == R::ZERO {
                    return None;
                }
                let t = k / det_b;
                return Some(SymMatrix2 { m11: b22 * t, m12: (-b12) * t, m22: b11 * t });
            }
            if det == R::ZERO {
                return None;
            }
        }
        Some(SymMatrix2 { m11: self.m22 / det, m12: (-self.m12) / det, m22: self.m11 / det })
    }

    /// The inverse without the singularity checks: same result as `try_inverse` on an invertible
    /// matrix; a singular matrix panics with the scalar's division-by-zero error instead of
    /// returning `None`. parry: `SdpMatrix2::inverse_unchecked`.
    fn inverse_unchecked(self: SymMatrix2<T>) -> SymMatrix2<T> {
        let det = R::diff_prod(self.m11, self.m22, self.m12, self.m12);
        if det < R::HALF && det > -R::HALF {
            let k = R::floor(R::TWO / R::norm4(self.m11, self.m12, self.m12, self.m22));
            if k >= R::TWO {
                let (b11, b12, b22) = (self.m11 * k, self.m12 * k, self.m22 * k);
                let t = k / R::diff_prod(b11, b22, b12, b12);
                return SymMatrix2 { m11: b22 * t, m12: (-b12) * t, m22: b11 * t };
            }
        }
        SymMatrix2 { m11: self.m22 / det, m12: (-self.m12) / det, m22: self.m11 / det }
    }

    // --- approximate equality ------------------------------------------------------------------

    /// Whether every component of `self` is within `ulps` smallest units of `other`'s.
    fn abs_diff_eq(self: SymMatrix2<T>, other: SymMatrix2<T>, ulps: u64) -> bool {
        R::abs_diff_eq(self.m11, other.m11, ulps)
            && R::abs_diff_eq(self.m12, other.m12, ulps)
            && R::abs_diff_eq(self.m22, other.m22, ulps)
    }
}

// --- operators -----------------------------------------------------------------------------------

/// `a + b`, component-wise. Exact; panics on overflow.
pub impl SymMatrix2Add<T, +Add<T>, +Copy<T>, +Drop<T>> of Add<SymMatrix2<T>> {
    #[inline(always)]
    fn add(lhs: SymMatrix2<T>, rhs: SymMatrix2<T>) -> SymMatrix2<T> {
        SymMatrix2 { m11: lhs.m11 + rhs.m11, m12: lhs.m12 + rhs.m12, m22: lhs.m22 + rhs.m22 }
    }
}

/// `a - b`, component-wise. Exact; panics on overflow.
pub impl SymMatrix2Sub<T, +Sub<T>, +Copy<T>, +Drop<T>> of Sub<SymMatrix2<T>> {
    #[inline(always)]
    fn sub(lhs: SymMatrix2<T>, rhs: SymMatrix2<T>) -> SymMatrix2<T> {
        SymMatrix2 { m11: lhs.m11 - rhs.m11, m12: lhs.m12 - rhs.m12, m22: lhs.m22 - rhs.m22 }
    }
}

/// `-a`, component-wise. Exact; panics on the scalar's `MIN`.
pub impl SymMatrix2Neg<T, +Neg<T>, +Copy<T>, +Drop<T>> of Neg<SymMatrix2<T>> {
    #[inline(always)]
    fn neg(a: SymMatrix2<T>) -> SymMatrix2<T> {
        SymMatrix2 { m11: -a.m11, m12: -a.m12, m22: -a.m22 }
    }
}

/// `a += b`.
pub impl SymMatrix2AddAssign<
    T, +Add<T>, +Copy<T>, +Drop<T>,
> of AddAssign<SymMatrix2<T>, SymMatrix2<T>> {
    #[inline(always)]
    fn add_assign(ref self: SymMatrix2<T>, rhs: SymMatrix2<T>) {
        self = self + rhs;
    }
}

/// `a -= b`.
pub impl SymMatrix2SubAssign<
    T, +Sub<T>, +Copy<T>, +Drop<T>,
> of SubAssign<SymMatrix2<T>, SymMatrix2<T>> {
    #[inline(always)]
    fn sub_assign(ref self: SymMatrix2<T>, rhs: SymMatrix2<T>) {
        self = self - rhs;
    }
}

#[cfg(test)]
mod tests {
    use nalgebra_testing::black_box;
    use simba::fixed::Fixed;
    use simba::scalar::Real;
    use crate::base::matrix2::Matrix2Trait;
    use crate::base::matrix_test_utils::{
        fx, int, m2, m2i, max_ulp_diff2, max_ulp_diff_s2, s2, s2i, v2, v2i,
    };
    use crate::base::{oracle_matrix2, oracle_sym_matrix};
    use super::SymMatrix2Trait;

    /// The oracle's `matrix2` cases without the `large` distribution (indices [12..16)), whose
    /// squares do not fit a quadratic form.
    fn quadratic_cases() -> Span<([[i64; 2]; 2], [[i64; 2]; 2], [[i64; 2]; 2], u64)> {
        oracle_matrix2::matrix2_mul_cases().slice(0, 12)
    }

    // --- constructors and accessors ----------------------------------------------------------

    #[test]
    fn test_new_zeros_identity() {
        assert!(SymMatrix2Trait::new(int(1), int(2), int(3)) == s2i((1, 2, 3)));
        assert!(SymMatrix2Trait::<Fixed>::zeros() == s2i((0, 0, 0)));
        assert!(SymMatrix2Trait::<Fixed>::zeros() == Default::default());
        assert!(SymMatrix2Trait::<Fixed>::identity() == s2i((1, 0, 1)));
        assert!(SymMatrix2Trait::<Fixed>::identity().to_matrix() == Matrix2Trait::identity());
    }

    #[test]
    fn test_serde_is_upper_triangle() {
        let mut out = array![];
        s2i((1, 2, 3)).serialize(ref out);
        let one: felt252 = 0x100000000;
        assert!(out == array![1 * one, 2 * one, 3 * one]);
    }

    #[test]
    fn test_from_diagonal() {
        assert!(SymMatrix2Trait::from_diagonal(v2i(2, -3)) == s2i((2, 0, -3)));
        assert!(SymMatrix2Trait::from_diagonal_element(int(7)) == s2i((7, 0, 7)));
        assert!(
            SymMatrix2Trait::from_diagonal(v2i(2, -3))
                .to_matrix() == Matrix2Trait::from_diagonal(v2i(2, -3)),
        );
    }

    #[test]
    fn test_from_matrix_unchecked_reads_the_upper_triangle() {
        // `m21` is ignored: an asymmetric matrix is silently symmetrised from its upper triangle.
        let m = m2i([[1, 2], [99, 4]]);
        assert!(SymMatrix2Trait::from_matrix_unchecked(m) == s2i((1, 2, 4)));
        assert!(SymMatrix2Trait::from_matrix_unchecked(m).to_matrix() == m2i([[1, 2], [2, 4]]));
    }

    #[test]
    fn test_to_matrix_is_a_round_trip() {
        let mut cases = oracle_matrix2::matrix2_transpose_cases();
        while let Some(case) = cases.pop_front() {
            let (a, _, _) = *case;
            let s = SymMatrix2Trait::from_matrix_unchecked(m2(a));
            assert!(SymMatrix2Trait::from_matrix_unchecked(s.to_matrix()) == s);
            // A symmetric matrix is its own transpose.
            assert!(s.to_matrix().transpose() == s.to_matrix());
        }
    }

    #[test]
    fn test_from_outer_self_matches_the_generic_outer_product() {
        let mut cases = oracle_matrix2::matrix2_outer_cases();
        while let Some(case) = cases.pop_front() {
            let (u, _, _, _) = *case;
            let v = v2(u);
            assert!(
                SymMatrix2Trait::from_outer_self(v).to_matrix() == Matrix2Trait::from_outer(v, v),
            );
        }
        assert!(SymMatrix2Trait::from_outer_self(v2i(2, -3)) == s2i((4, -6, 9)));
    }

    #[test]
    fn test_diagonal_trace_add_diagonal() {
        let s = s2i((1, 2, 4));
        assert!(s.diagonal() == v2i(1, 4));
        assert!(s.trace() == int(5));
        assert!(s.add_diagonal(int(10)) == s2i((11, 2, 14)));
    }

    // --- exact operations ----------------------------------------------------------------------

    #[test]
    fn test_add_sub_neg_match_the_generic_operators() {
        let mut cases = oracle_matrix2::matrix2_add_cases();
        while let Some(case) = cases.pop_front() {
            let (a, b, _, _) = *case;
            let (sa, sb) = (
                SymMatrix2Trait::from_matrix_unchecked(m2(a)),
                SymMatrix2Trait::from_matrix_unchecked(m2(b)),
            );
            assert!((sa + sb).to_matrix() == sa.to_matrix() + sb.to_matrix());
            assert!((sa - sb).to_matrix() == sa.to_matrix() - sb.to_matrix());
            assert!((-sa).to_matrix() == -sa.to_matrix());
            let mut acc = sa;
            acc += sb;
            assert!(acc == sa + sb);
            let mut acc = sa;
            acc -= sb;
            assert!(acc == sa - sb);
        }
        assert!(s2i((1, 2, 3)) + s2i((10, 20, 30)) == s2i((11, 22, 33)));
        assert!(s2i((1, 2, 3)) - s2i((10, 20, 30)) == s2i((-9, -18, -27)));
        assert!(-s2i((1, -2, 3)) == s2i((-1, 2, -3)));
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_add_overflow_panics() {
        let s = black_box(SymMatrix2Trait::from_diagonal_element(Real::<Fixed>::MAX));
        let _ = s + s;
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_neg_min_panics() {
        let _ = -black_box(SymMatrix2Trait::from_diagonal_element(Real::<Fixed>::MIN));
    }

    #[test]
    fn test_scale_matches_the_generic_scale() {
        let mut cases = oracle_matrix2::matrix2_scale_cases();
        while let Some(case) = cases.pop_front() {
            let (a, k, _, _) = *case;
            let s = SymMatrix2Trait::from_matrix_unchecked(m2(a));
            assert!(s.scale(fx(k)).to_matrix() == s.to_matrix().scale(fx(k)));
        }
        assert!(s2i((1, -2, 3)).scale(int(3)) == s2i((3, -6, 9)));
    }

    #[test]
    fn test_norm_matches_the_generic_norm() {
        let mut cases = oracle_matrix2::matrix2_transpose_cases();
        while let Some(case) = cases.pop_front() {
            let (a, _, _) = *case;
            let s = SymMatrix2Trait::from_matrix_unchecked(m2(a));
            assert!(s.norm() == s.to_matrix().norm());
        }
        // The off-diagonal square is counted twice: sqrt(1 + 4 + 4 + 16) = 5.
        assert!(s2i((1, 2, 4)).norm() == int(5));
    }

    // --- products ------------------------------------------------------------------------------

    #[test]
    fn test_mul_vec_matches_the_generic_product() {
        let mut cases = oracle_matrix2::matrix2_mul_vec_cases();
        while let Some(case) = cases.pop_front() {
            let (a, v, _, _) = *case;
            let s = SymMatrix2Trait::from_matrix_unchecked(m2(a));
            assert!(s.mul_vec(v2(v)) == s.to_matrix().mul_vec(v2(v)));
        }
        assert!(s2i((1, 2, 4)).mul_vec(v2i(1, 0)) == v2i(1, 2));
        assert!(SymMatrix2Trait::<Fixed>::identity().mul_vec(v2i(3, -5)) == v2i(3, -5));
    }

    #[test]
    fn test_mul_matrix_matches_the_generic_product() {
        let mut cases = quadratic_cases();
        while let Some(case) = cases.pop_front() {
            let (a, b, _, _) = *case;
            let s = SymMatrix2Trait::from_matrix_unchecked(m2(a));
            assert!(s.mul_matrix(m2(b)) == s.to_matrix() * m2(b));
        }
        assert!(s2i((1, 2, 4)).mul_matrix(Matrix2Trait::identity()) == m2i([[1, 2], [2, 4]]));
    }

    #[test]
    fn test_quadform_matches_the_generic_product() {
        let mut cases = quadratic_cases();
        while let Some(case) = cases.pop_front() {
            let (a, b, _, _) = *case;
            let (r, d) = (m2(a), m2(b).diagonal());
            let expected = (r * Matrix2Trait::from_diagonal(d)) * r.transpose();
            // Only the UPPER triangle: the generic product is not exactly symmetric, its `m21`
            // rounds `r21 * d.x` before multiplying by `r11` where `m12` rounds `r11 * d.x`.
            assert!(
                SymMatrix2Trait::quadform(r, d) == SymMatrix2Trait::from_matrix_unchecked(expected),
            );
        }
        // r * diag(2, 3) * r^T with r = [[1, 0], [1, 1]] is [[2, 2], [2, 5]].
        assert!(SymMatrix2Trait::quadform(m2i([[1, 0], [1, 1]]), v2i(2, 3)) == s2i((2, 2, 5)));
    }

    #[test]
    fn test_quadform_sym_matches_the_generic_product() {
        let mut cases = quadratic_cases();
        while let Some(case) = cases.pop_front() {
            let (a, b, _, _) = *case;
            let (r, s) = (m2(a), SymMatrix2Trait::from_matrix_unchecked(m2(b)));
            let expected = (r * s.to_matrix()) * r.transpose();
            // Upper triangle only, see `test_quadform_matches_the_generic_product`.
            assert!(s.quadform_sym(r) == SymMatrix2Trait::from_matrix_unchecked(expected));
        }
        // The identity rotation leaves the form unchanged.
        let s = s2i((1, 2, 4));
        assert!(s.quadform_sym(Matrix2Trait::identity()) == s);
    }

    #[test]
    fn test_quadform_matches_quadform_sym_on_a_diagonal_form() {
        let mut cases = quadratic_cases();
        while let Some(case) = cases.pop_front() {
            let (a, b, _, _) = *case;
            let (r, d) = (m2(a), m2(b).diagonal());
            let diag = SymMatrix2Trait::from_diagonal(d);
            assert!(SymMatrix2Trait::quadform(r, d) == diag.quadform_sym(r));
        }
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_quadform_overflow_panics() {
        let r = black_box(Matrix2Trait::from_diagonal_element(int(65536)));
        let _ = SymMatrix2Trait::quadform(r, black_box(v2i(1, 1)));
    }

    // --- determinant and inverse ---------------------------------------------------------------

    #[test]
    fn test_determinant_exact() {
        assert!(s2i((1, 2, 4)).determinant() == int(0));
        assert!(s2i((2, 1, 3)).determinant() == int(5));
        assert!(SymMatrix2Trait::<Fixed>::identity().determinant() == int(1));
        assert!(SymMatrix2Trait::<Fixed>::zeros().determinant() == int(0));
    }

    #[test]
    fn test_determinant_matches_the_generic_determinant() {
        let mut cases = oracle_matrix2::matrix2_transpose_cases().slice(0, 12);
        while let Some(case) = cases.pop_front() {
            let (a, _, _) = *case;
            let s = SymMatrix2Trait::from_matrix_unchecked(m2(a));
            assert!(s.determinant() == s.to_matrix().determinant());
        }
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_determinant_overflow_panics() {
        black_box(SymMatrix2Trait::from_diagonal_element(int(65536))).determinant();
    }

    #[test]
    fn test_try_inverse_exact() {
        // [[2, 1], [1, 1]] has determinant 1 and inverse [[1, -1], [-1, 2]].
        let s = s2i((2, 1, 1));
        let inv = s.try_inverse().unwrap();
        assert!(inv == s2i((1, -1, 2)));
        assert!(s.mul_matrix(inv.to_matrix()) == Matrix2Trait::identity());
        let d = SymMatrix2Trait::from_diagonal(v2i(2, -4)).try_inverse().unwrap();
        assert!(d == s2((0x80000000, 0, -0x40000000)));
        assert!(
            SymMatrix2Trait::<Fixed>::identity()
                .try_inverse()
                .unwrap() == SymMatrix2Trait::<Fixed>::identity(),
        );
    }

    #[test]
    fn test_try_inverse_oracle() {
        // SPD matrices of the oracle's `udu` suite (exactly symmetric in raw units) against
        // upstream's inverse; the `small` distribution goes through the integer pre-scaling.
        let mut cases = oracle_sym_matrix::udu2_inverse_cases();
        while let Some(case) = cases.pop_front() {
            let (a, expected, tol) = *case;
            let s = SymMatrix2Trait::from_matrix_unchecked(m2(a));
            let err = max_ulp_diff2(s.try_inverse().unwrap().to_matrix(), m2(expected));
            assert!(err <= tol.into(), "inverse error {err} > {tol}");
        }
    }

    #[test]
    fn test_try_inverse_product_is_identity() {
        let mut cases = oracle_sym_matrix::udu2_inverse_cases();
        let mut worst = 0;
        while let Some(case) = cases.pop_front() {
            let (a, _, _) = *case;
            let s = SymMatrix2Trait::from_matrix_unchecked(m2(a));
            let residual = max_ulp_diff2(
                s.mul_matrix(s.try_inverse().unwrap().to_matrix()), Matrix2Trait::identity(),
            );
            worst = core::cmp::max(worst, residual);
        }
        // Worst residual over the oracle's 24 SPD matrices: 92 ulp (2.1e-8), on the worst
        // conditioned `small` case, whose inverse has components above 4e4.
        assert!(worst <= 92, "worst residual {worst}");
    }

    #[test]
    fn test_try_inverse_is_bit_identical_to_matrix2() {
        let mut cases = oracle_sym_matrix::udu2_inverse_cases();
        while let Some(case) = cases.pop_front() {
            let (a, _, _) = *case;
            let s = SymMatrix2Trait::from_matrix_unchecked(m2(a));
            let generic = s.to_matrix().try_inverse().unwrap();
            assert!(s.try_inverse().unwrap() == SymMatrix2Trait::from_matrix_unchecked(generic));
            assert!(max_ulp_diff2(s.try_inverse().unwrap().to_matrix(), generic) == 0);
        }
    }

    #[test]
    fn test_try_inverse_singular_is_none() {
        // The criterion is a computed determinant of exactly zero, like `Matrix2::try_inverse`.
        assert!(SymMatrix2Trait::<Fixed>::zeros().try_inverse().is_none());
        assert!(s2i((1, 2, 4)).try_inverse().is_none());
        assert!(SymMatrix2Trait::from_outer_self(v2i(3, -1)).try_inverse().is_none());
        let mut cases = oracle_matrix2::matrix2_transpose_cases().slice(0, 12);
        while let Some(case) = cases.pop_front() {
            let (a, _, _) = *case;
            let s = SymMatrix2Trait::from_matrix_unchecked(m2(a));
            assert!(s.try_inverse().is_some() == s.to_matrix().try_inverse().is_some());
        }
    }

    #[test]
    fn test_inverse_unchecked_matches_try_inverse() {
        let mut cases = oracle_sym_matrix::udu2_inverse_cases();
        while let Some(case) = cases.pop_front() {
            let (a, _, _) = *case;
            let s = SymMatrix2Trait::from_matrix_unchecked(m2(a));
            assert!(s.inverse_unchecked() == s.try_inverse().unwrap());
        }
        assert!(s2i((2, 1, 1)).inverse_unchecked() == s2i((1, -1, 2)));
    }

    #[test]
    #[should_panic(expected: 'simba: division by zero')]
    fn test_inverse_unchecked_singular_panics() {
        let _ = black_box(s2i((1, 2, 4))).inverse_unchecked();
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_try_inverse_tiny_norm_panics() {
        let _ = black_box(SymMatrix2Trait::from_diagonal_element(fx(1))).try_inverse();
    }

    // --- approximate equality ------------------------------------------------------------------

    #[test]
    fn test_abs_diff_eq() {
        let id = SymMatrix2Trait::<Fixed>::identity();
        let mut s = id;
        s.m12 = fx(3);
        s.m22 = fx(0x100000000 - 2);
        assert!(s.abs_diff_eq(id, 3) && !s.abs_diff_eq(id, 2));
        assert!(id.abs_diff_eq(id, 0));
        assert!(max_ulp_diff_s2(s, id) == 3);
    }

    // --- gas benchmarks
    // ----------------------------------------------------------------------------
    //
    // Inputs are shared with the `matrix2` benchmarks wherever the same operation exists there, so
    // that the structured kernels and the generic `Matrix2` path can be read side by side. `a` is
    // the `unit` SPD case of the oracle's `udu2` vectors.

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_new__baseline() {
        let _a = black_box(s2((1400100946, 252374302, 1577459800)));
        let e = black_box(s2((1400100946, 252374302, 1577459800)));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_new__struct() {
        let a = black_box(s2((1400100946, 252374302, 1577459800)));
        let e = black_box(s2((1400100946, 252374302, 1577459800)));
        assert!(SymMatrix2Trait::new(a.m11, a.m12, a.m22) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_zeros__baseline() {
        let e = black_box(s2((0, 0, 0)));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_zeros__const() {
        let e = black_box(s2((0, 0, 0)));
        assert!(SymMatrix2Trait::<Fixed>::zeros() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_identity__baseline() {
        let e = black_box(s2((4294967296, 0, 4294967296)));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_identity__const() {
        let e = black_box(s2((4294967296, 0, 4294967296)));
        assert!(SymMatrix2Trait::<Fixed>::identity() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_from_diagonal__baseline() {
        let _d = black_box(v2((-7543252641, 4885438966)));
        let e = black_box(s2((-7543252641, 0, 4885438966)));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_from_diagonal__struct() {
        let d = black_box(v2((-7543252641, 4885438966)));
        let e = black_box(s2((-7543252641, 0, 4885438966)));
        assert!(SymMatrix2Trait::from_diagonal(d) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_from_diagonal_element__baseline() {
        let _k = black_box(fx(-7516192768));
        let e = black_box(s2((-7516192768, 0, -7516192768)));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_from_diagonal_element__struct() {
        let k = black_box(fx(-7516192768));
        let e = black_box(s2((-7516192768, 0, -7516192768)));
        assert!(SymMatrix2Trait::from_diagonal_element(k) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_from_matrix_unchecked__baseline() {
        let _m = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let e = black_box(s2((5594399379, 2839048663, 5944454799)));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_from_matrix_unchecked__struct() {
        let m = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let e = black_box(s2((5594399379, 2839048663, 5944454799)));
        assert!(SymMatrix2Trait::from_matrix_unchecked(m) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_from_outer_self__baseline() {
        let _v = black_box(v2((-7543252641, 4885438966)));
        let e = black_box(s2((13248217386, -8580298253, 5557088621)));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_from_outer_self__structured() {
        let v = black_box(v2((-7543252641, 4885438966)));
        let e = black_box(s2((13248217386, -8580298253, 5557088621)));
        assert!(SymMatrix2Trait::from_outer_self(v) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_from_outer_self__generic() {
        let v = black_box(v2((-7543252641, 4885438966)));
        let e = black_box(s2((13248217386, -8580298253, 5557088621)));
        assert!(SymMatrix2Trait::from_matrix_unchecked(Matrix2Trait::from_outer(v, v)) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_to_matrix__baseline() {
        let _a = black_box(s2((1400100946, 252374302, 1577459800)));
        let e = black_box(m2([[1400100946, 252374302], [252374302, 1577459800]]));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_to_matrix__struct() {
        let a = black_box(s2((1400100946, 252374302, 1577459800)));
        let e = black_box(m2([[1400100946, 252374302], [252374302, 1577459800]]));
        assert!(a.to_matrix() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_diagonal__baseline() {
        let _a = black_box(s2((1400100946, 252374302, 1577459800)));
        let e = black_box(v2((1400100946, 1577459800)));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_diagonal__struct() {
        let a = black_box(s2((1400100946, 252374302, 1577459800)));
        let e = black_box(v2((1400100946, 1577459800)));
        assert!(a.diagonal() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_trace__baseline() {
        let _a = black_box(s2((1400100946, 252374302, 1577459800)));
        let e = black_box(fx(2977560746));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_trace__sum() {
        let a = black_box(s2((1400100946, 252374302, 1577459800)));
        let e = black_box(fx(2977560746));
        assert!(a.trace() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_add_diagonal__baseline() {
        let _a = black_box(s2((1400100946, 252374302, 1577459800)));
        let _k = black_box(fx(-7516192768));
        let e = black_box(s2((-6116091822, 252374302, -5938732968)));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_add_diagonal__sum() {
        let a = black_box(s2((1400100946, 252374302, 1577459800)));
        let k = black_box(fx(-7516192768));
        let e = black_box(s2((-6116091822, 252374302, -5938732968)));
        assert!(a.add_diagonal(k) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_add__baseline() {
        let _a = black_box(s2((1400100946, 252374302, 1577459800)));
        let _b = black_box(s2((3493038984, -3134687279, -8272937965)));
        let e = black_box(s2((4893139930, -2882312977, -6695478165)));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_add__operator() {
        let a = black_box(s2((1400100946, 252374302, 1577459800)));
        let b = black_box(s2((3493038984, -3134687279, -8272937965)));
        let e = black_box(s2((4893139930, -2882312977, -6695478165)));
        assert!(a + b == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_add__assign() {
        let a = black_box(s2((1400100946, 252374302, 1577459800)));
        let b = black_box(s2((3493038984, -3134687279, -8272937965)));
        let e = black_box(s2((4893139930, -2882312977, -6695478165)));
        let mut r = a;
        r += b;
        assert!(r == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_add__generic() {
        let a = black_box(s2((1400100946, 252374302, 1577459800)));
        let b = black_box(s2((3493038984, -3134687279, -8272937965)));
        let e = black_box(s2((4893139930, -2882312977, -6695478165)));
        assert!(SymMatrix2Trait::from_matrix_unchecked(a.to_matrix() + b.to_matrix()) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_sub__baseline() {
        let _a = black_box(s2((1400100946, 252374302, 1577459800)));
        let _b = black_box(s2((3493038984, -3134687279, -8272937965)));
        let e = black_box(s2((-2092938038, 3387061581, 9850397765)));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_sub__operator() {
        let a = black_box(s2((1400100946, 252374302, 1577459800)));
        let b = black_box(s2((3493038984, -3134687279, -8272937965)));
        let e = black_box(s2((-2092938038, 3387061581, 9850397765)));
        assert!(a - b == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_sub__assign() {
        let a = black_box(s2((1400100946, 252374302, 1577459800)));
        let b = black_box(s2((3493038984, -3134687279, -8272937965)));
        let e = black_box(s2((-2092938038, 3387061581, 9850397765)));
        let mut r = a;
        r -= b;
        assert!(r == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_neg__baseline() {
        let _a = black_box(s2((1400100946, 252374302, 1577459800)));
        let e = black_box(s2((-1400100946, -252374302, -1577459800)));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_neg__operator() {
        let a = black_box(s2((1400100946, 252374302, 1577459800)));
        let e = black_box(s2((-1400100946, -252374302, -1577459800)));
        assert!(-a == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_scale__baseline() {
        let _a = black_box(s2((1400100946, 252374302, 1577459800)));
        let _k = black_box(fx(-7516192768));
        let e = black_box(s2((-2450176656, -441655029, -2760554650)));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_scale__structured() {
        let a = black_box(s2((1400100946, 252374302, 1577459800)));
        let k = black_box(fx(-7516192768));
        let e = black_box(s2((-2450176656, -441655029, -2760554650)));
        assert!(a.scale(k) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_scale__generic() {
        let a = black_box(s2((1400100946, 252374302, 1577459800)));
        let k = black_box(fx(-7516192768));
        let e = black_box(s2((-2450176656, -441655029, -2760554650)));
        assert!(SymMatrix2Trait::from_matrix_unchecked(a.to_matrix().scale(k)) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_mul_vec__baseline() {
        let _a = black_box(s2((1400100946, 252374302, 1577459800)));
        let _v = black_box(v2((-7543252641, 4885438966)));
        let e = black_box(v2((-2171927111, 1351083734)));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_mul_vec__structured() {
        let a = black_box(s2((1400100946, 252374302, 1577459800)));
        let v = black_box(v2((-7543252641, 4885438966)));
        let e = black_box(v2((-2171927111, 1351083734)));
        assert!(a.mul_vec(v) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_mul_vec__generic() {
        let a = black_box(s2((1400100946, 252374302, 1577459800)));
        let v = black_box(v2((-7543252641, 4885438966)));
        let e = black_box(v2((-2171927111, 1351083734)));
        assert!(a.to_matrix().mul_vec(v) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_mul_matrix__baseline() {
        let _a = black_box(s2((1400100946, 252374302, 1577459800)));
        let _b = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let e = black_box(m2([[1386267709, 1274790230], [-2405419346, 2350109024]]));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_mul_matrix__structured() {
        let a = black_box(s2((1400100946, 252374302, 1577459800)));
        let b = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let e = black_box(m2([[1386267709, 1274790230], [-2405419346, 2350109024]]));
        assert!(a.mul_matrix(b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_mul_matrix__generic() {
        let a = black_box(s2((1400100946, 252374302, 1577459800)));
        let b = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let e = black_box(m2([[1386267709, 1274790230], [-2405419346, 2350109024]]));
        assert!(a.to_matrix() * b == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_quadform__baseline() {
        let _r = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let _d = black_box(v2((-7543252641, 4885438966)));
        let e = black_box(s2((-10663446698, 21499658486, -13302844783)));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_quadform__structured() {
        let r = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let d = black_box(v2((-7543252641, 4885438966)));
        let e = black_box(s2((-10663446698, 21499658486, -13302844783)));
        assert!(SymMatrix2Trait::quadform(r, d) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_quadform__generic() {
        let r = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let d = black_box(v2((-7543252641, 4885438966)));
        let e = black_box(s2((-10663446698, 21499658486, -13302844783)));
        let m = (r * Matrix2Trait::from_diagonal(d)) * r.transpose();
        assert!(SymMatrix2Trait::from_matrix_unchecked(m) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_quadform_sym__baseline() {
        let _a = black_box(s2((1400100946, 252374302, 1577459800)));
        let _r = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let e = black_box(s2((3499307493, -1551924967, 6017098708)));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_quadform_sym__structured() {
        let a = black_box(s2((1400100946, 252374302, 1577459800)));
        let r = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let e = black_box(s2((3499307493, -1551924967, 6017098708)));
        assert!(a.quadform_sym(r) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_quadform_sym__generic() {
        let a = black_box(s2((1400100946, 252374302, 1577459800)));
        let r = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let e = black_box(s2((3499307493, -1551924967, 6017098708)));
        assert!(SymMatrix2Trait::from_matrix_unchecked((r * a.to_matrix()) * r.transpose()) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_norm__baseline() {
        let _a = black_box(s2((1400100946, 252374302, 1577459800)));
        let e = black_box(fx(2139169852));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_norm__fused() {
        let a = black_box(s2((1400100946, 252374302, 1577459800)));
        let e = black_box(fx(2139169852));
        assert!(a.norm() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_determinant__baseline() {
        let _a = black_box(s2((1400100946, 252374302, 1577459800)));
        let e = black_box(fx(499400815));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_determinant__structured() {
        let a = black_box(s2((1400100946, 252374302, 1577459800)));
        let e = black_box(fx(499400815));
        assert!(a.determinant() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_determinant__generic() {
        let a = black_box(s2((1400100946, 252374302, 1577459800)));
        let e = black_box(fx(499400815));
        assert!(a.to_matrix().determinant() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_try_inverse__baseline() {
        let _a = black_box(s2((1400100946, 252374302, 1577459800)));
        let e = black_box(s2((13566534218, -2170479783, 12041205356)));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_try_inverse__structured() {
        let a = black_box(s2((1400100946, 252374302, 1577459800)));
        let e = black_box(s2((13566534218, -2170479783, 12041205356)));
        assert!(a.try_inverse().unwrap() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_try_inverse__structured_prescaled() {
        let a = black_box(s2((301259457, 33609003, 288701628)));
        let e = black_box(s2((62037790974, -7222087099, 64736286220)));
        assert!(a.try_inverse().unwrap() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_try_inverse__generic() {
        let a = black_box(s2((1400100946, 252374302, 1577459800)));
        let e = black_box(s2((13566534218, -2170479783, 12041205356)));
        let inv = a.to_matrix().try_inverse().unwrap();
        assert!(SymMatrix2Trait::from_matrix_unchecked(inv) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_inverse_unchecked__baseline() {
        let _a = black_box(s2((1400100946, 252374302, 1577459800)));
        let e = black_box(s2((13566534218, -2170479783, 12041205356)));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_inverse_unchecked__structured() {
        let a = black_box(s2((1400100946, 252374302, 1577459800)));
        let e = black_box(s2((13566534218, -2170479783, 12041205356)));
        assert!(a.inverse_unchecked() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_abs_diff_eq__baseline() {
        let _a = black_box(s2((1400100946, 252374302, 1577459800)));
        let _b = black_box(s2((1400100946, 252374302, 1577459800)));
        let e = black_box(true);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_abs_diff_eq__all_compared() {
        let a = black_box(s2((1400100946, 252374302, 1577459800)));
        let b = black_box(s2((1400100946, 252374302, 1577459800)));
        let e = black_box(true);
        assert!(a.abs_diff_eq(b, 2) == e);
    }
}
