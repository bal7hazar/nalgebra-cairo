//! `SymMatrix3`: a symmetric 3x3 matrix stored as its 6 independent components.
//!
//! No upstream nalgebra equivalent; the model is rapier / parry's `SdpMatrix3` (inertia tensors,
//! effective masses). Structured kernels (`quadform`, `quadform_sym`, `from_outer_self`,
//! `Matrix3::mul_transpose`, `try_inverse`) compute only the 6 independent components, never
//! materialise a diagonal matrix, and are bit-identical to the upper triangle of the generic
//! `Matrix3` expression they replace.

use core::ops::{AddAssign, SubAssign};
use simba::scalar::Real;
use super::matrix3::Matrix3;
use super::vector3::Vector3;

/// A symmetric 3x3 matrix `[[m11, m12, m13], [m12, m22, m23], [m13, m23, m33]]`.
#[derive(Copy, Drop, PartialEq, Serde, Default, Debug, Hash)]
pub struct SymMatrix3<T> {
    pub m11: T,
    pub m12: T,
    pub m13: T,
    pub m22: T,
    pub m23: T,
    pub m33: T,
}

/// Methods of `SymMatrix3<T>` for any `Real` scalar.
#[generate_trait]
pub impl SymMatrix3Impl<
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
> of SymMatrix3Trait<T> {
    // --- constructors --------------------------------------------------------------------------

    /// The symmetric matrix with the given upper triangle, row by row. parry: `SdpMatrix3::new`.
    #[inline(always)]
    fn new(m11: T, m12: T, m13: T, m22: T, m23: T, m33: T) -> SymMatrix3<T> {
        SymMatrix3 { m11, m12, m13, m22, m23, m33 }
    }

    /// The zero matrix. parry: `SdpMatrix3::zero`.
    #[inline(always)]
    fn zeros() -> SymMatrix3<T> {
        SymMatrix3 {
            m11: R::ZERO, m12: R::ZERO, m13: R::ZERO, m22: R::ZERO, m23: R::ZERO, m33: R::ZERO,
        }
    }

    /// The identity matrix.
    #[inline(always)]
    fn identity() -> SymMatrix3<T> {
        SymMatrix3 {
            m11: R::ONE, m12: R::ZERO, m13: R::ZERO, m22: R::ONE, m23: R::ZERO, m33: R::ONE,
        }
    }

    /// The diagonal matrix `diag(d.x, d.y, d.z)` (principal inertia).
    #[inline(always)]
    fn from_diagonal(d: Vector3<T>) -> SymMatrix3<T> {
        SymMatrix3 { m11: d.x, m12: R::ZERO, m13: R::ZERO, m22: d.y, m23: R::ZERO, m33: d.z }
    }

    /// The matrix `e * I`. parry: `SdpMatrix3::diagonal`.
    #[inline(always)]
    fn from_diagonal_element(e: T) -> SymMatrix3<T> {
        SymMatrix3 { m11: e, m12: R::ZERO, m13: R::ZERO, m22: e, m23: R::ZERO, m33: e }
    }

    /// The UPPER triangle of `m`; the strictly lower triangle is ignored, the symmetry of `m` is
    /// not checked. parry: `SdpMatrix3::from_sdp_matrix`.
    #[inline(always)]
    fn from_matrix_unchecked(m: Matrix3<T>) -> SymMatrix3<T> {
        SymMatrix3 { m11: m.m11, m12: m.m12, m13: m.m13, m22: m.m22, m23: m.m23, m33: m.m33 }
    }

    /// The outer product `v * vᵀ`: 6 floored products. Panics on overflow.
    /// Upstream: `v * v.transpose()`.
    #[inline(always)]
    fn from_outer_self(v: Vector3<T>) -> SymMatrix3<T> {
        SymMatrix3 {
            m11: v.x * v.x,
            m12: v.x * v.y,
            m13: v.x * v.z,
            m22: v.y * v.y,
            m23: v.y * v.z,
            m33: v.z * v.z,
        }
    }

    /// `r * diag(d) * rᵀ` (world-space inertia from the principal inertia `d` and the rotation
    /// `r`) without materialising the diagonal matrix: 9 products `r_ik * d_k` then 6
    /// `sum_prod3` (two roundings per component, like `(r * diag(d)) * rᵀ`, to which it is
    /// bit-identical). Panics on overflow. Upstream: `quadform_tr` with a diagonal `mid`.
    fn quadform(r: Matrix3<T>, d: Vector3<T>) -> SymMatrix3<T> {
        let (t11, t12, t13) = (r.m11 * d.x, r.m12 * d.y, r.m13 * d.z);
        let (t21, t22, t23) = (r.m21 * d.x, r.m22 * d.y, r.m23 * d.z);
        let (t31, t32, t33) = (r.m31 * d.x, r.m32 * d.y, r.m33 * d.z);
        SymMatrix3 {
            m11: R::sum_prod3(t11, r.m11, t12, r.m12, t13, r.m13),
            m12: R::sum_prod3(t11, r.m21, t12, r.m22, t13, r.m23),
            m13: R::sum_prod3(t11, r.m31, t12, r.m32, t13, r.m33),
            m22: R::sum_prod3(t21, r.m21, t22, r.m22, t23, r.m23),
            m23: R::sum_prod3(t21, r.m31, t22, r.m32, t23, r.m33),
            m33: R::sum_prod3(t31, r.m31, t32, r.m32, t33, r.m33),
        }
    }

    // --- accessors and conversions -------------------------------------------------------------

    /// The full matrix. parry: `SdpMatrix3::into_matrix`.
    #[inline(always)]
    fn to_matrix(self: SymMatrix3<T>) -> Matrix3<T> {
        Matrix3 {
            m11: self.m11,
            m21: self.m12,
            m31: self.m13,
            m12: self.m12,
            m22: self.m22,
            m32: self.m23,
            m13: self.m13,
            m23: self.m23,
            m33: self.m33,
        }
    }

    /// The diagonal `(m11, m22, m33)`.
    #[inline(always)]
    fn diagonal(self: SymMatrix3<T>) -> Vector3<T> {
        Vector3 { x: self.m11, y: self.m22, z: self.m33 }
    }

    /// `m11 + m22 + m33`. Exact; panics on overflow.
    #[inline(always)]
    fn trace(self: SymMatrix3<T>) -> T {
        self.m11 + self.m22 + self.m33
    }

    /// `self + e * I`. Exact; panics on overflow. parry: `SdpMatrix3::add_diagonal`.
    #[inline(always)]
    fn add_diagonal(self: SymMatrix3<T>, e: T) -> SymMatrix3<T> {
        SymMatrix3 {
            m11: self.m11 + e,
            m12: self.m12,
            m13: self.m13,
            m22: self.m22 + e,
            m23: self.m23,
            m33: self.m33 + e,
        }
    }

    // --- products ------------------------------------------------------------------------------

    /// `self * k`: 6 floored products. Panics on overflow. parry: `SdpMatrix3 * Real`.
    #[inline(always)]
    fn scale(self: SymMatrix3<T>, k: T) -> SymMatrix3<T> {
        SymMatrix3 {
            m11: self.m11 * k,
            m12: self.m12 * k,
            m13: self.m13 * k,
            m22: self.m22 * k,
            m23: self.m23 * k,
            m33: self.m33 * k,
        }
    }

    /// `self * v`: one `sum_prod3` per component. Panics on overflow. parry: `mul_vec`.
    #[inline(always)]
    fn mul_vec(self: SymMatrix3<T>, v: Vector3<T>) -> Vector3<T> {
        Vector3 {
            x: R::sum_prod3(self.m11, v.x, self.m12, v.y, self.m13, v.z),
            y: R::sum_prod3(self.m12, v.x, self.m22, v.y, self.m23, v.z),
            z: R::sum_prod3(self.m13, v.x, self.m23, v.y, self.m33, v.z),
        }
    }

    /// `self * m` (a general matrix): 9 `sum_prod3`. Panics on overflow. parry: `mul_mat`.
    fn mul_matrix(self: SymMatrix3<T>, m: Matrix3<T>) -> Matrix3<T> {
        Matrix3 {
            m11: R::sum_prod3(self.m11, m.m11, self.m12, m.m21, self.m13, m.m31),
            m21: R::sum_prod3(self.m12, m.m11, self.m22, m.m21, self.m23, m.m31),
            m31: R::sum_prod3(self.m13, m.m11, self.m23, m.m21, self.m33, m.m31),
            m12: R::sum_prod3(self.m11, m.m12, self.m12, m.m22, self.m13, m.m32),
            m22: R::sum_prod3(self.m12, m.m12, self.m22, m.m22, self.m23, m.m32),
            m32: R::sum_prod3(self.m13, m.m12, self.m23, m.m22, self.m33, m.m32),
            m13: R::sum_prod3(self.m11, m.m13, self.m12, m.m23, self.m13, m.m33),
            m23: R::sum_prod3(self.m12, m.m13, self.m22, m.m23, self.m23, m.m33),
            m33: R::sum_prod3(self.m13, m.m13, self.m23, m.m23, self.m33, m.m33),
        }
    }

    /// `r * self * rᵀ` (change of frame of an inertia tensor): 15 `sum_prod3` instead of 18,
    /// bit-identical to the upper triangle of `(r * self.to_matrix()) * r.transpose()` (two
    /// roundings per component). Panics on overflow. Upstream: `quadform_tr`; parry's
    /// `quadform(m)` is `quadform_sym(m.transpose())`.
    fn quadform_sym(self: SymMatrix3<T>, r: Matrix3<T>) -> SymMatrix3<T> {
        let t11 = R::sum_prod3(r.m11, self.m11, r.m12, self.m12, r.m13, self.m13);
        let t12 = R::sum_prod3(r.m11, self.m12, r.m12, self.m22, r.m13, self.m23);
        let t13 = R::sum_prod3(r.m11, self.m13, r.m12, self.m23, r.m13, self.m33);
        let t21 = R::sum_prod3(r.m21, self.m11, r.m22, self.m12, r.m23, self.m13);
        let t22 = R::sum_prod3(r.m21, self.m12, r.m22, self.m22, r.m23, self.m23);
        let t23 = R::sum_prod3(r.m21, self.m13, r.m22, self.m23, r.m23, self.m33);
        let t31 = R::sum_prod3(r.m31, self.m11, r.m32, self.m12, r.m33, self.m13);
        let t32 = R::sum_prod3(r.m31, self.m12, r.m32, self.m22, r.m33, self.m23);
        let t33 = R::sum_prod3(r.m31, self.m13, r.m32, self.m23, r.m33, self.m33);
        SymMatrix3 {
            m11: R::sum_prod3(t11, r.m11, t12, r.m12, t13, r.m13),
            m12: R::sum_prod3(t11, r.m21, t12, r.m22, t13, r.m23),
            m13: R::sum_prod3(t11, r.m31, t12, r.m32, t13, r.m33),
            m22: R::sum_prod3(t21, r.m21, t22, r.m22, t23, r.m23),
            m23: R::sum_prod3(t21, r.m31, t22, r.m32, t23, r.m33),
            m33: R::sum_prod3(t31, r.m31, t32, r.m32, t33, r.m33),
        }
    }

    // --- norms ---------------------------------------------------------------------------------

    /// Frobenius norm of the full matrix (off-diagonal squares counted twice): square root of
    /// the exact, unscaled sum of squares, one rounding. Bit-identical to
    /// `self.to_matrix().norm()`.
    #[inline(always)]
    fn norm(self: SymMatrix3<T>) -> T {
        let w = R::wide_add_prod(R::wide_zero(), self.m11, self.m11);
        let w = R::wide_add_prod(w, self.m12, self.m12);
        let w = R::wide_add_prod(w, self.m12, self.m12);
        let w = R::wide_add_prod(w, self.m13, self.m13);
        let w = R::wide_add_prod(w, self.m13, self.m13);
        let w = R::wide_add_prod(w, self.m22, self.m22);
        let w = R::wide_add_prod(w, self.m23, self.m23);
        let w = R::wide_add_prod(w, self.m23, self.m23);
        R::wide_sqrt(R::wide_add_prod(w, self.m33, self.m33))
    }

    // --- determinant and inverse ---------------------------------------------------------------

    /// The determinant, by cofactor expansion along the first row (3 `diff_prod` then one
    /// `sum_prod3`): bit-identical to `self.to_matrix().determinant()`. Panics on overflow.
    fn determinant(self: SymMatrix3<T>) -> T {
        R::sum_prod3(
            self.m11,
            R::diff_prod(self.m22, self.m33, self.m23, self.m23),
            self.m12,
            R::diff_prod(self.m13, self.m23, self.m12, self.m33),
            self.m13,
            R::diff_prod(self.m12, self.m23, self.m13, self.m22),
        )
    }

    /// The adjugate (cofactor matrix, symmetric): `self * adjugate = determinant * I`. 6
    /// `diff_prod`, one rounding per component. Panics on overflow.
    #[inline(always)]
    fn adjugate(self: SymMatrix3<T>) -> SymMatrix3<T> {
        SymMatrix3 {
            m11: R::diff_prod(self.m22, self.m33, self.m23, self.m23),
            m12: R::diff_prod(self.m13, self.m23, self.m12, self.m33),
            m13: R::diff_prod(self.m12, self.m23, self.m13, self.m22),
            m22: R::diff_prod(self.m11, self.m33, self.m13, self.m13),
            m23: R::diff_prod(self.m13, self.m12, self.m11, self.m23),
            m33: R::diff_prod(self.m11, self.m22, self.m12, self.m12),
        }
    }

    /// The inverse, or `None` when the computed determinant is exactly zero. Same algorithm,
    /// criterion, rounding and panics as `Matrix3::try_inverse`, to which it is bit-identical
    /// on the upper triangle, with 6 cofactors and 6 divisions instead of 9.
    fn try_inverse(self: SymMatrix3<T>) -> Option<SymMatrix3<T>> {
        let adj = Self::adjugate(self);
        let det = R::sum_prod3(self.m11, adj.m11, self.m12, adj.m12, self.m13, adj.m13);
        if det < R::HALF && det > -R::HALF {
            let f = Self::norm(self);
            if f == R::ZERO {
                return None;
            }
            let k = R::floor(R::div(R::TWO, f));
            if k >= R::TWO {
                let b = Self::scale(self, k);
                let adj_b = Self::adjugate(b);
                let det_b = R::sum_prod3(b.m11, adj_b.m11, b.m12, adj_b.m12, b.m13, adj_b.m13);
                if det_b == R::ZERO {
                    return None;
                }
                return Some(Self::scale(adj_b, R::div(k, det_b)));
            }
            if det == R::ZERO {
                return None;
            }
        }
        Some(
            SymMatrix3 {
                m11: R::div(adj.m11, det),
                m12: R::div(adj.m12, det),
                m13: R::div(adj.m13, det),
                m22: R::div(adj.m22, det),
                m23: R::div(adj.m23, det),
                m33: R::div(adj.m33, det),
            },
        )
    }

    /// The inverse without the singularity checks: same result as `try_inverse` on an invertible
    /// matrix; a singular matrix panics with the scalar's division-by-zero error instead of
    /// returning `None`. parry: `SdpMatrix3::inverse_unchecked`.
    fn inverse_unchecked(self: SymMatrix3<T>) -> SymMatrix3<T> {
        let adj = Self::adjugate(self);
        let det = R::sum_prod3(self.m11, adj.m11, self.m12, adj.m12, self.m13, adj.m13);
        if det < R::HALF && det > -R::HALF {
            let k = R::floor(R::div(R::TWO, Self::norm(self)));
            if k >= R::TWO {
                let b = Self::scale(self, k);
                let adj_b = Self::adjugate(b);
                let det_b = R::sum_prod3(b.m11, adj_b.m11, b.m12, adj_b.m12, b.m13, adj_b.m13);
                return Self::scale(adj_b, R::div(k, det_b));
            }
        }
        SymMatrix3 {
            m11: R::div(adj.m11, det),
            m12: R::div(adj.m12, det),
            m13: R::div(adj.m13, det),
            m22: R::div(adj.m22, det),
            m23: R::div(adj.m23, det),
            m33: R::div(adj.m33, det),
        }
    }

    // --- approximate equality ------------------------------------------------------------------

    /// Whether every component of `self` is within `ulps` smallest units of `other`'s.
    fn abs_diff_eq(self: SymMatrix3<T>, other: SymMatrix3<T>, ulps: u64) -> bool {
        R::abs_diff_eq(self.m11, other.m11, ulps)
            && R::abs_diff_eq(self.m12, other.m12, ulps)
            && R::abs_diff_eq(self.m13, other.m13, ulps)
            && R::abs_diff_eq(self.m22, other.m22, ulps)
            && R::abs_diff_eq(self.m23, other.m23, ulps)
            && R::abs_diff_eq(self.m33, other.m33, ulps)
    }
}

// --- operators -----------------------------------------------------------------------------------

/// `a + b`, component-wise. Exact; panics on overflow.
pub impl SymMatrix3Add<T, +Add<T>, +Copy<T>, +Drop<T>> of Add<SymMatrix3<T>> {
    #[inline(always)]
    fn add(lhs: SymMatrix3<T>, rhs: SymMatrix3<T>) -> SymMatrix3<T> {
        SymMatrix3 {
            m11: lhs.m11 + rhs.m11,
            m12: lhs.m12 + rhs.m12,
            m13: lhs.m13 + rhs.m13,
            m22: lhs.m22 + rhs.m22,
            m23: lhs.m23 + rhs.m23,
            m33: lhs.m33 + rhs.m33,
        }
    }
}

/// `a - b`, component-wise. Exact; panics on overflow.
pub impl SymMatrix3Sub<T, +Sub<T>, +Copy<T>, +Drop<T>> of Sub<SymMatrix3<T>> {
    #[inline(always)]
    fn sub(lhs: SymMatrix3<T>, rhs: SymMatrix3<T>) -> SymMatrix3<T> {
        SymMatrix3 {
            m11: lhs.m11 - rhs.m11,
            m12: lhs.m12 - rhs.m12,
            m13: lhs.m13 - rhs.m13,
            m22: lhs.m22 - rhs.m22,
            m23: lhs.m23 - rhs.m23,
            m33: lhs.m33 - rhs.m33,
        }
    }
}

/// `-a`, component-wise. Exact; panics on the scalar's `MIN`.
pub impl SymMatrix3Neg<T, +Neg<T>, +Copy<T>, +Drop<T>> of Neg<SymMatrix3<T>> {
    #[inline(always)]
    fn neg(a: SymMatrix3<T>) -> SymMatrix3<T> {
        SymMatrix3 { m11: -a.m11, m12: -a.m12, m13: -a.m13, m22: -a.m22, m23: -a.m23, m33: -a.m33 }
    }
}

/// `a += b`.
pub impl SymMatrix3AddAssign<
    T, +Add<T>, +Copy<T>, +Drop<T>,
> of AddAssign<SymMatrix3<T>, SymMatrix3<T>> {
    #[inline(always)]
    fn add_assign(ref self: SymMatrix3<T>, rhs: SymMatrix3<T>) {
        self = self + rhs;
    }
}

/// `a -= b`.
pub impl SymMatrix3SubAssign<
    T, +Sub<T>, +Copy<T>, +Drop<T>,
> of SubAssign<SymMatrix3<T>, SymMatrix3<T>> {
    #[inline(always)]
    fn sub_assign(ref self: SymMatrix3<T>, rhs: SymMatrix3<T>) {
        self = self - rhs;
    }
}

#[cfg(test)]
mod tests {
    use fixed::Fixed;
    use nalgebra_testing::black_box;
    use simba::scalar::Real;
    use crate::base::matrix3::Matrix3Trait;
    use crate::base::matrix_test_utils::{
        fx, int, m3, m3i, max_ulp_diff3, max_ulp_diff_s3, s3, s3i, v3i, v3t,
    };
    use crate::base::{oracle_matrix3, oracle_sym_matrix};
    use super::SymMatrix3Trait;

    /// The oracle's `matrix3` cases without the `large` distribution (indices [12..16)), whose
    /// squares do not fit a quadratic form.
    fn quadratic_cases() -> Span<([[i64; 3]; 3], [[i64; 3]; 3], [[i64; 3]; 3], u64)> {
        oracle_matrix3::matrix3_mul_cases().slice(0, 12)
    }

    // --- constructors and accessors ----------------------------------------------------------

    #[test]
    fn test_new_zeros_identity() {
        let s = SymMatrix3Trait::new(int(1), int(2), int(3), int(4), int(5), int(6));
        assert!(s == s3i((1, 2, 3, 4, 5, 6)));
        assert!(s.to_matrix() == m3i([[1, 2, 3], [2, 4, 5], [3, 5, 6]]));
        assert!(SymMatrix3Trait::<Fixed>::zeros() == s3i((0, 0, 0, 0, 0, 0)));
        assert!(SymMatrix3Trait::<Fixed>::zeros() == Default::default());
        assert!(SymMatrix3Trait::<Fixed>::identity() == s3i((1, 0, 0, 1, 0, 1)));
        assert!(SymMatrix3Trait::<Fixed>::identity().to_matrix() == Matrix3Trait::identity());
    }

    #[test]
    fn test_serde_is_the_upper_triangle_row_by_row() {
        let mut out = array![];
        s3i((1, 2, 3, 4, 5, 6)).serialize(ref out);
        let one: felt252 = 0x100000000;
        assert!(out == array![1 * one, 2 * one, 3 * one, 4 * one, 5 * one, 6 * one]);
    }

    #[test]
    fn test_from_diagonal() {
        assert!(SymMatrix3Trait::from_diagonal(v3i(2, -3, 5)) == s3i((2, 0, 0, -3, 0, 5)));
        assert!(SymMatrix3Trait::from_diagonal_element(int(7)) == s3i((7, 0, 0, 7, 0, 7)));
        assert!(
            SymMatrix3Trait::from_diagonal(v3i(2, -3, 5))
                .to_matrix() == Matrix3Trait::from_diagonal(v3i(2, -3, 5)),
        );
    }

    #[test]
    fn test_from_matrix_unchecked_reads_the_upper_triangle() {
        // The strictly lower triangle is ignored: an asymmetric matrix is silently symmetrised.
        let m = m3i([[1, 2, 3], [97, 4, 5], [98, 99, 6]]);
        assert!(SymMatrix3Trait::from_matrix_unchecked(m) == s3i((1, 2, 3, 4, 5, 6)));
    }

    #[test]
    fn test_to_matrix_is_a_round_trip() {
        let mut cases = oracle_matrix3::matrix3_transpose_cases();
        while let Some(case) = cases.pop_front() {
            let (a, _, _) = *case;
            let s = SymMatrix3Trait::from_matrix_unchecked(m3(a));
            assert!(SymMatrix3Trait::from_matrix_unchecked(s.to_matrix()) == s);
            assert!(s.to_matrix().transpose() == s.to_matrix());
        }
    }

    #[test]
    fn test_from_outer_self_matches_the_generic_outer_product() {
        let mut cases = oracle_matrix3::matrix3_outer_cases();
        while let Some(case) = cases.pop_front() {
            let (u, _, _, _) = *case;
            let v = v3t(u);
            assert!(
                SymMatrix3Trait::from_outer_self(v).to_matrix() == Matrix3Trait::from_outer(v, v),
            );
        }
        assert!(SymMatrix3Trait::from_outer_self(v3i(2, -3, 1)) == s3i((4, -6, 2, 9, -3, 1)));
    }

    #[test]
    fn test_mul_transpose_matches_the_generic_product() {
        let mut cases = quadratic_cases();
        while let Some(case) = cases.pop_front() {
            let (a, b, _, _) = *case;
            assert!(m3(a).mul_transpose().to_matrix() == m3(a) * m3(a).transpose());
            assert!(m3(b).mul_transpose().to_matrix() == m3(b) * m3(b).transpose());
        }
    }

    #[test]
    fn test_diagonal_trace_add_diagonal() {
        let s = s3i((1, 2, 3, 4, 5, 6));
        assert!(s.diagonal() == v3i(1, 4, 6));
        assert!(s.trace() == int(11));
        assert!(s.add_diagonal(int(10)) == s3i((11, 2, 3, 14, 5, 16)));
    }

    // --- exact operations ----------------------------------------------------------------------

    #[test]
    fn test_add_sub_neg_match_the_generic_operators() {
        let mut cases = oracle_matrix3::matrix3_add_cases();
        while let Some(case) = cases.pop_front() {
            let (a, b, _, _) = *case;
            let (sa, sb) = (
                SymMatrix3Trait::from_matrix_unchecked(m3(a)),
                SymMatrix3Trait::from_matrix_unchecked(m3(b)),
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
        assert!(s3i((1, 2, 3, 4, 5, 6)) + s3i((1, 1, 1, 1, 1, 1)) == s3i((2, 3, 4, 5, 6, 7)));
        assert!(s3i((1, 2, 3, 4, 5, 6)) - s3i((1, 1, 1, 1, 1, 1)) == s3i((0, 1, 2, 3, 4, 5)));
        assert!(-s3i((1, -2, 3, -4, 5, -6)) == s3i((-1, 2, -3, 4, -5, 6)));
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_add_overflow_panics() {
        let s = black_box(SymMatrix3Trait::from_diagonal_element(Real::<Fixed>::MAX));
        let _ = s + s;
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_neg_min_panics() {
        let _ = -black_box(SymMatrix3Trait::from_diagonal_element(Real::<Fixed>::MIN));
    }

    #[test]
    fn test_scale_matches_the_generic_scale() {
        let mut cases = oracle_matrix3::matrix3_scale_cases();
        while let Some(case) = cases.pop_front() {
            let (a, k, _, _) = *case;
            let s = SymMatrix3Trait::from_matrix_unchecked(m3(a));
            assert!(s.scale(fx(k)).to_matrix() == s.to_matrix().scale(fx(k)));
        }
        assert!(s3i((1, -2, 3, 4, 5, 6)).scale(int(3)) == s3i((3, -6, 9, 12, 15, 18)));
    }

    #[test]
    fn test_norm_matches_the_generic_norm() {
        let mut cases = oracle_matrix3::matrix3_transpose_cases();
        while let Some(case) = cases.pop_front() {
            let (a, _, _) = *case;
            let s = SymMatrix3Trait::from_matrix_unchecked(m3(a));
            assert!(s.norm() == s.to_matrix().norm());
        }
        // Off-diagonal squares counted twice: sqrt(4 + 2*4 + 2*16 + 9 + 2*36 + 25) = sqrt(150).
        assert!(SymMatrix3Trait::from_diagonal(v3i(3, 4, 0)).norm() == int(5));
    }

    // --- products ------------------------------------------------------------------------------

    #[test]
    fn test_mul_vec_matches_the_generic_product() {
        let mut cases = oracle_matrix3::matrix3_mul_vec_cases();
        while let Some(case) = cases.pop_front() {
            let (a, v, _, _) = *case;
            let s = SymMatrix3Trait::from_matrix_unchecked(m3(a));
            assert!(s.mul_vec(v3t(v)) == s.to_matrix().mul_vec(v3t(v)));
        }
        assert!(s3i((1, 2, 3, 4, 5, 6)).mul_vec(v3i(1, 0, 0)) == v3i(1, 2, 3));
        assert!(SymMatrix3Trait::<Fixed>::identity().mul_vec(v3i(3, -5, 7)) == v3i(3, -5, 7));
    }

    #[test]
    fn test_mul_matrix_matches_the_generic_product() {
        let mut cases = quadratic_cases();
        while let Some(case) = cases.pop_front() {
            let (a, b, _, _) = *case;
            let s = SymMatrix3Trait::from_matrix_unchecked(m3(a));
            assert!(s.mul_matrix(m3(b)) == s.to_matrix() * m3(b));
        }
        let s = s3i((1, 2, 3, 4, 5, 6));
        assert!(s.mul_matrix(Matrix3Trait::identity()) == s.to_matrix());
    }

    #[test]
    fn test_quadform_matches_the_generic_product() {
        let mut cases = quadratic_cases();
        while let Some(case) = cases.pop_front() {
            let (a, b, _, _) = *case;
            let (r, d) = (m3(a), m3(b).diagonal());
            let expected = (r * Matrix3Trait::from_diagonal(d)) * r.transpose();
            // Only the UPPER triangle: the generic product is not exactly symmetric, its `m21`
            // rounds `r21 * d.x` before multiplying by `r11` where `m12` rounds `r11 * d.x`.
            assert!(
                SymMatrix3Trait::quadform(r, d) == SymMatrix3Trait::from_matrix_unchecked(expected),
            );
        }
        // diag(2, 3, 4) in the frame that swaps x and y.
        let swap = m3i([[0, 1, 0], [1, 0, 0], [0, 0, 1]]);
        assert!(SymMatrix3Trait::quadform(swap, v3i(2, 3, 4)) == s3i((3, 0, 0, 2, 0, 4)));
    }

    #[test]
    fn test_quadform_sym_matches_the_generic_product() {
        let mut cases = quadratic_cases();
        while let Some(case) = cases.pop_front() {
            let (a, b, _, _) = *case;
            let (r, s) = (m3(a), SymMatrix3Trait::from_matrix_unchecked(m3(b)));
            let expected = (r * s.to_matrix()) * r.transpose();
            // Upper triangle only, see `test_quadform_matches_the_generic_product`.
            assert!(s.quadform_sym(r) == SymMatrix3Trait::from_matrix_unchecked(expected));
        }
        let s = s3i((1, 2, 3, 4, 5, 6));
        assert!(s.quadform_sym(Matrix3Trait::identity()) == s);
    }

    #[test]
    fn test_quadform_matches_quadform_sym_on_a_diagonal_form() {
        let mut cases = quadratic_cases();
        while let Some(case) = cases.pop_front() {
            let (a, b, _, _) = *case;
            let (r, d) = (m3(a), m3(b).diagonal());
            assert!(
                SymMatrix3Trait::quadform(r, d) == SymMatrix3Trait::from_diagonal(d)
                    .quadform_sym(r),
            );
        }
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_quadform_overflow_panics() {
        let r = black_box(Matrix3Trait::from_diagonal_element(int(65536)));
        let _ = SymMatrix3Trait::quadform(r, black_box(v3i(1, 1, 1)));
    }

    // --- determinant and inverse ---------------------------------------------------------------

    #[test]
    fn test_determinant_exact() {
        assert!(SymMatrix3Trait::<Fixed>::identity().determinant() == int(1));
        assert!(SymMatrix3Trait::<Fixed>::zeros().determinant() == int(0));
        assert!(SymMatrix3Trait::from_diagonal(v3i(2, -3, 5)).determinant() == int(-30));
        // A rank-1 outer product is singular.
        assert!(SymMatrix3Trait::from_outer_self(v3i(2, -3, 1)).determinant() == int(0));
        // [[2, 1, 0], [1, 2, 1], [0, 1, 2]]: 2*(4-1) - 1*(2-0) = 4.
        assert!(s3i((2, 1, 0, 2, 1, 2)).determinant() == int(4));
    }

    #[test]
    fn test_determinant_matches_the_generic_determinant() {
        let mut cases = oracle_matrix3::matrix3_transpose_cases().slice(0, 12);
        while let Some(case) = cases.pop_front() {
            let (a, _, _) = *case;
            let s = SymMatrix3Trait::from_matrix_unchecked(m3(a));
            assert!(s.determinant() == s.to_matrix().determinant());
        }
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_determinant_overflow_panics() {
        black_box(SymMatrix3Trait::from_diagonal_element(int(2048))).determinant();
    }

    #[test]
    fn test_adjugate_identity() {
        let s = s3i((2, 1, 0, 2, 1, 2));
        let det = s.determinant();
        assert!(det == int(4));
        assert!(s.mul_matrix(s.adjugate().to_matrix()) == Matrix3Trait::from_diagonal_element(det));
        let mut cases = oracle_matrix3::matrix3_transpose_cases().slice(4, 8);
        while let Some(case) = cases.pop_front() {
            let (a, _, _) = *case;
            let s = SymMatrix3Trait::from_matrix_unchecked(m3(a));
            assert!(s.adjugate().to_matrix() == s.to_matrix().adjugate());
        }
    }

    #[test]
    fn test_try_inverse_exact() {
        // diag(2, -4, 8) inverts component-wise, exactly.
        let d = SymMatrix3Trait::from_diagonal(v3i(2, -4, 8)).try_inverse().unwrap();
        assert!(d == s3((0x80000000, 0, 0, -0x40000000, 0, 0x20000000)));
        assert!(
            SymMatrix3Trait::<Fixed>::identity()
                .try_inverse()
                .unwrap() == SymMatrix3Trait::<Fixed>::identity(),
        );
        // [[2, 1, 0], [1, 2, 1], [0, 1, 2]] has determinant 4 and inverse
        // [[3, -2, 1], [-2, 4, -2], [1, -2, 3]] / 4.
        let s = s3i((2, 1, 0, 2, 1, 2));
        let q = 0x40000000; // 0.25
        assert!(s.try_inverse().unwrap() == s3((3 * q, -2 * q, q, 4 * q, -2 * q, 3 * q)));
    }

    #[test]
    fn test_try_inverse_oracle() {
        // SPD matrices of the oracle's `udu` suite (exactly symmetric in raw units) against
        // upstream's inverse; the `small` distribution goes through the integer pre-scaling.
        let mut cases = oracle_sym_matrix::udu3_inverse_cases();
        while let Some(case) = cases.pop_front() {
            let (a, expected, tol) = *case;
            let s = SymMatrix3Trait::from_matrix_unchecked(m3(a));
            let err = max_ulp_diff3(s.try_inverse().unwrap().to_matrix(), m3(expected));
            assert!(err <= tol.into(), "inverse error {err} > {tol}");
        }
    }

    #[test]
    fn test_try_inverse_product_is_identity() {
        let mut cases = oracle_sym_matrix::udu3_inverse_cases();
        let mut worst = 0;
        while let Some(case) = cases.pop_front() {
            let (a, _, _) = *case;
            let s = SymMatrix3Trait::from_matrix_unchecked(m3(a));
            let residual = max_ulp_diff3(
                s.mul_matrix(s.try_inverse().unwrap().to_matrix()), Matrix3Trait::identity(),
            );
            worst = core::cmp::max(worst, residual);
        }
        // Worst residual over the oracle's 24 SPD matrices: 40 ulp (9.3e-9).
        assert!(worst <= 40, "worst residual {worst}");
    }

    #[test]
    fn test_try_inverse_is_bit_identical_to_matrix3() {
        let mut cases = oracle_sym_matrix::udu3_inverse_cases();
        while let Some(case) = cases.pop_front() {
            let (a, _, _) = *case;
            let s = SymMatrix3Trait::from_matrix_unchecked(m3(a));
            let generic = s.to_matrix().try_inverse().unwrap();
            assert!(s.try_inverse().unwrap() == SymMatrix3Trait::from_matrix_unchecked(generic));
            assert!(max_ulp_diff3(s.try_inverse().unwrap().to_matrix(), generic) == 0);
        }
    }

    #[test]
    fn test_try_inverse_singular_is_none() {
        // The criterion is a computed determinant of exactly zero, like `Matrix3::try_inverse`.
        assert!(SymMatrix3Trait::<Fixed>::zeros().try_inverse().is_none());
        assert!(SymMatrix3Trait::from_outer_self(v3i(3, -1, 2)).try_inverse().is_none());
        assert!(SymMatrix3Trait::from_diagonal(v3i(1, 0, 1)).try_inverse().is_none());
        let mut cases = oracle_matrix3::matrix3_transpose_cases().slice(0, 12);
        while let Some(case) = cases.pop_front() {
            let (a, _, _) = *case;
            let s = SymMatrix3Trait::from_matrix_unchecked(m3(a));
            assert!(s.try_inverse().is_some() == s.to_matrix().try_inverse().is_some());
        }
    }

    #[test]
    fn test_inverse_unchecked_matches_try_inverse() {
        let mut cases = oracle_sym_matrix::udu3_inverse_cases();
        while let Some(case) = cases.pop_front() {
            let (a, _, _) = *case;
            let s = SymMatrix3Trait::from_matrix_unchecked(m3(a));
            assert!(s.inverse_unchecked() == s.try_inverse().unwrap());
        }
        let q = 0x40000000;
        assert!(
            s3i((2, 1, 0, 2, 1, 2))
                .inverse_unchecked() == s3((3 * q, -2 * q, q, 4 * q, -2 * q, 3 * q)),
        );
    }

    #[test]
    #[should_panic(expected: 'simba: division by zero')]
    fn test_inverse_unchecked_singular_panics() {
        let _ = black_box(SymMatrix3Trait::from_outer_self(v3i(3, -1, 2))).inverse_unchecked();
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_try_inverse_tiny_norm_panics() {
        let _ = black_box(SymMatrix3Trait::from_diagonal_element(fx(1))).try_inverse();
    }

    // --- approximate equality ------------------------------------------------------------------

    #[test]
    fn test_abs_diff_eq() {
        let id = SymMatrix3Trait::<Fixed>::identity();
        let mut s = id;
        s.m23 = fx(3);
        s.m33 = fx(0x100000000 - 2);
        assert!(s.abs_diff_eq(id, 3) && !s.abs_diff_eq(id, 2));
        assert!(id.abs_diff_eq(id, 0));
        assert!(max_ulp_diff_s3(s, id) == 3);
    }

    // --- gas benchmarks
    // ----------------------------------------------------------------------------
    //
    // Inputs are shared with the `matrix3` benchmarks wherever the same operation exists there, so
    // that the structured kernels and the generic `Matrix3` path can be read side by side. `a` is
    // the `unit` SPD case of the oracle's `udu3` vectors (an inertia tensor).

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_new__baseline() {
        let _a = black_box(s3((1841663783, 52948593, 34555536, 1861238670, 30362422, 1130131990)));
        let e = black_box(s3((1841663783, 52948593, 34555536, 1861238670, 30362422, 1130131990)));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_new__struct() {
        let a = black_box(s3((1841663783, 52948593, 34555536, 1861238670, 30362422, 1130131990)));
        let e = black_box(s3((1841663783, 52948593, 34555536, 1861238670, 30362422, 1130131990)));
        assert!(SymMatrix3Trait::new(a.m11, a.m12, a.m13, a.m22, a.m23, a.m33) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_zeros__baseline() {
        let e = black_box(s3((0, 0, 0, 0, 0, 0)));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_zeros__const() {
        let e = black_box(s3((0, 0, 0, 0, 0, 0)));
        assert!(SymMatrix3Trait::<Fixed>::zeros() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_identity__baseline() {
        let e = black_box(s3((4294967296, 0, 0, 4294967296, 0, 4294967296)));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_identity__const() {
        let e = black_box(s3((4294967296, 0, 0, 4294967296, 0, 4294967296)));
        assert!(SymMatrix3Trait::<Fixed>::identity() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_from_diagonal__baseline() {
        let _d = black_box(v3t((6422282562, 6202159288, 2324644860)));
        let e = black_box(s3((6422282562, 0, 0, 6202159288, 0, 2324644860)));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_from_diagonal__struct() {
        let d = black_box(v3t((6422282562, 6202159288, 2324644860)));
        let e = black_box(s3((6422282562, 0, 0, 6202159288, 0, 2324644860)));
        assert!(SymMatrix3Trait::from_diagonal(d) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_from_diagonal_element__baseline() {
        let _k = black_box(fx(-7516192768));
        let e = black_box(s3((-7516192768, 0, 0, -7516192768, 0, -7516192768)));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_from_diagonal_element__struct() {
        let k = black_box(fx(-7516192768));
        let e = black_box(s3((-7516192768, 0, 0, -7516192768, 0, -7516192768)));
        assert!(SymMatrix3Trait::from_diagonal_element(k) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_from_matrix_unchecked__baseline() {
        let _m = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let e = black_box(
            s3((-8333418062, -3562322882, -7141719772, 8037214559, 5406550886, 4752733287)),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_from_matrix_unchecked__struct() {
        let m = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let e = black_box(
            s3((-8333418062, -3562322882, -7141719772, 8037214559, 5406550886, 4752733287)),
        );
        assert!(SymMatrix3Trait::from_matrix_unchecked(m) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_from_outer_self__baseline() {
        let _v = black_box(v3t((6422282562, 6202159288, 2324644860)));
        let e = black_box(
            s3((9603265977, 9274114724, 3476051182, 8956245107, 3356909777, 1258210680)),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_from_outer_self__structured() {
        let v = black_box(v3t((6422282562, 6202159288, 2324644860)));
        let e = black_box(
            s3((9603265977, 9274114724, 3476051182, 8956245107, 3356909777, 1258210680)),
        );
        assert!(SymMatrix3Trait::from_outer_self(v) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_from_outer_self__generic() {
        let v = black_box(v3t((6422282562, 6202159288, 2324644860)));
        let e = black_box(
            s3((9603265977, 9274114724, 3476051182, 8956245107, 3356909777, 1258210680)),
        );
        assert!(SymMatrix3Trait::from_matrix_unchecked(Matrix3Trait::from_outer(v, v)) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_to_matrix__baseline() {
        let _a = black_box(s3((1841663783, 52948593, 34555536, 1861238670, 30362422, 1130131990)));
        let e = black_box(
            m3(
                [
                    [1841663783, 52948593, 34555536], [52948593, 1861238670, 30362422],
                    [34555536, 30362422, 1130131990],
                ],
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_to_matrix__struct() {
        let a = black_box(s3((1841663783, 52948593, 34555536, 1861238670, 30362422, 1130131990)));
        let e = black_box(
            m3(
                [
                    [1841663783, 52948593, 34555536], [52948593, 1861238670, 30362422],
                    [34555536, 30362422, 1130131990],
                ],
            ),
        );
        assert!(a.to_matrix() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_diagonal__baseline() {
        let _a = black_box(s3((1841663783, 52948593, 34555536, 1861238670, 30362422, 1130131990)));
        let e = black_box(v3t((1841663783, 1861238670, 1130131990)));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_diagonal__struct() {
        let a = black_box(s3((1841663783, 52948593, 34555536, 1861238670, 30362422, 1130131990)));
        let e = black_box(v3t((1841663783, 1861238670, 1130131990)));
        assert!(a.diagonal() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_trace__baseline() {
        let _a = black_box(s3((1841663783, 52948593, 34555536, 1861238670, 30362422, 1130131990)));
        let e = black_box(fx(4833034443));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_trace__sum() {
        let a = black_box(s3((1841663783, 52948593, 34555536, 1861238670, 30362422, 1130131990)));
        let e = black_box(fx(4833034443));
        assert!(a.trace() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_add_diagonal__baseline() {
        let _a = black_box(s3((1841663783, 52948593, 34555536, 1861238670, 30362422, 1130131990)));
        let _k = black_box(fx(-7516192768));
        let e = black_box(
            s3((-5674528985, 52948593, 34555536, -5654954098, 30362422, -6386060778)),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_add_diagonal__sum() {
        let a = black_box(s3((1841663783, 52948593, 34555536, 1861238670, 30362422, 1130131990)));
        let k = black_box(fx(-7516192768));
        let e = black_box(
            s3((-5674528985, 52948593, 34555536, -5654954098, 30362422, -6386060778)),
        );
        assert!(a.add_diagonal(k) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_add__baseline() {
        let _a = black_box(s3((1841663783, 52948593, 34555536, 1861238670, 30362422, 1130131990)));
        let _b = black_box(
            s3((-6403338741, -8222594018, -3661741761, 3456238901, 7403023557, 3069760649)),
        );
        let e = black_box(
            s3((-4561674958, -8169645425, -3627186225, 5317477571, 7433385979, 4199892639)),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_add__operator() {
        let a = black_box(s3((1841663783, 52948593, 34555536, 1861238670, 30362422, 1130131990)));
        let b = black_box(
            s3((-6403338741, -8222594018, -3661741761, 3456238901, 7403023557, 3069760649)),
        );
        let e = black_box(
            s3((-4561674958, -8169645425, -3627186225, 5317477571, 7433385979, 4199892639)),
        );
        assert!(a + b == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_add__assign() {
        let a = black_box(s3((1841663783, 52948593, 34555536, 1861238670, 30362422, 1130131990)));
        let b = black_box(
            s3((-6403338741, -8222594018, -3661741761, 3456238901, 7403023557, 3069760649)),
        );
        let e = black_box(
            s3((-4561674958, -8169645425, -3627186225, 5317477571, 7433385979, 4199892639)),
        );
        let mut r = a;
        r += b;
        assert!(r == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_add__generic() {
        let a = black_box(s3((1841663783, 52948593, 34555536, 1861238670, 30362422, 1130131990)));
        let b = black_box(
            s3((-6403338741, -8222594018, -3661741761, 3456238901, 7403023557, 3069760649)),
        );
        let e = black_box(
            s3((-4561674958, -8169645425, -3627186225, 5317477571, 7433385979, 4199892639)),
        );
        assert!(SymMatrix3Trait::from_matrix_unchecked(a.to_matrix() + b.to_matrix()) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_sub__baseline() {
        let _a = black_box(s3((1841663783, 52948593, 34555536, 1861238670, 30362422, 1130131990)));
        let _b = black_box(
            s3((-6403338741, -8222594018, -3661741761, 3456238901, 7403023557, 3069760649)),
        );
        let e = black_box(
            s3((8245002524, 8275542611, 3696297297, -1595000231, -7372661135, -1939628659)),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_sub__operator() {
        let a = black_box(s3((1841663783, 52948593, 34555536, 1861238670, 30362422, 1130131990)));
        let b = black_box(
            s3((-6403338741, -8222594018, -3661741761, 3456238901, 7403023557, 3069760649)),
        );
        let e = black_box(
            s3((8245002524, 8275542611, 3696297297, -1595000231, -7372661135, -1939628659)),
        );
        assert!(a - b == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_sub__assign() {
        let a = black_box(s3((1841663783, 52948593, 34555536, 1861238670, 30362422, 1130131990)));
        let b = black_box(
            s3((-6403338741, -8222594018, -3661741761, 3456238901, 7403023557, 3069760649)),
        );
        let e = black_box(
            s3((8245002524, 8275542611, 3696297297, -1595000231, -7372661135, -1939628659)),
        );
        let mut r = a;
        r -= b;
        assert!(r == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_neg__baseline() {
        let _a = black_box(s3((1841663783, 52948593, 34555536, 1861238670, 30362422, 1130131990)));
        let e = black_box(
            s3((-1841663783, -52948593, -34555536, -1861238670, -30362422, -1130131990)),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_neg__operator() {
        let a = black_box(s3((1841663783, 52948593, 34555536, 1861238670, 30362422, 1130131990)));
        let e = black_box(
            s3((-1841663783, -52948593, -34555536, -1861238670, -30362422, -1130131990)),
        );
        assert!(-a == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_scale__baseline() {
        let _a = black_box(s3((1841663783, 52948593, 34555536, 1861238670, 30362422, 1130131990)));
        let _k = black_box(fx(-7516192768));
        let e = black_box(
            s3((-3222911621, -92660038, -60472188, -3257167673, -53134239, -1977730983)),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_scale__structured() {
        let a = black_box(s3((1841663783, 52948593, 34555536, 1861238670, 30362422, 1130131990)));
        let k = black_box(fx(-7516192768));
        let e = black_box(
            s3((-3222911621, -92660038, -60472188, -3257167673, -53134239, -1977730983)),
        );
        assert!(a.scale(k) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_scale__generic() {
        let a = black_box(s3((1841663783, 52948593, 34555536, 1861238670, 30362422, 1130131990)));
        let k = black_box(fx(-7516192768));
        let e = black_box(
            s3((-3222911621, -92660038, -60472188, -3257167673, -53134239, -1977730983)),
        );
        assert!(SymMatrix3Trait::from_matrix_unchecked(a.to_matrix().scale(k)) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_mul_vec__baseline() {
        let _a = black_box(s3((1841663783, 52948593, 34555536, 1861238670, 30362422, 1130131990)));
        let _v = black_box(v3t((6422282562, 6202159288, 2324644860)));
        let e = black_box(v3t((2849011252, 2783334669, 707198287)));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_mul_vec__structured() {
        let a = black_box(s3((1841663783, 52948593, 34555536, 1861238670, 30362422, 1130131990)));
        let v = black_box(v3t((6422282562, 6202159288, 2324644860)));
        let e = black_box(v3t((2849011252, 2783334669, 707198287)));
        assert!(a.mul_vec(v) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_mul_vec__generic() {
        let a = black_box(s3((1841663783, 52948593, 34555536, 1861238670, 30362422, 1130131990)));
        let v = black_box(v3t((6422282562, 6202159288, 2324644860)));
        let e = black_box(v3t((2849011252, 2783334669, 707198287)));
        assert!(a.to_matrix().mul_vec(v) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_mul_matrix__baseline() {
        let _a = black_box(s3((1841663783, 52948593, 34555536, 1861238670, 30362422, 1130131990)));
        let _b = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let e = black_box(
            m3(
                [
                    [-3626591127, -1481837766, -2957448397], [-2767137622, 3392106936, 2288502189],
                    [645231072, -1718673272, 1231344871],
                ],
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_mul_matrix__structured() {
        let a = black_box(s3((1841663783, 52948593, 34555536, 1861238670, 30362422, 1130131990)));
        let b = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let e = black_box(
            m3(
                [
                    [-3626591127, -1481837766, -2957448397], [-2767137622, 3392106936, 2288502189],
                    [645231072, -1718673272, 1231344871],
                ],
            ),
        );
        assert!(a.mul_matrix(b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_mul_matrix__generic() {
        let a = black_box(s3((1841663783, 52948593, 34555536, 1861238670, 30362422, 1130131990)));
        let b = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let e = black_box(
            m3(
                [
                    [-3626591127, -1481837766, -2957448397], [-2767137622, 3392106936, 2288502189],
                    [645231072, -1718673272, 1231344871],
                ],
            ),
        );
        assert!(a.to_matrix() * b == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_quadform__baseline() {
        let _r = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let _d = black_box(v3t((6422282562, 6202159288, 2324644860)));
        let e = black_box(
            s3((34871941038, 3481951390, -4662719457, 38764687225, -20898872038, 20538935376)),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_quadform__structured() {
        let r = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let d = black_box(v3t((6422282562, 6202159288, 2324644860)));
        let e = black_box(
            s3((34871941038, 3481951390, -4662719457, 38764687225, -20898872038, 20538935376)),
        );
        assert!(SymMatrix3Trait::quadform(r, d) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_quadform__generic() {
        let r = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let d = black_box(v3t((6422282562, 6202159288, 2324644860)));
        let e = black_box(
            s3((34871941038, 3481951390, -4662719457, 38764687225, -20898872038, 20538935376)),
        );
        let m = (r * Matrix3Trait::from_diagonal(d)) * r.transpose();
        assert!(SymMatrix3Trait::from_matrix_unchecked(m) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_quadform_sym__baseline() {
        let _a = black_box(s3((1841663783, 52948593, 34555536, 1861238670, 30362422, 1130131990)));
        let _r = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let e = black_box(
            s3((11815553290, -356672662, -2016999322, 11872012627, -5424449044, 6492729113)),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_quadform_sym__structured() {
        let a = black_box(s3((1841663783, 52948593, 34555536, 1861238670, 30362422, 1130131990)));
        let r = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let e = black_box(
            s3((11815553290, -356672662, -2016999322, 11872012627, -5424449044, 6492729113)),
        );
        assert!(a.quadform_sym(r) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_quadform_sym__generic() {
        let a = black_box(s3((1841663783, 52948593, 34555536, 1861238670, 30362422, 1130131990)));
        let r = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let e = black_box(
            s3((11815553290, -356672662, -2016999322, 11872012627, -5424449044, 6492729113)),
        );
        assert!(SymMatrix3Trait::from_matrix_unchecked((r * a.to_matrix()) * r.transpose()) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_norm__baseline() {
        let _a = black_box(s3((1841663783, 52948593, 34555536, 1861238670, 30362422, 1130131990)));
        let e = black_box(fx(2853589357));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_norm__wide() {
        let a = black_box(s3((1841663783, 52948593, 34555536, 1861238670, 30362422, 1130131990)));
        let e = black_box(fx(2853589357));
        assert!(a.norm() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_determinant__baseline() {
        let _a = black_box(s3((1841663783, 52948593, 34555536, 1861238670, 30362422, 1130131990)));
        let e = black_box(fx(209622987));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_determinant__structured() {
        let a = black_box(s3((1841663783, 52948593, 34555536, 1861238670, 30362422, 1130131990)));
        let e = black_box(fx(209622987));
        assert!(a.determinant() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_determinant__generic() {
        let a = black_box(s3((1841663783, 52948593, 34555536, 1861238670, 30362422, 1130131990)));
        let e = black_box(fx(209622987));
        assert!(a.to_matrix().determinant() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_adjugate__baseline() {
        let _a = black_box(s3((1841663783, 52948593, 34555536, 1861238670, 30362422, 1130131990)));
        let e = black_box(s3((489531896, -13688046, -14600450, 484317790, -12593276, 797438504)));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_adjugate__structured() {
        let a = black_box(s3((1841663783, 52948593, 34555536, 1861238670, 30362422, 1130131990)));
        let e = black_box(s3((489531896, -13688046, -14600450, 484317790, -12593276, 797438504)));
        assert!(a.adjugate() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_adjugate__generic() {
        let a = black_box(s3((1841663783, 52948593, 34555536, 1861238670, 30362422, 1130131990)));
        let e = black_box(s3((489531896, -13688046, -14600450, 484317790, -12593276, 797438504)));
        assert!(SymMatrix3Trait::from_matrix_unchecked(a.to_matrix().adjugate()) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_try_inverse__baseline() {
        let _a = black_box(s3((1841663783, 52948593, 34555536, 1861238670, 30362422, 1130131990)));
        let e = black_box(
            s3((10030023482, -280454497, -299148740, 9923191619, -258023736, 16338724775)),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_try_inverse__structured() {
        let a = black_box(s3((1841663783, 52948593, 34555536, 1861238670, 30362422, 1130131990)));
        let e = black_box(
            s3((10030023482, -280454497, -299148740, 9923191619, -258023736, 16338724775)),
        );
        assert!(a.try_inverse().unwrap() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_try_inverse__structured_prescaled() {
        let a = black_box(s3((927656796, 185981564, 385239268, 728183245, -57978140, 705759751)));
        let e = black_box(
            s3((28296730480, -8512599912, -16145092481, 28060217721, 6951752315, 35521319149)),
        );
        assert!(a.try_inverse().unwrap() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_try_inverse__generic() {
        let a = black_box(s3((1841663783, 52948593, 34555536, 1861238670, 30362422, 1130131990)));
        let e = black_box(
            s3((10030023482, -280454497, -299148740, 9923191619, -258023736, 16338724775)),
        );
        let inv = a.to_matrix().try_inverse().unwrap();
        assert!(SymMatrix3Trait::from_matrix_unchecked(inv) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_inverse_unchecked__baseline() {
        let _a = black_box(s3((1841663783, 52948593, 34555536, 1861238670, 30362422, 1130131990)));
        let e = black_box(
            s3((10030023482, -280454497, -299148740, 9923191619, -258023736, 16338724775)),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_inverse_unchecked__structured() {
        let a = black_box(s3((1841663783, 52948593, 34555536, 1861238670, 30362422, 1130131990)));
        let e = black_box(
            s3((10030023482, -280454497, -299148740, 9923191619, -258023736, 16338724775)),
        );
        assert!(a.inverse_unchecked() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_abs_diff_eq__baseline() {
        let _a = black_box(s3((1841663783, 52948593, 34555536, 1861238670, 30362422, 1130131990)));
        let _b = black_box(s3((1841663783, 52948593, 34555536, 1861238670, 30362422, 1130131990)));
        let e = black_box(true);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_abs_diff_eq__all_compared() {
        let a = black_box(s3((1841663783, 52948593, 34555536, 1861238670, 30362422, 1130131990)));
        let b = black_box(s3((1841663783, 52948593, 34555536, 1861238670, 30362422, 1130131990)));
        let e = black_box(true);
        assert!(a.abs_diff_eq(b, 2) == e);
    }
}
