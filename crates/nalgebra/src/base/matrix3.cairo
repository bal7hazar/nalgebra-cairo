//! `Matrix3`: a statically sized 3x3 matrix (upstream `nalgebra::Matrix3`).
//!
//! Every sum of products goes through a fused `Real` kernel: one rounding (floor) and one overflow
//! check per output scalar. Operators (`+`, `-`, unary `-`, `*` between matrices and their
//! assigning forms) are implemented in this module, so they need no import; the other operations
//! are methods of `Matrix3Trait`.

use core::ops::{AddAssign, MulAssign, SubAssign};
use simba::scalar::Real;
use super::sym_matrix3::SymMatrix3;
use super::vector3::Vector3;

/// A 3x3 matrix. `mRC` is the component at row `R`, column `C`.
///
/// Fields are declared in column-major order, so `Serde` matches upstream's storage order, while
/// `new` takes its arguments in row-major order like upstream.
#[derive(Copy, Drop, PartialEq, Serde, Default, Debug, Hash)]
pub struct Matrix3<T> {
    pub m11: T,
    pub m21: T,
    pub m31: T,
    pub m12: T,
    pub m22: T,
    pub m32: T,
    pub m13: T,
    pub m23: T,
    pub m33: T,
}

/// Methods of `Matrix3<T>` for any `Real` scalar.
#[generate_trait]
pub impl Matrix3Impl<
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
> of Matrix3Trait<T> {
    // --- constructors --------------------------------------------------------------------------

    /// The matrix with the given components, in ROW-major order. Upstream: `Matrix3::new`.
    #[inline(always)]
    fn new(m11: T, m12: T, m13: T, m21: T, m22: T, m23: T, m31: T, m32: T, m33: T) -> Matrix3<T> {
        Matrix3 { m11, m21, m31, m12, m22, m32, m13, m23, m33 }
    }

    /// The zero matrix. Upstream: `Matrix3::zeros`.
    #[inline(always)]
    fn zeros() -> Matrix3<T> {
        Matrix3 {
            m11: R::ZERO,
            m21: R::ZERO,
            m31: R::ZERO,
            m12: R::ZERO,
            m22: R::ZERO,
            m32: R::ZERO,
            m13: R::ZERO,
            m23: R::ZERO,
            m33: R::ZERO,
        }
    }

    /// The identity matrix. Upstream: `Matrix3::identity`.
    #[inline(always)]
    fn identity() -> Matrix3<T> {
        Matrix3 {
            m11: R::ONE,
            m21: R::ZERO,
            m31: R::ZERO,
            m12: R::ZERO,
            m22: R::ONE,
            m32: R::ZERO,
            m13: R::ZERO,
            m23: R::ZERO,
            m33: R::ONE,
        }
    }

    /// The diagonal matrix `diag(d)`. Upstream: `Matrix3::from_diagonal`.
    #[inline(always)]
    fn from_diagonal(d: Vector3<T>) -> Matrix3<T> {
        Matrix3 {
            m11: d.x,
            m21: R::ZERO,
            m31: R::ZERO,
            m12: R::ZERO,
            m22: d.y,
            m32: R::ZERO,
            m13: R::ZERO,
            m23: R::ZERO,
            m33: d.z,
        }
    }

    /// The matrix `e * I`. Upstream: `Matrix3::from_diagonal_element`.
    #[inline(always)]
    fn from_diagonal_element(e: T) -> Matrix3<T> {
        Matrix3 {
            m11: e,
            m21: R::ZERO,
            m31: R::ZERO,
            m12: R::ZERO,
            m22: e,
            m32: R::ZERO,
            m13: R::ZERO,
            m23: R::ZERO,
            m33: e,
        }
    }

    /// The matrix whose columns are the given vectors. Upstream: `Matrix3::from_columns`.
    #[inline(always)]
    fn from_columns(c1: Vector3<T>, c2: Vector3<T>, c3: Vector3<T>) -> Matrix3<T> {
        Matrix3 {
            m11: c1.x,
            m21: c1.y,
            m31: c1.z,
            m12: c2.x,
            m22: c2.y,
            m32: c2.z,
            m13: c3.x,
            m23: c3.y,
            m33: c3.z,
        }
    }

    /// The matrix whose rows are the given vectors. Upstream: `Matrix3::from_rows`.
    #[inline(always)]
    fn from_rows(r1: Vector3<T>, r2: Vector3<T>, r3: Vector3<T>) -> Matrix3<T> {
        Matrix3 {
            m11: r1.x,
            m21: r2.x,
            m31: r3.x,
            m12: r1.y,
            m22: r2.y,
            m32: r3.y,
            m13: r1.z,
            m23: r2.z,
            m33: r3.z,
        }
    }

    /// The outer product `a * bᵀ`: each component is one floored product. Panics with the
    /// scalar's overflow error. Upstream: `a * b.transpose()`.
    #[inline(always)]
    fn from_outer(a: Vector3<T>, b: Vector3<T>) -> Matrix3<T> {
        Matrix3 {
            m11: a.x * b.x,
            m21: a.y * b.x,
            m31: a.z * b.x,
            m12: a.x * b.y,
            m22: a.y * b.y,
            m32: a.z * b.y,
            m13: a.x * b.z,
            m23: a.y * b.z,
            m33: a.z * b.z,
        }
    }

    /// The skew-symmetric matrix `[v]×` such that `[v]× * u = v × u`. Exact; panics on the
    /// scalar's `MIN`. Prefer `cross_matrix_mul` / `mul_cross_matrix`, which never materialise it.
    /// Upstream: `Vector3::cross_matrix`.
    #[inline(always)]
    fn cross_matrix(v: Vector3<T>) -> Matrix3<T> {
        Matrix3 {
            m11: R::ZERO,
            m21: v.z,
            m31: -v.y,
            m12: -v.z,
            m22: R::ZERO,
            m32: v.x,
            m13: v.y,
            m23: -v.x,
            m33: R::ZERO,
        }
    }

    /// `[v]× * m` without materialising `[v]×`: column `j` is `v × column_j(m)`, 9 `diff_prod`
    /// (one rounding per component), bit-identical to `cross_matrix(v) * m`. Panics on overflow.
    /// Upstream: `v.cross_matrix() * m`; rapier: `gcross_matrix`.
    fn cross_matrix_mul(v: Vector3<T>, m: Matrix3<T>) -> Matrix3<T> {
        Matrix3 {
            m11: R::diff_prod(v.y, m.m31, v.z, m.m21),
            m21: R::diff_prod(v.z, m.m11, v.x, m.m31),
            m31: R::diff_prod(v.x, m.m21, v.y, m.m11),
            m12: R::diff_prod(v.y, m.m32, v.z, m.m22),
            m22: R::diff_prod(v.z, m.m12, v.x, m.m32),
            m32: R::diff_prod(v.x, m.m22, v.y, m.m12),
            m13: R::diff_prod(v.y, m.m33, v.z, m.m23),
            m23: R::diff_prod(v.z, m.m13, v.x, m.m33),
            m33: R::diff_prod(v.x, m.m23, v.y, m.m13),
        }
    }

    // --- accessors -----------------------------------------------------------------------------

    /// Column 1. Upstream: `column(0)`.
    #[inline(always)]
    fn column1(self: Matrix3<T>) -> Vector3<T> {
        Vector3 { x: self.m11, y: self.m21, z: self.m31 }
    }

    /// Column 2. Upstream: `column(1)`.
    #[inline(always)]
    fn column2(self: Matrix3<T>) -> Vector3<T> {
        Vector3 { x: self.m12, y: self.m22, z: self.m32 }
    }

    /// Column 3. Upstream: `column(2)`.
    #[inline(always)]
    fn column3(self: Matrix3<T>) -> Vector3<T> {
        Vector3 { x: self.m13, y: self.m23, z: self.m33 }
    }

    /// Row 1, as a (column) vector. Upstream: `row(0).transpose()`.
    #[inline(always)]
    fn row1(self: Matrix3<T>) -> Vector3<T> {
        Vector3 { x: self.m11, y: self.m12, z: self.m13 }
    }

    /// Row 2, as a (column) vector. Upstream: `row(1).transpose()`.
    #[inline(always)]
    fn row2(self: Matrix3<T>) -> Vector3<T> {
        Vector3 { x: self.m21, y: self.m22, z: self.m23 }
    }

    /// Row 3, as a (column) vector. Upstream: `row(2).transpose()`.
    #[inline(always)]
    fn row3(self: Matrix3<T>) -> Vector3<T> {
        Vector3 { x: self.m31, y: self.m32, z: self.m33 }
    }

    /// The diagonal. Upstream: `diagonal`.
    #[inline(always)]
    fn diagonal(self: Matrix3<T>) -> Vector3<T> {
        Vector3 { x: self.m11, y: self.m22, z: self.m33 }
    }

    // --- exact operations ----------------------------------------------------------------------

    /// The transpose. Exact. Upstream: `transpose`.
    #[inline(always)]
    fn transpose(self: Matrix3<T>) -> Matrix3<T> {
        Matrix3 {
            m11: self.m11,
            m21: self.m12,
            m31: self.m13,
            m12: self.m21,
            m22: self.m22,
            m32: self.m23,
            m13: self.m31,
            m23: self.m32,
            m33: self.m33,
        }
    }

    /// Sum of the diagonal. Exact; panics on overflow. Upstream: `trace`.
    #[inline(always)]
    fn trace(self: Matrix3<T>) -> T {
        self.m11 + self.m22 + self.m33
    }

    /// Component-wise absolute value. Panics on the scalar's `MIN`. Upstream: `abs`.
    #[inline(always)]
    fn abs(self: Matrix3<T>) -> Matrix3<T> {
        Matrix3 {
            m11: R::abs(self.m11),
            m21: R::abs(self.m21),
            m31: R::abs(self.m31),
            m12: R::abs(self.m12),
            m22: R::abs(self.m22),
            m32: R::abs(self.m32),
            m13: R::abs(self.m13),
            m23: R::abs(self.m23),
            m33: R::abs(self.m33),
        }
    }

    // --- products ------------------------------------------------------------------------------

    /// `self * k`: each component is one floored product. Panics on overflow.
    /// Upstream: `self * k`.
    #[inline(always)]
    fn scale(self: Matrix3<T>, k: T) -> Matrix3<T> {
        Matrix3 {
            m11: self.m11 * k,
            m21: self.m21 * k,
            m31: self.m31 * k,
            m12: self.m12 * k,
            m22: self.m22 * k,
            m32: self.m32 * k,
            m13: self.m13 * k,
            m23: self.m23 * k,
            m33: self.m33 * k,
        }
    }

    /// Component-wise (Hadamard) product: each component is one floored product. Panics on
    /// overflow. Upstream: `component_mul`.
    #[inline(always)]
    fn component_mul(self: Matrix3<T>, rhs: Matrix3<T>) -> Matrix3<T> {
        Matrix3 {
            m11: self.m11 * rhs.m11,
            m21: self.m21 * rhs.m21,
            m31: self.m31 * rhs.m31,
            m12: self.m12 * rhs.m12,
            m22: self.m22 * rhs.m22,
            m32: self.m32 * rhs.m32,
            m13: self.m13 * rhs.m13,
            m23: self.m23 * rhs.m23,
            m33: self.m33 * rhs.m33,
        }
    }

    /// `self * v`: one `sum_prod3` per component (one rounding each). Panics on overflow.
    /// Upstream: `self * v`.
    #[inline(always)]
    fn mul_vec(self: Matrix3<T>, v: Vector3<T>) -> Vector3<T> {
        Vector3 {
            x: R::sum_prod3(self.m11, v.x, self.m12, v.y, self.m13, v.z),
            y: R::sum_prod3(self.m21, v.x, self.m22, v.y, self.m23, v.z),
            z: R::sum_prod3(self.m31, v.x, self.m32, v.y, self.m33, v.z),
        }
    }

    /// `selfᵀ * v` without forming the transpose: one `sum_prod3` per component. Panics on
    /// overflow. Upstream: `self.tr_mul(&v)`.
    #[inline(always)]
    fn tr_mul_vec(self: Matrix3<T>, v: Vector3<T>) -> Vector3<T> {
        Vector3 {
            x: R::sum_prod3(self.m11, v.x, self.m21, v.y, self.m31, v.z),
            y: R::sum_prod3(self.m12, v.x, self.m22, v.y, self.m32, v.z),
            z: R::sum_prod3(self.m13, v.x, self.m23, v.y, self.m33, v.z),
        }
    }

    /// `selfᵀ * rhs` without forming the transpose: 9 `sum_prod3`. Panics on overflow.
    /// Upstream: `tr_mul`.
    fn tr_mul(self: Matrix3<T>, rhs: Matrix3<T>) -> Matrix3<T> {
        Matrix3 {
            m11: R::sum_prod3(self.m11, rhs.m11, self.m21, rhs.m21, self.m31, rhs.m31),
            m21: R::sum_prod3(self.m12, rhs.m11, self.m22, rhs.m21, self.m32, rhs.m31),
            m31: R::sum_prod3(self.m13, rhs.m11, self.m23, rhs.m21, self.m33, rhs.m31),
            m12: R::sum_prod3(self.m11, rhs.m12, self.m21, rhs.m22, self.m31, rhs.m32),
            m22: R::sum_prod3(self.m12, rhs.m12, self.m22, rhs.m22, self.m32, rhs.m32),
            m32: R::sum_prod3(self.m13, rhs.m12, self.m23, rhs.m22, self.m33, rhs.m32),
            m13: R::sum_prod3(self.m11, rhs.m13, self.m21, rhs.m23, self.m31, rhs.m33),
            m23: R::sum_prod3(self.m12, rhs.m13, self.m22, rhs.m23, self.m32, rhs.m33),
            m33: R::sum_prod3(self.m13, rhs.m13, self.m23, rhs.m23, self.m33, rhs.m33),
        }
    }

    /// `self * [v]×` without materialising `[v]×`: 9 `diff_prod` (one rounding per component),
    /// bit-identical to `self * cross_matrix(v)`. Panics on overflow.
    /// Upstream: `self * v.cross_matrix()`.
    fn mul_cross_matrix(self: Matrix3<T>, v: Vector3<T>) -> Matrix3<T> {
        Matrix3 {
            m11: R::diff_prod(self.m12, v.z, self.m13, v.y),
            m21: R::diff_prod(self.m22, v.z, self.m23, v.y),
            m31: R::diff_prod(self.m32, v.z, self.m33, v.y),
            m12: R::diff_prod(self.m13, v.x, self.m11, v.z),
            m22: R::diff_prod(self.m23, v.x, self.m21, v.z),
            m32: R::diff_prod(self.m33, v.x, self.m31, v.z),
            m13: R::diff_prod(self.m11, v.y, self.m12, v.x),
            m23: R::diff_prod(self.m21, v.y, self.m22, v.x),
            m33: R::diff_prod(self.m31, v.y, self.m32, v.x),
        }
    }

    /// `self * selfᵀ` as a symmetric matrix: 6 fused kernels instead of 9, bit-identical to the
    /// upper triangle of `self * self.transpose()`. Panics on overflow.
    /// Upstream: `self * self.transpose()`.
    fn mul_transpose(self: Matrix3<T>) -> SymMatrix3<T> {
        SymMatrix3 {
            m11: R::norm_squared3(self.m11, self.m12, self.m13),
            m12: R::sum_prod3(self.m11, self.m21, self.m12, self.m22, self.m13, self.m23),
            m13: R::sum_prod3(self.m11, self.m31, self.m12, self.m32, self.m13, self.m33),
            m22: R::norm_squared3(self.m21, self.m22, self.m23),
            m23: R::sum_prod3(self.m21, self.m31, self.m22, self.m32, self.m23, self.m33),
            m33: R::norm_squared3(self.m31, self.m32, self.m33),
        }
    }

    // --- norms ---------------------------------------------------------------------------------

    /// Squared Frobenius norm: the 9 squares are accumulated exactly (wide accumulator) and
    /// rounded once. Panics on overflow. Upstream: `norm_squared`.
    #[inline(always)]
    fn norm_squared(self: Matrix3<T>) -> T {
        let w = R::wide_add_prod(R::wide_zero(), self.m11, self.m11);
        let w = R::wide_add_prod(w, self.m21, self.m21);
        let w = R::wide_add_prod(w, self.m31, self.m31);
        let w = R::wide_add_prod(w, self.m12, self.m12);
        let w = R::wide_add_prod(w, self.m22, self.m22);
        let w = R::wide_add_prod(w, self.m32, self.m32);
        let w = R::wide_add_prod(w, self.m13, self.m13);
        let w = R::wide_add_prod(w, self.m23, self.m23);
        R::wide_rescale(R::wide_add_prod(w, self.m33, self.m33))
    }

    /// Frobenius norm: square root of the exact, unscaled sum of squares (one rounding, no
    /// intermediate overflow: only the result must fit). Upstream: `norm`.
    #[inline(always)]
    fn norm(self: Matrix3<T>) -> T {
        let w = R::wide_add_prod(R::wide_zero(), self.m11, self.m11);
        let w = R::wide_add_prod(w, self.m21, self.m21);
        let w = R::wide_add_prod(w, self.m31, self.m31);
        let w = R::wide_add_prod(w, self.m12, self.m12);
        let w = R::wide_add_prod(w, self.m22, self.m22);
        let w = R::wide_add_prod(w, self.m32, self.m32);
        let w = R::wide_add_prod(w, self.m13, self.m13);
        let w = R::wide_add_prod(w, self.m23, self.m23);
        R::wide_sqrt(R::wide_add_prod(w, self.m33, self.m33))
    }

    // --- determinant and inverse ---------------------------------------------------------------

    /// The determinant, by cofactor expansion along the first row: 3 `diff_prod` (2x2 minors,
    /// one rounding each) then one `sum_prod3` (second rounding). The error against the exact
    /// floor is at most `|m11| + |m12| + |m13| + 1` ulp (values, not raw: 4 ulp for a rotation).
    /// Panics on overflow. Upstream: `determinant`.
    fn determinant(self: Matrix3<T>) -> T {
        R::sum_prod3(
            self.m11,
            R::diff_prod(self.m22, self.m33, self.m23, self.m32),
            self.m12,
            R::diff_prod(self.m23, self.m31, self.m21, self.m33),
            self.m13,
            R::diff_prod(self.m21, self.m32, self.m22, self.m31),
        )
    }

    /// The adjugate (transposed cofactor matrix): `self * adjugate = determinant * I`. 9
    /// `diff_prod`, one rounding per component. Panics on overflow.
    /// No upstream equivalent (upstream `adjoint` is the conjugate transpose).
    #[inline(always)]
    fn adjugate(self: Matrix3<T>) -> Matrix3<T> {
        Matrix3 {
            m11: R::diff_prod(self.m22, self.m33, self.m23, self.m32),
            m21: R::diff_prod(self.m23, self.m31, self.m21, self.m33),
            m31: R::diff_prod(self.m21, self.m32, self.m22, self.m31),
            m12: R::diff_prod(self.m13, self.m32, self.m12, self.m33),
            m22: R::diff_prod(self.m11, self.m33, self.m13, self.m31),
            m32: R::diff_prod(self.m12, self.m31, self.m11, self.m32),
            m13: R::diff_prod(self.m12, self.m23, self.m13, self.m22),
            m23: R::diff_prod(self.m13, self.m21, self.m11, self.m23),
            m33: R::diff_prod(self.m11, self.m22, self.m12, self.m21),
        }
    }

    /// The inverse, or `None` when the matrix is singular. Upstream: `try_inverse`.
    ///
    /// Singularity criterion, like upstream: the computed determinant is EXACTLY zero (no
    /// epsilon). The zero matrix is singular. A nearly singular matrix is inverted with the
    /// precision its conditioning allows, and panics with the scalar's overflow error when a
    /// component of the inverse does not fit.
    ///
    /// Algorithm: `adjugate / determinant`, one division per component (a single reciprocal
    /// followed by 9 multiplications is 15 % cheaper but loses the low bits of `1 / det` when
    /// `|det| >> 1`: up to 900 ulp on the oracle's `medium` matrices, against 1). Because a
    /// fixed-point determinant of a small matrix has few significant bits (a triple product of
    /// 0.05 is 2^-13), a matrix of Frobenius norm `f <= 1` is first multiplied by the INTEGER
    /// `k = floor(2 / f)` (exact products), which gives `inverse = adjugate(k * self) * (k /
    /// det(k * self))`: 27 ulp instead of 84 960 on the oracle's `small` matrices. Matrices
    /// with `|det| >= 1/2` skip the norm computation (`f <= 1` implies `|det| < 1/2`). Panics
    /// with the overflow error when `0 < f <= 2^-30`.
    ///
    /// Re-ranked on `fixed` 0.3.0 (WP 7.2): the unscaled branch divides through ONE prepared
    /// divisor (`Real::div9`, bit-identical to per-element division). `adjugate / det` without the
    /// pre-scaling — upstream's 3x3 formula — costs 66 860 gas through `Real::div9`
    /// (`bench_matrix3_try_inverse__alt_div_n`) against 88 360, and `adjugate * (1 / det)` is
    /// cheaper still, but they leave respectively 2 and 9 of the 30 oracle cases outside their
    /// tolerance (`test_try_inverse_candidates_error`), so the pre-scaled algorithm stays. The
    /// charged gas is that of the costliest branch (the pre-scaled one): the three
    /// `bench_matrix3_try_inverse__prescaled_*` benchmarks measure the same figure.
    fn try_inverse(self: Matrix3<T>) -> Option<Matrix3<T>> {
        let adj = Self::adjugate(self);
        let det = R::sum_prod3(self.m11, adj.m11, self.m12, adj.m21, self.m13, adj.m31);
        if det < R::HALF && det > -R::HALF {
            let f = Self::norm(self);
            if f == R::ZERO {
                return None;
            }
            let k = R::floor(R::div(R::TWO, f));
            if k >= R::TWO {
                let b = Self::scale(self, k);
                let adj_b = Self::adjugate(b);
                let det_b = R::sum_prod3(b.m11, adj_b.m11, b.m12, adj_b.m21, b.m13, adj_b.m31);
                if det_b == R::ZERO {
                    return None;
                }
                return Some(Self::scale(adj_b, R::div(k, det_b)));
            }
            if det == R::ZERO {
                return None;
            }
        }
        let (m11, m21, m31, m12, m22, m32, m13, m23, m33) = R::div9(
            adj.m11, adj.m21, adj.m31, adj.m12, adj.m22, adj.m32, adj.m13, adj.m23, adj.m33, det,
        );
        Some(Matrix3 { m11, m21, m31, m12, m22, m32, m13, m23, m33 })
    }

    // --- approximate equality ------------------------------------------------------------------

    /// Whether every component is within `ulps` smallest units of the identity's.
    /// Upstream: `is_identity(eps)`, with the tolerance in raw units instead of a float epsilon.
    fn is_identity(self: Matrix3<T>, ulps: u64) -> bool {
        R::abs_diff_eq(self.m11, R::ONE, ulps)
            && R::abs_diff_eq(self.m21, R::ZERO, ulps)
            && R::abs_diff_eq(self.m31, R::ZERO, ulps)
            && R::abs_diff_eq(self.m12, R::ZERO, ulps)
            && R::abs_diff_eq(self.m22, R::ONE, ulps)
            && R::abs_diff_eq(self.m32, R::ZERO, ulps)
            && R::abs_diff_eq(self.m13, R::ZERO, ulps)
            && R::abs_diff_eq(self.m23, R::ZERO, ulps)
            && R::abs_diff_eq(self.m33, R::ONE, ulps)
    }

    /// Whether every component of `self` is within `ulps` smallest units of `other`'s.
    /// Upstream: `abs_diff_eq`, with the tolerance in raw units instead of a float epsilon.
    fn abs_diff_eq(self: Matrix3<T>, other: Matrix3<T>, ulps: u64) -> bool {
        R::abs_diff_eq(self.m11, other.m11, ulps)
            && R::abs_diff_eq(self.m21, other.m21, ulps)
            && R::abs_diff_eq(self.m31, other.m31, ulps)
            && R::abs_diff_eq(self.m12, other.m12, ulps)
            && R::abs_diff_eq(self.m22, other.m22, ulps)
            && R::abs_diff_eq(self.m32, other.m32, ulps)
            && R::abs_diff_eq(self.m13, other.m13, ulps)
            && R::abs_diff_eq(self.m23, other.m23, ulps)
            && R::abs_diff_eq(self.m33, other.m33, ulps)
    }
}

// --- operators -----------------------------------------------------------------------------------

/// `a + b`, component-wise. Exact; panics on overflow.
pub impl Matrix3Add<T, +Add<T>, +Copy<T>, +Drop<T>> of Add<Matrix3<T>> {
    #[inline(always)]
    fn add(lhs: Matrix3<T>, rhs: Matrix3<T>) -> Matrix3<T> {
        Matrix3 {
            m11: lhs.m11 + rhs.m11,
            m21: lhs.m21 + rhs.m21,
            m31: lhs.m31 + rhs.m31,
            m12: lhs.m12 + rhs.m12,
            m22: lhs.m22 + rhs.m22,
            m32: lhs.m32 + rhs.m32,
            m13: lhs.m13 + rhs.m13,
            m23: lhs.m23 + rhs.m23,
            m33: lhs.m33 + rhs.m33,
        }
    }
}

/// `a - b`, component-wise. Exact; panics on overflow.
pub impl Matrix3Sub<T, +Sub<T>, +Copy<T>, +Drop<T>> of Sub<Matrix3<T>> {
    #[inline(always)]
    fn sub(lhs: Matrix3<T>, rhs: Matrix3<T>) -> Matrix3<T> {
        Matrix3 {
            m11: lhs.m11 - rhs.m11,
            m21: lhs.m21 - rhs.m21,
            m31: lhs.m31 - rhs.m31,
            m12: lhs.m12 - rhs.m12,
            m22: lhs.m22 - rhs.m22,
            m32: lhs.m32 - rhs.m32,
            m13: lhs.m13 - rhs.m13,
            m23: lhs.m23 - rhs.m23,
            m33: lhs.m33 - rhs.m33,
        }
    }
}

/// `-a`, component-wise. Exact; panics on the scalar's `MIN`.
pub impl Matrix3Neg<T, +Neg<T>, +Copy<T>, +Drop<T>> of Neg<Matrix3<T>> {
    #[inline(always)]
    fn neg(a: Matrix3<T>) -> Matrix3<T> {
        Matrix3 {
            m11: -a.m11,
            m21: -a.m21,
            m31: -a.m31,
            m12: -a.m12,
            m22: -a.m22,
            m32: -a.m32,
            m13: -a.m13,
            m23: -a.m23,
            m33: -a.m33,
        }
    }
}

/// `a * b` (matrix product): 9 `sum_prod3`, one rounding per component. Panics on overflow.
pub impl Matrix3Mul<T, impl R: Real<T>, +Copy<T>, +Drop<T>> of Mul<Matrix3<T>> {
    fn mul(lhs: Matrix3<T>, rhs: Matrix3<T>) -> Matrix3<T> {
        Matrix3 {
            m11: R::sum_prod3(lhs.m11, rhs.m11, lhs.m12, rhs.m21, lhs.m13, rhs.m31),
            m21: R::sum_prod3(lhs.m21, rhs.m11, lhs.m22, rhs.m21, lhs.m23, rhs.m31),
            m31: R::sum_prod3(lhs.m31, rhs.m11, lhs.m32, rhs.m21, lhs.m33, rhs.m31),
            m12: R::sum_prod3(lhs.m11, rhs.m12, lhs.m12, rhs.m22, lhs.m13, rhs.m32),
            m22: R::sum_prod3(lhs.m21, rhs.m12, lhs.m22, rhs.m22, lhs.m23, rhs.m32),
            m32: R::sum_prod3(lhs.m31, rhs.m12, lhs.m32, rhs.m22, lhs.m33, rhs.m32),
            m13: R::sum_prod3(lhs.m11, rhs.m13, lhs.m12, rhs.m23, lhs.m13, rhs.m33),
            m23: R::sum_prod3(lhs.m21, rhs.m13, lhs.m22, rhs.m23, lhs.m23, rhs.m33),
            m33: R::sum_prod3(lhs.m31, rhs.m13, lhs.m32, rhs.m23, lhs.m33, rhs.m33),
        }
    }
}

/// `a += b`.
pub impl Matrix3AddAssign<T, +Add<T>, +Copy<T>, +Drop<T>> of AddAssign<Matrix3<T>, Matrix3<T>> {
    #[inline(always)]
    fn add_assign(ref self: Matrix3<T>, rhs: Matrix3<T>) {
        self = self + rhs;
    }
}

/// `a -= b`.
pub impl Matrix3SubAssign<T, +Sub<T>, +Copy<T>, +Drop<T>> of SubAssign<Matrix3<T>, Matrix3<T>> {
    #[inline(always)]
    fn sub_assign(ref self: Matrix3<T>, rhs: Matrix3<T>) {
        self = self - rhs;
    }
}

/// `a *= b` (matrix product, `a = a * b`).
pub impl Matrix3MulAssign<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>,
> of MulAssign<Matrix3<T>, Matrix3<T>> {
    #[inline(always)]
    fn mul_assign(ref self: Matrix3<T>, rhs: Matrix3<T>) {
        self = self * rhs;
    }
}

#[cfg(test)]
mod tests {
    use fixed::Fixed;
    use nalgebra_testing::black_box;
    use simba::scalar::Real;
    use crate::base::matrix_test_utils::{fx, int, m3, m3i, max_ulp_diff3, s3, ulp_diff, v3i, v3t};
    use crate::base::sym_matrix3::SymMatrix3Trait;
    use crate::base::{oracle_matrix3, oracle_matrix3_inverse};
    use super::{Matrix3, Matrix3Trait};

    // --- losing candidates of the determinant / inverse study (kept as evidence) -----------------

    /// Determinant as six triple products, each rounded twice, summed exactly: more gas AND less
    /// precision than the cofactor expansion.
    ///
    /// This is as close to "exact cofactors in a wide accumulator" as `Real` allows: a determinant
    /// term is a product of THREE scalars, and the accumulator only takes products of two
    /// (`wide_add_prod(w, a, b)`), so an unrounded minor cannot be multiplied by `m1j`. A fully
    /// exact 3x3 determinant would need a 96-bit accumulator op in `simba` (`Wide * Fixed`), which
    /// does not exist; see the report of this work package.
    fn determinant_triple_products(m: Matrix3<Fixed>) -> Fixed {
        let w = Real::wide_add_prod(Real::<Fixed>::wide_zero(), m.m11 * m.m22, m.m33);
        let w = Real::wide_add_prod(w, m.m12 * m.m23, m.m31);
        let w = Real::wide_add_prod(w, m.m13 * m.m21, m.m32);
        let w = Real::wide_sub_prod(w, m.m13 * m.m22, m.m31);
        let w = Real::wide_sub_prod(w, m.m12 * m.m21, m.m33);
        Real::wide_rescale(Real::wide_sub_prod(w, m.m11 * m.m23, m.m32))
    }

    /// `adjugate / determinant` without the integer pre-scaling of small matrices.
    fn try_inverse_div(m: Matrix3<Fixed>) -> Option<Matrix3<Fixed>> {
        let adj = m.adjugate();
        let det = m.determinant();
        if det == Real::ZERO {
            return None;
        }
        Some(
            Matrix3 {
                m11: adj.m11 / det,
                m21: adj.m21 / det,
                m31: adj.m31 / det,
                m12: adj.m12 / det,
                m22: adj.m22 / det,
                m32: adj.m32 / det,
                m13: adj.m13 / det,
                m23: adj.m23 / det,
                m33: adj.m33 / det,
            },
        )
    }

    /// Upstream's `adjugate / determinant` (the formula of `try_inverse_div`) through ONE prepared
    /// divisor (`Real::div9`): bit-identical to `try_inverse_div`, so it fails the same oracle
    /// cases; kept to price upstream's formula at its cheapest (WP 7.2).
    fn try_inverse_div_n(m: Matrix3<Fixed>) -> Option<Matrix3<Fixed>> {
        let adj = m.adjugate();
        let det = m.determinant();
        if det == Real::ZERO {
            return None;
        }
        let (m11, m21, m31, m12, m22, m32, m13, m23, m33) = Real::div9(
            adj.m11, adj.m21, adj.m31, adj.m12, adj.m22, adj.m32, adj.m13, adj.m23, adj.m33, det,
        );
        Some(Matrix3 { m11, m21, m31, m12, m22, m32, m13, m23, m33 })
    }

    /// `adjugate * (1 / determinant)`: one reciprocal, 9 multiplications.
    fn try_inverse_recip(m: Matrix3<Fixed>) -> Option<Matrix3<Fixed>> {
        let det = m.determinant();
        if det == Real::ZERO {
            return None;
        }
        Some(m.adjugate().scale(det.recip()))
    }

    /// `(cases above the oracle tolerance, worst error in ulp)` of an inverse candidate.
    fn inverse_failures(variant: u8) -> (u32, u128) {
        let mut cases = oracle_matrix3_inverse::matrix3_try_inverse_cases();
        let mut failures = 0;
        let mut worst = 0;
        while let Some(case) = cases.pop_front() {
            let (a, expected, tol) = *case;
            let got = match variant {
                0 => m3(a).try_inverse(),
                1 => try_inverse_div(m3(a)),
                3 => try_inverse_div_n(m3(a)),
                _ => try_inverse_recip(m3(a)),
            };
            let err = max_ulp_diff3(got.unwrap(), m3(expected));
            if err > tol.into() {
                failures += 1;
            }
            worst = core::cmp::max(worst, err);
        }
        (failures, worst)
    }

    // --- constructors and accessors --------------------------------------------------------------

    #[test]
    fn test_new_is_row_major() {
        let m = Matrix3Trait::new(
            int(1), int(2), int(3), int(4), int(5), int(6), int(7), int(8), int(9),
        );
        assert!(m == m3i([[1, 2, 3], [4, 5, 6], [7, 8, 9]]));
        assert!(m.m12 == int(2) && m.m21 == int(4));
    }

    #[test]
    fn test_serde_is_column_major() {
        let m = m3i([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        let mut out = array![];
        m.serialize(ref out);
        let one: felt252 = 0x100000000;
        assert!(
            out == array![
                1 * one, 4 * one, 7 * one, 2 * one, 5 * one, 8 * one, 3 * one, 6 * one, 9 * one,
            ],
        );
    }

    #[test]
    fn test_zeros_identity() {
        assert!(Matrix3Trait::<Fixed>::zeros() == m3i([[0, 0, 0], [0, 0, 0], [0, 0, 0]]));
        assert!(Matrix3Trait::<Fixed>::identity() == m3i([[1, 0, 0], [0, 1, 0], [0, 0, 1]]));
        assert!(Matrix3Trait::<Fixed>::zeros() == Default::default());
    }

    #[test]
    fn test_from_diagonal() {
        assert!(
            Matrix3Trait::from_diagonal(v3i(2, -3, 4)) == m3i([[2, 0, 0], [0, -3, 0], [0, 0, 4]]),
        );
        assert!(
            Matrix3Trait::from_diagonal_element(int(7)) == m3i([[7, 0, 0], [0, 7, 0], [0, 0, 7]]),
        );
    }

    #[test]
    fn test_from_columns_from_rows() {
        let (r1, r2, r3) = (v3i(1, 2, 3), v3i(4, 5, 6), v3i(7, 8, 9));
        assert!(Matrix3Trait::from_rows(r1, r2, r3) == m3i([[1, 2, 3], [4, 5, 6], [7, 8, 9]]));
        assert!(Matrix3Trait::from_columns(r1, r2, r3) == m3i([[1, 4, 7], [2, 5, 8], [3, 6, 9]]));
    }

    #[test]
    fn test_columns_rows_diagonal() {
        let m = m3i([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        assert!(m.column1() == v3i(1, 4, 7));
        assert!(m.column2() == v3i(2, 5, 8));
        assert!(m.column3() == v3i(3, 6, 9));
        assert!(m.row1() == v3i(1, 2, 3));
        assert!(m.row2() == v3i(4, 5, 6));
        assert!(m.row3() == v3i(7, 8, 9));
        assert!(m.diagonal() == v3i(1, 5, 9));
        assert!(Matrix3Trait::from_columns(m.column1(), m.column2(), m.column3()) == m);
        assert!(Matrix3Trait::from_rows(m.row1(), m.row2(), m.row3()) == m);
    }

    // --- exact operations
    // --------------------------------------------------------------------------

    #[test]
    fn test_add_oracle() {
        let mut cases = oracle_matrix3::matrix3_add_cases();
        while let Some(case) = cases.pop_front() {
            let (a, b, expected, _) = *case;
            assert!(m3(a) + m3(b) == m3(expected));
            let mut acc = m3(a);
            acc += m3(b);
            assert!(acc == m3(expected));
        }
    }

    #[test]
    fn test_sub_oracle() {
        let mut cases = oracle_matrix3::matrix3_sub_cases();
        while let Some(case) = cases.pop_front() {
            let (a, b, expected, _) = *case;
            assert!(m3(a) - m3(b) == m3(expected));
            let mut acc = m3(a);
            acc -= m3(b);
            assert!(acc == m3(expected));
        }
    }

    #[test]
    fn test_neg() {
        let m = m3i([[1, -2, 3], [-4, 5, -6], [7, -8, 0]]);
        assert!(-m == m3i([[-1, 2, -3], [4, -5, 6], [-7, 8, 0]]));
        assert!(m + (-m) == Matrix3Trait::zeros());
    }

    #[test]
    #[should_panic(expected: 'i64_add Overflow')]
    fn test_add_overflow_panics() {
        let m = black_box(Matrix3Trait::from_diagonal_element(Real::<Fixed>::MAX));
        let _ = m + m;
    }

    #[test]
    #[should_panic(expected: 'i64_neg Underflow')]
    fn test_neg_min_panics() {
        let _ = -black_box(Matrix3Trait::from_diagonal_element(Real::<Fixed>::MIN));
    }

    #[test]
    fn test_transpose_oracle() {
        let mut cases = oracle_matrix3::matrix3_transpose_cases();
        while let Some(case) = cases.pop_front() {
            let (a, expected, _) = *case;
            assert!(m3(a).transpose() == m3(expected));
            assert!(m3(a).transpose().transpose() == m3(a));
        }
    }

    #[test]
    fn test_trace_oracle() {
        let mut cases = oracle_matrix3::matrix3_trace_cases();
        while let Some(case) = cases.pop_front() {
            let (a, expected, _) = *case;
            assert!(m3(a).trace() == fx(expected));
        }
    }

    #[test]
    fn test_abs() {
        let m = m3i([[1, -2, 3], [-4, 5, -6], [7, -8, 0]]);
        assert!(m.abs() == m3i([[1, 2, 3], [4, 5, 6], [7, 8, 0]]));
    }

    // --- products
    // ----------------------------------------------------------------------------------

    #[test]
    fn test_scale_oracle() {
        let mut cases = oracle_matrix3::matrix3_scale_cases();
        while let Some(case) = cases.pop_front() {
            let (a, k, expected, _) = *case;
            assert!(m3(a).scale(fx(k)) == m3(expected));
        }
    }

    #[test]
    fn test_component_mul_exact() {
        let a = m3i([[1, -2, 3], [-4, 5, -6], [7, -8, 0]]);
        let b = m3i([[2, 2, 2], [3, 3, 3], [-1, -1, -1]]);
        assert!(a.component_mul(b) == m3i([[2, -4, 6], [-12, 15, -18], [-7, 8, 0]]));
        // 0.5 ulp floors to 0, -0.5 ulp to -1 ulp, 1.5 ulp to 1 ulp, -1.5 ulp to -2 ulp.
        let h = Matrix3Trait::from_diagonal_element(Real::<Fixed>::HALF);
        let e = Matrix3Trait::from_diagonal(v3t((1, -1, 3)));
        assert!(e.component_mul(h) == Matrix3Trait::from_diagonal(v3t((0, -1, 1))));
    }

    #[test]
    fn test_mul_oracle() {
        let mut cases = oracle_matrix3::matrix3_mul_cases();
        while let Some(case) = cases.pop_front() {
            let (a, b, expected, _) = *case;
            assert!(m3(a) * m3(b) == m3(expected));
            let mut acc = m3(a);
            acc *= m3(b);
            assert!(acc == m3(expected));
        }
    }

    #[test]
    fn test_mul_exact_and_identity() {
        let a = m3i([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        let b = m3i([[-1, 0, 2], [3, 1, 0], [0, -2, 1]]);
        assert!(a * b == m3i([[5, -4, 5], [11, -7, 14], [17, -10, 23]]));
        assert!(a * Matrix3Trait::identity() == a);
        assert!(Matrix3Trait::identity() * a == a);
    }

    #[test]
    #[should_panic(expected: 'Fixed: overflow')]
    fn test_mul_overflow_panics() {
        let m = black_box(Matrix3Trait::from_diagonal_element(int(65536)));
        let _ = m * m;
    }

    #[test]
    fn test_mul_vec_oracle() {
        let mut cases = oracle_matrix3::matrix3_mul_vec_cases();
        while let Some(case) = cases.pop_front() {
            let (a, v, expected, _) = *case;
            assert!(m3(a).mul_vec(v3t(v)) == v3t(expected));
            assert!(m3(a).transpose().tr_mul_vec(v3t(v)) == v3t(expected));
        }
    }

    #[test]
    fn test_mul_vec_tr_mul_vec_exact() {
        let a = m3i([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        assert!(a.mul_vec(v3i(1, 0, -1)) == v3i(-2, -2, -2));
        assert!(a.tr_mul_vec(v3i(1, 0, -1)) == v3i(-6, -6, -6));
    }

    #[test]
    fn test_tr_mul_oracle() {
        let mut cases = oracle_matrix3::matrix3_tr_mul_cases();
        while let Some(case) = cases.pop_front() {
            let (a, b, expected, _) = *case;
            assert!(m3(a).tr_mul(m3(b)) == m3(expected));
            assert!(m3(a).transpose() * m3(b) == m3(expected));
        }
    }

    #[test]
    fn test_from_outer_oracle() {
        let mut cases = oracle_matrix3::matrix3_outer_cases();
        while let Some(case) = cases.pop_front() {
            let (u, v, expected, _) = *case;
            assert!(Matrix3Trait::from_outer(v3t(u), v3t(v)) == m3(expected));
        }
    }

    #[test]
    fn test_cross_matrix_oracle() {
        let mut cases = oracle_matrix3::matrix3_cross_matrix_cases();
        while let Some(case) = cases.pop_front() {
            let (v, expected, _) = *case;
            assert!(Matrix3Trait::cross_matrix(v3t(v)) == m3(expected));
        }
    }

    #[test]
    fn test_cross_matrix_kernels_match_materialised_products() {
        let mut cases = oracle_matrix3::matrix3_mul_vec_cases();
        while let Some(case) = cases.pop_front() {
            let (a, v, _, _) = *case;
            let cm = Matrix3Trait::cross_matrix(v3t(v));
            assert!(Matrix3Trait::cross_matrix_mul(v3t(v), m3(a)) == cm * m3(a));
            assert!(m3(a).mul_cross_matrix(v3t(v)) == m3(a) * cm);
        }
        // [x]× * I = [x]× = I * [x]×
        let x = v3i(1, 2, 3);
        let expected = m3i([[0, -3, 2], [3, 0, -1], [-2, 1, 0]]);
        assert!(Matrix3Trait::cross_matrix_mul(x, Matrix3Trait::identity()) == expected);
        assert!(Matrix3Trait::identity().mul_cross_matrix(x) == expected);
    }

    #[test]
    fn test_mul_transpose_matches_generic_product() {
        // small, unit and medium cases (`large` squares do not fit).
        let mut cases = oracle_matrix3::matrix3_mul_cases().slice(0, 12);
        while let Some(case) = cases.pop_front() {
            let (a, b, _, _) = *case;
            assert!(m3(a).mul_transpose().to_matrix() == m3(a) * m3(a).transpose());
            assert!(m3(b).mul_transpose().to_matrix() == m3(b) * m3(b).transpose());
        }
        let one = 0x100000000;
        assert!(
            m3i([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
                .mul_transpose() == s3(
                    (14 * one, 32 * one, 50 * one, 77 * one, 122 * one, 194 * one),
                ),
        );
    }

    // --- norms
    // -------------------------------------------------------------------------------------

    #[test]
    fn test_norm_exact() {
        let m = m3i([[2, -2, 2], [-2, 2, -2], [2, -2, 2]]);
        assert!(m.norm_squared() == int(36));
        assert!(m.norm() == int(6));
        assert!(Matrix3Trait::<Fixed>::zeros().norm() == int(0));
        // The squared norm (3 * 2^58) does not fit, the norm does.
        assert!(Matrix3Trait::from_diagonal_element(int(0x20000000)).norm() >= int(0x20000000));
    }

    #[test]
    #[should_panic(expected: 'Fixed: overflow')]
    fn test_norm_squared_overflow_panics() {
        black_box(Matrix3Trait::from_diagonal_element(int(0x20000000))).norm_squared();
    }

    // --- determinant and inverse
    // -------------------------------------------------------------------

    #[test]
    fn test_determinant_exact() {
        assert!(m3i([[1, 2, 3], [0, 1, 4], [5, 6, 0]]).determinant() == int(1));
        assert!(m3i([[2, 0, 0], [0, 3, 0], [0, 0, -4]]).determinant() == int(-24));
        assert!(m3i([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).determinant() == int(0));
        assert!(Matrix3Trait::<Fixed>::identity().determinant() == int(1));
    }

    #[test]
    fn test_determinant_oracle() {
        let mut cases = oracle_matrix3_inverse::matrix3_determinant_cases();
        let mut worst = 0;
        let mut worst_alt = 0;
        while let Some(case) = cases.pop_front() {
            let (a, expected, tol) = *case;
            let err = ulp_diff(m3(a).determinant(), fx(expected));
            assert!(err <= tol.into(), "determinant error {err} > {tol}");
            worst = core::cmp::max(worst, err);
            worst_alt =
                core::cmp::max(
                    worst_alt, ulp_diff(determinant_triple_products(m3(a)), fx(expected)),
                );
        }
        // Worst error over the oracle (exact expectation, |a_ij| up to 1e3): 556 ulp, against
        // 612 for the triple products.
        assert!(worst == 556, "worst {worst}");
        assert!(worst_alt == 612, "worst alt {worst_alt}");
    }

    #[test]
    #[should_panic(expected: 'Fixed: overflow')]
    fn test_determinant_overflow_panics() {
        black_box(Matrix3Trait::from_diagonal_element(int(2048))).determinant();
    }

    #[test]
    fn test_adjugate_identity() {
        let m = m3i([[2, -1, 3], [0, 4, 1], [5, 2, -2]]);
        let det = m.determinant();
        assert!(det == int(-85));
        assert!(m * m.adjugate() == Matrix3Trait::from_diagonal_element(det));
        assert!(m.adjugate() * m == Matrix3Trait::from_diagonal_element(det));
    }

    #[test]
    fn test_try_inverse_exact() {
        let m = m3i([[1, 2, 3], [0, 1, 4], [5, 6, 0]]);
        let inv = m.try_inverse().unwrap();
        assert!(inv == m3i([[-24, 18, 5], [20, -15, -4], [-5, 4, 1]]));
        assert!(m * inv == Matrix3Trait::identity());
        let d = Matrix3Trait::from_diagonal(v3i(2, -4, 8)).try_inverse().unwrap();
        assert!(d == Matrix3Trait::from_diagonal(v3t((0x80000000, -0x40000000, 0x20000000))));
        assert!(
            Matrix3Trait::<Fixed>::identity().try_inverse().unwrap() == Matrix3Trait::identity(),
        );
    }

    #[test]
    fn test_try_inverse_oracle() {
        let mut cases = oracle_matrix3_inverse::matrix3_try_inverse_cases();
        while let Some(case) = cases.pop_front() {
            let (a, expected, tol) = *case;
            let err = max_ulp_diff3(m3(a).try_inverse().unwrap(), m3(expected));
            assert!(err <= tol.into(), "inverse error {err} > {tol}");
        }
    }

    #[test]
    fn test_try_inverse_candidates_error() {
        // Oracle, 30 well-conditioned matrices (10 small, 10 unit, 10 medium):
        // (cases above the oracle tolerance, worst error in ulp) of the shipped algorithm, of
        // `adjugate / det` without pre-scaling and of `adjugate * (1 / det)`.
        assert!(inverse_failures(0) == (0, 27));
        assert!(inverse_failures(1) == (2, 84960));
        // Upstream's formula through one prepared divisor: the same bits as `try_inverse_div`.
        assert!(inverse_failures(3) == (2, 84960));
        assert!(inverse_failures(2) == (9, 84960));
    }

    #[test]
    fn test_try_inverse_product_is_identity() {
        let mut cases = oracle_matrix3_inverse::matrix3_try_inverse_cases();
        while let Some(case) = cases.pop_front() {
            let (a, _, _) = *case;
            let inv = m3(a).try_inverse().unwrap();
            // Worst residual over the oracle: 104 ulp.
            assert!((m3(a) * inv).is_identity(104));
            assert!((inv * m3(a)).is_identity(104));
        }
    }

    #[test]
    fn test_try_inverse_singular_oracle() {
        let mut cases = oracle_matrix3_inverse::matrix3_try_inverse_singular_cases();
        while let Some(case) = cases.pop_front() {
            let (a, is_some, _) = *case;
            assert!(m3(a).try_inverse().is_some() == is_some);
        }
        assert!(Matrix3Trait::<Fixed>::zeros().try_inverse().is_none());
        assert!(m3i([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).try_inverse().is_none());
    }

    #[test]
    fn test_try_inverse_near_singular_oracle() {
        let mut cases = oracle_matrix3_inverse::matrix3_try_inverse_near_singular_cases();
        while let Some(case) = cases.pop_front() {
            let (a, expected, tol) = *case;
            let inv = m3(a).try_inverse().unwrap();
            assert!(max_ulp_diff3(inv, m3(expected)) <= tol.into());
        }
    }

    #[test]
    fn test_try_inverse_small_scale() {
        // 2^-12 * I: without pre-scaling the determinant (below the resolution).
        let m = Matrix3Trait::from_diagonal_element(fx(0x100000));
        let expected = Matrix3Trait::from_diagonal_element(int(4096));
        assert!(m.determinant() == int(0));
        assert!(try_inverse_div(m).is_none());
        // Relative error below 2^-33 (1496 ulp on 4096).
        let inv = m.try_inverse().unwrap();
        assert!(inv.abs_diff_eq(expected, 1496));
        assert!(!inv.abs_diff_eq(expected, 1495));
    }

    #[test]
    #[should_panic(expected: 'Fixed: overflow')]
    fn test_try_inverse_tiny_norm_panics() {
        let _ = black_box(Matrix3Trait::from_diagonal_element(fx(1))).try_inverse();
    }

    // --- approximate equality
    // ----------------------------------------------------------------------

    #[test]
    fn test_is_identity_abs_diff_eq() {
        let id = Matrix3Trait::<Fixed>::identity();
        let mut m = id;
        m.m21 = fx(3);
        m.m33 = fx(0x100000000 - 2);
        assert!(m.is_identity(3) && !m.is_identity(2));
        assert!(m.abs_diff_eq(id, 3) && !m.abs_diff_eq(id, 2));
        assert!(id.is_identity(0) && id.abs_diff_eq(id, 0));
    }

    // --- gas benchmarks
    // ----------------------------------------------------------------------------

    #[test]
    #[inline(never)]
    fn bench_matrix3_new__baseline() {
        let _a = black_box(
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
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_new__struct() {
        let a = black_box(
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
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        assert!(
            Matrix3Trait::new(a.m11, a.m12, a.m13, a.m21, a.m22, a.m23, a.m31, a.m32, a.m33) == e,
        );
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_zeros__baseline() {
        let e = black_box(m3([[0, 0, 0], [0, 0, 0], [0, 0, 0]]));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_zeros__const() {
        let e = black_box(m3([[0, 0, 0], [0, 0, 0], [0, 0, 0]]));
        assert!(Matrix3Trait::<Fixed>::zeros() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_identity__baseline() {
        let e = black_box(m3([[4294967296, 0, 0], [0, 4294967296, 0], [0, 0, 4294967296]]));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_identity__const() {
        let e = black_box(m3([[4294967296, 0, 0], [0, 4294967296, 0], [0, 0, 4294967296]]));
        assert!(Matrix3Trait::<Fixed>::identity() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_from_diagonal__baseline() {
        let _v = black_box(v3t((6422282562, 6202159288, 2324644860)));
        let e = black_box(m3([[6422282562, 0, 0], [0, 6202159288, 0], [0, 0, 2324644860]]));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_from_diagonal__struct() {
        let v = black_box(v3t((6422282562, 6202159288, 2324644860)));
        let e = black_box(m3([[6422282562, 0, 0], [0, 6202159288, 0], [0, 0, 2324644860]]));
        assert!(Matrix3Trait::from_diagonal(v) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_from_diagonal_element__baseline() {
        let _k = black_box(fx(-7516192768));
        let e = black_box(m3([[-7516192768, 0, 0], [0, -7516192768, 0], [0, 0, -7516192768]]));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_from_diagonal_element__struct() {
        let k = black_box(fx(-7516192768));
        let e = black_box(m3([[-7516192768, 0, 0], [0, -7516192768, 0], [0, 0, -7516192768]]));
        assert!(Matrix3Trait::from_diagonal_element(k) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_from_columns__baseline() {
        let _c1 = black_box(v3t((-8333418062, -6195210852, 2873393302)));
        let _c2 = black_box(v3t((-3562322882, 8037214559, -6638673079)));
        let _c3 = black_box(v3t((-7141719772, 5406550886, 4752733287)));
        let e = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_from_columns__struct() {
        let c1 = black_box(v3t((-8333418062, -6195210852, 2873393302)));
        let c2 = black_box(v3t((-3562322882, 8037214559, -6638673079)));
        let c3 = black_box(v3t((-7141719772, 5406550886, 4752733287)));
        let e = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        assert!(Matrix3Trait::from_columns(c1, c2, c3) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_from_rows__baseline() {
        let _r1 = black_box(v3t((-8333418062, -3562322882, -7141719772)));
        let _r2 = black_box(v3t((-6195210852, 8037214559, 5406550886)));
        let _r3 = black_box(v3t((2873393302, -6638673079, 4752733287)));
        let e = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_from_rows__struct() {
        let r1 = black_box(v3t((-8333418062, -3562322882, -7141719772)));
        let r2 = black_box(v3t((-6195210852, 8037214559, 5406550886)));
        let r3 = black_box(v3t((2873393302, -6638673079, 4752733287)));
        let e = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        assert!(Matrix3Trait::from_rows(r1, r2, r3) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_from_outer__baseline() {
        let _u = black_box(v3t((-2161644290, 7569327059, 4908735083)));
        let _v = black_box(v3t((6422282562, 6202159288, 2324644860)));
        let e = black_box(
            m3(
                [
                    [-3232315749, -3121528358, -1169986858], [11318446411, 10930507472, 4096887363],
                    [7340052101, 7088472341, 2656845790],
                ],
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_from_outer__products() {
        let u = black_box(v3t((-2161644290, 7569327059, 4908735083)));
        let v = black_box(v3t((6422282562, 6202159288, 2324644860)));
        let e = black_box(
            m3(
                [
                    [-3232315749, -3121528358, -1169986858], [11318446411, 10930507472, 4096887363],
                    [7340052101, 7088472341, 2656845790],
                ],
            ),
        );
        assert!(Matrix3Trait::from_outer(u, v) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_cross_matrix__baseline() {
        let _v = black_box(v3t((6422282562, 6202159288, 2324644860)));
        let e = black_box(
            m3(
                [
                    [0, -2324644860, 6202159288], [2324644860, 0, -6422282562],
                    [-6202159288, 6422282562, 0],
                ],
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_cross_matrix__struct() {
        let v = black_box(v3t((6422282562, 6202159288, 2324644860)));
        let e = black_box(
            m3(
                [
                    [0, -2324644860, 6202159288], [2324644860, 0, -6422282562],
                    [-6202159288, 6422282562, 0],
                ],
            ),
        );
        assert!(Matrix3Trait::cross_matrix(v) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_cross_matrix_mul__baseline() {
        let _v = black_box(v3t((6422282562, 6202159288, 2324644860)));
        let _a = black_box(
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
                    [7502480414, -13936724843, 3936909644], [-8807047541, 7998733495, -10972227499],
                    [2770170478, 17162262662, 18397458151],
                ],
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_cross_matrix_mul__structured() {
        let v = black_box(v3t((6422282562, 6202159288, 2324644860)));
        let a = black_box(
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
                    [7502480414, -13936724843, 3936909644], [-8807047541, 7998733495, -10972227499],
                    [2770170478, 17162262662, 18397458151],
                ],
            ),
        );
        assert!(Matrix3Trait::cross_matrix_mul(v, a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_cross_matrix_mul__materialised() {
        let v = black_box(v3t((6422282562, 6202159288, 2324644860)));
        let a = black_box(
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
                    [7502480414, -13936724843, 3936909644], [-8807047541, 7998733495, -10972227499],
                    [2770170478, 17162262662, 18397458151],
                ],
            ),
        );
        assert!(Matrix3Trait::cross_matrix(v) * a == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_mul_cross_matrix__baseline() {
        let _a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let _v = black_box(v3t((6422282562, 6202159288, 2324644860)));
        let e = black_box(
            m3(
                [
                    [8384917871, -6168592929, -6707138873],
                    [-3457213818, 11437587099, -20964291747],
                    [-10456369759, 5551561978, 14076167090],
                ],
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_mul_cross_matrix__structured() {
        let a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let v = black_box(v3t((6422282562, 6202159288, 2324644860)));
        let e = black_box(
            m3(
                [
                    [8384917871, -6168592929, -6707138873],
                    [-3457213818, 11437587099, -20964291747],
                    [-10456369759, 5551561978, 14076167090],
                ],
            ),
        );
        assert!(a.mul_cross_matrix(v) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_mul_cross_matrix__materialised() {
        let a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let v = black_box(v3t((6422282562, 6202159288, 2324644860)));
        let e = black_box(
            m3(
                [
                    [8384917871, -6168592929, -6707138873],
                    [-3457213818, 11437587099, -20964291747],
                    [-10456369759, 5551561978, 14076167090],
                ],
            ),
        );
        assert!(a * Matrix3Trait::cross_matrix(v) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_column__baseline() {
        let _a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let e = black_box(v3t((-3562322882, 8037214559, -6638673079)));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_column__second() {
        let a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let e = black_box(v3t((-3562322882, 8037214559, -6638673079)));
        assert!(a.column2() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_row__baseline() {
        let _a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let e = black_box(v3t((-6195210852, 8037214559, 5406550886)));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_row__second() {
        let a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let e = black_box(v3t((-6195210852, 8037214559, 5406550886)));
        assert!(a.row2() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_diagonal__baseline() {
        let _a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let e = black_box(v3t((-8333418062, 8037214559, 4752733287)));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_diagonal__struct() {
        let a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let e = black_box(v3t((-8333418062, 8037214559, 4752733287)));
        assert!(a.diagonal() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_transpose__baseline() {
        let _a = black_box(
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
                    [-8333418062, -6195210852, 2873393302], [-3562322882, 8037214559, -6638673079],
                    [-7141719772, 5406550886, 4752733287],
                ],
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_transpose__struct() {
        let a = black_box(
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
                    [-8333418062, -6195210852, 2873393302], [-3562322882, 8037214559, -6638673079],
                    [-7141719772, 5406550886, 4752733287],
                ],
            ),
        );
        assert!(a.transpose() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_trace__baseline() {
        let _a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let e = black_box(fx(4456529784));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_trace__sum() {
        let a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let e = black_box(fx(4456529784));
        assert!(a.trace() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_abs__baseline() {
        let _a = black_box(
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
                    [8333418062, 3562322882, 7141719772], [6195210852, 8037214559, 5406550886],
                    [2873393302, 6638673079, 4752733287],
                ],
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_abs__componentwise() {
        let a = black_box(
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
                    [8333418062, 3562322882, 7141719772], [6195210852, 8037214559, 5406550886],
                    [2873393302, 6638673079, 4752733287],
                ],
            ),
        );
        assert!(a.abs() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_add__baseline() {
        let _a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let _b = black_box(
            m3(
                [
                    [-6403338741, -8222594018, -3661741761], [4499486460, 3456238901, 7403023557],
                    [-5213305701, 2927835069, 3069760649],
                ],
            ),
        );
        let e = black_box(
            m3(
                [
                    [-14736756803, -11784916900, -10803461533],
                    [-1695724392, 11493453460, 12809574443], [-2339912399, -3710838010, 7822493936],
                ],
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_add__operator() {
        let a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let b = black_box(
            m3(
                [
                    [-6403338741, -8222594018, -3661741761], [4499486460, 3456238901, 7403023557],
                    [-5213305701, 2927835069, 3069760649],
                ],
            ),
        );
        let e = black_box(
            m3(
                [
                    [-14736756803, -11784916900, -10803461533],
                    [-1695724392, 11493453460, 12809574443], [-2339912399, -3710838010, 7822493936],
                ],
            ),
        );
        assert!(a + b == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_add__assign() {
        let a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let b = black_box(
            m3(
                [
                    [-6403338741, -8222594018, -3661741761], [4499486460, 3456238901, 7403023557],
                    [-5213305701, 2927835069, 3069760649],
                ],
            ),
        );
        let e = black_box(
            m3(
                [
                    [-14736756803, -11784916900, -10803461533],
                    [-1695724392, 11493453460, 12809574443], [-2339912399, -3710838010, 7822493936],
                ],
            ),
        );
        let mut r = a;
        r += b;
        assert!(r == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_sub__baseline() {
        let _a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let _b = black_box(
            m3(
                [
                    [-6403338741, -8222594018, -3661741761], [4499486460, 3456238901, 7403023557],
                    [-5213305701, 2927835069, 3069760649],
                ],
            ),
        );
        let e = black_box(
            m3(
                [
                    [-1930079321, 4660271136, -3479978011], [-10694697312, 4580975658, -1996472671],
                    [8086699003, -9566508148, 1682972638],
                ],
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_sub__operator() {
        let a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let b = black_box(
            m3(
                [
                    [-6403338741, -8222594018, -3661741761], [4499486460, 3456238901, 7403023557],
                    [-5213305701, 2927835069, 3069760649],
                ],
            ),
        );
        let e = black_box(
            m3(
                [
                    [-1930079321, 4660271136, -3479978011], [-10694697312, 4580975658, -1996472671],
                    [8086699003, -9566508148, 1682972638],
                ],
            ),
        );
        assert!(a - b == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_sub__assign() {
        let a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let b = black_box(
            m3(
                [
                    [-6403338741, -8222594018, -3661741761], [4499486460, 3456238901, 7403023557],
                    [-5213305701, 2927835069, 3069760649],
                ],
            ),
        );
        let e = black_box(
            m3(
                [
                    [-1930079321, 4660271136, -3479978011], [-10694697312, 4580975658, -1996472671],
                    [8086699003, -9566508148, 1682972638],
                ],
            ),
        );
        let mut r = a;
        r -= b;
        assert!(r == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_neg__baseline() {
        let _a = black_box(
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
                    [8333418062, 3562322882, 7141719772], [6195210852, -8037214559, -5406550886],
                    [-2873393302, 6638673079, -4752733287],
                ],
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_neg__operator() {
        let a = black_box(
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
                    [8333418062, 3562322882, 7141719772], [6195210852, -8037214559, -5406550886],
                    [-2873393302, 6638673079, -4752733287],
                ],
            ),
        );
        assert!(-a == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_scale__baseline() {
        let _a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let _k = black_box(fx(-7516192768));
        let e = black_box(
            m3(
                [
                    [14583481608, 6234065043, 12498009601],
                    [10841618991, -14065125479, -9461464051],
                    [-5028438279, 11617677888, -8317283253],
                ],
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_scale__products() {
        let a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let k = black_box(fx(-7516192768));
        let e = black_box(
            m3(
                [
                    [14583481608, 6234065043, 12498009601],
                    [10841618991, -14065125479, -9461464051],
                    [-5028438279, 11617677888, -8317283253],
                ],
            ),
        );
        assert!(a.scale(k) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_component_mul__baseline() {
        let _a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let _b = black_box(
            m3(
                [
                    [-6403338741, -8222594018, -3661741761], [4499486460, 3456238901, 7403023557],
                    [-5213305701, 2927835069, 3069760649],
                ],
            ),
        );
        let e = black_box(
            m3(
                [
                    [12424238659, 6819966905, 6088785253], [-6490216439, 6467693861, 9319005434],
                    [-3487774563, -4525515217, 3396941726],
                ],
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_component_mul__products() {
        let a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let b = black_box(
            m3(
                [
                    [-6403338741, -8222594018, -3661741761], [4499486460, 3456238901, 7403023557],
                    [-5213305701, 2927835069, 3069760649],
                ],
            ),
        );
        let e = black_box(
            m3(
                [
                    [12424238659, 6819966905, 6088785253], [-6490216439, 6467693861, 9319005434],
                    [-3487774563, -4525515217, 3396941726],
                ],
            ),
        );
        assert!(a.component_mul(b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_mul_vec__baseline() {
        let _a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let _v = black_box(v3t((6422282562, 6202159288, 2324644860)));
        let e = black_box(v3t((-21470622535, 5268724875, -2717586978)));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_mul_vec__fused() {
        let a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let v = black_box(v3t((6422282562, 6202159288, 2324644860)));
        let e = black_box(v3t((-21470622535, 5268724875, -2717586978)));
        assert!(a.mul_vec(v) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_tr_mul_vec__baseline() {
        let _a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let _v = black_box(v3t((6422282562, 6202159288, 2324644860)));
        let e = black_box(v3t((-19851986100, 2686233155, -299288788)));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_tr_mul_vec__fused() {
        let a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let v = black_box(v3t((6422282562, 6202159288, 2324644860)));
        let e = black_box(v3t((-19851986100, 2686233155, -299288788)));
        assert!(a.tr_mul_vec(v) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_tr_mul_vec__transpose_mul_vec() {
        let a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let v = black_box(v3t((6422282562, 6202159288, 2324644860)));
        let e = black_box(v3t((-19851986100, 2686233155, -299288788)));
        assert!(a.transpose().mul_vec(v) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_mul__baseline() {
        let _a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let _b = black_box(
            m3(
                [
                    [-6403338741, -8222594018, -3661741761], [4499486460, 3456238901, 7403023557],
                    [-5213305701, 2927835069, 3069760649],
                ],
            ),
        );
        let e = black_box(
            m3(
                [
                    [17361027083, 8218990867, -4139846565], [11093767635, 22013840869, 22999422663],
                    [-17007668927, -7603403072, -10495568584],
                ],
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_mul__fused() {
        let a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let b = black_box(
            m3(
                [
                    [-6403338741, -8222594018, -3661741761], [4499486460, 3456238901, 7403023557],
                    [-5213305701, 2927835069, 3069760649],
                ],
            ),
        );
        let e = black_box(
            m3(
                [
                    [17361027083, 8218990867, -4139846565], [11093767635, 22013840869, 22999422663],
                    [-17007668927, -7603403072, -10495568584],
                ],
            ),
        );
        assert!(a * b == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_mul__assign() {
        let a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let b = black_box(
            m3(
                [
                    [-6403338741, -8222594018, -3661741761], [4499486460, 3456238901, 7403023557],
                    [-5213305701, 2927835069, 3069760649],
                ],
            ),
        );
        let e = black_box(
            m3(
                [
                    [17361027083, 8218990867, -4139846565], [11093767635, 22013840869, 22999422663],
                    [-17007668927, -7603403072, -10495568584],
                ],
            ),
        );
        let mut r = a;
        r *= b;
        assert!(r == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_tr_mul__baseline() {
        let _a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let _b = black_box(
            m3(
                [
                    [-6403338741, -8222594018, -3661741761], [4499486460, 3456238901, 7403023557],
                    [-5213305701, 2927835069, 3069760649],
                ],
            ),
        );
        let e = black_box(
            m3(
                [
                    [2446247659, 12927457326, -1519880552], [21789113621, 8762145550, 12145577416],
                    [10542595260, 21263261548, 18804732413],
                ],
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_tr_mul__fused() {
        let a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let b = black_box(
            m3(
                [
                    [-6403338741, -8222594018, -3661741761], [4499486460, 3456238901, 7403023557],
                    [-5213305701, 2927835069, 3069760649],
                ],
            ),
        );
        let e = black_box(
            m3(
                [
                    [2446247659, 12927457326, -1519880552], [21789113621, 8762145550, 12145577416],
                    [10542595260, 21263261548, 18804732413],
                ],
            ),
        );
        assert!(a.tr_mul(b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_tr_mul__transpose_mul() {
        let a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let b = black_box(
            m3(
                [
                    [-6403338741, -8222594018, -3661741761], [4499486460, 3456238901, 7403023557],
                    [-5213305701, 2927835069, 3069760649],
                ],
            ),
        );
        let e = black_box(
            m3(
                [
                    [2446247659, 12927457326, -1519880552], [21789113621, 8762145550, 12145577416],
                    [10542595260, 21263261548, 18804732413],
                ],
            ),
        );
        assert!(a.transpose() * b == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_mul_transpose__baseline() {
        let _a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let e = black_box(
            s3((30999109664, -3635869986, -7971837166, 30782131443, -10584905494, 17442936779)),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_mul_transpose__structured() {
        let a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let e = black_box(
            s3((30999109664, -3635869986, -7971837166, 30782131443, -10584905494, 17442936779)),
        );
        assert!(a.mul_transpose() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_mul_transpose__generic() {
        let a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let e = black_box(
            s3((30999109664, -3635869986, -7971837166, 30782131443, -10584905494, 17442936779)),
        );
        assert!(SymMatrix3Trait::from_matrix_unchecked(a * a.transpose()) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_norm_squared__baseline() {
        let _a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let e = black_box(fx(79224177887));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_norm_squared__wide() {
        let a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let e = black_box(fx(79224177887));
        assert!(a.norm_squared() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_norm__baseline() {
        let _a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let e = black_box(fx(18446280196));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_norm__wide() {
        let a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let e = black_box(fx(18446280196));
        assert!(a.norm() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_determinant__baseline() {
        let _a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let e = black_box(fx(-49139065034));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_determinant__cofactors() {
        let a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let e = black_box(fx(-49139065034));
        assert!(a.determinant() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_determinant__alt_triple_products() {
        let a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let e = black_box(fx(-49139065036));
        assert!(determinant_triple_products(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_adjugate__baseline() {
        let _a = black_box(
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
                    [17250669418, 14980862226, 8880080234], [10472566806, -4443699415, 20791662074],
                    [4198844782, -15264095938, -20732826170],
                ],
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_adjugate__cofactors() {
        let a = black_box(
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
                    [17250669418, 14980862226, 8880080234], [10472566806, -4443699415, 20791662074],
                    [4198844782, -15264095938, -20732826170],
                ],
            ),
        );
        assert!(a.adjugate() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_try_inverse__baseline() {
        let _a = black_box(
            m3(
                [
                    [1651849619, 3942926787, -4111385247], [-2706174340, -1356311774, -3161153896],
                    [1713407532, -2388220103, 1097729905],
                ],
            ),
        );
        let e = black_box(
            m3(
                [
                    [2746796816, -1668618017, 5482570468], [743254844, -2691902150, -4968171083],
                    [-2670352862, -3252013209, -2561850150],
                ],
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_try_inverse__prescaled_det_ge_half() {
        let a = black_box(
            m3(
                [
                    [1651849619, 3942926787, -4111385247], [-2706174340, -1356311774, -3161153896],
                    [1713407532, -2388220103, 1097729905],
                ],
            ),
        );
        let e = black_box(
            m3(
                [
                    [2746796817, -1668618017, 5482570468], [743254844, -2691902150, -4968171083],
                    [-2670352862, -3252013208, -2561850149],
                ],
            ),
        );
        assert!(a.try_inverse().unwrap() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_try_inverse__prescaled_norm_gt_one() {
        let a = black_box(
            m3(
                [
                    [-549311473, 47395325, -2269605289], [1632143220, 2059389761, -2812332219],
                    [-1861974849, -80861466, 2028588103],
                ],
            ),
        );
        let e = black_box(
            m3(
                [
                    [-6951842770, -153772323, -7990975570], [-3388673950, 9398089719, 9237754150],
                    [-6515945503, 233474262, 2126960560],
                ],
            ),
        );
        assert!(a.try_inverse().unwrap() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_try_inverse__prescaled_small() {
        let a = black_box(
            m3(
                [
                    [719963565, -491093223, 202288240], [14792173, 6300332, -1708200869],
                    [-556953536, -690401814, -251454189],
                ],
            ),
        );
        let e = black_box(
            m3(
                [
                    [16495394595, 3675684940, -11699880121],
                    [-13341091189, 955040073, -17220417569], [93636196, -10763579368, -164828975],
                ],
            ),
        );
        assert!(a.try_inverse().unwrap() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_try_inverse__alt_div() {
        let a = black_box(
            m3(
                [
                    [1651849619, 3942926787, -4111385247], [-2706174340, -1356311774, -3161153896],
                    [1713407532, -2388220103, 1097729905],
                ],
            ),
        );
        let e = black_box(
            m3(
                [
                    [2746796817, -1668618017, 5482570468], [743254844, -2691902150, -4968171083],
                    [-2670352862, -3252013208, -2561850149],
                ],
            ),
        );
        assert!(try_inverse_div(a).unwrap() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_try_inverse__alt_div_n() {
        let a = black_box(
            m3(
                [
                    [1651849619, 3942926787, -4111385247], [-2706174340, -1356311774, -3161153896],
                    [1713407532, -2388220103, 1097729905],
                ],
            ),
        );
        let e = black_box(
            m3(
                [
                    [2746796817, -1668618017, 5482570468], [743254844, -2691902150, -4968171083],
                    [-2670352862, -3252013208, -2561850149],
                ],
            ),
        );
        assert!(try_inverse_div_n(a).unwrap() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_try_inverse__alt_recip() {
        let a = black_box(
            m3(
                [
                    [1651849619, 3942926787, -4111385247], [-2706174340, -1356311774, -3161153896],
                    [1713407532, -2388220103, 1097729905],
                ],
            ),
        );
        let e = black_box(
            m3(
                [
                    [2746796816, -1668618017, 5482570468], [743254844, -2691902150, -4968171083],
                    [-2670352862, -3252013209, -2561850150],
                ],
            ),
        );
        assert!(try_inverse_recip(a).unwrap() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_try_inverse_singular__baseline() {
        let _a = black_box(
            m3(
                [
                    [38654705664, 0, -4294967296], [-21474836480, -21474836480, 17179869184],
                    [-17179869184, 21474836480, -12884901888],
                ],
            ),
        );
        let e = black_box(true);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_try_inverse_singular__none() {
        let a = black_box(
            m3(
                [
                    [38654705664, 0, -4294967296], [-21474836480, -21474836480, 17179869184],
                    [-17179869184, 21474836480, -12884901888],
                ],
            ),
        );
        let e = black_box(true);
        assert!(a.try_inverse().is_none() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_is_identity__baseline() {
        let _a = black_box(m3([[4294967296, 0, 0], [0, 4294967296, 0], [0, 0, 4294967296]]));
        let e = black_box(true);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_is_identity__all_compared() {
        let a = black_box(m3([[4294967296, 0, 0], [0, 4294967296, 0], [0, 0, 4294967296]]));
        let e = black_box(true);
        assert!(a.is_identity(2) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_abs_diff_eq__baseline() {
        let _a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let _b = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let e = black_box(true);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_abs_diff_eq__all_compared() {
        let a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let b = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let e = black_box(true);
        assert!(a.abs_diff_eq(b, 2) == e);
    }
}
