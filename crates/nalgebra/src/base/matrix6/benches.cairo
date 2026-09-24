//! Gas benchmarks of `Matrix6` (`bench_matrix6_<op>__<variant>`, net = raw - `baseline` of the
//! group), and the alternative implementations that lost (`alt_*`), kept as evidence together
//! with the tests showing why (AGENTS.md rule 8).
//!
//! The block-composition variants (`alt_mul_blocks`, `alt_mul_vec_blocks`) are the natural way to
//! multiply a 2x2 grid of `Matrix3` (the 3x3 blocks are read and written through the test-only
//! `m6_block*` / `m6_from_blocks` helpers, moves only). They are NOT shipped: each output scalar
//! would be rounded twice (once per 3x3 product, once more when the two are added), which breaks
//! the oracle's tolerance of 0. They also cost more gas, so nothing is given up.

use fixed::Fixed;
use nalgebra_testing::black_box;
use simba::scalar::Real;
use crate::base::matrix3::Matrix3Trait;
use crate::base::matrix_test_utils::{
    fx, int, m6_block11, m6_block12, m6_block21, m6_block22, m6_from_blocks, m6i, v6_from_halves,
    v6_head, v6_tail, v6i,
};
use crate::base::vector6::Vector6;
use crate::base::{MatrixMul, MatrixTrMul};
use super::{Matrix6, Matrix6Trait};

/// `[[1, .., 6], .., [31, .., 36]]`.
fn a6() -> Matrix6<Fixed> {
    m6i(
        [
            [1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18],
            [19, 20, 21, 22, 23, 24], [25, 26, 27, 28, 29, 30], [31, 32, 33, 34, 35, 36],
        ],
    )
}

/// A permutation matrix: `a6() * b6()` permutes columns, with exact products.
fn b6() -> Matrix6<Fixed> {
    m6i(
        [
            [0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0],
        ],
    )
}

/// `a6() * b6()`.
fn ab6() -> Matrix6<Fixed> {
    m6i(
        [
            [2, 1, 6, 4, 5, 3], [8, 7, 12, 10, 11, 9], [14, 13, 18, 16, 17, 15],
            [20, 19, 24, 22, 23, 21], [26, 25, 30, 28, 29, 27], [32, 31, 36, 34, 35, 33],
        ],
    )
}

// --- alternative implementations (losers)

/// The matrix product composed from four rounded 3x3 block products per output block: the natural
/// block formula, and the one AGENTS.md rule 4 forbids — every output scalar is rounded once
/// inside each `Matrix3` product and once more by the sum.
fn alt_mul_blocks(lhs: Matrix6<Fixed>, rhs: Matrix6<Fixed>) -> Matrix6<Fixed> {
    let (l11, l21, l12, l22) = (m6_block11(lhs), m6_block21(lhs), m6_block12(lhs), m6_block22(lhs));
    let (r11, r21, r12, r22) = (m6_block11(rhs), m6_block21(rhs), m6_block12(rhs), m6_block22(rhs));
    m6_from_blocks(
        l11 * r11 + l12 * r21, l21 * r11 + l22 * r21, l11 * r12 + l12 * r22, l21 * r12 + l22 * r22,
    )
}

/// `m * v` composed from two rounded `Matrix3 * Vector3` per block row: same two-rounding defect.
fn alt_mul_vec_blocks(m: Matrix6<Fixed>, v: Vector6<Fixed>) -> Vector6<Fixed> {
    let (a, b) = (v6_head(v), v6_tail(v));
    v6_from_halves(
        MatrixMul::mul_mat(m6_block11(m), a) + MatrixMul::mul_mat(m6_block12(m), b),
        MatrixMul::mul_mat(m6_block21(m), a) + MatrixMul::mul_mat(m6_block22(m), b),
    )
}

/// `abs_diff_eq` as the four 3x3 blocks compared by the non-inlined `Matrix3` kernel (the form
/// of the former block layout): the call returns at the first differing block, and still costs
/// more than the shipped inlined chain in both cases measured.
fn alt_abs_diff_eq_blocks(a: Matrix6<Fixed>, b: Matrix6<Fixed>, ulps: u64) -> bool {
    Matrix3Trait::abs_diff_eq(m6_block11(a), m6_block11(b), ulps)
        && Matrix3Trait::abs_diff_eq(m6_block21(a), m6_block21(b), ulps)
        && Matrix3Trait::abs_diff_eq(m6_block12(a), m6_block12(b), ulps)
        && Matrix3Trait::abs_diff_eq(m6_block22(a), m6_block22(b), ulps)
}

/// `is_identity` through the four 3x3 blocks, see `alt_abs_diff_eq_blocks`.
fn alt_is_identity_blocks(a: Matrix6<Fixed>, ulps: u64) -> bool {
    Matrix3Trait::is_identity(m6_block11(a), ulps)
        && Matrix3Trait::abs_diff_eq(m6_block21(a), Matrix3Trait::zeros(), ulps)
        && Matrix3Trait::abs_diff_eq(m6_block12(a), Matrix3Trait::zeros(), ulps)
        && Matrix3Trait::is_identity(m6_block22(a), ulps)
}

/// The shipped 36-term `abs_diff_eq` chain as a call (`#[inline(never)]`, the attribute of the
/// smaller shapes' comparisons): about twice the inlined chain when the first component differs.
#[inline(never)]
fn alt_abs_diff_eq_not_inlined(a: Matrix6<Fixed>, b: Matrix6<Fixed>, ulps: u64) -> bool {
    Real::abs_diff_eq(a.m11, b.m11, ulps)
        && Real::abs_diff_eq(a.m21, b.m21, ulps)
        && Real::abs_diff_eq(a.m31, b.m31, ulps)
        && Real::abs_diff_eq(a.m41, b.m41, ulps)
        && Real::abs_diff_eq(a.m51, b.m51, ulps)
        && Real::abs_diff_eq(a.m61, b.m61, ulps)
        && Real::abs_diff_eq(a.m12, b.m12, ulps)
        && Real::abs_diff_eq(a.m22, b.m22, ulps)
        && Real::abs_diff_eq(a.m32, b.m32, ulps)
        && Real::abs_diff_eq(a.m42, b.m42, ulps)
        && Real::abs_diff_eq(a.m52, b.m52, ulps)
        && Real::abs_diff_eq(a.m62, b.m62, ulps)
        && Real::abs_diff_eq(a.m13, b.m13, ulps)
        && Real::abs_diff_eq(a.m23, b.m23, ulps)
        && Real::abs_diff_eq(a.m33, b.m33, ulps)
        && Real::abs_diff_eq(a.m43, b.m43, ulps)
        && Real::abs_diff_eq(a.m53, b.m53, ulps)
        && Real::abs_diff_eq(a.m63, b.m63, ulps)
        && Real::abs_diff_eq(a.m14, b.m14, ulps)
        && Real::abs_diff_eq(a.m24, b.m24, ulps)
        && Real::abs_diff_eq(a.m34, b.m34, ulps)
        && Real::abs_diff_eq(a.m44, b.m44, ulps)
        && Real::abs_diff_eq(a.m54, b.m54, ulps)
        && Real::abs_diff_eq(a.m64, b.m64, ulps)
        && Real::abs_diff_eq(a.m15, b.m15, ulps)
        && Real::abs_diff_eq(a.m25, b.m25, ulps)
        && Real::abs_diff_eq(a.m35, b.m35, ulps)
        && Real::abs_diff_eq(a.m45, b.m45, ulps)
        && Real::abs_diff_eq(a.m55, b.m55, ulps)
        && Real::abs_diff_eq(a.m65, b.m65, ulps)
        && Real::abs_diff_eq(a.m16, b.m16, ulps)
        && Real::abs_diff_eq(a.m26, b.m26, ulps)
        && Real::abs_diff_eq(a.m36, b.m36, ulps)
        && Real::abs_diff_eq(a.m46, b.m46, ulps)
        && Real::abs_diff_eq(a.m56, b.m56, ulps)
        && Real::abs_diff_eq(a.m66, b.m66, ulps)
}

/// The shipped 36-term `is_identity` chain as a call, see `alt_abs_diff_eq_not_inlined`.
#[inline(never)]
fn alt_is_identity_not_inlined(a: Matrix6<Fixed>, ulps: u64) -> bool {
    Real::abs_diff_eq(a.m11, Real::one(), ulps)
        && Real::abs_diff_eq(a.m21, Real::zero(), ulps)
        && Real::abs_diff_eq(a.m31, Real::zero(), ulps)
        && Real::abs_diff_eq(a.m41, Real::zero(), ulps)
        && Real::abs_diff_eq(a.m51, Real::zero(), ulps)
        && Real::abs_diff_eq(a.m61, Real::zero(), ulps)
        && Real::abs_diff_eq(a.m12, Real::zero(), ulps)
        && Real::abs_diff_eq(a.m22, Real::one(), ulps)
        && Real::abs_diff_eq(a.m32, Real::zero(), ulps)
        && Real::abs_diff_eq(a.m42, Real::zero(), ulps)
        && Real::abs_diff_eq(a.m52, Real::zero(), ulps)
        && Real::abs_diff_eq(a.m62, Real::zero(), ulps)
        && Real::abs_diff_eq(a.m13, Real::zero(), ulps)
        && Real::abs_diff_eq(a.m23, Real::zero(), ulps)
        && Real::abs_diff_eq(a.m33, Real::one(), ulps)
        && Real::abs_diff_eq(a.m43, Real::zero(), ulps)
        && Real::abs_diff_eq(a.m53, Real::zero(), ulps)
        && Real::abs_diff_eq(a.m63, Real::zero(), ulps)
        && Real::abs_diff_eq(a.m14, Real::zero(), ulps)
        && Real::abs_diff_eq(a.m24, Real::zero(), ulps)
        && Real::abs_diff_eq(a.m34, Real::zero(), ulps)
        && Real::abs_diff_eq(a.m44, Real::one(), ulps)
        && Real::abs_diff_eq(a.m54, Real::zero(), ulps)
        && Real::abs_diff_eq(a.m64, Real::zero(), ulps)
        && Real::abs_diff_eq(a.m15, Real::zero(), ulps)
        && Real::abs_diff_eq(a.m25, Real::zero(), ulps)
        && Real::abs_diff_eq(a.m35, Real::zero(), ulps)
        && Real::abs_diff_eq(a.m45, Real::zero(), ulps)
        && Real::abs_diff_eq(a.m55, Real::one(), ulps)
        && Real::abs_diff_eq(a.m65, Real::zero(), ulps)
        && Real::abs_diff_eq(a.m16, Real::zero(), ulps)
        && Real::abs_diff_eq(a.m26, Real::zero(), ulps)
        && Real::abs_diff_eq(a.m36, Real::zero(), ulps)
        && Real::abs_diff_eq(a.m46, Real::zero(), ulps)
        && Real::abs_diff_eq(a.m56, Real::zero(), ulps)
        && Real::abs_diff_eq(a.m66, Real::one(), ulps)
}

// --- why the alternatives lost

#[test]
fn test_mul_alt_blocks_rounds_twice() {
    // Row 1 of `lhs` is (0.5, 0, 0, 0.5, 0, 0), column 1 of `rhs` is (1, 0, 0, 1, 0, 0) ulp: two
    // products of 2^-33, exact sum 1 ulp. Each 3x3 block product floors its half to 0.
    let h = fx(0x80000000);
    let zero: Fixed = Real::zero();
    let mut lhs = Matrix6Trait::<Fixed>::zeros();
    lhs.m11 = h;
    lhs.m14 = h;
    let mut rhs = Matrix6Trait::<Fixed>::zeros();
    rhs.m11 = fx(1);
    rhs.m41 = fx(1);
    assert!((lhs * rhs).m11 == fx(1));
    assert!(alt_mul_blocks(lhs, rhs).m11 == zero);
}

#[test]
fn test_mul_vec_alt_blocks_rounds_twice() {
    let h = fx(0x80000000);
    let zero: Fixed = Real::zero();
    let mut m = Matrix6Trait::<Fixed>::zeros();
    m.m11 = h;
    m.m14 = h;
    let v = Vector6 { x: fx(1), y: zero, z: zero, w: fx(1), a: zero, b: zero };
    assert!(m.mul_mat(v).x == fx(1));
    assert!(alt_mul_vec_blocks(m, v).x == zero);
}

#[test]
fn test_alt_comparisons_agree() {
    // Same answers as the shipped inlined chains: the alternatives lost on gas only.
    let i = Matrix6Trait::<Fixed>::identity();
    let off = i
        + Matrix6Trait::from_diagonal(
            Vector6 { x: fx(0), y: fx(0), z: fx(0), w: fx(0), a: fx(0), b: fx(3) },
        );
    assert!(alt_abs_diff_eq_blocks(a6(), a6(), 0) && !alt_abs_diff_eq_blocks(a6(), b6(), 1));
    assert!(alt_abs_diff_eq_not_inlined(a6(), a6(), 0));
    assert!(!alt_abs_diff_eq_not_inlined(a6(), b6(), 1));
    assert!(alt_is_identity_blocks(i, 0) && !alt_is_identity_blocks(a6(), 1));
    assert!(alt_is_identity_not_inlined(i, 0) && !alt_is_identity_not_inlined(a6(), 1));
    assert!(alt_is_identity_blocks(off, 3) == off.is_identity(3));
    assert!(alt_is_identity_blocks(off, 2) == off.is_identity(2));
    assert!(alt_is_identity_not_inlined(off, 2) == off.is_identity(2));
    assert!(alt_abs_diff_eq_blocks(i, off, 2) == i.abs_diff_eq(off, 2));
    assert!(alt_abs_diff_eq_not_inlined(i, off, 3) == i.abs_diff_eq(off, 3));
}

#[test]
fn test_mul_alt_blocks_agrees_on_exact_products() {
    // With integer components every product is exact, so both implementations agree: the drift is
    // only visible on fractional data (the two tests above, and the oracle's tolerance of 0).
    assert!(alt_mul_blocks(a6(), b6()) == a6() * b6());
}

// --- gas benchmarks

#[test]
#[inline(never)]
fn bench_matrix6_new__baseline() {
    let _x: Fixed = black_box(int(1));
    let e = black_box(a6());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_matrix6_new__new() {
    let x: Fixed = black_box(int(1));
    let e = black_box(a6());
    let r = Matrix6Trait::new(
        x,
        int(2),
        int(3),
        int(4),
        int(5),
        int(6),
        int(7),
        int(8),
        int(9),
        int(10),
        int(11),
        int(12),
        int(13),
        int(14),
        int(15),
        int(16),
        int(17),
        int(18),
        int(19),
        int(20),
        int(21),
        int(22),
        int(23),
        int(24),
        int(25),
        int(26),
        int(27),
        int(28),
        int(29),
        int(30),
        int(31),
        int(32),
        int(33),
        int(34),
        int(35),
        int(36),
    );
    assert!(r == e);
}

#[test]
#[inline(never)]
fn bench_matrix6_fill__baseline() {
    let _s: Fixed = black_box(int(7));
    let e = black_box(Matrix6Trait::<Fixed>::zeros());
    assert!(black_box(e) == e);
}

#[test]
#[inline(never)]
fn bench_matrix6_fill__zeros() {
    let _s: Fixed = black_box(int(7));
    let e = black_box(Matrix6Trait::<Fixed>::zeros());
    assert!(black_box(Matrix6Trait::<Fixed>::zeros()) == e);
}

#[test]
#[inline(never)]
fn bench_matrix6_fill__identity() {
    let _s: Fixed = black_box(int(7));
    let e = black_box(
        m6i(
            [
                [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1],
            ],
        ),
    );
    assert!(black_box(Matrix6Trait::<Fixed>::identity()) == e);
}

#[test]
#[inline(never)]
fn bench_matrix6_fill__from_diagonal_element() {
    let s: Fixed = black_box(int(7));
    let e = black_box(
        m6i(
            [
                [7, 0, 0, 0, 0, 0], [0, 7, 0, 0, 0, 0], [0, 0, 7, 0, 0, 0], [0, 0, 0, 7, 0, 0],
                [0, 0, 0, 0, 7, 0], [0, 0, 0, 0, 0, 7],
            ],
        ),
    );
    assert!(black_box(Matrix6Trait::from_diagonal_element(s)) == e);
}

#[test]
#[inline(never)]
fn bench_matrix6_from_diagonal__baseline() {
    let _d = black_box(v6i(1, -2, 3, -4, 5, -6));
    let e = black_box(
        m6i(
            [
                [1, 0, 0, 0, 0, 0], [0, -2, 0, 0, 0, 0], [0, 0, 3, 0, 0, 0], [0, 0, 0, -4, 0, 0],
                [0, 0, 0, 0, 5, 0], [0, 0, 0, 0, 0, -6],
            ],
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_matrix6_from_diagonal__blocks() {
    let d = black_box(v6i(1, -2, 3, -4, 5, -6));
    let e = black_box(
        m6i(
            [
                [1, 0, 0, 0, 0, 0], [0, -2, 0, 0, 0, 0], [0, 0, 3, 0, 0, 0], [0, 0, 0, -4, 0, 0],
                [0, 0, 0, 0, 5, 0], [0, 0, 0, 0, 0, -6],
            ],
        ),
    );
    assert!(Matrix6Trait::from_diagonal(d) == e);
}

#[test]
#[inline(never)]
fn bench_matrix6_diagonal__baseline() {
    let _a = black_box(a6());
    let e = black_box(v6i(1, 8, 15, 22, 29, 36));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_matrix6_diagonal__blocks() {
    let a = black_box(a6());
    let e = black_box(v6i(1, 8, 15, 22, 29, 36));
    assert!(a.diagonal() == e);
}

#[test]
#[inline(never)]
fn bench_matrix6_transpose__baseline() {
    let _a = black_box(a6());
    let e = black_box(
        m6i(
            [
                [1, 7, 13, 19, 25, 31], [2, 8, 14, 20, 26, 32], [3, 9, 15, 21, 27, 33],
                [4, 10, 16, 22, 28, 34], [5, 11, 17, 23, 29, 35], [6, 12, 18, 24, 30, 36],
            ],
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_matrix6_transpose__blocks() {
    let a = black_box(a6());
    let e = black_box(
        m6i(
            [
                [1, 7, 13, 19, 25, 31], [2, 8, 14, 20, 26, 32], [3, 9, 15, 21, 27, 33],
                [4, 10, 16, 22, 28, 34], [5, 11, 17, 23, 29, 35], [6, 12, 18, 24, 30, 36],
            ],
        ),
    );
    assert!(a.transpose() == e);
}

#[test]
#[inline(never)]
fn bench_matrix6_trace__baseline() {
    let _a = black_box(a6());
    let e = black_box(int(111));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_matrix6_trace__blocks() {
    let a = black_box(a6());
    let e = black_box(int(111));
    assert!(a.trace() == e);
}

#[test]
#[inline(never)]
fn bench_matrix6_abs__baseline() {
    let _a = black_box(-a6());
    let e = black_box(a6());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_matrix6_abs__componentwise() {
    let a = black_box(-a6());
    let e = black_box(a6());
    assert!(a.abs() == e);
}

#[test]
#[inline(never)]
fn bench_matrix6_add__baseline() {
    let _a = black_box(a6());
    let _b = black_box(b6());
    let e = black_box(
        m6i(
            [
                [1, 3, 3, 4, 5, 6], [8, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 19],
                [19, 20, 21, 23, 23, 24], [25, 26, 27, 28, 30, 30], [31, 32, 34, 34, 35, 36],
            ],
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_matrix6_add__operator() {
    let a = black_box(a6());
    let b = black_box(b6());
    let e = black_box(
        m6i(
            [
                [1, 3, 3, 4, 5, 6], [8, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 19],
                [19, 20, 21, 23, 23, 24], [25, 26, 27, 28, 30, 30], [31, 32, 34, 34, 35, 36],
            ],
        ),
    );
    assert!(a + b == e);
}

#[test]
#[inline(never)]
fn bench_matrix6_add__assign() {
    let a = black_box(a6());
    let b = black_box(b6());
    let e = black_box(
        m6i(
            [
                [1, 3, 3, 4, 5, 6], [8, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 19],
                [19, 20, 21, 23, 23, 24], [25, 26, 27, 28, 30, 30], [31, 32, 34, 34, 35, 36],
            ],
        ),
    );
    let mut acc = a;
    acc += b;
    assert!(acc == e);
}

#[test]
#[inline(never)]
fn bench_matrix6_sub__baseline() {
    let _a = black_box(a6());
    let _b = black_box(b6());
    let e = black_box(
        m6i(
            [
                [1, 1, 3, 4, 5, 6], [6, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 17],
                [19, 20, 21, 21, 23, 24], [25, 26, 27, 28, 28, 30], [31, 32, 32, 34, 35, 36],
            ],
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_matrix6_sub__operator() {
    let a = black_box(a6());
    let b = black_box(b6());
    let e = black_box(
        m6i(
            [
                [1, 1, 3, 4, 5, 6], [6, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 17],
                [19, 20, 21, 21, 23, 24], [25, 26, 27, 28, 28, 30], [31, 32, 32, 34, 35, 36],
            ],
        ),
    );
    assert!(a - b == e);
}

#[test]
#[inline(never)]
fn bench_matrix6_sub__assign() {
    let a = black_box(a6());
    let b = black_box(b6());
    let e = black_box(
        m6i(
            [
                [1, 1, 3, 4, 5, 6], [6, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 17],
                [19, 20, 21, 21, 23, 24], [25, 26, 27, 28, 28, 30], [31, 32, 32, 34, 35, 36],
            ],
        ),
    );
    let mut acc = a;
    acc -= b;
    assert!(acc == e);
}

#[test]
#[inline(never)]
fn bench_matrix6_neg__baseline() {
    let _a = black_box(a6());
    let e = black_box(
        m6i(
            [
                [-1, -2, -3, -4, -5, -6], [-7, -8, -9, -10, -11, -12],
                [-13, -14, -15, -16, -17, -18], [-19, -20, -21, -22, -23, -24],
                [-25, -26, -27, -28, -29, -30], [-31, -32, -33, -34, -35, -36],
            ],
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_matrix6_neg__operator() {
    let a = black_box(a6());
    let e = black_box(
        m6i(
            [
                [-1, -2, -3, -4, -5, -6], [-7, -8, -9, -10, -11, -12],
                [-13, -14, -15, -16, -17, -18], [-19, -20, -21, -22, -23, -24],
                [-25, -26, -27, -28, -29, -30], [-31, -32, -33, -34, -35, -36],
            ],
        ),
    );
    assert!(-a == e);
}

#[test]
#[inline(never)]
fn bench_matrix6_scale__baseline() {
    let _a = black_box(a6());
    let _k = black_box(int(2));
    let e = black_box(
        m6i(
            [
                [2, 4, 6, 8, 10, 12], [14, 16, 18, 20, 22, 24], [26, 28, 30, 32, 34, 36],
                [38, 40, 42, 44, 46, 48], [50, 52, 54, 56, 58, 60], [62, 64, 66, 68, 70, 72],
            ],
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_matrix6_scale__blocks() {
    let a = black_box(a6());
    let k = black_box(int(2));
    let e = black_box(
        m6i(
            [
                [2, 4, 6, 8, 10, 12], [14, 16, 18, 20, 22, 24], [26, 28, 30, 32, 34, 36],
                [38, 40, 42, 44, 46, 48], [50, 52, 54, 56, 58, 60], [62, 64, 66, 68, 70, 72],
            ],
        ),
    );
    assert!(a.scale(k) == e);
}

#[test]
#[inline(never)]
fn bench_matrix6_mul__baseline() {
    let _a = black_box(a6());
    let _b = black_box(b6());
    let e = black_box(ab6());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_matrix6_mul__fused() {
    let a = black_box(a6());
    let b = black_box(b6());
    let e = black_box(ab6());
    assert!(a * b == e);
}

#[test]
#[inline(never)]
fn bench_matrix6_mul__assign() {
    let a = black_box(a6());
    let b = black_box(b6());
    let e = black_box(ab6());
    let mut acc = a;
    acc *= b;
    assert!(acc == e);
}

#[test]
#[inline(never)]
fn bench_matrix6_mul__alt_blocks() {
    let a = black_box(a6());
    let b = black_box(b6());
    let e = black_box(ab6());
    assert!(alt_mul_blocks(a, b) == e);
}

#[test]
#[inline(never)]
fn bench_matrix6_tr_mul__baseline() {
    let _a = black_box(a6());
    let _b = black_box(b6());
    let e = black_box(
        m6i(
            [
                [7, 1, 31, 19, 25, 13], [8, 2, 32, 20, 26, 14], [9, 3, 33, 21, 27, 15],
                [10, 4, 34, 22, 28, 16], [11, 5, 35, 23, 29, 17], [12, 6, 36, 24, 30, 18],
            ],
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_matrix6_tr_mul__fused() {
    let a = black_box(a6());
    let b = black_box(b6());
    let e = black_box(
        m6i(
            [
                [7, 1, 31, 19, 25, 13], [8, 2, 32, 20, 26, 14], [9, 3, 33, 21, 27, 15],
                [10, 4, 34, 22, 28, 16], [11, 5, 35, 23, 29, 17], [12, 6, 36, 24, 30, 18],
            ],
        ),
    );
    assert!(a.tr_mul(b) == e);
}

#[test]
#[inline(never)]
fn bench_matrix6_tr_mul__alt_transpose_then_mul() {
    let a = black_box(a6());
    let b = black_box(b6());
    let e = black_box(
        m6i(
            [
                [7, 1, 31, 19, 25, 13], [8, 2, 32, 20, 26, 14], [9, 3, 33, 21, 27, 15],
                [10, 4, 34, 22, 28, 16], [11, 5, 35, 23, 29, 17], [12, 6, 36, 24, 30, 18],
            ],
        ),
    );
    assert!(a.transpose() * b == e);
}

#[test]
#[inline(never)]
fn bench_matrix6_mul_vec__baseline() {
    let _a = black_box(a6());
    let _v = black_box(v6i(1, 0, 0, 0, 0, 2));
    let e = black_box(v6i(13, 31, 49, 67, 85, 103));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_matrix6_mul_vec__fused() {
    let a = black_box(a6());
    let v = black_box(v6i(1, 0, 0, 0, 0, 2));
    let e = black_box(v6i(13, 31, 49, 67, 85, 103));
    assert!(a.mul_mat(v) == e);
}

#[test]
#[inline(never)]
fn bench_matrix6_mul_vec__alt_blocks() {
    let a = black_box(a6());
    let v = black_box(v6i(1, 0, 0, 0, 0, 2));
    let e = black_box(v6i(13, 31, 49, 67, 85, 103));
    assert!(alt_mul_vec_blocks(a, v) == e);
}

#[test]
#[inline(never)]
fn bench_matrix6_tr_mul_vec__baseline() {
    let _a = black_box(a6());
    let _v = black_box(v6i(1, 0, 0, 0, 0, 2));
    let e = black_box(v6i(63, 66, 69, 72, 75, 78));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_matrix6_tr_mul_vec__fused() {
    let a = black_box(a6());
    let v = black_box(v6i(1, 0, 0, 0, 0, 2));
    let e = black_box(v6i(63, 66, 69, 72, 75, 78));
    assert!(a.tr_mul(v) == e);
}

#[test]
#[inline(never)]
fn bench_matrix6_tr_mul_vec__alt_transpose_then_mul_vec() {
    let a = black_box(a6());
    let v = black_box(v6i(1, 0, 0, 0, 0, 2));
    let e = black_box(v6i(63, 66, 69, 72, 75, 78));
    assert!(a.transpose().mul_mat(v) == e);
}

#[test]
#[inline(never)]
fn bench_matrix6_abs_diff_eq__baseline() {
    let _a = black_box(a6());
    let _b = black_box(a6());
    assert!(black_box(true));
}

#[test]
#[inline(never)]
fn bench_matrix6_abs_diff_eq__all_compared() {
    let a = black_box(a6());
    let b = black_box(a6());
    assert!(a.abs_diff_eq(b, 1));
}

#[test]
#[inline(never)]
fn bench_matrix6_abs_diff_eq__alt_blocks() {
    let a = black_box(a6());
    let b = black_box(a6());
    assert!(alt_abs_diff_eq_blocks(a, b, 1));
}

#[test]
#[inline(never)]
fn bench_matrix6_abs_diff_eq__alt_not_inlined() {
    let a = black_box(a6());
    let b = black_box(a6());
    assert!(alt_abs_diff_eq_not_inlined(a, b, 1));
}

#[test]
#[inline(never)]
fn bench_matrix6_abs_diff_eq__first_differs() {
    let a = black_box(a6());
    let b = black_box(b6());
    assert!(!a.abs_diff_eq(b, 1));
}

#[test]
#[inline(never)]
fn bench_matrix6_abs_diff_eq__alt_blocks_first_differs() {
    let a = black_box(a6());
    let b = black_box(b6());
    assert!(!alt_abs_diff_eq_blocks(a, b, 1));
}

#[test]
#[inline(never)]
fn bench_matrix6_abs_diff_eq__alt_not_inlined_first_differs() {
    let a = black_box(a6());
    let b = black_box(b6());
    assert!(!alt_abs_diff_eq_not_inlined(a, b, 1));
}

#[test]
#[inline(never)]
fn bench_matrix6_is_identity__baseline() {
    let _a = black_box(Matrix6Trait::<Fixed>::identity());
    assert!(black_box(true));
}

#[test]
#[inline(never)]
fn bench_matrix6_is_identity__all_compared() {
    let a = black_box(Matrix6Trait::<Fixed>::identity());
    assert!(a.is_identity(1));
}

#[test]
#[inline(never)]
fn bench_matrix6_is_identity__alt_blocks() {
    let a = black_box(Matrix6Trait::<Fixed>::identity());
    assert!(alt_is_identity_blocks(a, 1));
}

#[test]
#[inline(never)]
fn bench_matrix6_is_identity__alt_not_inlined() {
    let a = black_box(Matrix6Trait::<Fixed>::identity());
    assert!(alt_is_identity_not_inlined(a, 1));
}

#[test]
#[inline(never)]
fn bench_matrix6_is_identity__first_differs() {
    let a = black_box(a6());
    assert!(!a.is_identity(1));
}

#[test]
#[inline(never)]
fn bench_matrix6_is_identity__alt_blocks_first_differs() {
    let a = black_box(a6());
    assert!(!alt_is_identity_blocks(a, 1));
}

#[test]
#[inline(never)]
fn bench_matrix6_is_identity__alt_not_inlined_first_differs() {
    let a = black_box(a6());
    assert!(!alt_is_identity_not_inlined(a, 1));
}
