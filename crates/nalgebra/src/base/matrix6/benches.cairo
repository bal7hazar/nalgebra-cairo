//! Gas benchmarks of `Matrix6` (`bench_matrix6_<op>__<variant>`, net = raw - `baseline` of the
//! group), and the alternative implementations that lost (`alt_*`), kept as evidence together
//! with the tests showing why (AGENTS.md rule 8).
//!
//! The block-composition variants (`alt_mul_blocks`, `alt_mul_vec_blocks`) are the natural way to
//! multiply a 2x2 grid of `Matrix3`. They are NOT shipped: each output scalar would be rounded
//! twice (once per 3x3 product, once more when the two are added), which breaks the oracle's
//! tolerance of 0. They also cost more gas, so nothing is given up.

use fixed::Fixed;
use nalgebra_testing::black_box;
use simba::scalar::Real;
use crate::base::matrix3::Matrix3Trait;
use crate::base::matrix_test_utils::{fx, int, m6i, v6i};
use crate::base::vector3::Vector3;
use crate::base::vector6::Vector6;
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
    Matrix6 {
        m11: lhs.m11 * rhs.m11 + lhs.m12 * rhs.m21,
        m21: lhs.m21 * rhs.m11 + lhs.m22 * rhs.m21,
        m12: lhs.m11 * rhs.m12 + lhs.m12 * rhs.m22,
        m22: lhs.m21 * rhs.m12 + lhs.m22 * rhs.m22,
    }
}

/// `mul_vec` composed from two rounded `Matrix3::mul_vec` per block row: same two-rounding defect.
fn alt_mul_vec_blocks(m: Matrix6<Fixed>, v: Vector6<Fixed>) -> Vector6<Fixed> {
    Vector6 {
        a: Matrix3Trait::mul_vec(m.m11, v.a) + Matrix3Trait::mul_vec(m.m12, v.b),
        b: Matrix3Trait::mul_vec(m.m21, v.a) + Matrix3Trait::mul_vec(m.m22, v.b),
    }
}

// --- why the alternatives lost

#[test]
fn test_mul_alt_blocks_rounds_twice() {
    // Row 1 of `lhs` is (0.5, 0, 0, 0.5, 0, 0), column 1 of `rhs` is (1, 0, 0, 1, 0, 0) ulp: two
    // products of 2^-33, exact sum 1 ulp. Each 3x3 block product floors its half to 0.
    let h = fx(0x80000000);
    let zero: Fixed = Real::zero();
    let mut lhs = Matrix6Trait::<Fixed>::zeros();
    lhs.m11.m11 = h;
    lhs.m12.m11 = h;
    let mut rhs = Matrix6Trait::<Fixed>::zeros();
    rhs.m11.m11 = fx(1);
    rhs.m21.m11 = fx(1);
    assert!((lhs * rhs).m11.m11 == fx(1));
    assert!(alt_mul_blocks(lhs, rhs).m11.m11 == zero);
}

#[test]
fn test_mul_vec_alt_blocks_rounds_twice() {
    let h = fx(0x80000000);
    let zero: Fixed = Real::zero();
    let mut m = Matrix6Trait::<Fixed>::zeros();
    m.m11.m11 = h;
    m.m12.m11 = h;
    let v = Vector6 {
        a: Vector3 { x: fx(1), y: zero, z: zero }, b: Vector3 { x: fx(1), y: zero, z: zero },
    };
    assert!(m.mul_vec(v).a.x == fx(1));
    assert!(alt_mul_vec_blocks(m, v).a.x == zero);
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
    assert!(a.mul_vec(v) == e);
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
    assert!(a.tr_mul_vec(v) == e);
}

#[test]
#[inline(never)]
fn bench_matrix6_tr_mul_vec__alt_transpose_then_mul_vec() {
    let a = black_box(a6());
    let v = black_box(v6i(1, 0, 0, 0, 0, 2));
    let e = black_box(v6i(63, 66, 69, 72, 75, 78));
    assert!(a.transpose().mul_vec(v) == e);
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
