//! Unit tests of `Matrix6`: exact cases (expected values from the bit-exact Q32.32 model, floor
//! rounding), panics, identities, and the oracle vectors of `tools/oracle` (upstream nalgebra 0.35
//! on the same raw inputs).
//!
//! Every `dim6` matrix op of the oracle has a tolerance of 0: the expected value is the exact
//! floor of the mathematical result, which one rescale per output scalar reproduces bit for bit.
//! The tests therefore assert EQUALITY, not `abs_diff_eq` — a two-rounding implementation (block
//! composition) fails them.
//!
//! `crate::base::oracle_dim6_matrix` is emitted from `tools/oracle` (committed vectors, 2 cases
//! per distribution — a 6x6 case carries up to 108 scalars) with
//! `cargo run --release -- emit-cairo dim6 --from vectors --max-per-dist 2 --out
//! crates/nalgebra/src/base/oracle_dim6_matrix.cairo --ops matrix6_add,matrix6_sub,matrix6_scale,
//! matrix6_mul,matrix6_mul_vec,matrix6_transpose,matrix6_trace`.

use nalgebra_testing::black_box;
use simba::fixed::Fixed;
use simba::scalar::Real;
use crate::base::matrix3::{Matrix3, Matrix3Trait};
use crate::base::oracle_dim6_matrix as oracle;
use crate::base::vector3::Vector3;
use crate::base::vector6::Vector6;
use super::{Matrix6, Matrix6Trait};

/// 2^32: the raw value of 1.
const ONE_RAW: i64 = 0x100000000;

fn fx(raw: i64) -> Fixed {
    Fixed { raw }
}

fn int(v: i64) -> Fixed {
    Fixed { raw: v * ONE_RAW }
}

fn v6(x: i64, y: i64, z: i64, w: i64, a: i64, b: i64) -> Vector6<Fixed> {
    Vector6 {
        a: Vector3 { x: fx(x), y: fx(y), z: fx(z) }, b: Vector3 { x: fx(w), y: fx(a), z: fx(b) },
    }
}

fn v6i(x: i64, y: i64, z: i64, w: i64, a: i64, b: i64) -> Vector6<Fixed> {
    v6(x * ONE_RAW, y * ONE_RAW, z * ONE_RAW, w * ONE_RAW, a * ONE_RAW, b * ONE_RAW)
}

/// A `Vector6` from an oracle tuple of raws.
fn vt(t: (i64, i64, i64, i64, i64, i64)) -> Vector6<Fixed> {
    let (x, y, z, w, a, b) = t;
    v6(x, y, z, w, a, b)
}

/// `Matrix6` from raw ROW-major rows (oracle layout).
fn m6(rows: [[i64; 6]; 6]) -> Matrix6<Fixed> {
    let [r1, r2, r3, r4, r5, r6] = rows;
    let [a11, a12, a13, a14, a15, a16] = r1;
    let [a21, a22, a23, a24, a25, a26] = r2;
    let [a31, a32, a33, a34, a35, a36] = r3;
    let [a41, a42, a43, a44, a45, a46] = r4;
    let [a51, a52, a53, a54, a55, a56] = r5;
    let [a61, a62, a63, a64, a65, a66] = r6;
    Matrix6Trait::new(
        fx(a11),
        fx(a12),
        fx(a13),
        fx(a14),
        fx(a15),
        fx(a16),
        fx(a21),
        fx(a22),
        fx(a23),
        fx(a24),
        fx(a25),
        fx(a26),
        fx(a31),
        fx(a32),
        fx(a33),
        fx(a34),
        fx(a35),
        fx(a36),
        fx(a41),
        fx(a42),
        fx(a43),
        fx(a44),
        fx(a45),
        fx(a46),
        fx(a51),
        fx(a52),
        fx(a53),
        fx(a54),
        fx(a55),
        fx(a56),
        fx(a61),
        fx(a62),
        fx(a63),
        fx(a64),
        fx(a65),
        fx(a66),
    )
}

/// `Matrix6` from integer ROW-major rows.
fn m6i(rows: [[i64; 6]; 6]) -> Matrix6<Fixed> {
    let [r1, r2, r3, r4, r5, r6] = rows;
    let [a11, a12, a13, a14, a15, a16] = r1;
    let [a21, a22, a23, a24, a25, a26] = r2;
    let [a31, a32, a33, a34, a35, a36] = r3;
    let [a41, a42, a43, a44, a45, a46] = r4;
    let [a51, a52, a53, a54, a55, a56] = r5;
    let [a61, a62, a63, a64, a65, a66] = r6;
    Matrix6Trait::new(
        int(a11),
        int(a12),
        int(a13),
        int(a14),
        int(a15),
        int(a16),
        int(a21),
        int(a22),
        int(a23),
        int(a24),
        int(a25),
        int(a26),
        int(a31),
        int(a32),
        int(a33),
        int(a34),
        int(a35),
        int(a36),
        int(a41),
        int(a42),
        int(a43),
        int(a44),
        int(a45),
        int(a46),
        int(a51),
        int(a52),
        int(a53),
        int(a54),
        int(a55),
        int(a56),
        int(a61),
        int(a62),
        int(a63),
        int(a64),
        int(a65),
        int(a66),
    )
}

/// `[[1, .., 6], [7, .., 12], .., [31, .., 36]]`: every component distinct, so a layout mistake
/// cannot hide.
fn a6() -> Matrix6<Fixed> {
    m6i(
        [
            [1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18],
            [19, 20, 21, 22, 23, 24], [25, 26, 27, 28, 29, 30], [31, 32, 33, 34, 35, 36],
        ],
    )
}

/// A permutation matrix (exact products, no rounding anywhere).
fn b6() -> Matrix6<Fixed> {
    m6i(
        [
            [0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0],
        ],
    )
}

// --- constructors and accessors

#[test]
fn test_new_is_row_major() {
    let m = a6();
    // Row 1 lives in blocks (1, 1) and (1, 2); row 6 in blocks (2, 1) and (2, 2).
    assert!(m.m11.m11 == int(1) && m.m11.m12 == int(2) && m.m11.m13 == int(3));
    assert!(m.m12.m11 == int(4) && m.m12.m12 == int(5) && m.m12.m13 == int(6));
    assert!(m.m21.m31 == int(31) && m.m21.m32 == int(32) && m.m21.m33 == int(33));
    assert!(m.m22.m31 == int(34) && m.m22.m32 == int(35) && m.m22.m33 == int(36));
    // The scalar at row 4, column 2 is block (2, 1), local (1, 2).
    assert!(m.m21.m12 == int(20));
}

#[test]
fn test_serde_is_block_order() {
    // Block by block (m11, m21, m12, m22), each block column-major: NOT upstream's flat
    // column-major order.
    let mut out = array![];
    a6().serialize(ref out);
    let one: felt252 = 0x100000000;
    assert!(
        out == array![
            1 * one, 7 * one, 13 * one, 2 * one, 8 * one, 14 * one, 3 * one, 9 * one, 15 * one,
            19 * one, 25 * one, 31 * one, 20 * one, 26 * one, 32 * one, 21 * one, 27 * one,
            33 * one, 4 * one, 10 * one, 16 * one, 5 * one, 11 * one, 17 * one, 6 * one, 12 * one,
            18 * one, 22 * one, 28 * one, 34 * one, 23 * one, 29 * one, 35 * one, 24 * one,
            30 * one, 36 * one,
        ],
    );
}

#[test]
fn test_zeros_identity_default() {
    let z = Matrix6Trait::<Fixed>::zeros();
    assert!(
        z == m6i(
            [
                [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
            ],
        ),
    );
    assert!(z == Default::default());
    let i = Matrix6Trait::<Fixed>::identity();
    assert!(
        i == m6i(
            [
                [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1],
            ],
        ),
    );
    assert!(i.is_identity(0));
    assert!(!z.is_identity(0));
}

#[test]
fn test_from_blocks_and_block_accessors() {
    let m = a6();
    let r = Matrix6Trait::from_blocks(m.block11(), m.block12(), m.block21(), m.block22());
    assert!(r == m);
    // Block (1, 2) holds rows 1-3, columns 4-6.
    assert!(
        m
            .block12() == Matrix3Trait::new(
                int(4), int(5), int(6), int(10), int(11), int(12), int(16), int(17), int(18),
            ),
    );
    // Block (2, 1) holds rows 4-6, columns 1-3.
    assert!(
        m
            .block21() == Matrix3Trait::new(
                int(19), int(20), int(21), int(25), int(26), int(27), int(31), int(32), int(33),
            ),
    );
}

#[test]
fn test_from_diagonal_and_diagonal() {
    let d = v6i(1, -2, 3, -4, 5, -6);
    let m = Matrix6Trait::from_diagonal(d);
    assert!(
        m == m6i(
            [
                [1, 0, 0, 0, 0, 0], [0, -2, 0, 0, 0, 0], [0, 0, 3, 0, 0, 0], [0, 0, 0, -4, 0, 0],
                [0, 0, 0, 0, 5, 0], [0, 0, 0, 0, 0, -6],
            ],
        ),
    );
    assert!(m.diagonal() == d);
    assert!(a6().diagonal() == v6i(1, 8, 15, 22, 29, 36));
    assert!(Matrix6Trait::from_diagonal_element(int(7)).diagonal() == v6i(7, 7, 7, 7, 7, 7));
    assert!(Matrix6Trait::<Fixed>::from_diagonal_element(Real::ONE) == Matrix6Trait::identity());
}

// --- exact operations

#[test]
fn test_transpose_swaps_blocks() {
    let t = a6().transpose();
    assert!(
        t == m6i(
            [
                [1, 7, 13, 19, 25, 31], [2, 8, 14, 20, 26, 32], [3, 9, 15, 21, 27, 33],
                [4, 10, 16, 22, 28, 34], [5, 11, 17, 23, 29, 35], [6, 12, 18, 24, 30, 36],
            ],
        ),
    );
    assert!(t.transpose() == a6());
    // The off-diagonal blocks are swapped, each transposed.
    assert!(t.block12() == a6().block21().transpose());
}

#[test]
fn test_trace_and_abs() {
    assert!(a6().trace() == int(111));
    assert!(Matrix6Trait::<Fixed>::identity().trace() == int(6));
    let m = Matrix6Trait::from_diagonal(v6i(1, -2, 3, -4, 5, -6));
    assert!(m.abs() == Matrix6Trait::from_diagonal(v6i(1, 2, 3, 4, 5, 6)));
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_trace_overflow_panics() {
    let _ = black_box(Matrix6Trait::from_diagonal_element(Real::<Fixed>::MAX)).trace();
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_abs_of_min_panics() {
    let _ = black_box(Matrix6Trait::from_diagonal_element(Real::<Fixed>::MIN)).abs();
}

// --- additive operators

#[test]
fn test_add_sub_neg_exact() {
    assert!(a6() + a6() == a6().scale(int(2)));
    assert!(a6() - a6() == Matrix6Trait::zeros());
    assert!(-a6() + a6() == Matrix6Trait::zeros());
    assert!((-a6()).m22.m33 == int(-36));
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_add_overflow_panics() {
    let m = black_box(Matrix6Trait::from_diagonal_element(Real::<Fixed>::MAX));
    let _ = m + m;
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_neg_min_panics() {
    let _ = -black_box(Matrix6Trait::from_diagonal_element(Real::<Fixed>::MIN));
}

// --- products

#[test]
fn test_scale_exact_and_rounding() {
    assert!(a6().scale(int(-1)) == -a6());
    // 0.5 ulp floors to 0, -0.5 ulp to -1 ulp: one floor per component.
    let h = Matrix6Trait::from_diagonal_element(Real::<Fixed>::HALF);
    let e = Matrix6Trait::from_diagonal(v6(1, -1, 3, -3, 0, 0));
    assert!(e.scale(Real::HALF) == Matrix6Trait::from_diagonal(v6(0, -1, 1, -2, 0, 0)));
    assert!(h.scale(Real::TWO) == Matrix6Trait::identity());
}

#[test]
fn test_mul_exact_and_identity() {
    let i = Matrix6Trait::<Fixed>::identity();
    assert!(a6() * i == a6());
    assert!(i * a6() == a6());
    // `b6` permutes columns.
    assert!(
        a6()
            * b6() == m6i(
                [
                    [2, 1, 6, 4, 5, 3], [8, 7, 12, 10, 11, 9], [14, 13, 18, 16, 17, 15],
                    [20, 19, 24, 22, 23, 21], [26, 25, 30, 28, 29, 27], [32, 31, 36, 34, 35, 33],
                ],
            ),
    );
    let mut acc = a6();
    acc *= b6();
    assert!(acc == a6() * b6());
}

#[test]
fn test_mul_is_a_single_rescale_per_scalar() {
    // Row 1 of `lhs` is (0.5, 0, 0, 0.5, 0, 0) and column 1 of `rhs` is (1, 0, 0, 1, 0, 0) ulp:
    // the two products are 2^-33 each, so the exact sum is 1 ulp. Rounding the 3x3 block products
    // first (`m11 * n11 + m12 * n21`) gives 0 — see `benches::alt_mul_blocks`.
    let h = 0x80000000;
    let lhs = m6(
        [
            [h, 0, 0, h, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
        ],
    );
    let rhs = m6(
        [
            [1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
        ],
    );
    assert!((lhs * rhs).m11.m11 == fx(1));
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_mul_overflow_panics() {
    let m = black_box(Matrix6Trait::from_diagonal_element(int(65536)));
    let _ = m * m;
}

#[test]
fn test_mul_vec_and_tr_mul_vec_exact() {
    let v = v6i(1, 0, 0, 0, 0, 2);
    assert!(a6().mul_vec(v) == v6i(13, 31, 49, 67, 85, 103));
    assert!(a6().tr_mul_vec(v) == v6i(63, 66, 69, 72, 75, 78));
    // `tr_mul_vec` is `transpose().mul_vec()`, without forming the transpose.
    assert!(a6().tr_mul_vec(v) == a6().transpose().mul_vec(v));
    assert!(Matrix6Trait::<Fixed>::identity().mul_vec(v) == v);
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_mul_vec_overflow_panics() {
    let m = black_box(Matrix6Trait::from_diagonal_element(int(65536)));
    let _ = m.mul_vec(black_box(v6i(65536, 0, 0, 0, 0, 0)));
}

#[test]
fn test_tr_mul_matches_transpose_product() {
    assert!(a6().tr_mul(b6()) == a6().transpose() * b6());
    assert!(
        a6()
            .tr_mul(
                b6(),
            ) == m6i(
                [
                    [7, 1, 31, 19, 25, 13], [8, 2, 32, 20, 26, 14], [9, 3, 33, 21, 27, 15],
                    [10, 4, 34, 22, 28, 16], [11, 5, 35, 23, 29, 17], [12, 6, 36, 24, 30, 18],
                ],
            ),
    );
    assert!(Matrix6Trait::<Fixed>::identity().tr_mul(a6()) == a6());
}

// --- approximate equality

#[test]
fn test_abs_diff_eq_and_is_identity_count_raw_units() {
    let i = Matrix6Trait::<Fixed>::identity();
    let off = i + Matrix6Trait::from_diagonal(v6(0, 0, 0, 0, 0, 3));
    assert!(off.is_identity(3));
    assert!(!off.is_identity(2));
    assert!(i.abs_diff_eq(off, 3));
    assert!(!i.abs_diff_eq(off, 2));
    assert!(a6().abs_diff_eq(a6(), 0));
    // An off-diagonal block is compared too.
    let skew = Matrix6Trait::from_blocks(
        Matrix3Trait::identity(),
        Matrix3 {
            m11: fx(4),
            m21: fx(0),
            m31: fx(0),
            m12: fx(0),
            m22: fx(0),
            m32: fx(0),
            m13: fx(0),
            m23: fx(0),
            m33: fx(0),
        },
        Matrix3Trait::zeros(),
        Matrix3Trait::identity(),
    );
    assert!(skew.is_identity(4));
    assert!(!skew.is_identity(3));
}

// --- oracle vectors (upstream nalgebra on the same raw inputs; every tolerance is 0)

#[test]
fn test_add_oracle() {
    let mut cases = oracle::matrix6_add_cases();
    assert!(cases.len() >= 8);
    while let Some(case) = cases.pop_front() {
        let (a, b, expected, tol) = *case;
        assert!(tol == 0);
        assert!(m6(a) + m6(b) == m6(expected));
        let mut acc = m6(a);
        acc += m6(b);
        assert!(acc == m6(expected));
    }
}

#[test]
fn test_sub_oracle() {
    let mut cases = oracle::matrix6_sub_cases();
    assert!(cases.len() >= 8);
    while let Some(case) = cases.pop_front() {
        let (a, b, expected, tol) = *case;
        assert!(tol == 0);
        assert!(m6(a) - m6(b) == m6(expected));
        let mut acc = m6(a);
        acc -= m6(b);
        assert!(acc == m6(expected));
    }
}

#[test]
fn test_scale_oracle() {
    let mut cases = oracle::matrix6_scale_cases();
    assert!(cases.len() >= 8);
    while let Some(case) = cases.pop_front() {
        let (a, k, expected, tol) = *case;
        assert!(tol == 0);
        assert!(m6(a).scale(fx(k)) == m6(expected));
    }
}

#[test]
fn test_transpose_oracle() {
    let mut cases = oracle::matrix6_transpose_cases();
    assert!(cases.len() >= 8);
    while let Some(case) = cases.pop_front() {
        let (a, expected, tol) = *case;
        assert!(tol == 0);
        assert!(m6(a).transpose() == m6(expected));
        assert!(m6(a).transpose().transpose() == m6(a));
    }
}

#[test]
fn test_trace_oracle() {
    let mut cases = oracle::matrix6_trace_cases();
    assert!(cases.len() >= 8);
    while let Some(case) = cases.pop_front() {
        let (a, expected, tol) = *case;
        assert!(tol == 0);
        assert!(m6(a).trace() == fx(expected));
    }
}

#[test]
fn test_mul_oracle() {
    let mut cases = oracle::matrix6_mul_cases();
    assert!(cases.len() >= 8);
    while let Some(case) = cases.pop_front() {
        let (a, b, expected, tol) = *case;
        assert!(tol == 0);
        assert!(m6(a) * m6(b) == m6(expected));
    }
}

#[test]
fn test_tr_mul_oracle() {
    let mut cases = oracle::matrix6_mul_cases();
    while let Some(case) = cases.pop_front() {
        let (a, b, expected, _) = *case;
        // `aᵀ b` with `a` transposed is `a b`, bit for bit.
        assert!(m6(a).transpose().tr_mul(m6(b)) == m6(expected));
    }
}

#[test]
fn test_mul_vec_oracle() {
    let mut cases = oracle::matrix6_mul_vec_cases();
    assert!(cases.len() >= 8);
    while let Some(case) = cases.pop_front() {
        let (a, v, expected, tol) = *case;
        assert!(tol == 0);
        assert!(m6(a).mul_vec(vt(v)) == vt(expected));
        assert!(m6(a).transpose().tr_mul_vec(vt(v)) == vt(expected));
    }
}

#[test]
fn test_mul_vec_matches_mul_by_a_one_column_matrix() {
    // `m * v` must agree, scalar for scalar, with the first column of `m * diag(v, 0, .., 0)`.
    let mut cases = oracle::matrix6_mul_vec_cases();
    while let Some(case) = cases.pop_front() {
        let (a, v, expected, _) = *case;
        let (vx, vy, vz, vw, va, vb) = v;
        let col = m6(
            [
                [vx, 0, 0, 0, 0, 0], [vy, 0, 0, 0, 0, 0], [vz, 0, 0, 0, 0, 0], [vw, 0, 0, 0, 0, 0],
                [va, 0, 0, 0, 0, 0], [vb, 0, 0, 0, 0, 0],
            ],
        );
        let p = m6(a) * col;
        let e = vt(expected);
        assert!(p.m11.m11 == e.a.x && p.m11.m21 == e.a.y && p.m11.m31 == e.a.z);
        assert!(p.m21.m11 == e.b.x && p.m21.m21 == e.b.y && p.m21.m31 == e.b.z);
    }
}

#[test]
fn test_oracle_emission_is_complete() {
    // Guards against a silently truncated emission: 2 cases per distribution, 4 distributions.
    assert!(oracle::matrix6_mul_cases().len() == 8);
    assert!(oracle::matrix6_mul_vec_cases().len() == 8);
    assert!(oracle::matrix6_trace_cases().len() == 8);
}
