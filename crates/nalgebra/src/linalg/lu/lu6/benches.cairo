//! Gas benchmarks of `Lu6` (`bench_lu6_<op>__<variant>`, net = raw - the `baseline` of the group),
//! and the alternative implementations that lost, kept as evidence together with the tests that
//! show why (AGENTS.md rule 8):
//!
//! - `alt_no_pivot`: the elimination without partial pivoting. Cheaper and shorter, and wrong on a
//! matrix as ordinary as a permuted identity.
//!
//! - `alt_recip`: one reciprocal per pivot instead of one correctly rounded division per output
//! scalar. DEARER for `solve`, where a single right-hand side does not amortise the reciprocal,
//! cheaper for `try_inverse`, where 6 columns share it, and a second rounding per output in both.
//!
//! - `alt_solve_columns`: the inverse as 6 calls to `solve`. Bit-identical, dearer.
//!
//! In-crate part (WP 8.1c): the tests of crate-internal items only; the tests of the public API are
//! in `crates/tests_linalg/src/lu/lu6/benches.cairo`.

use fixed::Fixed;
use nalgebra_testing::black_box;
use simba::scalar::Real;
use crate::base::matrix6::Matrix6;
use crate::base::matrix_test_utils::{m6, v6t};
use crate::base::vector6::Vector6;
use crate::linalg::lu::Perm6;
use super::{Lu6, Lu6InternalTrait, Lu6Trait};

/// The oracle's first `unit` 6x6 case whose factorisation actually swaps rows (all the benchmarks
/// share it, so their inputs exercise the permutation).
fn a_bench() -> Matrix6<Fixed> {
    m6(
        [
            [-2527254097, 4325213708, -1213189640, 4008510819, 2405075077, 466428599],
            [-700750028, -3536112306, 1731465811, 244560054, 2632174365, 1874080084],
            [-4134312449, -1036601848, -444052936, -1797900266, -2730917964, 404854629],
            [-617580144, 580405440, -3475076872, -924247759, -985644408, 938584824],
            [-1076219818, -1905997652, -1595683778, 2067806730, -3245944919, 3717734456],
            [-308280913, -120179493, 2005545225, -1090342151, 932840990, 2332476429],
        ],
    )
}

/// Its right-hand side.
fn b_bench() -> Vector6<Fixed> {
    v6t((3579353502, 7767965432, 6840630971, -6086016248, -4735434050, -2809324573))
}

/// `a_bench()` already factored, so that the benchmarks of the derived operations do not pay for
/// `new`.
fn f_bench() -> Lu6<Fixed> {
    Lu6 {
        lu: m6(
            [
                [-4134312449, -1036601848, -444052936, -1797900266, -2730917964, 404854629],
                [2625460419, 4958875604, -941745570, 5107545028, 4074451472, 218945968],
                [641578630, 636814517, -3269112063, -1412974282, -1181820813, 845644872],
                [1118040539, -1417102436, 2352776611, 4995058989, -543304442, 3221342200],
                [727980405, -2910510776, -1535245700, 3014081670, 5814961885, -4536658],
                [320260371, -37142392, -2667692283, -1538893104, 179474863, 3983829884],
            ],
        ),
        p: Perm6 { p1: 3, p2: 3, p3: 4, p4: 5, p5: 5 },
    }
}

/// The same elimination WITHOUT partial pivoting: no `abs`, no comparison, no row swap, and the
/// identity permutation. Kept as evidence (AGENTS.md rule 8): it is cheaper and shorter, and it is
/// wrong — the pivot of step `k` is whatever sits at `a_kk`, so a matrix as ordinary as a
/// permuted identity factors with a zero pivot, and nothing bounds `|l_ik|`, which is what keeps
/// `U` from growing. Upstream has no unpivoted variant either.
pub fn new_no_pivot(matrix: Matrix6<Fixed>) -> Lu6<Fixed> {
    let mut a11 = matrix.m11;
    let mut a12 = matrix.m12;
    let mut a13 = matrix.m13;
    let mut a14 = matrix.m14;
    let mut a15 = matrix.m15;
    let mut a16 = matrix.m16;
    let mut a21 = matrix.m21;
    let mut a22 = matrix.m22;
    let mut a23 = matrix.m23;
    let mut a24 = matrix.m24;
    let mut a25 = matrix.m25;
    let mut a26 = matrix.m26;
    let mut a31 = matrix.m31;
    let mut a32 = matrix.m32;
    let mut a33 = matrix.m33;
    let mut a34 = matrix.m34;
    let mut a35 = matrix.m35;
    let mut a36 = matrix.m36;
    let mut a41 = matrix.m41;
    let mut a42 = matrix.m42;
    let mut a43 = matrix.m43;
    let mut a44 = matrix.m44;
    let mut a45 = matrix.m45;
    let mut a46 = matrix.m46;
    let mut a51 = matrix.m51;
    let mut a52 = matrix.m52;
    let mut a53 = matrix.m53;
    let mut a54 = matrix.m54;
    let mut a55 = matrix.m55;
    let mut a56 = matrix.m56;
    let mut a61 = matrix.m61;
    let mut a62 = matrix.m62;
    let mut a63 = matrix.m63;
    let mut a64 = matrix.m64;
    let mut a65 = matrix.m65;
    let mut a66 = matrix.m66;
    if a11 != Real::zero() {
        let l = a21 / a11;
        let nl = -l;
        a22 = Real::mul_add(nl, a12, a22);
        a23 = Real::mul_add(nl, a13, a23);
        a24 = Real::mul_add(nl, a14, a24);
        a25 = Real::mul_add(nl, a15, a25);
        a26 = Real::mul_add(nl, a16, a26);
        a21 = l;
        let l = a31 / a11;
        let nl = -l;
        a32 = Real::mul_add(nl, a12, a32);
        a33 = Real::mul_add(nl, a13, a33);
        a34 = Real::mul_add(nl, a14, a34);
        a35 = Real::mul_add(nl, a15, a35);
        a36 = Real::mul_add(nl, a16, a36);
        a31 = l;
        let l = a41 / a11;
        let nl = -l;
        a42 = Real::mul_add(nl, a12, a42);
        a43 = Real::mul_add(nl, a13, a43);
        a44 = Real::mul_add(nl, a14, a44);
        a45 = Real::mul_add(nl, a15, a45);
        a46 = Real::mul_add(nl, a16, a46);
        a41 = l;
        let l = a51 / a11;
        let nl = -l;
        a52 = Real::mul_add(nl, a12, a52);
        a53 = Real::mul_add(nl, a13, a53);
        a54 = Real::mul_add(nl, a14, a54);
        a55 = Real::mul_add(nl, a15, a55);
        a56 = Real::mul_add(nl, a16, a56);
        a51 = l;
        let l = a61 / a11;
        let nl = -l;
        a62 = Real::mul_add(nl, a12, a62);
        a63 = Real::mul_add(nl, a13, a63);
        a64 = Real::mul_add(nl, a14, a64);
        a65 = Real::mul_add(nl, a15, a65);
        a66 = Real::mul_add(nl, a16, a66);
        a61 = l;
    }
    if a22 != Real::zero() {
        let l = a32 / a22;
        let nl = -l;
        a33 = Real::mul_add(nl, a23, a33);
        a34 = Real::mul_add(nl, a24, a34);
        a35 = Real::mul_add(nl, a25, a35);
        a36 = Real::mul_add(nl, a26, a36);
        a32 = l;
        let l = a42 / a22;
        let nl = -l;
        a43 = Real::mul_add(nl, a23, a43);
        a44 = Real::mul_add(nl, a24, a44);
        a45 = Real::mul_add(nl, a25, a45);
        a46 = Real::mul_add(nl, a26, a46);
        a42 = l;
        let l = a52 / a22;
        let nl = -l;
        a53 = Real::mul_add(nl, a23, a53);
        a54 = Real::mul_add(nl, a24, a54);
        a55 = Real::mul_add(nl, a25, a55);
        a56 = Real::mul_add(nl, a26, a56);
        a52 = l;
        let l = a62 / a22;
        let nl = -l;
        a63 = Real::mul_add(nl, a23, a63);
        a64 = Real::mul_add(nl, a24, a64);
        a65 = Real::mul_add(nl, a25, a65);
        a66 = Real::mul_add(nl, a26, a66);
        a62 = l;
    }
    if a33 != Real::zero() {
        let l = a43 / a33;
        let nl = -l;
        a44 = Real::mul_add(nl, a34, a44);
        a45 = Real::mul_add(nl, a35, a45);
        a46 = Real::mul_add(nl, a36, a46);
        a43 = l;
        let l = a53 / a33;
        let nl = -l;
        a54 = Real::mul_add(nl, a34, a54);
        a55 = Real::mul_add(nl, a35, a55);
        a56 = Real::mul_add(nl, a36, a56);
        a53 = l;
        let l = a63 / a33;
        let nl = -l;
        a64 = Real::mul_add(nl, a34, a64);
        a65 = Real::mul_add(nl, a35, a65);
        a66 = Real::mul_add(nl, a36, a66);
        a63 = l;
    }
    if a44 != Real::zero() {
        let l = a54 / a44;
        let nl = -l;
        a55 = Real::mul_add(nl, a45, a55);
        a56 = Real::mul_add(nl, a46, a56);
        a54 = l;
        let l = a64 / a44;
        let nl = -l;
        a65 = Real::mul_add(nl, a45, a65);
        a66 = Real::mul_add(nl, a46, a66);
        a64 = l;
    }
    if a55 != Real::zero() {
        let l = a65 / a55;
        let nl = -l;
        a66 = Real::mul_add(nl, a56, a66);
        a65 = l;
    }
    Lu6 {
        lu: Matrix6 {
            m11: a11,
            m21: a21,
            m31: a31,
            m12: a12,
            m22: a22,
            m32: a32,
            m13: a13,
            m23: a23,
            m33: a33,
            m41: a41,
            m51: a51,
            m61: a61,
            m42: a42,
            m52: a52,
            m62: a62,
            m43: a43,
            m53: a53,
            m63: a63,
            m14: a14,
            m24: a24,
            m34: a34,
            m15: a15,
            m25: a25,
            m35: a35,
            m16: a16,
            m26: a26,
            m36: a36,
            m44: a44,
            m54: a54,
            m64: a64,
            m45: a45,
            m55: a55,
            m65: a65,
            m46: a46,
            m56: a56,
            m66: a66,
        },
        p: Perm6 { p1: 1, p2: 2, p3: 3, p4: 4, p5: 5 },
    }
}

/// `solve` with ONE reciprocal per pivot and 6 multiplications instead of 6 correctly rounded
/// divisions. Kept as evidence, and it loses on both counts: a reciprocal (2 190) plus a
/// multiplication (1 750) is dearer than a division (2 740), and a single right-hand side gives
/// nothing to amortise it over, so it costs 73 460 against 65 020 gas; and rounding `1 / u_ii`
/// before using it puts 1 of the oracle cases outside the tolerance against 0
/// (`test_solve_candidates_error`).
pub fn solve_recip(f: Lu6<Fixed>, b: Vector6<Fixed>) -> Option<Vector6<Fixed>> {
    if !f.is_invertible() {
        return None;
    }
    let r1 = Real::recip(f.lu.m11);
    let r2 = Real::recip(f.lu.m22);
    let r3 = Real::recip(f.lu.m33);
    let r4 = Real::recip(f.lu.m44);
    let r5 = Real::recip(f.lu.m55);
    let r6 = Real::recip(f.lu.m66);
    let pb = f.permute(b);
    let y1 = pb.x;
    let y2 = Real::mul_add(-f.lu.m21, y1, pb.y);
    let y3 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub_prod(Real::wide_add(Real::<Fixed>::wide_zero(), pb.z), f.lu.m31, y1),
            f.lu.m32,
            y2,
        ),
    );
    let y4 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub_prod(
                Real::wide_sub_prod(Real::wide_add(Real::<Fixed>::wide_zero(), pb.w), f.lu.m41, y1),
                f.lu.m42,
                y2,
            ),
            f.lu.m43,
            y3,
        ),
    );
    let y5 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub_prod(
                Real::wide_sub_prod(
                    Real::wide_sub_prod(
                        Real::wide_add(Real::<Fixed>::wide_zero(), pb.a), f.lu.m51, y1,
                    ),
                    f.lu.m52,
                    y2,
                ),
                f.lu.m53,
                y3,
            ),
            f.lu.m54,
            y4,
        ),
    );
    let y6 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub_prod(
                Real::wide_sub_prod(
                    Real::wide_sub_prod(
                        Real::wide_sub_prod(
                            Real::wide_add(Real::<Fixed>::wide_zero(), pb.b), f.lu.m61, y1,
                        ),
                        f.lu.m62,
                        y2,
                    ),
                    f.lu.m63,
                    y3,
                ),
                f.lu.m64,
                y4,
            ),
            f.lu.m65,
            y5,
        ),
    );
    let x6 = y6 * r6;
    let x5 = Real::mul_add(-f.lu.m56, x6, y5) * r5;
    let x4 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub_prod(Real::wide_add(Real::<Fixed>::wide_zero(), y4), f.lu.m45, x5),
            f.lu.m46,
            x6,
        ),
    )
        * r4;
    let x3 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub_prod(
                Real::wide_sub_prod(Real::wide_add(Real::<Fixed>::wide_zero(), y3), f.lu.m34, x4),
                f.lu.m35,
                x5,
            ),
            f.lu.m36,
            x6,
        ),
    )
        * r3;
    let x2 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub_prod(
                Real::wide_sub_prod(
                    Real::wide_sub_prod(
                        Real::wide_add(Real::<Fixed>::wide_zero(), y2), f.lu.m23, x3,
                    ),
                    f.lu.m24,
                    x4,
                ),
                f.lu.m25,
                x5,
            ),
            f.lu.m26,
            x6,
        ),
    )
        * r2;
    let x1 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub_prod(
                Real::wide_sub_prod(
                    Real::wide_sub_prod(
                        Real::wide_sub_prod(
                            Real::wide_add(Real::<Fixed>::wide_zero(), y1), f.lu.m12, x2,
                        ),
                        f.lu.m13,
                        x3,
                    ),
                    f.lu.m14,
                    x4,
                ),
                f.lu.m15,
                x5,
            ),
            f.lu.m16,
            x6,
        ),
    )
        * r1;
    Some(Vector6 { x: x1, y: x2, z: x3, w: x4, a: x5, b: x6 })
}

#[test]
#[inline(never)]
fn bench_lu6_permute__transpositions() {
    let f = black_box(f_bench());
    let b = black_box(b_bench());
    let e = black_box(
        v6t((6840630971, 3579353502, -6086016248, -4735434050, 7767965432, -2809324573)),
    );
    assert!(f.permute(b) == e);
}

#[test]
#[inline(never)]
fn bench_lu6_permute_rows__transpositions() {
    let f = black_box(f_bench());
    let a = black_box(a_bench());
    let e = black_box(
        m6(
            [
                [-4134312449, -1036601848, -444052936, -1797900266, -2730917964, 404854629],
                [-2527254097, 4325213708, -1213189640, 4008510819, 2405075077, 466428599],
                [-617580144, 580405440, -3475076872, -924247759, -985644408, 938584824],
                [-1076219818, -1905997652, -1595683778, 2067806730, -3245944919, 3717734456],
                [-700750028, -3536112306, 1731465811, 244560054, 2632174365, 1874080084],
                [-308280913, -120179493, 2005545225, -1090342151, 932840990, 2332476429],
            ],
        ),
    );
    assert!(f.permute_rows(a) == e);
}

#[test]
#[inline(never)]
fn bench_lu6_solve__alt_recip() {
    let f = black_box(f_bench());
    let b = black_box(b_bench());
    let e = black_box(
        Some(v6t((-11208158818, -7248812507, 3383102673, 4212967580, 4206763122, -9649832809))),
    );
    assert!(solve_recip(f, b) == e);
}
