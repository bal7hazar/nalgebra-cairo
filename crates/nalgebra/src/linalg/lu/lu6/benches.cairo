//! Gas benchmarks of `Lu6` (`bench_lu6_<op>__<variant>`, net = raw - the `baseline` of the group),
//! and the alternative implementations that lost, kept as evidence together with the tests that
//! show why (AGENTS.md rule 8):
//!
//! - `alt_no_pivot`: the elimination without partial pivoting. Cheaper and shorter, and wrong on a
//! matrix as ordinary as a permuted identity.
//!
//! - `alt_recip`: one reciprocal per pivot instead of one truncated division per output scalar.
//! DEARER for `solve`, where a single right-hand side does not amortise the reciprocal, cheaper for
//! `try_inverse`, where 6 columns share it, and a second rounding per output in both.
//!
//! - `alt_solve_columns`: the inverse as 6 calls to `solve`. Bit-identical, dearer.

use fixed::Fixed;
use nalgebra_testing::black_box;
use simba::scalar::Real;
use crate::base::matrix3::Matrix3;
use crate::base::matrix6::Matrix6;
use crate::base::matrix_test_utils::{fx, int, m6, v6t};
use crate::base::vector3::Vector3;
use crate::base::vector6::Vector6;
use crate::linalg::lu::Perm6;
use super::{Lu6, Lu6Trait, Matrix6LuTrait};

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

/// A matrix whose fixed-point factorisation is EXACT: `L` has at most 2 fractional bits, `U`
/// integer entries, so no division and no update rounds. Built by `L * U` then a row shuffle, which
/// partial pivoting undoes exactly because `|l_ik| < 1`.
fn a_exact() -> Matrix6<Fixed> {
    m6(
        [
            [-3221225472, 6442450944, -10737418240, 9663676416, -25769803776, 21474836480],
            [9663676416, -7516192768, -10737418240, 8589934592, -24696061952, 22548578304],
            [-9663676416, 11811160064, 34359738368, -6442450944, 0, 12884901888],
            [-3221225472, 6442450944, 15032385536, -4294967296, -30064771072, 31138512896],
            [-3221225472, 7516192768, 19327352832, 8589934592, -22548578304, 16106127360],
            [-12884901888, 12884901888, 8589934592, -17179869184, -4294967296, 12884901888],
        ],
    )
}

/// An exactly singular integer matrix (one row is an integer combination of the others, and every
/// multiplier of the elimination is dyadic, so the last pivot is exactly zero).
fn a_singular() -> Matrix6<Fixed> {
    m6(
        [
            [4294967296, 0, -25769803776, -25769803776, 21474836480, 25769803776],
            [17179869184, 0, -8589934592, 0, 4294967296, 4294967296],
            [-8589934592, -8589934592, -12884901888, 25769803776, 25769803776, -17179869184],
            [-17179869184, 0, -4294967296, 25769803776, -17179869184, 8589934592],
            [0, -8589934592, 17179869184, 103079215104, -30064771072, -12884901888],
            [12884901888, 0, 17179869184, 25769803776, -21474836480, 17179869184],
        ],
    )
}

/// `a_singular()` already factored (its last pivot is exactly zero), so the benchmark of the
/// rejection path measures the rejection and not `new`.
fn f_singular() -> Lu6<Fixed> {
    Lu6 {
        lu: m6(
            [
                [17179869184, 0, -8589934592, 0, 4294967296, 4294967296],
                [-2147483648, -8589934592, -17179869184, 25769803776, 27917287424, -15032385536],
                [0, 4294967296, 34359738368, 77309411328, -57982058496, 2147483648],
                [-4294967296, 0, -1610612736, 54760833024, -34628173824, 13690208256],
                [1073741824, 0, -2952790016, 2147483648, -2147483648, 19327352832],
                [3221225472, 0, 2952790016, -2147483648, 4294967296, 0],
            ],
        ),
        p: Perm6 { p1: 2, p2: 3, p3: 5, p4: 4, p5: 5 },
    }
}

/// The same elimination WITHOUT partial pivoting: no `abs`, no comparison, no row swap, and the
/// identity permutation. Kept as evidence (AGENTS.md rule 8): it is cheaper and shorter, and it is
/// wrong — the pivot of step `k` is whatever sits at `a_kk`, so a matrix as ordinary as a
/// permuted identity factors with a zero pivot, and nothing bounds `|l_ik|`, which is what keeps
/// `U` from growing. Upstream has no unpivoted variant either.
pub fn new_no_pivot(matrix: Matrix6<Fixed>) -> Lu6<Fixed> {
    let mut a11 = matrix.m11.m11;
    let mut a12 = matrix.m11.m12;
    let mut a13 = matrix.m11.m13;
    let mut a14 = matrix.m12.m11;
    let mut a15 = matrix.m12.m12;
    let mut a16 = matrix.m12.m13;
    let mut a21 = matrix.m11.m21;
    let mut a22 = matrix.m11.m22;
    let mut a23 = matrix.m11.m23;
    let mut a24 = matrix.m12.m21;
    let mut a25 = matrix.m12.m22;
    let mut a26 = matrix.m12.m23;
    let mut a31 = matrix.m11.m31;
    let mut a32 = matrix.m11.m32;
    let mut a33 = matrix.m11.m33;
    let mut a34 = matrix.m12.m31;
    let mut a35 = matrix.m12.m32;
    let mut a36 = matrix.m12.m33;
    let mut a41 = matrix.m21.m11;
    let mut a42 = matrix.m21.m12;
    let mut a43 = matrix.m21.m13;
    let mut a44 = matrix.m22.m11;
    let mut a45 = matrix.m22.m12;
    let mut a46 = matrix.m22.m13;
    let mut a51 = matrix.m21.m21;
    let mut a52 = matrix.m21.m22;
    let mut a53 = matrix.m21.m23;
    let mut a54 = matrix.m22.m21;
    let mut a55 = matrix.m22.m22;
    let mut a56 = matrix.m22.m23;
    let mut a61 = matrix.m21.m31;
    let mut a62 = matrix.m21.m32;
    let mut a63 = matrix.m21.m33;
    let mut a64 = matrix.m22.m31;
    let mut a65 = matrix.m22.m32;
    let mut a66 = matrix.m22.m33;
    if a11 != Real::ZERO {
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
    if a22 != Real::ZERO {
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
    if a33 != Real::ZERO {
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
    if a44 != Real::ZERO {
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
    if a55 != Real::ZERO {
        let l = a65 / a55;
        let nl = -l;
        a66 = Real::mul_add(nl, a56, a66);
        a65 = l;
    }
    Lu6 {
        lu: Matrix6 {
            m11: Matrix3 {
                m11: a11,
                m21: a21,
                m31: a31,
                m12: a12,
                m22: a22,
                m32: a32,
                m13: a13,
                m23: a23,
                m33: a33,
            },
            m21: Matrix3 {
                m11: a41,
                m21: a51,
                m31: a61,
                m12: a42,
                m22: a52,
                m32: a62,
                m13: a43,
                m23: a53,
                m33: a63,
            },
            m12: Matrix3 {
                m11: a14,
                m21: a24,
                m31: a34,
                m12: a15,
                m22: a25,
                m32: a35,
                m13: a16,
                m23: a26,
                m33: a36,
            },
            m22: Matrix3 {
                m11: a44,
                m21: a54,
                m31: a64,
                m12: a45,
                m22: a55,
                m32: a65,
                m13: a46,
                m23: a56,
                m33: a66,
            },
        },
        p: Perm6 { p1: 1, p2: 2, p3: 3, p4: 4, p5: 5 },
    }
}

/// `solve` with ONE reciprocal per pivot and 6 multiplications instead of 6 truncated divisions.
/// Kept as evidence, and it loses on both counts: a reciprocal (2 190) plus a multiplication (1
/// 750) is dearer than a division (2 740), and a single right-hand side gives nothing to amortise
/// it over, so it costs 73 460 against 65 020 gas; and rounding `1 / u_ii` before using it puts 1
/// of the oracle cases outside the tolerance against 0 (`test_solve_candidates_error`).
pub fn solve_recip(f: Lu6<Fixed>, b: Vector6<Fixed>) -> Option<Vector6<Fixed>> {
    if !f.is_invertible() {
        return None;
    }
    let r1 = Real::recip(f.lu.m11.m11);
    let r2 = Real::recip(f.lu.m11.m22);
    let r3 = Real::recip(f.lu.m11.m33);
    let r4 = Real::recip(f.lu.m22.m11);
    let r5 = Real::recip(f.lu.m22.m22);
    let r6 = Real::recip(f.lu.m22.m33);
    let pb = f.permute(b);
    let y1 = pb.a.x;
    let y2 = Real::mul_add(-f.lu.m11.m21, y1, pb.a.y);
    let y3 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub_prod(
                Real::wide_add(Real::<Fixed>::wide_zero(), pb.a.z), f.lu.m11.m31, y1,
            ),
            f.lu.m11.m32,
            y2,
        ),
    );
    let y4 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub_prod(
                Real::wide_sub_prod(
                    Real::wide_add(Real::<Fixed>::wide_zero(), pb.b.x), f.lu.m21.m11, y1,
                ),
                f.lu.m21.m12,
                y2,
            ),
            f.lu.m21.m13,
            y3,
        ),
    );
    let y5 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub_prod(
                Real::wide_sub_prod(
                    Real::wide_sub_prod(
                        Real::wide_add(Real::<Fixed>::wide_zero(), pb.b.y), f.lu.m21.m21, y1,
                    ),
                    f.lu.m21.m22,
                    y2,
                ),
                f.lu.m21.m23,
                y3,
            ),
            f.lu.m22.m21,
            y4,
        ),
    );
    let y6 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub_prod(
                Real::wide_sub_prod(
                    Real::wide_sub_prod(
                        Real::wide_sub_prod(
                            Real::wide_add(Real::<Fixed>::wide_zero(), pb.b.z), f.lu.m21.m31, y1,
                        ),
                        f.lu.m21.m32,
                        y2,
                    ),
                    f.lu.m21.m33,
                    y3,
                ),
                f.lu.m22.m31,
                y4,
            ),
            f.lu.m22.m32,
            y5,
        ),
    );
    let x6 = y6 * r6;
    let x5 = Real::mul_add(-f.lu.m22.m23, x6, y5) * r5;
    let x4 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub_prod(Real::wide_add(Real::<Fixed>::wide_zero(), y4), f.lu.m22.m12, x5),
            f.lu.m22.m13,
            x6,
        ),
    )
        * r4;
    let x3 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub_prod(
                Real::wide_sub_prod(
                    Real::wide_add(Real::<Fixed>::wide_zero(), y3), f.lu.m12.m31, x4,
                ),
                f.lu.m12.m32,
                x5,
            ),
            f.lu.m12.m33,
            x6,
        ),
    )
        * r3;
    let x2 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub_prod(
                Real::wide_sub_prod(
                    Real::wide_sub_prod(
                        Real::wide_add(Real::<Fixed>::wide_zero(), y2), f.lu.m11.m23, x3,
                    ),
                    f.lu.m12.m21,
                    x4,
                ),
                f.lu.m12.m22,
                x5,
            ),
            f.lu.m12.m23,
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
                            Real::wide_add(Real::<Fixed>::wide_zero(), y1), f.lu.m11.m12, x2,
                        ),
                        f.lu.m11.m13,
                        x3,
                    ),
                    f.lu.m12.m11,
                    x4,
                ),
                f.lu.m12.m12,
                x5,
            ),
            f.lu.m12.m13,
            x6,
        ),
    )
        * r1;
    Some(Vector6 { a: Vector3 { x: x1, y: x2, z: x3 }, b: Vector3 { x: x4, y: x5, z: x6 } })
}

/// `try_inverse` as 6 full calls to `solve` on the unit vectors — the obvious route, and the one
/// upstream takes. Kept as evidence: it gives BIT-IDENTICAL results
/// (`test_try_inverse_candidates_agree`) for strictly more gas, because it pays the permutation 6
/// times and multiplies by the leading zeros of each unit vector.
pub fn try_inverse_solve_columns(f: Lu6<Fixed>) -> Option<Matrix6<Fixed>> {
    if !f.is_invertible() {
        return None;
    }
    let c1 = f
        .solve(
            Vector6 {
                a: Vector3 { x: int(1), y: int(0), z: int(0) },
                b: Vector3 { x: int(0), y: int(0), z: int(0) },
            },
        )
        .unwrap();
    let c2 = f
        .solve(
            Vector6 {
                a: Vector3 { x: int(0), y: int(1), z: int(0) },
                b: Vector3 { x: int(0), y: int(0), z: int(0) },
            },
        )
        .unwrap();
    let c3 = f
        .solve(
            Vector6 {
                a: Vector3 { x: int(0), y: int(0), z: int(1) },
                b: Vector3 { x: int(0), y: int(0), z: int(0) },
            },
        )
        .unwrap();
    let c4 = f
        .solve(
            Vector6 {
                a: Vector3 { x: int(0), y: int(0), z: int(0) },
                b: Vector3 { x: int(1), y: int(0), z: int(0) },
            },
        )
        .unwrap();
    let c5 = f
        .solve(
            Vector6 {
                a: Vector3 { x: int(0), y: int(0), z: int(0) },
                b: Vector3 { x: int(0), y: int(1), z: int(0) },
            },
        )
        .unwrap();
    let c6 = f
        .solve(
            Vector6 {
                a: Vector3 { x: int(0), y: int(0), z: int(0) },
                b: Vector3 { x: int(0), y: int(0), z: int(1) },
            },
        )
        .unwrap();
    Some(
        Matrix6 {
            m11: Matrix3 {
                m11: c1.a.x,
                m21: c1.a.y,
                m31: c1.a.z,
                m12: c2.a.x,
                m22: c2.a.y,
                m32: c2.a.z,
                m13: c3.a.x,
                m23: c3.a.y,
                m33: c3.a.z,
            },
            m21: Matrix3 {
                m11: c1.b.x,
                m21: c1.b.y,
                m31: c1.b.z,
                m12: c2.b.x,
                m22: c2.b.y,
                m32: c2.b.z,
                m13: c3.b.x,
                m23: c3.b.y,
                m33: c3.b.z,
            },
            m12: Matrix3 {
                m11: c4.a.x,
                m21: c4.a.y,
                m31: c4.a.z,
                m12: c5.a.x,
                m22: c5.a.y,
                m32: c5.a.z,
                m13: c6.a.x,
                m23: c6.a.y,
                m33: c6.a.z,
            },
            m22: Matrix3 {
                m11: c4.b.x,
                m21: c4.b.y,
                m31: c4.b.z,
                m12: c5.b.x,
                m22: c5.b.y,
                m32: c5.b.z,
                m13: c6.b.x,
                m23: c6.b.y,
                m33: c6.b.z,
            },
        },
    )
}

/// `try_inverse` with ONE reciprocal per pivot instead of one division per entry of the back
/// substitution. Here the reciprocal IS amortised (6 columns share it), so this is the candidate
/// the gas argument favours; it is not shipped because it rounds `1 / u_ii` before using it, which
/// is the second rounding per output scalar that AGENTS.md rule 4 and DESIGN D2 forbid (the same
/// call `Vector6::unscale` makes).
pub fn try_inverse_recip(f: Lu6<Fixed>) -> Option<Matrix6<Fixed>> {
    if !f.is_invertible() {
        return None;
    }
    let r1 = Real::recip(f.lu.m11.m11);
    let r2 = Real::recip(f.lu.m11.m22);
    let r3 = Real::recip(f.lu.m11.m33);
    let r4 = Real::recip(f.lu.m22.m11);
    let r5 = Real::recip(f.lu.m22.m22);
    let r6 = Real::recip(f.lu.m22.m33);
    let y21 = -f.lu.m11.m21;
    let y31 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub(Real::<Fixed>::wide_zero(), f.lu.m11.m31), f.lu.m11.m32, y21,
        ),
    );
    let y41 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub_prod(
                Real::wide_sub(Real::<Fixed>::wide_zero(), f.lu.m21.m11), f.lu.m21.m12, y21,
            ),
            f.lu.m21.m13,
            y31,
        ),
    );
    let y51 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub_prod(
                Real::wide_sub_prod(
                    Real::wide_sub(Real::<Fixed>::wide_zero(), f.lu.m21.m21), f.lu.m21.m22, y21,
                ),
                f.lu.m21.m23,
                y31,
            ),
            f.lu.m22.m21,
            y41,
        ),
    );
    let y61 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub_prod(
                Real::wide_sub_prod(
                    Real::wide_sub_prod(
                        Real::wide_sub(Real::<Fixed>::wide_zero(), f.lu.m21.m31), f.lu.m21.m32, y21,
                    ),
                    f.lu.m21.m33,
                    y31,
                ),
                f.lu.m22.m31,
                y41,
            ),
            f.lu.m22.m32,
            y51,
        ),
    );
    let x61 = y61 * r6;
    let x51 = Real::mul_add(-f.lu.m22.m23, x61, y51) * r5;
    let x41 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub_prod(Real::wide_add(Real::<Fixed>::wide_zero(), y41), f.lu.m22.m12, x51),
            f.lu.m22.m13,
            x61,
        ),
    )
        * r4;
    let x31 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub_prod(
                Real::wide_sub_prod(
                    Real::wide_add(Real::<Fixed>::wide_zero(), y31), f.lu.m12.m31, x41,
                ),
                f.lu.m12.m32,
                x51,
            ),
            f.lu.m12.m33,
            x61,
        ),
    )
        * r3;
    let x21 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub_prod(
                Real::wide_sub_prod(
                    Real::wide_sub_prod(
                        Real::wide_add(Real::<Fixed>::wide_zero(), y21), f.lu.m11.m23, x31,
                    ),
                    f.lu.m12.m21,
                    x41,
                ),
                f.lu.m12.m22,
                x51,
            ),
            f.lu.m12.m23,
            x61,
        ),
    )
        * r2;
    let x11 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub_prod(
                Real::wide_sub_prod(
                    Real::wide_sub_prod(
                        Real::wide_sub_prod(
                            Real::wide_add(Real::<Fixed>::wide_zero(), int(1)), f.lu.m11.m12, x21,
                        ),
                        f.lu.m11.m13,
                        x31,
                    ),
                    f.lu.m12.m11,
                    x41,
                ),
                f.lu.m12.m12,
                x51,
            ),
            f.lu.m12.m13,
            x61,
        ),
    )
        * r1;
    let y32 = -f.lu.m11.m32;
    let y42 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub(Real::<Fixed>::wide_zero(), f.lu.m21.m12), f.lu.m21.m13, y32,
        ),
    );
    let y52 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub_prod(
                Real::wide_sub(Real::<Fixed>::wide_zero(), f.lu.m21.m22), f.lu.m21.m23, y32,
            ),
            f.lu.m22.m21,
            y42,
        ),
    );
    let y62 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub_prod(
                Real::wide_sub_prod(
                    Real::wide_sub(Real::<Fixed>::wide_zero(), f.lu.m21.m32), f.lu.m21.m33, y32,
                ),
                f.lu.m22.m31,
                y42,
            ),
            f.lu.m22.m32,
            y52,
        ),
    );
    let x62 = y62 * r6;
    let x52 = Real::mul_add(-f.lu.m22.m23, x62, y52) * r5;
    let x42 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub_prod(Real::wide_add(Real::<Fixed>::wide_zero(), y42), f.lu.m22.m12, x52),
            f.lu.m22.m13,
            x62,
        ),
    )
        * r4;
    let x32 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub_prod(
                Real::wide_sub_prod(
                    Real::wide_add(Real::<Fixed>::wide_zero(), y32), f.lu.m12.m31, x42,
                ),
                f.lu.m12.m32,
                x52,
            ),
            f.lu.m12.m33,
            x62,
        ),
    )
        * r3;
    let x22 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub_prod(
                Real::wide_sub_prod(
                    Real::wide_sub_prod(
                        Real::wide_add(Real::<Fixed>::wide_zero(), int(1)), f.lu.m11.m23, x32,
                    ),
                    f.lu.m12.m21,
                    x42,
                ),
                f.lu.m12.m22,
                x52,
            ),
            f.lu.m12.m23,
            x62,
        ),
    )
        * r2;
    let x12 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub_prod(
                Real::wide_sub_prod(
                    Real::wide_sub_prod(
                        Real::wide_sub_prod(Real::<Fixed>::wide_zero(), f.lu.m11.m12, x22),
                        f.lu.m11.m13,
                        x32,
                    ),
                    f.lu.m12.m11,
                    x42,
                ),
                f.lu.m12.m12,
                x52,
            ),
            f.lu.m12.m13,
            x62,
        ),
    )
        * r1;
    let y43 = -f.lu.m21.m13;
    let y53 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub(Real::<Fixed>::wide_zero(), f.lu.m21.m23), f.lu.m22.m21, y43,
        ),
    );
    let y63 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub_prod(
                Real::wide_sub(Real::<Fixed>::wide_zero(), f.lu.m21.m33), f.lu.m22.m31, y43,
            ),
            f.lu.m22.m32,
            y53,
        ),
    );
    let x63 = y63 * r6;
    let x53 = Real::mul_add(-f.lu.m22.m23, x63, y53) * r5;
    let x43 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub_prod(Real::wide_add(Real::<Fixed>::wide_zero(), y43), f.lu.m22.m12, x53),
            f.lu.m22.m13,
            x63,
        ),
    )
        * r4;
    let x33 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub_prod(
                Real::wide_sub_prod(
                    Real::wide_add(Real::<Fixed>::wide_zero(), int(1)), f.lu.m12.m31, x43,
                ),
                f.lu.m12.m32,
                x53,
            ),
            f.lu.m12.m33,
            x63,
        ),
    )
        * r3;
    let x23 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub_prod(
                Real::wide_sub_prod(
                    Real::wide_sub_prod(Real::<Fixed>::wide_zero(), f.lu.m11.m23, x33),
                    f.lu.m12.m21,
                    x43,
                ),
                f.lu.m12.m22,
                x53,
            ),
            f.lu.m12.m23,
            x63,
        ),
    )
        * r2;
    let x13 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub_prod(
                Real::wide_sub_prod(
                    Real::wide_sub_prod(
                        Real::wide_sub_prod(Real::<Fixed>::wide_zero(), f.lu.m11.m12, x23),
                        f.lu.m11.m13,
                        x33,
                    ),
                    f.lu.m12.m11,
                    x43,
                ),
                f.lu.m12.m12,
                x53,
            ),
            f.lu.m12.m13,
            x63,
        ),
    )
        * r1;
    let y54 = -f.lu.m22.m21;
    let y64 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub(Real::<Fixed>::wide_zero(), f.lu.m22.m31), f.lu.m22.m32, y54,
        ),
    );
    let x64 = y64 * r6;
    let x54 = Real::mul_add(-f.lu.m22.m23, x64, y54) * r5;
    let x44 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub_prod(
                Real::wide_add(Real::<Fixed>::wide_zero(), int(1)), f.lu.m22.m12, x54,
            ),
            f.lu.m22.m13,
            x64,
        ),
    )
        * r4;
    let x34 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub_prod(
                Real::wide_sub_prod(Real::<Fixed>::wide_zero(), f.lu.m12.m31, x44),
                f.lu.m12.m32,
                x54,
            ),
            f.lu.m12.m33,
            x64,
        ),
    )
        * r3;
    let x24 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub_prod(
                Real::wide_sub_prod(
                    Real::wide_sub_prod(Real::<Fixed>::wide_zero(), f.lu.m11.m23, x34),
                    f.lu.m12.m21,
                    x44,
                ),
                f.lu.m12.m22,
                x54,
            ),
            f.lu.m12.m23,
            x64,
        ),
    )
        * r2;
    let x14 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub_prod(
                Real::wide_sub_prod(
                    Real::wide_sub_prod(
                        Real::wide_sub_prod(Real::<Fixed>::wide_zero(), f.lu.m11.m12, x24),
                        f.lu.m11.m13,
                        x34,
                    ),
                    f.lu.m12.m11,
                    x44,
                ),
                f.lu.m12.m12,
                x54,
            ),
            f.lu.m12.m13,
            x64,
        ),
    )
        * r1;
    let y65 = -f.lu.m22.m32;
    let x65 = y65 * r6;
    let x55 = Real::mul_add(-f.lu.m22.m23, x65, int(1)) * r5;
    let x45 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub_prod(Real::<Fixed>::wide_zero(), f.lu.m22.m12, x55), f.lu.m22.m13, x65,
        ),
    )
        * r4;
    let x35 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub_prod(
                Real::wide_sub_prod(Real::<Fixed>::wide_zero(), f.lu.m12.m31, x45),
                f.lu.m12.m32,
                x55,
            ),
            f.lu.m12.m33,
            x65,
        ),
    )
        * r3;
    let x25 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub_prod(
                Real::wide_sub_prod(
                    Real::wide_sub_prod(Real::<Fixed>::wide_zero(), f.lu.m11.m23, x35),
                    f.lu.m12.m21,
                    x45,
                ),
                f.lu.m12.m22,
                x55,
            ),
            f.lu.m12.m23,
            x65,
        ),
    )
        * r2;
    let x15 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub_prod(
                Real::wide_sub_prod(
                    Real::wide_sub_prod(
                        Real::wide_sub_prod(Real::<Fixed>::wide_zero(), f.lu.m11.m12, x25),
                        f.lu.m11.m13,
                        x35,
                    ),
                    f.lu.m12.m11,
                    x45,
                ),
                f.lu.m12.m12,
                x55,
            ),
            f.lu.m12.m13,
            x65,
        ),
    )
        * r1;
    let x66 = int(1) * r6;
    let x56 = Real::wide_rescale(Real::wide_sub_prod(Real::<Fixed>::wide_zero(), f.lu.m22.m23, x66))
        * r5;
    let x46 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub_prod(Real::<Fixed>::wide_zero(), f.lu.m22.m12, x56), f.lu.m22.m13, x66,
        ),
    )
        * r4;
    let x36 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub_prod(
                Real::wide_sub_prod(Real::<Fixed>::wide_zero(), f.lu.m12.m31, x46),
                f.lu.m12.m32,
                x56,
            ),
            f.lu.m12.m33,
            x66,
        ),
    )
        * r3;
    let x26 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub_prod(
                Real::wide_sub_prod(
                    Real::wide_sub_prod(Real::<Fixed>::wide_zero(), f.lu.m11.m23, x36),
                    f.lu.m12.m21,
                    x46,
                ),
                f.lu.m12.m22,
                x56,
            ),
            f.lu.m12.m23,
            x66,
        ),
    )
        * r2;
    let x16 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub_prod(
                Real::wide_sub_prod(
                    Real::wide_sub_prod(
                        Real::wide_sub_prod(Real::<Fixed>::wide_zero(), f.lu.m11.m12, x26),
                        f.lu.m11.m13,
                        x36,
                    ),
                    f.lu.m12.m11,
                    x46,
                ),
                f.lu.m12.m12,
                x56,
            ),
            f.lu.m12.m13,
            x66,
        ),
    )
        * r1;
    let mut c11 = x11;
    let mut c12 = x12;
    let mut c13 = x13;
    let mut c14 = x14;
    let mut c15 = x15;
    let mut c16 = x16;
    let mut c21 = x21;
    let mut c22 = x22;
    let mut c23 = x23;
    let mut c24 = x24;
    let mut c25 = x25;
    let mut c26 = x26;
    let mut c31 = x31;
    let mut c32 = x32;
    let mut c33 = x33;
    let mut c34 = x34;
    let mut c35 = x35;
    let mut c36 = x36;
    let mut c41 = x41;
    let mut c42 = x42;
    let mut c43 = x43;
    let mut c44 = x44;
    let mut c45 = x45;
    let mut c46 = x46;
    let mut c51 = x51;
    let mut c52 = x52;
    let mut c53 = x53;
    let mut c54 = x54;
    let mut c55 = x55;
    let mut c56 = x56;
    let mut c61 = x61;
    let mut c62 = x62;
    let mut c63 = x63;
    let mut c64 = x64;
    let mut c65 = x65;
    let mut c66 = x66;
    if f.p.p5 == 6 {
        let t = c15;
        c15 = c16;
        c16 = t;
        let t = c25;
        c25 = c26;
        c26 = t;
        let t = c35;
        c35 = c36;
        c36 = t;
        let t = c45;
        c45 = c46;
        c46 = t;
        let t = c55;
        c55 = c56;
        c56 = t;
        let t = c65;
        c65 = c66;
        c66 = t;
    }
    if f.p.p4 == 5 {
        let t = c14;
        c14 = c15;
        c15 = t;
        let t = c24;
        c24 = c25;
        c25 = t;
        let t = c34;
        c34 = c35;
        c35 = t;
        let t = c44;
        c44 = c45;
        c45 = t;
        let t = c54;
        c54 = c55;
        c55 = t;
        let t = c64;
        c64 = c65;
        c65 = t;
    } else if f.p.p4 == 6 {
        let t = c14;
        c14 = c16;
        c16 = t;
        let t = c24;
        c24 = c26;
        c26 = t;
        let t = c34;
        c34 = c36;
        c36 = t;
        let t = c44;
        c44 = c46;
        c46 = t;
        let t = c54;
        c54 = c56;
        c56 = t;
        let t = c64;
        c64 = c66;
        c66 = t;
    }
    if f.p.p3 == 4 {
        let t = c13;
        c13 = c14;
        c14 = t;
        let t = c23;
        c23 = c24;
        c24 = t;
        let t = c33;
        c33 = c34;
        c34 = t;
        let t = c43;
        c43 = c44;
        c44 = t;
        let t = c53;
        c53 = c54;
        c54 = t;
        let t = c63;
        c63 = c64;
        c64 = t;
    } else if f.p.p3 == 5 {
        let t = c13;
        c13 = c15;
        c15 = t;
        let t = c23;
        c23 = c25;
        c25 = t;
        let t = c33;
        c33 = c35;
        c35 = t;
        let t = c43;
        c43 = c45;
        c45 = t;
        let t = c53;
        c53 = c55;
        c55 = t;
        let t = c63;
        c63 = c65;
        c65 = t;
    } else if f.p.p3 == 6 {
        let t = c13;
        c13 = c16;
        c16 = t;
        let t = c23;
        c23 = c26;
        c26 = t;
        let t = c33;
        c33 = c36;
        c36 = t;
        let t = c43;
        c43 = c46;
        c46 = t;
        let t = c53;
        c53 = c56;
        c56 = t;
        let t = c63;
        c63 = c66;
        c66 = t;
    }
    if f.p.p2 == 3 {
        let t = c12;
        c12 = c13;
        c13 = t;
        let t = c22;
        c22 = c23;
        c23 = t;
        let t = c32;
        c32 = c33;
        c33 = t;
        let t = c42;
        c42 = c43;
        c43 = t;
        let t = c52;
        c52 = c53;
        c53 = t;
        let t = c62;
        c62 = c63;
        c63 = t;
    } else if f.p.p2 == 4 {
        let t = c12;
        c12 = c14;
        c14 = t;
        let t = c22;
        c22 = c24;
        c24 = t;
        let t = c32;
        c32 = c34;
        c34 = t;
        let t = c42;
        c42 = c44;
        c44 = t;
        let t = c52;
        c52 = c54;
        c54 = t;
        let t = c62;
        c62 = c64;
        c64 = t;
    } else if f.p.p2 == 5 {
        let t = c12;
        c12 = c15;
        c15 = t;
        let t = c22;
        c22 = c25;
        c25 = t;
        let t = c32;
        c32 = c35;
        c35 = t;
        let t = c42;
        c42 = c45;
        c45 = t;
        let t = c52;
        c52 = c55;
        c55 = t;
        let t = c62;
        c62 = c65;
        c65 = t;
    } else if f.p.p2 == 6 {
        let t = c12;
        c12 = c16;
        c16 = t;
        let t = c22;
        c22 = c26;
        c26 = t;
        let t = c32;
        c32 = c36;
        c36 = t;
        let t = c42;
        c42 = c46;
        c46 = t;
        let t = c52;
        c52 = c56;
        c56 = t;
        let t = c62;
        c62 = c66;
        c66 = t;
    }
    if f.p.p1 == 2 {
        let t = c11;
        c11 = c12;
        c12 = t;
        let t = c21;
        c21 = c22;
        c22 = t;
        let t = c31;
        c31 = c32;
        c32 = t;
        let t = c41;
        c41 = c42;
        c42 = t;
        let t = c51;
        c51 = c52;
        c52 = t;
        let t = c61;
        c61 = c62;
        c62 = t;
    } else if f.p.p1 == 3 {
        let t = c11;
        c11 = c13;
        c13 = t;
        let t = c21;
        c21 = c23;
        c23 = t;
        let t = c31;
        c31 = c33;
        c33 = t;
        let t = c41;
        c41 = c43;
        c43 = t;
        let t = c51;
        c51 = c53;
        c53 = t;
        let t = c61;
        c61 = c63;
        c63 = t;
    } else if f.p.p1 == 4 {
        let t = c11;
        c11 = c14;
        c14 = t;
        let t = c21;
        c21 = c24;
        c24 = t;
        let t = c31;
        c31 = c34;
        c34 = t;
        let t = c41;
        c41 = c44;
        c44 = t;
        let t = c51;
        c51 = c54;
        c54 = t;
        let t = c61;
        c61 = c64;
        c64 = t;
    } else if f.p.p1 == 5 {
        let t = c11;
        c11 = c15;
        c15 = t;
        let t = c21;
        c21 = c25;
        c25 = t;
        let t = c31;
        c31 = c35;
        c35 = t;
        let t = c41;
        c41 = c45;
        c45 = t;
        let t = c51;
        c51 = c55;
        c55 = t;
        let t = c61;
        c61 = c65;
        c65 = t;
    } else if f.p.p1 == 6 {
        let t = c11;
        c11 = c16;
        c16 = t;
        let t = c21;
        c21 = c26;
        c26 = t;
        let t = c31;
        c31 = c36;
        c36 = t;
        let t = c41;
        c41 = c46;
        c46 = t;
        let t = c51;
        c51 = c56;
        c56 = t;
        let t = c61;
        c61 = c66;
        c66 = t;
    }
    Some(
        Matrix6 {
            m11: Matrix3 {
                m11: c11,
                m21: c21,
                m31: c31,
                m12: c12,
                m22: c22,
                m32: c32,
                m13: c13,
                m23: c23,
                m33: c33,
            },
            m21: Matrix3 {
                m11: c41,
                m21: c51,
                m31: c61,
                m12: c42,
                m22: c52,
                m32: c62,
                m13: c43,
                m23: c53,
                m33: c63,
            },
            m12: Matrix3 {
                m11: c14,
                m21: c24,
                m31: c34,
                m12: c15,
                m22: c25,
                m32: c35,
                m13: c16,
                m23: c26,
                m33: c36,
            },
            m22: Matrix3 {
                m11: c44,
                m21: c54,
                m31: c64,
                m12: c45,
                m22: c55,
                m32: c65,
                m13: c46,
                m23: c56,
                m33: c66,
            },
        },
    )
}

#[test]
#[inline(never)]
fn bench_lu6_new__baseline() {
    let _a = black_box(a_bench());
    let e = black_box(
        m6(
            [
                [-4134312449, -1036601848, -444052936, -1797900266, -2730917964, 404854629],
                [2625460419, 4958875604, -941745570, 5107545028, 4074451472, 218945968],
                [641578630, 636814517, -3269112063, -1412974282, -1181820813, 845644872],
                [1118040539, -1417102436, 2352776611, 4995058989, -543304442, 3221342200],
                [727980405, -2910510776, -1535245700, 3014081670, 5814961885, -4536658],
                [320260371, -37142392, -2667692283, -1538893104, 179474863, 3983829884],
            ],
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_lu6_new__pivot() {
    let a = black_box(a_bench());
    let e = black_box(
        m6(
            [
                [-4134312449, -1036601848, -444052936, -1797900266, -2730917964, 404854629],
                [2625460419, 4958875604, -941745570, 5107545028, 4074451472, 218945968],
                [641578630, 636814517, -3269112063, -1412974282, -1181820813, 845644872],
                [1118040539, -1417102436, 2352776611, 4995058989, -543304442, 3221342200],
                [727980405, -2910510776, -1535245700, 3014081670, 5814961885, -4536658],
                [320260371, -37142392, -2667692283, -1538893104, 179474863, 3983829884],
            ],
        ),
    );
    assert!(Lu6Trait::new(a).lu == e);
}

#[test]
#[inline(never)]
fn bench_lu6_new__alt_no_pivot() {
    let a = black_box(a_bench());
    let e = black_box(
        m6(
            [
                [-2527254097, 4325213708, -1213189640, 4008510819, 2405075077, 466428599],
                [1190896655, -4735395603, 2067855679, -866908742, 1965301798, 1744750051],
                [7026098713, 7357685064, -2001839112, -6870290742, -10032106641, -3347093562],
                [1049552763, 432216817, 7266216798, 9806583874, 15201154539, 6311630561],
                [1828992552, 3399289645, 5826509812, 4540451266, -8286129527, 6464586],
                [523911086, 587531568, -4013521424, -3451539334, -1664843283, 3983829886],
            ],
        ),
    );
    assert!(new_no_pivot(a).lu == e);
}

#[test]
#[inline(never)]
fn bench_lu6_factors__baseline() {
    let _f = black_box(f_bench());
    let e = black_box(
        m6(
            [
                [4294967296, 0, 0, 0, 0, 0], [2625460419, 4294967296, 0, 0, 0, 0],
                [641578630, 636814517, 4294967296, 0, 0, 0],
                [1118040539, -1417102436, 2352776611, 4294967296, 0, 0],
                [727980405, -2910510776, -1535245700, 3014081670, 4294967296, 0],
                [320260371, -37142392, -2667692283, -1538893104, 179474863, 4294967296],
            ],
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_lu6_factors__l() {
    let f = black_box(f_bench());
    let e = black_box(
        m6(
            [
                [4294967296, 0, 0, 0, 0, 0], [2625460419, 4294967296, 0, 0, 0, 0],
                [641578630, 636814517, 4294967296, 0, 0, 0],
                [1118040539, -1417102436, 2352776611, 4294967296, 0, 0],
                [727980405, -2910510776, -1535245700, 3014081670, 4294967296, 0],
                [320260371, -37142392, -2667692283, -1538893104, 179474863, 4294967296],
            ],
        ),
    );
    assert!(f.l() == e);
}

#[test]
#[inline(never)]
fn bench_lu6_factors__u() {
    let f = black_box(f_bench());
    let e = black_box(
        m6(
            [
                [-4134312449, -1036601848, -444052936, -1797900266, -2730917964, 404854629],
                [0, 4958875604, -941745570, 5107545028, 4074451472, 218945968],
                [0, 0, -3269112063, -1412974282, -1181820813, 845644872],
                [0, 0, 0, 4995058989, -543304442, 3221342200], [0, 0, 0, 0, 5814961885, -4536658],
                [0, 0, 0, 0, 0, 3983829884],
            ],
        ),
    );
    assert!(f.u() == e);
}

#[test]
#[inline(never)]
fn bench_lu6_p__baseline() {
    let _f = black_box(f_bench());
    let e = black_box(Perm6 { p1: 3, p2: 3, p3: 4, p4: 5, p5: 5 });
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_lu6_p__field() {
    let f = black_box(f_bench());
    let e = black_box(Perm6 { p1: 3, p2: 3, p3: 4, p4: 5, p5: 5 });
    assert!(f.p() == e);
}

#[test]
#[inline(never)]
fn bench_lu6_permute__baseline() {
    let _f = black_box(f_bench());
    let _b = black_box(b_bench());
    let e = black_box(
        v6t((6840630971, 3579353502, -6086016248, -4735434050, 7767965432, -2809324573)),
    );
    assert!(e == e);
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
fn bench_lu6_permute_rows__baseline() {
    let _f = black_box(f_bench());
    let _a = black_box(a_bench());
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
    assert!(e == e);
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
fn bench_lu6_is_invertible__baseline() {
    let _f = black_box(f_bench());
    let e = black_box(true);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_lu6_is_invertible__pivots() {
    let f = black_box(f_bench());
    let e = black_box(true);
    assert!(f.is_invertible() == e);
}

#[test]
#[inline(never)]
fn bench_lu6_solve__baseline() {
    let _f = black_box(f_bench());
    let _b = black_box(b_bench());
    let e = black_box(
        Some(v6t((-11208158819, -7248812508, 3383102672, 4212967580, 4206763123, -9649832809))),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_lu6_solve__substitution() {
    let f = black_box(f_bench());
    let b = black_box(b_bench());
    let e = black_box(
        Some(v6t((-11208158819, -7248812508, 3383102672, 4212967580, 4206763123, -9649832809))),
    );
    assert!(f.solve(b) == e);
}

#[test]
#[inline(never)]
fn bench_lu6_solve__alt_recip() {
    let f = black_box(f_bench());
    let b = black_box(b_bench());
    let e = black_box(
        Some(v6t((-11208158820, -7248812506, 3383102673, 4212967579, 4206763122, -9649832809))),
    );
    assert!(solve_recip(f, b) == e);
}

#[test]
#[inline(never)]
fn bench_lu6_solve_singular__baseline() {
    let _s = black_box(f_singular());
    let _b = black_box(b_bench());
    let e = black_box(true);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_lu6_solve_singular__none() {
    let s = black_box(f_singular());
    let b = black_box(b_bench());
    let e = black_box(true);
    assert!(s.solve(b).is_none() == e);
}

#[test]
#[inline(never)]
fn bench_lu6_try_inverse__baseline() {
    let _f = black_box(f_bench());
    let e = black_box(
        Some(
            m6(
                [
                    [-1634328750, -1328909174, -3540643764, 384598172, 758821033, 644873742],
                    [1278427182, -3347589152, -334138950, 113652953, -560814343, 3340191840],
                    [-131306984, -1399877949, 769507538, -4750312465, 277298268, 2486987574],
                    [1485628124, 469812391, -922548165, -2942963830, 2293487810, -2985781543],
                    [1066424137, 3172138808, -962445434, 2354884600, -2224819313, 3612501],
                    [230737289, -193491864, -1193168035, 1823649346, 1794867665, 4630404563],
                ],
            ),
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_lu6_try_inverse__columns() {
    let f = black_box(f_bench());
    let e = black_box(
        Some(
            m6(
                [
                    [-1634328750, -1328909174, -3540643764, 384598172, 758821033, 644873742],
                    [1278427182, -3347589152, -334138950, 113652953, -560814343, 3340191840],
                    [-131306984, -1399877949, 769507538, -4750312465, 277298268, 2486987574],
                    [1485628124, 469812391, -922548165, -2942963830, 2293487810, -2985781543],
                    [1066424137, 3172138808, -962445434, 2354884600, -2224819313, 3612501],
                    [230737289, -193491864, -1193168035, 1823649346, 1794867665, 4630404563],
                ],
            ),
        ),
    );
    assert!(f.try_inverse() == e);
}

#[test]
#[inline(never)]
fn bench_lu6_try_inverse__alt_solve_columns() {
    let f = black_box(f_bench());
    let e = black_box(
        Some(
            m6(
                [
                    [-1634328750, -1328909174, -3540643764, 384598172, 758821033, 644873742],
                    [1278427182, -3347589152, -334138950, 113652953, -560814343, 3340191840],
                    [-131306984, -1399877949, 769507538, -4750312465, 277298268, 2486987574],
                    [1485628124, 469812391, -922548165, -2942963830, 2293487810, -2985781543],
                    [1066424137, 3172138808, -962445434, 2354884600, -2224819313, 3612501],
                    [230737289, -193491864, -1193168035, 1823649346, 1794867665, 4630404563],
                ],
            ),
        ),
    );
    assert!(try_inverse_solve_columns(f) == e);
}

#[test]
#[inline(never)]
fn bench_lu6_try_inverse__alt_recip() {
    let f = black_box(f_bench());
    let e = black_box(
        Some(
            m6(
                [
                    [-1634328751, -1328909174, -3540643764, 384598172, 758821032, 644873743],
                    [1278427182, -3347589150, -334138950, 113652953, -560814343, 3340191840],
                    [-131306984, -1399877949, 769507538, -4750312465, 277298268, 2486987574],
                    [1485628124, 469812391, -922548165, -2942963830, 2293487809, -2985781543],
                    [1066424137, 3172138807, -962445434, 2354884600, -2224819312, 3612501],
                    [230737289, -193491864, -1193168035, 1823649346, 1794867665, 4630404563],
                ],
            ),
        ),
    );
    assert!(try_inverse_recip(f) == e);
}

#[test]
#[inline(never)]
fn bench_lu6_determinant__baseline() {
    let _f = black_box(f_bench());
    let e = black_box(fx(5306464826));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_lu6_determinant__pivots() {
    let f = black_box(f_bench());
    let e = black_box(fx(5306464826));
    assert!(f.determinant() == e);
}

#[test]
#[inline(never)]
fn bench_matrix6_lu__baseline() {
    let _a = black_box(a_bench());
    let e = black_box(fx(5306464826));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_matrix6_lu__determinant() {
    let a = black_box(a_bench());
    let e = black_box(fx(5306464826));
    assert!(a.determinant() == e);
}

#[test]
#[inline(never)]
fn bench_matrix6_lu__try_inverse() {
    let a = black_box(a_bench());
    let e = black_box(
        Some(
            m6(
                [
                    [-1634328750, -1328909174, -3540643764, 384598172, 758821033, 644873742],
                    [1278427182, -3347589152, -334138950, 113652953, -560814343, 3340191840],
                    [-131306984, -1399877949, 769507538, -4750312465, 277298268, 2486987574],
                    [1485628124, 469812391, -922548165, -2942963830, 2293487810, -2985781543],
                    [1066424137, 3172138808, -962445434, 2354884600, -2224819313, 3612501],
                    [230737289, -193491864, -1193168035, 1823649346, 1794867665, 4630404563],
                ],
            ),
        ),
    );
    assert!(a.try_inverse() == e);
}

#[test]
#[inline(never)]
fn bench_matrix6_lu__solve() {
    let a = black_box(a_bench());
    let b = black_box(b_bench());
    let e = black_box(
        Some(v6t((-11208158819, -7248812508, 3383102672, 4212967580, 4206763123, -9649832809))),
    );
    assert!(a.solve(b) == e);
}
