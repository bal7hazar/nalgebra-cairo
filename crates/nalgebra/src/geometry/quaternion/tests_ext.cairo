//! Unit tests of the WP 8.4-P08 completion of `Quaternion`: the algebra (`squared`, `half`,
//! `inner`, `outer`, divisions, projections, `sqrt`), the transcendental functions, the polar
//! decomposition, the approximate comparisons and the trait impls (`Index`, `One`, arrays):
//! exact cases, identities, panics, and the oracle vectors of `tools/oracle` (upstream nalgebra
//! 0.35 on the same raw inputs, tolerance in ulp).
//!
//! `oracle_ext.cairo` is emitted from `tools/oracle` (committed vectors, at most 8 cases per
//! distribution):
//!
//! ```text
//! cargo run --release -- emit-cairo quaternion_functions --from vectors --max-per-dist 8 \
//!     --out crates/nalgebra/src/geometry/quaternion/oracle_ext.cairo
//! ```
//!
//! Every oracle test checks EVERY case and prints the ones that leave the tolerance (with the
//! excess) before failing, so that a regression shows by how much.
//!
//! In-crate part (WP 8.1c): the tests of crate-internal items only; the tests of the public API are
//! in `crates/tests_geometry/src/quaternion/tests_ext.cairo`.

use fixed::Fixed;
use crate::base::matrix_test_utils::{excess, max_ulp_diff_q, qi, qt};
use crate::geometry::quaternion::QuaternionInternalTrait;
use super::{Quaternion, oracle_ext};

/// `1 + 2i - 3j + 4k`.
fn a() -> Quaternion<Fixed> {
    qi(1, 2, -3, 4)
}

/// `-2 + i + 5j - k`.
fn b() -> Quaternion<Fixed> {
    qi(-2, 1, 5, -1)
}

/// The excess of `err` over `tol`, printed when positive.
fn report(op: ByteArray, index: usize, err: u128, tol: u64) -> u128 {
    let e = excess(err, tol.into());
    if e > 0 {
        println!("{op} case {index}: error {err} ulp, tolerance {tol}");
    }
    e
}

/// The excess of `got` over the tolerance around `expected` (printed when positive).
fn check_q(
    op: ByteArray, index: usize, got: Quaternion<Fixed>, expected: (i64, i64, i64, i64), tol: u64,
) -> u128 {
    report(op, index, max_ulp_diff_q(got, qt(expected)), tol)
}

#[test]
fn test_mul_conj_oracle_is_bit_exact() {
    let mut cases = oracle_ext::quaternion_mul_conj_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (a, b, e, tol) = *case;
        let (a, b) = (qt(a), qt(b));
        worst = core::cmp::max(worst, check_q("mul_conj", n, a.mul_conj(b), e, tol));
        n += 1;
    }
    assert!(worst == 0);
}
