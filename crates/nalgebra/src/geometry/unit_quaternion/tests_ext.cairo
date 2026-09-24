//! Unit tests of the WP 8.4-P08 completion of `UnitQuaternion`: the division, the heterogeneous
//! operators (rotations, translations, isometries, similarities), the observer frames,
//! `rotation_between_axis`, `from_matrix_eps`, `mean_of`, `ln` / `exp`, the constructors and the
//! trait impls: exact cases, identities, panics, the measurements behind `FROM_MATRIX_MAX_ITER`
//! and `MEAN_OF_SQUARINGS`, and the oracle vectors of `tools/oracle` (upstream nalgebra 0.35 on
//! the same raw inputs, tolerance in ulp).
//!
//! `oracle_ext.cairo` is emitted from `tools/oracle` (committed vectors, at most 8 cases per
//! distribution):
//!
//! ```text
//! cargo run --release -- emit-cairo unit_quaternion_completion --from vectors --max-per-dist 8 \
//!     --out crates/nalgebra/src/geometry/unit_quaternion/oracle_ext.cairo
//! ```
//!
//! In-crate part (WP 8.1c): the tests of crate-internal items only; the tests of the public API are
//! in `crates/tests_geometry/src/unit_quaternion/tests_ext.cairo`.

use core::num::traits::One;
use fixed::Fixed;
use crate::base::matrix_test_utils::{excess, fx, m3, max_ulp_diff_q, qt, uqt};
use crate::geometry::quaternion::Quaternion;
use super::{
    FROM_MATRIX_MAX_ITER, Sym4, UnitQuaternionAngleInternalTrait, UnitQuaternionAngleTrait,
    UnitQuaternionInternalTrait, UnitQuaternionTrait, oracle_ext,
};

/// `min(|a - b|, |a + b|)` component-wise: a distance between two rotations up to the sign.
fn err_up_to_sign(a: Quaternion<Fixed>, b: Quaternion<Fixed>) -> u128 {
    core::cmp::min(max_ulp_diff_q(a, b), max_ulp_diff_q(a, -b))
}

/// The measurement behind `FROM_MATRIX_MAX_ITER` and the closed form of `max_iter = 0`: from the
/// identity, with `eps` = 32 ulp (just above the rounding noise of `ω`; at 1 ulp the iteration
/// never stops before its bound), Müller's iteration reaches the closed form on 17 of the 24
/// oracle cases, in 21 to 57 iterations (its convergence is linear), and needs more than 64 on
/// the other 7.
#[test]
fn test_from_matrix_eps_iterations_on_the_oracle_set() {
    let mut cases = oracle_ext::unit_quaternion_from_matrix_cases();
    let (mut converged, mut fewest, mut most) = (0, FROM_MATRIX_MAX_ITER, 0);
    let mut worst = 0;
    while let Some(case) = cases.pop_front() {
        let (m, e, tol) = *case;
        let (got, iterations) = UnitQuaternionAngleInternalTrait::from_matrix_eps_count(
            m3(m), fx(32), 0, UnitQuaternionTrait::identity(),
        );
        if iterations < FROM_MATRIX_MAX_ITER {
            converged += 1;
            fewest = core::cmp::min(fewest, iterations);
            most = core::cmp::max(most, iterations);
            let got = UnitQuaternionInternalTrait::shepperd_sign(got);
            worst =
                core::cmp::max(
                    worst, excess(max_ulp_diff_q(got.quaternion, qt(e)), tol.into() + 64),
                );
        }
    }
    println!("from_matrix_eps: {converged} of 24 converged, in {fewest} to {most} iterations");
    assert!(converged == 17 && fewest == 21 && most == 57);
    assert!(worst == 0);
}

/// The measurement behind `MEAN_OF_SQUARINGS` for `from_matrix`.
#[test]
fn test_from_matrix_squarings_on_the_oracle_set() {
    let mut cases = oracle_ext::unit_quaternion_from_matrix_cases();
    let mut worst: u128 = 0;
    while let Some(case) = cases.pop_front() {
        let (m, e, tol) = *case;
        let full = UnitQuaternionInternalTrait::shepperd_sign(
            UnitQuaternionInternalTrait::closest_rotation(m3(m)),
        );
        worst = core::cmp::max(worst, excess(max_ulp_diff_q(full.quaternion, qt(e)), tol.into()));
    }
    assert!(worst == 0);
}

#[test]
fn test_from_matrix_of_a_rotation_is_that_rotation() {
    let q = uqt((-1509276477, -2563574020, -2263667719, 2114881862));
    let m = q.to_rotation_matrix().matrix;
    let got = UnitQuaternionAngleTrait::from_matrix(m);
    // About one ulp per squaring of the closed form.
    let err = err_up_to_sign(got.quaternion, q.quaternion);
    println!("from_matrix of a rotation: {err} ulp");
    assert!(err <= 16);
    // From the answer itself, the iteration stops at once (above the noise floor of `ω`).
    let (_, iterations) = UnitQuaternionAngleInternalTrait::from_matrix_eps_count(m, fx(32), 0, q);
    assert!(iterations == 1);
    // max_iter bounds the work: one iteration from the identity is not the answer yet, 64 are.
    let one = UnitQuaternionAngleTrait::from_matrix_eps(m, fx(32), 1, One::one());
    assert!(err_up_to_sign(one.quaternion, q.quaternion) > 8);
    let many = UnitQuaternionAngleTrait::from_matrix_eps(m, fx(32), 64, One::one());
    assert!(err_up_to_sign(many.quaternion, q.quaternion) <= 64);
}

/// The measurement behind `MEAN_OF_SQUARINGS`: the worst error on the oracle set after `k`
/// squarings is 28 571 386 ulp at 4, 447 at 6, then 7 at 8, 9 at 10, 10 at 12, 12 at 14 and 14 at
/// 16 (about one ulp more per squaring once converged): from 8 squarings every case is within
/// its tolerance, and 12 leave room for eigenvalue ratios up to about 0.99 (the oracle rejects
/// ratios above 0.8).
#[test]
fn test_mean_of_squarings_on_the_oracle_set() {
    let mut k: usize = 4;
    while k <= 16 {
        let mut cases = oracle_ext::unit_quaternion_mean_of_cases();
        let mut worst: u128 = 0;
        let mut over: u128 = 0;
        while let Some(case) = cases.pop_front() {
            let ((a, b, c), e, tol) = *case;
            let span = array![uqt(a), uqt(b), uqt(c)].span();
            let mut s: Sym4<Fixed> = UnitQuaternionInternalTrait::outer_sum(span, 3);
            let mut i = 0;
            while i < k {
                s = UnitQuaternionInternalTrait::normalized_square(s);
                i += 1;
            }
            let got = UnitQuaternionInternalTrait::dominant_column(s);
            let err = max_ulp_diff_q(got.quaternion, qt(e));
            worst = core::cmp::max(worst, err);
            over = core::cmp::max(over, excess(err, tol.into()));
        }
        println!("mean_of: {k} squarings, worst error {worst} ulp, worst excess {over}");
        if k >= 8 {
            assert!(over == 0);
        }
        k += 2;
    }
}
