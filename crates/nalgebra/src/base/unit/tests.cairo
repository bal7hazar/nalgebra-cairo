//! Unit tests of `Unit`: exact cases (expected values from a bit-exact integer model of the
//! Q32.32 kernels, floor rounding), panics, robustness at both ends of the range, the Newton step
//! of `renormalize_fast`, and the oracle vectors of `tools/oracle` (`normalize` of upstream
//! nalgebra 0.35 on the same raw inputs, which is what `Unit::new_normalize` computes).
//!
//! `oracle.cairo` is emitted from `tools/oracle` (committed vectors, 4 cases per distribution)
//! as documented in its header.
//!
//! In-crate part (WP 8.1c): the tests of crate-internal items only; the tests of the public API are
//! in `crates/tests_base/src/unit/tests.cairo`.

use fixed::Fixed;
use nalgebra_testing::black_box;
use crate::base::matrix_test_utils::{u2, u3, u4};
use crate::base::unit::UnitInternalTrait;
use crate::base::vector3::Vector3;
use super::{Unit, Unit3Trait, UnitTrait, oracle};

/// `p / 13`, rounded to nearest
fn np() -> Unit<Vector3<Fixed>> {
    u3(991146299, -1321528399, 3964585196)
}

// --- renormalization

#[test]
fn test_renormalize_matches_new_normalize() {
    // A floored unit vector is a few ulp short of norm 1: renormalizing moves it by that much.
    assert!(np().renormalized() == UnitTrait::new_normalize(np().value));
    assert!(np().renormalized().abs_diff_eq(np(), 2));
    // `na` scaled by 1 + 3e-6: back to the exact unit vector.
    let drifted = u3(1393475576, -2090213367, 3483688943);
    assert!(drifted.renormalized() == UnitTrait::new_normalize(drifted.value));
    assert!(drifted.renormalized() == u3(1393471396, -2090207097, 3483678493));
}

#[test]
#[should_panic(expected: 'Fixed: division by zero')]
fn test_renormalize_zero() {
    let _ = black_box(u3(0, 0, 0)).renormalized();
}

#[test]
fn test_renormalize_fast_is_a_fixed_point_of_normalized_vectors() {
    // |v|² is 1 or 1 ulp short: the correction rounds to zero. These are the FLOORED quotients
    // (3, -4, 12) / 13 and a / |a|; `new_normalize` rounds to nearest, which for `p` is the same
    // vector, so one fast step leaves `np()` unchanged too.
    let np_floor = u3(991146299, -1321528399, 3964585196);
    let na_floor = u3(1393471396, -2090207096, 3483678492);
    assert!(np_floor.renormalized_fast() == np_floor);
    assert!(na_floor.renormalized_fast() == na_floor);
    assert!(np().renormalized_fast() == np_floor);
    assert!(Unit3Trait::<Fixed>::x_axis().renormalized_fast() == Unit3Trait::<Fixed>::x_axis());
    assert!(Unit3Trait::<Fixed>::z_axis().renormalized_fast() == Unit3Trait::<Fixed>::z_axis());
}

#[test]
fn test_renormalize_fast_fixes_small_drift() {
    // na scaled by 1 + 3e-6 and by 1 - 5e-6: one step gives the exact unit vector within about
    // 1 ulp (1.18 ulp measured, floor products), and `renormalize` (divisions rounded to nearest)
    // is within 0.82 ulp of it; the two can straddle the exact value, hence 2 ulp between them.
    let up = u3(1393475576, -2090213367, 3483688943);
    assert!(up.renormalized_fast() == u3(1393471395, -2090207097, 3483678491));
    assert!(up.renormalized_fast().abs_diff_eq(up.renormalized(), 2));
    let down = u3(1393464429, -2090196645, 3483661074);
    assert!(down.renormalized_fast() == u3(1393471396, -2090207097, 3483678492));
    assert!(down.renormalized_fast().abs_diff_eq(down.renormalized(), 2));
}

#[test]
fn test_renormalize_fast_error_is_squared() {
    // |v|² = 1 + e becomes 1 - 3e²/4 + e³/4: na scaled by 1.001 (e = 2e-3, error 3e-6 = 12 892
    // ulp) and by 1.0001 (e = 2e-4, error 3e-8 = 128 ulp).
    let v001 = u3(1394864867, -2092297303, 3487162170);
    assert!(v001.renormalized_fast() == u3(1393469304, -2090203960, 3483673264));
    let v0001 = u3(1393610743, -2090416117, 3484026860);
    assert!(v0001.renormalized_fast() == u3(1393471375, -2090207066, 3483678440));
}

#[test]
fn test_renormalize_fast_converges_in_four_steps() {
    // (3, -4, 12) / 13 scaled by 1.2 (|v|² = 1.44): four steps, then within 20 ulp.
    let start = u3(1189375559, -1585834079, 4757502235);
    let r = start.renormalized_fast().renormalized_fast().renormalized_fast().renormalized_fast();
    assert!(r == u3(991146293, -1321528396, 3964585178));
    assert!(r.abs_diff_eq(np(), 20));
}

#[test]
fn test_renormalize_fast_2d_and_4d() {
    // (0.6, -0.8) scaled by 1.001, (0.5, -0.5, 0.5, -0.5) scaled by 0.999.
    assert!(u2(2579557357, -3439409811).renormalized_fast() == u2(2576976510, -3435968683));
    assert!(
        u4(2145336164, -2145336164, 2145336164, -2145336164)
            .renormalized_fast() == u4(2147480428, -2147480429, 2147480428, -2147480429),
    );
}

#[test]
fn test_renormalize_fast_zero_stays_zero() {
    assert!(u3(0, 0, 0).renormalized_fast() == u3(0, 0, 0));
}

#[test]
#[should_panic(expected: 'Fixed: overflow')]
fn test_renormalize_fast_overflow() {
    // The squared norm (1e10) does not fit: `renormalize` still works on such a vector.
    let _ = black_box(u3(0x186a000000000, 0, 0)).renormalized_fast();
}

/// Oracle unit vectors have a norm of 1 within a few ulp: `renormalize_fast` keeps them there.
#[test]
fn test_renormalize_fast_on_oracle_unit_vectors_stays_close() {
    let mut cases = oracle::vector3_normalize_cases();
    while let Some(case) = cases.pop_front() {
        let (_, (ex, ey, ez), _) = *case;
        let u = u3(ex, ey, ez);
        assert!(u.renormalized_fast().abs_diff_eq(u, 2));
    }
}
