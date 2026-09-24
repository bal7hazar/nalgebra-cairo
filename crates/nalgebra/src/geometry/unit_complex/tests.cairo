//! Unit tests of `UnitComplex`: exact cases (identity, quarter turns, axes), the identities a
//! rotation must satisfy (`c · c⁻¹ = 1`, `R · Rᵀ = I`, `c → R → c`), panics on
//! degenerate inputs, and the oracle vectors of `tools/oracle` (upstream nalgebra 0.35 on the same
//! raw inputs).
//!
//! `oracle.cairo` is emitted from `tools/oracle` (committed vectors, 6 cases per distribution,
//! the `unit_complex_*` ops of the `rotation2` suite) with
//! `cargo run --release -- emit-cairo rotation2 --from vectors --max-per-dist 6 --out
//! <oracle.cairo> --ops <list>`, `<list>` being the comma-separated op names of the oracle tests
//! at the bottom of this file.
//!
//! In-crate part (WP 8.1c): the tests of crate-internal items only; the tests of the public API are
//! in `crates/tests_geometry/src/unit_complex/tests.cairo`.

use fixed::Fixed;
use nalgebra_testing::black_box;
use simba::scalar::Real;
use crate::base::matrix_test_utils::{ONE_RAW, fx, uct, ulp_diff};
use crate::geometry::unit_complex::UnitComplexInternalTrait;
use super::{UnitComplex, UnitComplexAngleTrait, UnitComplexTrait};

/// The raw value of 1.
fn id() -> UnitComplex<Fixed> {
    UnitComplexTrait::<Fixed>::identity()
}

/// The quarter turn `(0, 1)`, an exactly representable rotation.
fn quarter() -> UnitComplex<Fixed> {
    uct((0, ONE_RAW))
}

// --- renormalization

#[test]
fn test_renormalize_is_exact_on_a_unit_pair() {
    assert!(quarter().renormalized() == quarter());
    assert!(id().renormalized() == id());
}

#[test]
fn test_renormalize_recovers_from_a_large_drift() {
    // A unit pair scaled by 4 (exact: a power of two): renormalize divides it back, which
    // `renormalize_fast` cannot do (`|c|² - 1 = 15` is far outside its convergence radius).
    let c = UnitComplexAngleTrait::<Fixed>::new(fx(0x1f0a3d70a));
    let four = Real::<Fixed>::from_int(4);
    let drifted = UnitComplex { re: c.re * four, im: c.im * four };
    let n = drifted.renormalized();
    assert!(ulp_diff(Real::norm_squared2(n.re, n.im), Real::one()) <= 2);
    assert!(n.abs_diff_eq(c, 2));
}

#[test]
fn test_renormalize_fast_converges_quadratically() {
    // A pair about 2^-20 off the unit circle: `|c|² = 1 + e` with `e ~ 2.7e-6`, and one Newton
    // step leaves `3e²/4 ~ 5e-12`, i.e. 0 ulp. The step costs no division and no square root.
    let c = UnitComplexAngleTrait::<Fixed>::new(fx(0x1f0a3d70a));
    let drifted = UnitComplex { re: c.re + fx(0x1000), im: c.im + fx(0x1000) };
    assert!(ulp_diff(Real::norm_squared2(drifted.re, drifted.im), Real::one()) > 1000);
    let n = drifted.renormalized_fast();
    assert!(ulp_diff(Real::norm_squared2(n.re, n.im), Real::one()) <= 2);
    // And the exact renormalization agrees to a few ulp.
    assert!(n.abs_diff_eq(drifted.renormalized(), 4));
}

#[test]
fn test_renormalize_fast_leaves_the_square_of_a_large_drift() {
    // 1e-4 off the unit circle: `e ~ 2.8e-4`, one step leaves `3e²/4 ~ 6e-8` = 250 ulp on the
    // squared norm, where `renormalize` would land within 2. The error is squared at every step,
    // so a second step would fix it; `renormalize` is the right tool above 2^-16.
    let c = UnitComplexAngleTrait::<Fixed>::new(fx(0x1f0a3d70a));
    let drifted = UnitComplex { re: c.re + fx(0x68db8), im: c.im + fx(0x68db8) };
    let once = drifted.renormalized_fast();
    let e = ulp_diff(Real::norm_squared2(once.re, once.im), Real::one());
    assert!(e > 2 && e < 400);
    let twice = once.renormalized_fast();
    assert!(ulp_diff(Real::norm_squared2(twice.re, twice.im), Real::one()) <= 2);
    let exact = drifted.renormalized();
    assert!(ulp_diff(Real::norm_squared2(exact.re, exact.im), Real::one()) <= 2);
}

#[test]
fn test_renormalize_fast_of_zero_stays_zero() {
    let z = UnitComplex { re: Real::<Fixed>::zero(), im: Real::zero() };
    assert!(z.renormalized_fast() == z);
}

#[test]
#[should_panic(expected: 'Fixed: division by zero')]
fn test_renormalize_of_zero_panics() {
    let z = UnitComplex { re: Real::<Fixed>::zero(), im: Real::zero() };
    let _ = black_box(z).renormalized();
}
