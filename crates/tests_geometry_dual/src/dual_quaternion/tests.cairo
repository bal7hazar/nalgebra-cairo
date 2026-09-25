//! Unit tests of `DualQuaternion` (WP 8.4-P12): the oracle vectors of `tools/oracle` (suite
//! `dual_quaternion`) and exact cases.

use core::num::traits::{One, Zero};
use fixed::Fixed;
use nalgebra::geometry::dual_quaternion::{DualQuaternion, DualQuaternionTrait};
use nalgebra::geometry::quaternion::{Quaternion, QuaternionTrait};
use nalgebra::geometry::unit_dual_quaternion::UnitDualQuaternionTrait;
use nalgebra_tests_utils::{ONE_RAW, fx, q, qi};
use crate::common::{dq_err, dqt, report, udqt};
use crate::oracle;

/// `(1 + 2i - 3j + 0.5k) + ε(-1.5 + 0.25i + 4j - 2k)`.
fn a() -> DualQuaternion<Fixed> {
    DualQuaternion {
        real: q(ONE_RAW, 2 * ONE_RAW, -3 * ONE_RAW, ONE_RAW / 2),
        dual: q(-3 * ONE_RAW / 2, ONE_RAW / 4, 4 * ONE_RAW, -2 * ONE_RAW),
    }
}

/// `(-2 + 0.5i + j - 1k) + ε(3 - 1i + 0.75j + 2k)`.
fn b() -> DualQuaternion<Fixed> {
    DualQuaternion {
        real: q(-2 * ONE_RAW, ONE_RAW / 2, ONE_RAW, -ONE_RAW),
        dual: q(3 * ONE_RAW, -ONE_RAW, 3 * ONE_RAW / 4, 2 * ONE_RAW),
    }
}

// --- oracle vectors

#[test]
fn test_dual_quaternion_oracle() {
    let (mut worst, mut n) = (0, 0);
    let mut cases = oracle::dual_quaternion_mul_cases();
    while let Some(case) = cases.pop_front() {
        let (x, y, e, tol) = *case;
        worst = core::cmp::max(worst, report("dq mul", n, dq_err(dqt(x) * dqt(y), e), tol));
        n += 1;
    }
    let mut cases = oracle::dual_quaternion_normalize_cases();
    while let Some(case) = cases.pop_front() {
        let (x, e, tol) = *case;
        worst =
            core::cmp::max(worst, report("dq normalize", n, dq_err(dqt(x).normalize(), e), tol));
        n += 1;
    }
    let mut cases = oracle::dual_quaternion_try_inverse_cases();
    while let Some(case) = cases.pop_front() {
        let (x, e, tol) = *case;
        let got = dqt(x).try_inverse().unwrap();
        worst = core::cmp::max(worst, report("dq try_inverse", n, dq_err(got, e), tol));
        n += 1;
    }
    let mut cases = oracle::dual_quaternion_lerp_cases();
    while let Some(case) = cases.pop_front() {
        let (x, y, t, e, tol) = *case;
        let got = dqt(x).lerp(dqt(y), fx(t));
        worst = core::cmp::max(worst, report("dq lerp", n, dq_err(got, e), tol));
        n += 1;
    }
    let mut cases = oracle::dual_quaternion_div_unit_dual_quaternion_cases();
    while let Some(case) = cases.pop_front() {
        let (x, y, e, tol) = *case;
        let got = dqt(x).div_unit_dual_quaternion(udqt(y));
        worst = core::cmp::max(worst, report("dq div_udq", n, dq_err(got, e), tol));
        n += 1;
    }
    assert!(worst == 0);
}

// --- construction

#[test]
fn test_from_real_and_dual_from_real_identity() {
    let (r, d) = (a().real, a().dual);
    assert!(DualQuaternionTrait::from_real_and_dual(r, d) == a());
    let x: DualQuaternion<Fixed> = DualQuaternionTrait::from_real(r);
    assert!(x.real == r && x.dual == Zero::zero());
    let i: DualQuaternion<Fixed> = DualQuaternionTrait::identity();
    assert!(i.real == QuaternionTrait::identity() && i.dual == Zero::zero());
    assert!(i == One::one());
}

#[test]
fn test_zero_one_default() {
    let z: DualQuaternion<Fixed> = Zero::zero();
    assert!(z.is_zero() && !a().is_zero() && a().is_non_zero());
    let d: DualQuaternion<Fixed> = Default::default();
    assert!(d == z);
    let o: DualQuaternion<Fixed> = One::one();
    assert!(o.is_one() && !a().is_one() && a().is_non_one());
    assert!(a() * o == a() && o * a() == a());
}

#[test]
fn test_cast_is_identity() {
    let c: DualQuaternion<Fixed> = a().cast();
    assert!(c == a());
}

// --- algebra

#[test]
fn test_add_sub_neg() {
    let s = a() + b();
    assert!(s.real == a().real + b().real && s.dual == a().dual + b().dual);
    assert!(s - b() == a());
    let n = -a();
    assert!(n.real == -a().real && n.dual == -a().dual);
    assert!(a() + n == Zero::zero());
}

#[test]
fn test_mul_matches_two_hamilton_products_on_integers() {
    // Integer inputs: every product is exact, so the fused kernel equals the naive formula.
    let x = DualQuaternion { real: qi(1, 2, -3, 4), dual: qi(-2, 1, 0, 3) };
    let y = DualQuaternion { real: qi(0, -1, 2, 1), dual: qi(5, -3, 1, -2) };
    let p = x * y;
    assert!(p.real == x.real * y.real);
    assert!(p.dual == x.real * y.dual + x.dual * y.real);
}

#[test]
fn test_mul_is_not_commutative_and_associates() {
    let x = DualQuaternion { real: qi(1, 2, -3, 4), dual: qi(-2, 1, 0, 3) };
    let y = DualQuaternion { real: qi(0, -1, 2, 1), dual: qi(5, -3, 1, -2) };
    let z = DualQuaternion { real: qi(2, 0, 1, -1), dual: qi(1, 1, -1, 0) };
    assert!(x * y != y * x);
    assert!((x * y) * z == x * (y * z));
}

#[test]
fn test_scale_unscale_and_assign_ops() {
    let k = fx(3 * ONE_RAW / 2);
    let s = a().scale(k);
    assert!(s.real == a().real.scale(k) && s.dual == a().dual.scale(k));
    let mut m = a();
    m *= k;
    assert!(m == s);
    let u = s.unscale(k);
    assert!(u == a());
    let mut d = s;
    d /= k;
    assert!(d == u);
}

#[test]
#[should_panic(expected: 'Fixed: division by zero')]
fn test_unscale_by_zero_panics() {
    let _ = a().unscale(fx(0));
}

#[test]
fn test_conjugate_and_conjugate_mut() {
    let c = a().conjugate();
    assert!(c.real == a().real.conjugate() && c.dual == a().dual.conjugate());
    let mut m = a();
    m.conjugate_mut();
    assert!(m == c);
    assert!(c.conjugate() == a());
}

#[test]
fn test_normalize_and_normalize_mut() {
    // |real| = |(1, 2, -3, 0.5)| = sqrt(14.25).
    let n = a().normalize();
    let norm = a().real.norm();
    assert!(n.real == a().real.unscale(norm) && n.dual == a().dual.unscale(norm));
    let mut m = a();
    let got = m.normalize_mut();
    assert!(got == norm && m == n);
    // The real part of the normalised dual quaternion has a unit norm (within 2 ulp).
    let r = n.real.norm();
    assert!(r.raw >= ONE_RAW - 2 && r.raw <= ONE_RAW + 2);
}

#[test]
#[should_panic(expected: 'Fixed: division by zero')]
fn test_normalize_zero_real_panics() {
    let x = DualQuaternion { real: Zero::zero(), dual: qi(1, 0, 0, 0) };
    let _ = x.normalize();
}

#[test]
fn test_try_inverse_is_the_inverse() {
    let x = DualQuaternion { real: qi(1, 1, 0, 1), dual: qi(0, 1, 2, -1) };
    let inv = x.try_inverse().unwrap();
    let p = x * inv;
    let one: DualQuaternion<Fixed> = One::one();
    assert!(p.abs_diff_eq(one, 8));
    assert!((inv * x).abs_diff_eq(one, 8));
}

#[test]
fn test_try_inverse_of_a_zero_real_part_is_none() {
    let x = DualQuaternion { real: Zero::zero(), dual: qi(1, 2, 3, 4) };
    assert!(x.try_inverse().is_none());
    let mut m = x;
    assert!(!m.try_inverse_mut());
    assert!(m == x);
}

#[test]
fn test_try_inverse_mut() {
    let mut m = a();
    assert!(m.try_inverse_mut());
    assert!(m == a().try_inverse().unwrap());
}

#[test]
fn test_lerp_endpoints() {
    assert!(a().lerp(b(), fx(0)) == a());
    assert!(a().lerp(b(), fx(ONE_RAW)) == b());
    let h = a().lerp(b(), fx(ONE_RAW / 2));
    assert!(h.real == a().real.lerp(b().real, fx(ONE_RAW / 2)));
    assert!(h.dual == a().dual.lerp(b().dual, fx(ONE_RAW / 2)));
}

// --- index, serde, comparisons

#[test]
fn test_index_is_the_array_view() {
    let mut x = a();
    assert!(x[0] == x.real.i && x[1] == x.real.j && x[2] == x.real.k && x[3] == x.real.w);
    assert!(x[4] == x.dual.i && x[5] == x.dual.j && x[6] == x.dual.k && x[7] == x.dual.w);
}

#[test]
#[should_panic(expected: 'nalgebra: index out of bounds')]
fn test_index_out_of_bounds_panics() {
    let mut x = a();
    let _ = x[8];
}

#[test]
fn test_serde_is_the_array_view() {
    let mut out = array![];
    a().serialize(ref out);
    let x = a();
    let expected = array![
        x.real.i.raw.into(), x.real.j.raw.into(), x.real.k.raw.into(), x.real.w.raw.into(),
        x.dual.i.raw.into(), x.dual.j.raw.into(), x.dual.k.raw.into(), x.dual.w.raw.into(),
    ];
    assert!(out == expected);
    let mut span = out.span();
    let back: DualQuaternion<Fixed> = Serde::deserialize(ref span).unwrap();
    assert!(back == a());
}

#[test]
fn test_abs_diff_eq_accepts_the_whole_negation_only() {
    let x = a();
    let near = DualQuaternion {
        real: x.real, dual: Quaternion { i: fx(x.dual.i.raw + 3), ..x.dual },
    };
    assert!(x.abs_diff_eq(near, 3) && !x.abs_diff_eq(near, 2));
    assert!(x.abs_diff_eq(-x, 0));
    // Only one part negated: a different transform.
    let half = DualQuaternion { real: x.real, dual: -x.dual };
    assert!(!x.abs_diff_eq(half, 0));
}

#[test]
fn test_relative_eq_and_ulps_eq() {
    let x = a();
    let near = DualQuaternion {
        real: x.real, dual: Quaternion { k: fx(x.dual.k.raw + 64), ..x.dual },
    };
    // |dual.k| = 2: a relative tolerance of 2^-20 is 8192 ulp there.
    assert!(x.relative_eq(near, 0, fx(ONE_RAW / 1048576)));
    assert!(!x.relative_eq(near, 0, fx(0)));
    assert!(x.relative_eq(-near, 0, fx(ONE_RAW / 1048576)));
    assert!(x.ulps_eq(near, 0, 64) && !x.ulps_eq(near, 0, 63));
    assert!(x.ulps_eq(-near, 0, 64));
    let half = DualQuaternion { real: -x.real, dual: x.dual };
    assert!(!x.relative_eq(half, 0, fx(ONE_RAW / 1048576)) && !x.ulps_eq(half, 0, 64));
}

#[test]
fn test_mul_div_unit_dual_quaternion() {
    let u = UnitDualQuaternionTrait::from_parts(
        nalgebra_tests_utils::t3(ONE_RAW, -2 * ONE_RAW, ONE_RAW / 2),
        nalgebra_tests_utils::uq(3037000499, 3037000499, 0, 0),
    );
    assert!(a().mul_unit_dual_quaternion(u) == a() * u.dual_quaternion);
    assert!(a().div_unit_dual_quaternion(u) == a() * u.inverse().dual_quaternion);
    // `(a * u) / u` is `a` up to rounding.
    assert!(a().mul_unit_dual_quaternion(u).div_unit_dual_quaternion(u).abs_diff_eq(a(), 16));
}
