//! Unit tests of `Vector6`: exact cases (expected values from the bit-exact Q32.32 model, floor
//! rounding), panics, robustness at both ends of the range, identities, and the oracle vectors of
//! `tools/oracle` (upstream nalgebra 0.35 on the same raw inputs).
//!
//! `crate::base::oracle_dim6_vector` is emitted from `tools/oracle` (committed vectors, 4 cases
//! per distribution) with
//! `cargo run --release -- emit-cairo dim6 --from vectors --max-per-dist 4 --out
//! crates/nalgebra/src/base/oracle_dim6_vector.cairo --ops vector6_add,vector6_sub,vector6_neg,
//! vector6_scale,vector6_dot,vector6_norm_squared,vector6_norm`.

use fixed::Fixed;
use nalgebra_testing::black_box;
use simba::scalar::Real;
use crate::base::matrix_test_utils::{fx, int, v6, v6i, v6t};
use crate::base::oracle_dim6_vector as oracle;
use super::{Vector6, Vector6Trait};

const MAX: i64 = 0x7fffffffffffffff;
const MIN: i64 = -0x8000000000000000;

/// (1.5, -2.25, 3.75, -4.5, 0.25, 2)
fn a() -> Vector6<Fixed> {
    v6(0x180000000, -0x240000000, 0x3c0000000, -0x480000000, 0x40000000, 0x200000000)
}

/// (-4.5, 0.25, 2, 1.5, -2.25, 3.75)
fn b() -> Vector6<Fixed> {
    v6(-0x480000000, 0x40000000, 0x200000000, 0x180000000, -0x240000000, 0x3c0000000)
}

/// (2, -2, 2, -2, 2, 4), of norm exactly 6.
fn p() -> Vector6<Fixed> {
    v6i(2, -2, 2, -2, 2, 4)
}

// --- constructors, conversions

#[test]
fn test_new_is_upstream_component_order() {
    let r = Vector6Trait::new(
        fx(0x180000000),
        fx(-0x240000000),
        fx(0x3c0000000),
        fx(-0x480000000),
        fx(0x40000000),
        fx(0x200000000),
    );
    assert!(r == a());
    // Upstream's component names, in order.
    assert!(r.x == fx(0x180000000) && r.y == fx(-0x240000000) && r.z == fx(0x3c0000000));
    assert!(r.w == fx(-0x480000000) && r.a == fx(0x40000000) && r.b == fx(0x200000000));
}

#[test]
fn test_zeros_is_zero_and_default() {
    assert!(Vector6Trait::<Fixed>::zeros() == v6(0, 0, 0, 0, 0, 0));
    assert!(Vector6Trait::<Fixed>::zeros() == Default::default());
}

#[test]
fn test_serde_is_upstream_order() {
    let mut out = array![];
    v6i(1, 2, 3, 4, 5, 6).serialize(ref out);
    let one: felt252 = 0x100000000;
    assert!(out == array![1 * one, 2 * one, 3 * one, 4 * one, 5 * one, 6 * one]);
}

#[test]
fn test_into_array() {
    let arr: [Fixed; 6] = a().into();
    let [x, y, z, w, p5, p6] = arr;
    assert!(x == fx(0x180000000) && y == fx(-0x240000000) && z == fx(0x3c0000000));
    assert!(w == fx(-0x480000000) && p5 == fx(0x40000000) && p6 == fx(0x200000000));
    let from_arr: Vector6<Fixed> = arr.into();
    assert!(from_arr == a());
}

// --- additive operators

#[test]
fn test_add_sub_neg_exact() {
    // (1.5, -2.25, 3.75, -4.5, 0.25, 2) +- (-4.5, 0.25, 2, 1.5, -2.25, 3.75)
    assert!(
        a()
            + b() == v6(
                -0x300000000, -0x200000000, 0x5c0000000, -0x300000000, -0x200000000, 0x5c0000000,
            ),
    );
    assert!(
        a()
            - b() == v6(
                0x600000000, -0x280000000, 0x1c0000000, -0x600000000, 0x280000000, -0x1c0000000,
            ),
    );
    assert!(
        -a() == v6(-0x180000000, 0x240000000, -0x3c0000000, 0x480000000, -0x40000000, -0x200000000),
    );
    assert!(a() + (-a()) == Vector6Trait::zeros());
    assert!((a() - b()) + b() == a());
}

#[test]
fn test_add_sub_assign_match_operators() {
    let mut acc = a();
    acc += b();
    assert!(acc == a() + b());
    let mut acc = a();
    acc -= b();
    assert!(acc == a() - b());
}

#[test]
#[should_panic(expected: 'i64_add Overflow')]
fn test_add_overflow_panics() {
    let m = black_box(v6(MAX, 0, 0, 0, 0, 0));
    let _ = m + m;
}

#[test]
#[should_panic(expected: 'i64_sub Underflow')]
fn test_sub_overflow_panics() {
    let _ = black_box(v6(MIN, 0, 0, 0, 0, 0)) - black_box(v6(0x100000000, 0, 0, 0, 0, 0));
}

#[test]
#[should_panic(expected: 'i64_neg Underflow')]
fn test_neg_min_panics() {
    let _ = -black_box(v6(0, 0, 0, 0, 0, MIN));
}

// --- scaling

#[test]
fn test_scale_unscale_exact() {
    assert!(
        a()
            .scale(
                int(2),
            ) == v6(0x300000000, -0x480000000, 0x780000000, -0x900000000, 0x80000000, 0x400000000),
    );
    assert!(p().unscale(int(2)) == v6i(1, -1, 1, -1, 1, 2));
    // Floor rounding, once per component: 1 / 3 and -1 / 3 floor apart.
    assert!(v6i(1, -1, 0, 0, 0, 0).unscale(int(3)) == v6(1431655765, -1431655765, 0, 0, 0, 0));
}

#[test]
fn test_mul_div_assign_are_scale_unscale() {
    let mut v = a();
    v *= int(2);
    assert!(v == a().scale(int(2)));
    let mut v = p();
    v /= int(2);
    assert!(v == p().unscale(int(2)));
}

#[test]
#[should_panic(expected: 'Fixed: division by zero')]
fn test_unscale_by_zero_panics() {
    let _ = black_box(a()).unscale(Real::zero());
}

#[test]
#[should_panic(expected: 'Fixed: overflow')]
fn test_scale_overflow_panics() {
    let _ = black_box(v6(0, 0, 0, 0, 0, MAX)).scale(black_box(int(2)));
}

// --- component-wise reductions

#[test]
fn test_component_mul_abs_inf_sup_sum() {
    assert!(
        a()
            .component_mul(
                b(),
            ) == v6(-0x6c0000000, -0x90000000, 0x780000000, -0x6c0000000, -0x90000000, 0x780000000),
    );
    assert!(
        a()
            .abs() == v6(
                0x180000000, 0x240000000, 0x3c0000000, 0x480000000, 0x40000000, 0x200000000,
            ),
    );
    assert!(
        a()
            .inf(
                b(),
            ) == v6(
                -0x480000000, -0x240000000, 0x200000000, -0x480000000, -0x240000000, 0x200000000,
            ),
    );
    assert!(
        a()
            .sup(
                b(),
            ) == v6(0x180000000, 0x40000000, 0x3c0000000, 0x180000000, 0x40000000, 0x3c0000000),
    );
    // 1.5 - 2.25 + 3.75 - 4.5 + 0.25 + 2 = 0.75
    assert!(a().sum() == fx(0xc0000000));
    assert!(p().sum() == int(6));
}

#[test]
#[should_panic(expected: 'Fixed: overflow')]
fn test_abs_of_min_panics() {
    let _ = black_box(v6(0, 0, MIN, 0, 0, 0)).abs();
}

#[test]
#[should_panic(expected: 'i64_add Overflow')]
fn test_sum_overflow_panics() {
    let _ = black_box(v6(MAX, MAX, 0, 0, 0, 0)).sum();
}

// --- dot product: ONE rescale over the six products

#[test]
fn test_dot_exact() {
    // 1.5*-4.5 - 2.25*0.25 + 3.75*2 - 4.5*1.5 + 0.25*-2.25 + 2*3.75 = 0.375
    assert!(a().dot(b()) == fx(0x60000000));
    assert!(p().dot(p()) == int(36));
    assert!(Vector6Trait::<Fixed>::zeros().dot(a()) == Real::zero());
    // Symmetry.
    assert!(a().dot(b()) == b().dot(a()));
}

#[test]
fn test_dot_is_a_single_rescale() {
    // Two products of 2^-33 each: the exact sum is 1 ulp, but flooring each half separately
    // (`head.dot(head) + tail.dot(tail)`) gives 0 + 0. See `benches::alt_dot_two_sum_prod3`.
    let u = v6(0x80000000, 0, 0, 0x80000000, 0, 0);
    let w = v6(1, 0, 0, 1, 0, 0);
    assert!(u.dot(w) == fx(1));
}

#[test]
fn test_dot_of_huge_components_does_not_overflow_intermediates() {
    // 1e9 * 1e-9 six times: every product is representable only once summed and rescaled.
    let u = v6i(1000000000, 0, 0, 0, 0, 0);
    assert!(u.dot(v6(1, 0, 0, 0, 0, 0)) == fx(1000000000));
}

#[test]
#[should_panic(expected: 'Fixed: overflow')]
fn test_dot_overflow_panics() {
    let u = black_box(v6i(65536, 65536, 65536, 65536, 65536, 65536));
    let _ = u.dot(u);
}

// --- norms

#[test]
fn test_norm_squared_and_norm_exact() {
    assert!(p().norm_squared() == int(36));
    assert!(p().norm() == int(6));
    // 2.25 + 5.0625 + 14.0625 + 20.25 + 0.0625 + 4 = 45.6875
    assert!(a().norm_squared() == fx(0x2db0000000));
    assert!(Vector6Trait::<Fixed>::zeros().norm() == Real::zero());
    assert!(a().norm_squared() == a().dot(a()));
}

#[test]
fn test_norm_of_huge_vector_does_not_overflow() {
    // Six components of 1e6: the sum of squares is 6e12, far outside Q32.32; the norm is not.
    let v = v6i(1000000, 1000000, 1000000, 1000000, 1000000, 1000000);
    assert!(v.norm() == fx(10520478337141201));
}

#[test]
#[should_panic(expected: 'Fixed: overflow')]
fn test_norm_squared_overflow_panics() {
    let _ = black_box(v6i(1000000, 0, 0, 0, 0, 0)).norm_squared();
}

#[test]
fn test_normalize_exact() {
    // (2, -2, 2, -2, 2, 4) / 6: the floored quotients of +-1/3 and 2/3.
    assert!(
        p()
            .normalize() == v6(
                1431655765, -1431655765, 1431655765, -1431655765, 1431655765, 2863311531,
            ),
    );
    assert!(p().normalize().norm().abs_diff_eq(Real::one(), 2));
}

#[test]
fn test_normalize_tiny_is_exact() {
    // A vector of 6 raw units has a norm of floor(sqrt(6)) = 2 raw units.
    assert!(v6(1, -1, 1, -1, 1, 1).norm() == fx(2));
    assert!(v6(1, 0, 0, 0, 0, 0).normalize() == v6(0x100000000, 0, 0, 0, 0, 0));
}

#[test]
#[should_panic(expected: 'Fixed: division by zero')]
fn test_normalize_of_zero_panics() {
    let _ = black_box(Vector6Trait::<Fixed>::zeros()).normalize();
}

// --- interpolation and approximate equality

#[test]
fn test_lerp_endpoints_and_middle() {
    assert!(a().lerp(b(), Real::zero()) == a());
    assert!(a().lerp(b(), Real::one()) == b());
    // (-1.5, -1, 2.875, -1.5, -1, 2.875)
    assert!(
        a()
            .lerp(
                b(), Real::HALF,
            ) == v6(
                -0x180000000, -0x100000000, 0x2e0000000, -0x180000000, -0x100000000, 0x2e0000000,
            ),
    );
}

#[test]
fn test_abs_diff_eq_counts_raw_units() {
    let u = v6(1, 2, 3, 4, 5, 6);
    assert!(u.abs_diff_eq(v6(0, 0, 0, 0, 0, 0), 6));
    assert!(!u.abs_diff_eq(v6(0, 0, 0, 0, 0, 0), 5));
    assert!(u.abs_diff_eq(u, 0));
    // Never overflows, whatever the distance: `MAX - MIN` is 2^64 - 1 raw units, and the
    // comparison against a `u64` tolerance is done on a wider type.
    assert!(!v6(MIN, 0, 0, 0, 0, 0).abs_diff_eq(v6(MAX, 0, 0, 0, 0, 0), 0xfffffffffffffffe));
    // `fixed`'s tolerance is a `Fixed`: a `u64` tolerance above `MAX` raw is clamped to it,
    // so a pair `2^63` raw or more apart is never equal (the exact answer would be `true`).
    assert!(!v6(MIN, 0, 0, 0, 0, 0).abs_diff_eq(v6(MAX, 0, 0, 0, 0, 0), 0xffffffffffffffff));
    assert!(
        v6(MAX, 0, 0, 0, 0, 0)
            .abs_diff_eq(v6(MAX, 0, 0, 0, 0, 0).scale(Real::zero()), 0xffffffffffffffff),
    );
}

// --- oracle vectors (upstream nalgebra on the same raw inputs; `tol` in ulp, 0 = bit for bit)

#[test]
fn test_add_oracle() {
    let mut cases = oracle::vector6_add_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (a, b, expected, tol) = *case;
        let (a, b) = (v6t(a), v6t(b));
        assert!((a + b).abs_diff_eq(v6t(expected), tol));
        let mut acc = a;
        acc += b;
        assert!(acc == v6t(expected));
    }
}

#[test]
fn test_sub_oracle() {
    let mut cases = oracle::vector6_sub_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (a, b, expected, tol) = *case;
        let (a, b) = (v6t(a), v6t(b));
        assert!((a - b).abs_diff_eq(v6t(expected), tol));
        let mut acc = a;
        acc -= b;
        assert!(acc == v6t(expected));
    }
}

#[test]
fn test_neg_oracle() {
    let mut cases = oracle::vector6_neg_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (a, expected, tol) = *case;
        assert!((-v6t(a)).abs_diff_eq(v6t(expected), tol));
    }
}

#[test]
fn test_scale_oracle() {
    let mut cases = oracle::vector6_scale_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (a, k, expected, tol) = *case;
        assert!(v6t(a).scale(fx(k)).abs_diff_eq(v6t(expected), tol));
    }
}

#[test]
fn test_dot_oracle() {
    let mut cases = oracle::vector6_dot_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (a, b, expected, tol) = *case;
        // The oracle tolerance is 0: the wide accumulator reproduces the exact floor bit for bit.
        assert!(tol == 0);
        assert!(v6t(a).dot(v6t(b)) == fx(expected));
    }
}

#[test]
fn test_norm_squared_oracle() {
    let mut cases = oracle::vector6_norm_squared_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (a, expected, tol) = *case;
        assert!(tol == 0);
        assert!(v6t(a).norm_squared() == fx(expected));
    }
}

#[test]
fn test_norm_oracle() {
    let mut cases = oracle::vector6_norm_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (a, expected, tol) = *case;
        assert!(v6t(a).norm().abs_diff_eq(fx(expected), tol));
    }
}
