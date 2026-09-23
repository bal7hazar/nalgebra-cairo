//! Unit tests of `Vector2`: exact cases (expected values from a bit-exact integer model of the
//! Q32.32 kernels, floor rounding), panics, robustness at both ends of the range, identities, and
//! the oracle vectors of `tools/oracle` (upstream nalgebra 0.35 on the same raw inputs).
//!
//! `oracle.cairo` is emitted from `tools/oracle` (committed vectors, 4 cases per distribution,
//! every op of the suite) with
//! `cargo run --release -- emit-cairo vector2 --from vectors --max-per-dist 4 --out
//! <oracle.cairo>`.

use fixed::Fixed;
use nalgebra_testing::black_box;
use simba::scalar::Real;
use crate::base::matrix_test_utils::{fx, v2, v2t, v3};
use super::{Vector2, Vector2AngleTrait, Vector2Trait, oracle};

const MAX: i64 = 0x7fffffffffffffff;
const MIN: i64 = -0x8000000000000000;

/// (1.5, -2.25)
fn a() -> Vector2<Fixed> {
    v2(0x180000000, -0x240000000)
}

/// (-4.5, 0.25)
fn b() -> Vector2<Fixed> {
    v2(-0x480000000, 0x40000000)
}

/// (3, -4), of norm 5
fn p() -> Vector2<Fixed> {
    v2(0x300000000, -0x400000000)
}

// --- constructors, resizing, conversions

#[test]
fn test_new_sets_fields() {
    let r = Vector2Trait::<Fixed>::new(fx(0x180000000), fx(-0x240000000));
    assert!(r.x == fx(0x180000000) && r.y == fx(-0x240000000));
    assert!(r == a());
}

#[test]
fn test_zeros_is_zero() {
    assert!(Vector2Trait::<Fixed>::zeros() == v2(0, 0));
    assert!(Vector2Trait::<Fixed>::zeros() == Default::default());
    assert!(Vector2Trait::<Fixed>::zeros().is_zero());
}

#[test]
fn test_repeat_fills_components() {
    assert!(Vector2Trait::<Fixed>::repeat(fx(-0x240000000)) == v2(-0x240000000, -0x240000000));
    assert!(
        Vector2Trait::<Fixed>::from_element(fx(-0x240000000)) == v2(-0x240000000, -0x240000000),
    );
}

#[test]
fn test_axes_are_orthonormal() {
    assert!(Vector2Trait::<Fixed>::x() == v2(0x100000000, 0));
    assert!(Vector2Trait::<Fixed>::y() == v2(0, 0x100000000));
    // Orthonormal: unit norms, zero dot products.
    assert!(Vector2Trait::<Fixed>::x().norm() == Real::ONE);
    assert!(Vector2Trait::<Fixed>::x().dot(Vector2Trait::<Fixed>::y()) == Real::ZERO);
    assert!(Vector2Trait::<Fixed>::y().norm() == Real::ONE);
}

#[test]
fn test_push_appends_component() {
    assert!(a().push(fx(0x700000000)) == v3(0x180000000, -0x240000000, 0x700000000));
}

#[test]
fn test_to_homogeneous_appends_zero() {
    assert!(a().to_homogeneous() == v3(0x180000000, -0x240000000, 0));
}

#[test]
fn test_tuple_conversions_roundtrip() {
    let r: Vector2<Fixed> = (fx(0x180000000), fx(-0x240000000)).into();
    assert!(r == a());
    let t: (Fixed, Fixed) = r.into();
    assert!(t == (fx(0x180000000), fx(-0x240000000)));
}

#[test]
fn test_array_conversions_roundtrip() {
    let r: Vector2<Fixed> = [fx(0x180000000), fx(-0x240000000)].into();
    assert!(r == a());
    let [x, y]: [Fixed; 2] = r.into();
    assert!(x == fx(0x180000000) && y == fx(-0x240000000));
}

// --- operators

#[test]
fn test_add_exact() {
    // (1.5, -2.25) + (-4.5, 0.25) = (-3, -2)
    assert!(a() + b() == v2(-0x300000000, -0x200000000));
}

#[test]
fn test_sub_exact() {
    // (1.5, -2.25) - (-4.5, 0.25) = (6, -2.5)
    assert!(a() - b() == v2(0x600000000, -0x280000000));
}

#[test]
fn test_neg_exact() {
    assert!(-a() == v2(-0x180000000, 0x240000000));
    assert!(-(-a()) == a());
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_add_overflow() {
    let _ = black_box(v2(0, 9223372036854775807)) + v2(0, 1);
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_sub_overflow() {
    let _ = black_box(v2(0, -0x8000000000000000)) - v2(0, 1);
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_neg_overflow() {
    let _ = -black_box(v2(0, MIN));
}

#[test]
fn test_add_assign_matches_add() {
    let mut r = a();
    r += b();
    assert!(r == a() + b());
}

#[test]
fn test_sub_assign_matches_sub() {
    let mut r = a();
    r -= b();
    assert!(r == a() - b());
}

#[test]
fn test_mul_assign_matches_scale() {
    let mut r = a();
    r *= fx(0x280000000);
    assert!(r == a().scale(fx(0x280000000)));
}

#[test]
fn test_div_assign_matches_unscale() {
    let mut r = a();
    r /= fx(0x280000000);
    assert!(r == a().unscale(fx(0x280000000)));
}

// --- component-wise

#[test]
fn test_scale_exact() {
    // (1.5, -2.25) * 2.5 = (3.75, -5.625)
    assert!(a().scale(fx(0x280000000)) == v2(0x3c0000000, -0x5a0000000));
    assert!(a().scale(Real::ONE) == a());
    assert!(a().scale(Real::ZERO).is_zero());
}

#[test]
fn test_scale_rounds_toward_negative_infinity() {
    // [3, -3] ulp * 0.5: floor, whatever the sign.
    assert!(v2(3, -3).scale(fx(0x80000000)) == v2(1, -2));
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_scale_overflow() {
    let _ = black_box(v2(0, 0x4000000000000000)).scale(fx(0x200000000));
}

#[test]
fn test_unscale_exact() {
    // (1.5, -2.25) / 2.5 = (0.5999999999, -0.9000000001)
    assert!(a().unscale(fx(0x280000000)) == v2(2576980377, -3865470567));
    assert!(a().unscale(Real::ONE) == a());
}

#[test]
fn test_unscale_rounds_toward_negative_infinity() {
    // +-1 / 3, +-2 / 3: exact floors.
    assert!(v2(0x100000000, -0x100000000).unscale(fx(0x300000000)) == v2(1431655765, -1431655766));
}

#[test]
#[should_panic(expected: 'simba: division by zero')]
fn test_unscale_by_zero() {
    let _ = black_box(a()).unscale(Real::ZERO);
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_unscale_overflow() {
    let _ = black_box(v2(0, 0x4000000000000000)).unscale(fx(0x80000000));
}

#[test]
fn test_component_mul_exact() {
    // (1.5, -2.25) .* (-4.5, 0.25) = (-6.75, -0.5625)
    assert!(a().component_mul(b()) == v2(-0x6c0000000, -0x90000000));
    assert!(a().component_mul(Vector2Trait::<Fixed>::repeat(Real::ONE)) == a());
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_component_mul_overflow() {
    let _ = black_box(v2(0, 0x1000000000000)).component_mul(v2(0, 0x800000000000));
}

#[test]
fn test_component_div_exact() {
    // (1.5, -2.25) ./ (-4.5, 0.25) = floor of the exact quotients
    assert!(a().component_div(b()) == v2(-1431655766, -0x900000000));
    assert!(a().component_div(Vector2Trait::<Fixed>::repeat(Real::ONE)) == a());
}

#[test]
#[should_panic(expected: 'simba: division by zero')]
fn test_component_div_by_zero() {
    let _ = black_box(a()).component_div(v2(0x100000000, 0));
}

#[test]
fn test_abs_exact() {
    assert!(a().abs() == v2(0x180000000, 0x240000000));
    assert!(a().abs().abs() == a().abs());
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_abs_overflow() {
    let _ = black_box(v2(MIN, 0)).abs();
}

#[test]
fn test_inf_sup_exact() {
    // (1.5, -2.25) and (-4.5, 0.25)
    assert!(a().inf(b()) == v2(-0x480000000, -0x240000000));
    assert!(a().sup(b()) == v2(0x180000000, 0x40000000));
    assert!(a().inf_sup(b()) == (v2(-0x480000000, -0x240000000), v2(0x180000000, 0x40000000)));
    assert!(b().inf(a()) == v2(-0x480000000, -0x240000000));
    assert!(a().inf(a()) == a() && a().sup(a()) == a());
    assert!(v2(MIN, MIN).inf(v2(MAX, MAX)) == v2(MIN, MIN));
}

// --- reductions

#[test]
fn test_min_max_exact() {
    // (1.5, -2.25)
    assert!(a().min() == fx(-0x240000000));
    assert!(a().max() == fx(0x180000000));
    assert!(a().amin() == fx(0x180000000));
    assert!(a().amax() == fx(0x240000000));
}

#[test]
fn test_min_max_every_position() {
    assert!(
        v2(-0x700000000, 0x500000000).min() == fx(-0x700000000)
            && v2(-0x700000000, 0x500000000).imin() == 0,
    );
    assert!(
        v2(0x700000000, -0x500000000).max() == fx(0x700000000)
            && v2(0x700000000, -0x500000000).imax() == 0,
    );
    assert!(
        v2(-0x700000000, 0x500000000).amax() == fx(0x700000000)
            && v2(-0x700000000, 0x500000000).iamax() == 0,
    );
    assert!(
        v2(0x300000000, -0x500000000).amin() == fx(0x300000000)
            && v2(0x300000000, -0x500000000).iamin() == 0,
    );
    assert!(
        v2(0x500000000, -0x700000000).min() == fx(-0x700000000)
            && v2(0x500000000, -0x700000000).imin() == 1,
    );
    assert!(
        v2(-0x500000000, 0x700000000).max() == fx(0x700000000)
            && v2(-0x500000000, 0x700000000).imax() == 1,
    );
    assert!(
        v2(0x500000000, -0x700000000).amax() == fx(0x700000000)
            && v2(0x500000000, -0x700000000).iamax() == 1,
    );
    assert!(
        v2(-0x500000000, 0x300000000).amin() == fx(0x300000000)
            && v2(-0x500000000, 0x300000000).iamin() == 1,
    );
}

#[test]
fn test_imin_imax_ties_pick_first() {
    // Upstream returns the first extremum.
    assert!(v2(0x100000000, 0x100000000).imax() == 0 && v2(0x100000000, 0x100000000).imin() == 0);
    assert!(
        v2(-0x100000000, -0x100000000).iamax() == 0 && v2(-0x100000000, -0x100000000).iamin() == 0,
    );
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_amax_overflow() {
    let _ = black_box(v2(0, MIN)).amax();
}

#[test]
fn test_sum_exact() {
    // (1.5, -2.25)
    assert!(a().sum() == fx(-0xc0000000));
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_sum_overflow() {
    let _ = black_box(v2(0x4000000000000000, 0x4000000000000000)).sum();
}

#[test]
fn test_is_zero_cases() {
    assert!(v2(0, 0).is_zero());
    assert!(!v2(1, 0).is_zero());
    assert!(!v2(0, 1).is_zero());
}

#[test]
fn test_abs_diff_eq_counts_ulps() {
    let r = a();
    assert!(r.abs_diff_eq(r, 0));
    assert!(
        r.abs_diff_eq(v2(6442450947, -0x240000000), 3)
            && !r.abs_diff_eq(v2(6442450947, -0x240000000), 2),
    );
    assert!(
        r.abs_diff_eq(v2(0x180000000, -9663676419), 3)
            && !r.abs_diff_eq(v2(0x180000000, -9663676419), 2),
    );
    // The difference itself may exceed the scalar range.
    assert!(!v2(MIN, MIN).abs_diff_eq(v2(MAX, MAX), 0xfffffffffffffffe));
    assert!(v2(MIN, MIN).abs_diff_eq(v2(MAX, MAX), 0xffffffffffffffff));
}

// --- products

#[test]
fn test_dot_exact() {
    // (1.5, -2.25) . (-4.5, 0.25) = -7.3125
    assert!(a().dot(b()) == fx(-0x750000000));
    assert!(b().dot(a()) == fx(-0x750000000));
    assert!(a().dot(a()) == a().norm_squared());
}

#[test]
fn test_dot_rounds_once() {
    // 2 products of 0.5 ulp each: the fused sum is floor(1.0) = 1 ulp, where rounding every
    // product would give 0.
    assert!(v2(0x80000000, 0x80000000).dot(v2(1, 1)) == fx(1));
}

#[test]
fn test_dot_intermediate_products_may_overflow() {
    // 60000^2 = 3.6e9 does not fit Q32.32, the cancelled sum does.
    assert!(v2(0xea6000000000, 0xea6000000000).dot(v2(0xea6000000000, -0xea6000000000)) == fx(0));
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_dot_overflow() {
    let _ = black_box(v2(0xea6000000000, 0xea6000000000)).dot(v2(0xea6000000000, 0xea6000000000));
}

#[test]
fn test_perp_exact() {
    // (1.5, -2.25) perp (-4.5, 0.25) = -9.75
    assert!(a().perp(b()) == fx(-0x9c0000000));
    assert!(b().perp(a()) == fx(0x9c0000000));
    assert!(a().perp(a()) == Real::ZERO);
    assert!(Vector2Trait::<Fixed>::x().perp(Vector2Trait::<Fixed>::y()) == Real::ONE);
}

#[test]
fn test_perp_rounds_once() {
    // 1.25 ulp - 0.5 ulp = 0.75 ulp floors to 0; rounding each product would give 1 - 0 = 1.
    assert!(v2(0x40000000, 0x80000000).perp(v2(1, 5)) == Real::ZERO);
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_perp_overflow() {
    let _ = black_box(v2(0xea6000000000, 0xea6000000000)).perp(v2(-0xea6000000000, 0xea6000000000));
}

// --- norms

#[test]
fn test_norm_squared_exact() {
    // |(3, -4)|^2 = 25
    assert!(p().norm_squared() == fx(0x1900000000));
    assert!(p().magnitude_squared() == fx(0x1900000000));
    assert!(a().norm_squared() == fx(0x750000000));
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_norm_squared_overflow() {
    let _ = black_box(v2(0xea6000000000, 0)).norm_squared();
}

#[test]
fn test_norm_exact() {
    // |(3, -4)| = 5
    assert!(p().norm() == fx(0x500000000));
    assert!(p().magnitude() == fx(0x500000000));
    assert!(v2(0, 0).norm() == Real::ZERO);
    assert!(a().norm() == fx(11614293609));
}

#[test]
fn test_norm_large_magnitude_does_not_overflow() {
    // |(1e6, ..)| = 1414213.562: the squared norm (2e12) is far outside Q32.32.
    assert!(v2(0xf424000000000, 0xf424000000000).norm() == fx(6074000999952099));
    // Largest components.
    assert!(v2(MAX, 0).norm() == fx(MAX));
}

#[test]
fn test_norm_tiny_magnitude_keeps_precision() {
    // |(3, -4, ..) ulp| = 5 ulp: the squares vanish in Q32.32, not in the unscaled kernel.
    assert!(v2(3, -4).norm() == fx(5));
    assert!(v2(3, -4).norm_squared() == Real::ZERO);
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_norm_overflow() {
    let _ = black_box(v2(MAX, MAX)).norm();
}

#[test]
fn test_metric_distance_exact() {
    // Distance between (4.5, -6.25) and (1.5, -2.25).
    assert!(v2(0x480000000, -0x640000000).metric_distance(a()) == fx(0x500000000));
    assert!(a().metric_distance(v2(0x480000000, -0x640000000)) == fx(0x500000000));
    assert!(a().metric_distance(a()) == Real::ZERO);
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_metric_distance_overflow() {
    let _ = black_box(v2(MAX, 0)).metric_distance(v2(-1, 0));
}

#[test]
fn test_normalize_exact() {
    // (3, -4) / 5, floored.
    assert!(p().normalize() == v2(2576980377, -3435973837));
    assert!(v2(-0x500000000, 0).normalize() == -Vector2Trait::<Fixed>::x());
    assert!(v2(0, -0x500000000).normalize() == -Vector2Trait::<Fixed>::y());
}

#[test]
fn test_normalize_large_magnitude() {
    // (1e6, ..) / |(1e6, ..)|: no overflow, every component is 1 / sqrt(2) within 1 ulp.
    assert!(v2(0xf424000000000, 0xf424000000000).normalize() == v2(3037000499, 3037000499));
    assert!(
        v2(0xf424000000000, 0xf424000000000)
            .normalize()
            .abs_diff_eq(Vector2Trait::<Fixed>::repeat(fx(3037000499)), 1),
    );
}

#[test]
fn test_normalize_tiny_magnitude() {
    // (3, -4, ..) ulp: norm 5 ulp, result (0.6, -0.8, ..) floored.
    assert!(v2(3, -4).normalize() == v2(2576980377, -3435973837));
}

#[test]
#[should_panic(expected: 'simba: division by zero')]
fn test_normalize_zero() {
    let _ = black_box(Vector2Trait::<Fixed>::zeros()).normalize();
}

#[test]
fn test_normalize_is_unit_within_tolerance() {
    let r = a().normalize();
    assert!(r == v2(2382419201, -3573628803));
    assert!(r.norm().abs_diff_eq(Real::ONE, 4));
}

#[test]
fn test_try_normalize_some() {
    assert!(p().try_normalize(Real::ZERO) == Some(p().normalize()));
    assert!(p().try_normalize(fx(0x500000000) - Real::EPSILON) == Some(p().normalize()));
}

#[test]
fn test_try_normalize_none() {
    assert!(Vector2Trait::<Fixed>::zeros().try_normalize(Real::ZERO) == None);
    // The threshold is inclusive, like upstream (`norm <= min_norm`).
    assert!(p().try_normalize(fx(0x500000000)) == None);
    assert!(v2(3, -4).try_normalize(Real::EPSILON) == Some(v2(3, -4).normalize()));
    assert!(v2(3, -4).try_normalize(fx(5)) == None);
}

#[test]
fn test_cap_magnitude_exact() {
    // |(3, -4)| = 5 capped to 2.5: the ratio 0.5 is exact.
    assert!(p().cap_magnitude(fx(0x280000000)) == v2(0x180000000, -0x200000000));
    assert!(p().cap_magnitude(fx(0x280000000)).norm() == fx(0x280000000));
    // Not longer than the cap: unchanged (inclusive).
    assert!(p().cap_magnitude(fx(0x500000000)) == p());
    assert!(p().cap_magnitude(fx(MAX)) == p());
    assert!(p().cap_magnitude(Real::ZERO).is_zero());
    assert!(Vector2Trait::<Fixed>::zeros().cap_magnitude(Real::ZERO).is_zero());
}

#[test]
fn test_cap_magnitude_never_exceeds_cap_by_more_than_rounding() {
    let r = a().cap_magnitude(fx(0x200000000));
    assert!(r == v2(4764838402, -7147257604));
    assert!(r.norm() <= fx(0x200000000) && r.norm().abs_diff_eq(fx(0x200000000), 16));
}

#[test]
fn test_cap_magnitude_large_magnitude() {
    assert!(
        v2(0xf424000000000, 0xf424000000000)
            .cap_magnitude(fx(0xa00000000)) == v2(30370000000, 30370000000),
    );
}

// --- interpolation

#[test]
fn test_lerp_exact() {
    // From (1.5, -2.25) to (-4.5, 0.25).
    assert!(a().lerp(b(), Real::ZERO) == a());
    assert!(a().lerp(b(), Real::ONE) == b());
    assert!(a().lerp(b(), Real::HALF) == v2(-0x180000000, -0x100000000));
    assert!(a().lerp(b(), fx(0x40000000)) == v2(0, -0x1a0000000));
    // Not clamped: extrapolation.
    assert!(a().lerp(b(), Real::TWO) == v2(-0xa80000000, 0x2c0000000));
    assert!(a().lerp(b(), Real::NEG_ONE) == v2(0x780000000, -0x4c0000000));
}

#[test]
fn test_lerp_full_range_does_not_overflow() {
    // `rhs - self` does not fit the scalar; the kernel works on the exact difference.
    let (lo, hi) = (v2(MIN, MAX), v2(MAX, MIN));
    assert!(lo.lerp(hi, Real::ONE) == hi);
    assert!(lo.lerp(hi, Real::HALF) == v2(-1, -1));
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_lerp_overflow() {
    let _ = black_box(v2(0, 0)).lerp(v2(0x4000000000000000, 0x4000000000000000), fx(0x400000000));
}

// --- angle

#[test]
fn test_angle_cardinal_directions() {
    // 2 * atan2(|u - v|, |u + v|) on the normalized vectors, whatever their lengths. `atan2` is
    // accurate to 1.12 ulp (DESIGN D6) and normalizing the inputs costs a couple more, so the
    // cardinal angles land within 4 ulp of the exact 0, pi/2 and pi. Parallel vectors give
    // EXACTLY zero (`|u - v| = 0` and `atan2(0, x) = 0` for `x > 0`).
    let half_turn = Real::<Fixed>::PI;
    let quarter_turn = Real::<Fixed>::FRAC_PI_2;
    assert!(v2(0x300000000, 0).angle(v2(0x300000000, 0)) == Real::ZERO);
    assert!(v2(0x300000000, 0).angle(v2(0x1500000000, 0)) == Real::ZERO);
    assert!(Real::abs_diff_eq(v2(0x300000000, 0).angle(v2(0, 0x80000000)), quarter_turn, 4));
    assert!(Real::abs_diff_eq(v2(0, 0x80000000).angle(v2(-0x300000000, 0)), quarter_turn, 4));
    assert!(Real::abs_diff_eq(v2(0x300000000, 0).angle(v2(-0x300000000, 0)), half_turn, 4));
    // Lengths whose product overflows (upstream divides by `|a| * |b|`).
    assert!(
        Real::abs_diff_eq(v2(0xf424000000000, 0).angle(v2(0, 0xf424000000000)), quarter_turn, 4),
    );
}

#[test]
fn test_angle_of_zero_vector_is_zero() {
    assert!(Vector2Trait::<Fixed>::zeros().angle(v2(0x300000000, 0)) == Real::ZERO);
    assert!(v2(0x300000000, 0).angle(Vector2Trait::<Fixed>::zeros()) == Real::ZERO);
}

#[test]
fn test_angle_oracle() {
    let mut cases = oracle::vector2_angle_cases();
    assert!(cases.len() >= 12);
    while let Some(case) = cases.pop_front() {
        let (a, b, expected, tol) = *case;
        assert!(Real::abs_diff_eq(v2t(a).angle(v2t(b)), fx(expected), tol));
    }
}

// --- oracle vectors (upstream nalgebra on the same raw inputs; `tol` in ulp, 0 = bit for bit)

#[test]
fn test_add_oracle() {
    let mut cases = oracle::vector2_add_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (a, b, expected, tol) = *case;
        let (a, b) = (v2t(a), v2t(b));
        assert!((a + b).abs_diff_eq(v2t(expected), tol));
    }
}

#[test]
fn test_sub_oracle() {
    let mut cases = oracle::vector2_sub_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (a, b, expected, tol) = *case;
        let (a, b) = (v2t(a), v2t(b));
        assert!((a - b).abs_diff_eq(v2t(expected), tol));
    }
}

#[test]
fn test_neg_oracle() {
    let mut cases = oracle::vector2_neg_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (a, expected, tol) = *case;
        let a = v2t(a);
        assert!((-a).abs_diff_eq(v2t(expected), tol));
    }
}

#[test]
fn test_scale_oracle() {
    let mut cases = oracle::vector2_scale_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (a, k, expected, tol) = *case;
        let a = v2t(a);
        assert!((a.scale(fx(k))).abs_diff_eq(v2t(expected), tol));
    }
}

#[test]
fn test_dot_oracle() {
    let mut cases = oracle::vector2_dot_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (a, b, expected, tol) = *case;
        let (a, b) = (v2t(a), v2t(b));
        assert!((a.dot(b)).abs_diff_eq(fx(expected), tol));
    }
}

#[test]
fn test_norm_squared_oracle() {
    let mut cases = oracle::vector2_norm_squared_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (a, expected, tol) = *case;
        let a = v2t(a);
        assert!((a.norm_squared()).abs_diff_eq(fx(expected), tol));
    }
}

#[test]
fn test_norm_oracle() {
    let mut cases = oracle::vector2_norm_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (a, expected, tol) = *case;
        let a = v2t(a);
        assert!((a.norm()).abs_diff_eq(fx(expected), tol));
    }
}

#[test]
fn test_normalize_oracle() {
    let mut cases = oracle::vector2_normalize_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (a, expected, tol) = *case;
        let a = v2t(a);
        assert!((a.normalize()).abs_diff_eq(v2t(expected), tol));
    }
}

#[test]
fn test_lerp_oracle() {
    let mut cases = oracle::vector2_lerp_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (a, b, t, expected, tol) = *case;
        let (a, b) = (v2t(a), v2t(b));
        assert!((a.lerp(b, fx(t))).abs_diff_eq(v2t(expected), tol));
    }
}

#[test]
fn test_component_mul_oracle() {
    let mut cases = oracle::vector2_component_mul_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (a, b, expected, tol) = *case;
        let (a, b) = (v2t(a), v2t(b));
        assert!((a.component_mul(b)).abs_diff_eq(v2t(expected), tol));
    }
}

#[test]
fn test_component_div_oracle() {
    let mut cases = oracle::vector2_component_div_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (a, b, expected, tol) = *case;
        let (a, b) = (v2t(a), v2t(b));
        assert!((a.component_div(b)).abs_diff_eq(v2t(expected), tol));
    }
}

#[test]
fn test_abs_oracle() {
    let mut cases = oracle::vector2_abs_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (a, expected, tol) = *case;
        let a = v2t(a);
        assert!((a.abs()).abs_diff_eq(v2t(expected), tol));
    }
}

#[test]
fn test_min_oracle() {
    let mut cases = oracle::vector2_min_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (a, expected, tol) = *case;
        let a = v2t(a);
        assert!((a.min()).abs_diff_eq(fx(expected), tol));
    }
}

#[test]
fn test_max_oracle() {
    let mut cases = oracle::vector2_max_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (a, expected, tol) = *case;
        let a = v2t(a);
        assert!((a.max()).abs_diff_eq(fx(expected), tol));
    }
}

#[test]
fn test_inf_oracle() {
    let mut cases = oracle::vector2_inf_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (a, b, expected, tol) = *case;
        let (a, b) = (v2t(a), v2t(b));
        assert!((a.inf(b)).abs_diff_eq(v2t(expected), tol));
    }
}

#[test]
fn test_sup_oracle() {
    let mut cases = oracle::vector2_sup_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (a, b, expected, tol) = *case;
        let (a, b) = (v2t(a), v2t(b));
        assert!((a.sup(b)).abs_diff_eq(v2t(expected), tol));
    }
}

#[test]
fn test_perp_oracle() {
    let mut cases = oracle::vector2_perp_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (a, b, expected, tol) = *case;
        let (a, b) = (v2t(a), v2t(b));
        assert!((a.perp(b)).abs_diff_eq(fx(expected), tol));
    }
}
