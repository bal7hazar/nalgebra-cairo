//! Fused kernels: every sum of products of the library goes through one of these.
//!
//! The products are accumulated UNSCALED and EXACT (typed wide intervals, no range check), then
//! rounded once (floor) and checked once. Compared to `a * b + c * d`: one rounding instead of
//! two (error < 1 ulp), one overflow check instead of three, intermediate products may exceed the
//! `Fixed` range as long as the result fits, and a fraction of the gas (`sum_prod3`: x3).
//!
//! For more than 4 products, or mixed signs, use `simba::fixed::wide::Wide`.

use super::kernels;
use super::types::Fixed;

/// `floor(a^2)`. Panics with `errors::OVERFLOW`.
#[inline(always)]
pub fn sqr(a: Fixed) -> Fixed {
    Fixed { raw: kernels::mul(a.raw, a.raw) }
}

/// `floor(a0*b0 + a1*b1)`, one rounding. Panics with `errors::OVERFLOW` (result only).
/// Upstream: `Vector2::dot`.
#[inline(always)]
pub fn sum_prod2(a0: Fixed, b0: Fixed, a1: Fixed, b1: Fixed) -> Fixed {
    Fixed { raw: kernels::sum_prod2(a0.raw, b0.raw, a1.raw, b1.raw) }
}

/// `floor(a0*b0 + a1*b1 + a2*b2)`, one rounding. Panics with `errors::OVERFLOW` (result only).
/// Upstream: `Vector3::dot`, a row of `Matrix3 * Vector3`.
#[inline(always)]
pub fn sum_prod3(a0: Fixed, b0: Fixed, a1: Fixed, b1: Fixed, a2: Fixed, b2: Fixed) -> Fixed {
    Fixed { raw: kernels::sum_prod3(a0.raw, b0.raw, a1.raw, b1.raw, a2.raw, b2.raw) }
}

/// `floor(a0*b0 + a1*b1 + a2*b2 + a3*b3)`, one rounding. Panics with `errors::OVERFLOW` (result
/// only). Upstream: `Vector4::dot`, quaternion products.
#[inline(always)]
pub fn sum_prod4(
    a0: Fixed, b0: Fixed, a1: Fixed, b1: Fixed, a2: Fixed, b2: Fixed, a3: Fixed, b3: Fixed,
) -> Fixed {
    Fixed {
        raw: kernels::sum_prod4(a0.raw, b0.raw, a1.raw, b1.raw, a2.raw, b2.raw, a3.raw, b3.raw),
    }
}

/// `floor(a*b - c*d)`, one rounding. Panics with `errors::OVERFLOW` (result only).
/// Upstream: a component of `cross`, `Matrix2::determinant`, `perp`.
#[inline(always)]
pub fn diff_prod(a: Fixed, b: Fixed, c: Fixed, d: Fixed) -> Fixed {
    Fixed { raw: kernels::diff_prod(a.raw, b.raw, c.raw, d.raw) }
}

/// `floor(a*b + c)`, one rounding; equal to `a * b + c` whenever the latter does not overflow.
/// Panics with `errors::OVERFLOW` (result only). Upstream: `f64::mul_add`.
#[inline(always)]
pub fn mul_add(a: Fixed, b: Fixed, c: Fixed) -> Fixed {
    Fixed { raw: kernels::mul_add(a.raw, b.raw, c.raw) }
}

/// `floor(a*b - c)`, one rounding; equal to `a * b - c` whenever the latter does not overflow.
/// Panics with `errors::OVERFLOW` (result only).
#[inline(always)]
pub fn mul_sub(a: Fixed, b: Fixed, c: Fixed) -> Fixed {
    Fixed { raw: kernels::mul_sub(a.raw, b.raw, c.raw) }
}

/// `floor(a + (b - a) * t)`: exact difference and product, one rounding; `t` is not clamped.
/// `lerp(a, b, 0) = a` and `lerp(a, b, 1) = b` exactly. `b - a` may exceed the `Fixed` range.
/// Panics with `errors::OVERFLOW` (result only). Upstream: `lerp` (`a * (1 - t) + b * t`).
#[inline(always)]
pub fn lerp(a: Fixed, b: Fixed, t: Fixed) -> Fixed {
    Fixed { raw: kernels::lerp(a.raw, b.raw, t.raw) }
}

/// `floor(x^2 + y^2)`, one rounding. Panics with `errors::OVERFLOW`.
/// Upstream: `Vector2::norm_squared`.
#[inline(always)]
pub fn norm_squared2(x: Fixed, y: Fixed) -> Fixed {
    Fixed { raw: kernels::sum_prod2(x.raw, x.raw, y.raw, y.raw) }
}

/// `floor(x^2 + y^2 + z^2)`, one rounding. Panics with `errors::OVERFLOW`.
/// Upstream: `Vector3::norm_squared`.
#[inline(always)]
pub fn norm_squared3(x: Fixed, y: Fixed, z: Fixed) -> Fixed {
    Fixed { raw: kernels::sum_prod3(x.raw, x.raw, y.raw, y.raw, z.raw, z.raw) }
}

/// `floor(x^2 + y^2 + z^2 + w^2)`, one rounding. Panics with `errors::OVERFLOW`.
/// Upstream: `Vector4::norm_squared`.
#[inline(always)]
pub fn norm_squared4(x: Fixed, y: Fixed, z: Fixed, w: Fixed) -> Fixed {
    Fixed { raw: kernels::sum_prod4(x.raw, x.raw, y.raw, y.raw, z.raw, z.raw, w.raw, w.raw) }
}

/// `floor(sqrt(x^2 + y^2))`: integer square root of the UNSCALED exact sum of squares. No
/// rescale, one rounding, and no intermediate overflow: only the result must fit (the squared
/// norm may be far outside the `Fixed` range). Panics with `errors::OVERFLOW`.
/// Upstream: `Vector2::norm`.
#[inline(always)]
pub fn norm2(x: Fixed, y: Fixed) -> Fixed {
    Fixed { raw: kernels::norm2(x.raw, y.raw) }
}

/// `floor(sqrt(x^2 + y^2 + z^2))`, see `norm2`. Upstream: `Vector3::norm`.
#[inline(always)]
pub fn norm3(x: Fixed, y: Fixed, z: Fixed) -> Fixed {
    Fixed { raw: kernels::norm3(x.raw, y.raw, z.raw) }
}

/// `floor(sqrt(x^2 + y^2 + z^2 + w^2))`, see `norm2`. Upstream: `Vector4::norm`,
/// `Quaternion::norm`.
#[inline(always)]
pub fn norm4(x: Fixed, y: Fixed, z: Fixed, w: Fixed) -> Fixed {
    Fixed { raw: kernels::norm4(x.raw, y.raw, z.raw, w.raw) }
}

#[cfg(test)]
mod tests {
    use nalgebra_testing::black_box;
    use crate::fixed::convert::from_raw as fx;
    use crate::fixed::math::sqrt;
    use crate::fixed::types::{EPSILON, Fixed, HALF, MAX, MIN, NEG_ONE, ONE, TWO, ZERO};
    use super::{
        diff_prod, lerp, mul_add, mul_sub, norm2, norm3, norm4, norm_squared2, norm_squared3,
        norm_squared4, sqr, sum_prod2, sum_prod3, sum_prod4,
    };

    const QUARTER: Fixed = Fixed { raw: 0x40000000 };
    const THREE: Fixed = Fixed { raw: 0x300000000 };
    /// 60000: its square (3.6e9) does not fit a `Fixed`.
    const BIG: Fixed = Fixed { raw: 0xea6000000000 };

    // --- sqr
    // ---------------------------------------------------------------------------------------

    #[test]
    fn test_sqr_values() {
        assert!(sqr(fx(-0x180000000)) == fx(0x240000000));
        assert!(sqr(ZERO) == ZERO);
        assert!(sqr(-EPSILON) == ZERO);
        assert!(sqr(fx(65536)) == EPSILON);
        assert!(sqr(fx(-0xb50400000000)) == fx(0x7ffea81000000000));
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_sqr_overflow_panics() {
        sqr(black_box(fx(0xb50500000000)));
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_sqr_min_panics() {
        sqr(black_box(MIN));
    }

    // --- sum_prod
    // ----------------------------------------------------------------------------------

    #[test]
    fn test_sum_prod2_exact_values() {
        // 1.5 * -4.5 + -2.25 * 0.25 = -7.3125
        assert!(
            sum_prod2(
                fx(0x180000000), fx(-0x480000000), fx(-0x240000000), QUARTER,
            ) == fx(-0x750000000),
        );
        assert!(sum_prod2(ONE, ONE, NEG_ONE, ONE) == ZERO);
        assert!(sum_prod2(ZERO, MAX, ZERO, MIN) == ZERO);
    }

    #[test]
    fn test_sum_prod2_rounds_once() {
        // 0.5 ulp + 0.5 ulp = 1 ulp exactly: the unfused form floors each product to 0.
        assert!(sum_prod2(EPSILON, HALF, EPSILON, HALF) == EPSILON);
        assert!(EPSILON * HALF + EPSILON * HALF == ZERO);
        // -0.5 ulp + 0.25 ulp = -0.25 ulp: floor is -1 ulp.
        assert!(sum_prod2(-EPSILON, HALF, EPSILON, QUARTER) == -EPSILON);
    }

    #[test]
    fn test_sum_prod2_survives_intermediate_overflow() {
        assert!(sum_prod2(BIG, BIG, BIG, -BIG) == ZERO);
        assert!(sum_prod2(MAX, MAX, MAX, -MAX) == ZERO);
        assert!(sum_prod2(MIN, MIN, MIN, MAX) == HALF);
    }

    #[test]
    fn test_sum_prod2_result_can_be_min() {
        // 2 * (65536 * -16384) = -2^31
        let a = fx(0x1000000000000);
        let b = fx(-0x400000000000);
        assert!(sum_prod2(a, b, a, b) == MIN);
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_sum_prod2_overflow_panics() {
        let a = black_box(fx(0x1000000000000));
        let b = fx(0x400000000000);
        sum_prod2(a, b, a, b);
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_sum_prod2_one_ulp_over_max_panics() {
        sum_prod2(black_box(MAX), ONE, EPSILON, ONE);
    }

    #[test]
    fn test_sum_prod3_values() {
        // (1.5, -2.25, 3.75) . (-4.5, 0.25, 2) = 0.1875
        assert!(
            sum_prod3(
                fx(0x180000000), fx(-0x480000000), fx(-0x240000000), QUARTER, fx(0x3c0000000), TWO,
            ) == fx(0x30000000),
        );
        assert!(sum_prod3(EPSILON, HALF, EPSILON, QUARTER, EPSILON, QUARTER) == EPSILON);
        assert!(sum_prod3(-EPSILON, QUARTER, -EPSILON, QUARTER, -EPSILON, QUARTER) == -EPSILON);
        assert!(sum_prod3(BIG, BIG, BIG, -BIG, ONE, ONE) == ONE);
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_sum_prod3_overflow_panics() {
        sum_prod3(black_box(MIN), MIN, MIN, MIN, MIN, MIN);
    }

    #[test]
    fn test_sum_prod4_values() {
        // (1.5, -2.25, 3.75, -0.5) . (-4.5, 0.25, 2, 7.125) = -3.375
        assert!(
            sum_prod4(
                fx(0x180000000),
                fx(-0x480000000),
                fx(-0x240000000),
                QUARTER,
                fx(0x3c0000000),
                TWO,
                -HALF,
                fx(0x720000000),
            ) == fx(-0x360000000),
        );
        assert!(
            sum_prod4(
                EPSILON, QUARTER, EPSILON, QUARTER, EPSILON, QUARTER, EPSILON, QUARTER,
            ) == EPSILON,
        );
        let q1 = fx(0x40000001);
        assert!(
            sum_prod4(
                -EPSILON, QUARTER, -EPSILON, QUARTER, -EPSILON, QUARTER, -EPSILON, q1,
            ) == fx(-2),
        );
        assert!(sum_prod4(BIG, BIG, BIG, -BIG, BIG, BIG, -BIG, BIG) == ZERO);
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_sum_prod4_largest_accumulation_panics() {
        // 4 * 2^126 = 2^128: the largest value the typed accumulator can hold, still detected.
        sum_prod4(black_box(MIN), MIN, MIN, MIN, MIN, MIN, MIN, MIN);
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_sum_prod4_most_negative_accumulation_panics() {
        sum_prod4(black_box(MIN), MAX, MIN, MAX, MIN, MAX, MIN, MAX);
    }

    // --- diff_prod
    // ---------------------------------------------------------------------------------

    #[test]
    fn test_diff_prod_values() {
        // 1.5 * 2.5 - (-0.5 * 3) = 5.25
        assert!(diff_prod(fx(0x180000000), fx(0x280000000), -HALF, THREE) == fx(0x540000000));
        // 0.5 ulp - 1 ulp = -0.5 ulp: floor is -1 ulp.
        assert!(diff_prod(EPSILON, HALF, EPSILON, ONE) == -EPSILON);
        assert!(diff_prod(MIN, MIN, MIN, MIN) == ZERO);
        assert!(diff_prod(MIN, MAX, MAX, MIN) == ZERO);
        assert!(diff_prod(BIG, BIG, BIG, BIG) == ZERO);
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_diff_prod_overflow_panics() {
        diff_prod(black_box(MIN), MIN, MAX, MIN);
    }

    // --- mul_add / mul_sub
    // -------------------------------------------------------------------------

    #[test]
    fn test_mul_add_values() {
        assert!(mul_add(TWO, THREE, ONE) == fx(0x700000000));
        assert!(mul_add(TWO, -THREE, ONE) == fx(-0x500000000));
        assert!(mul_add(-EPSILON, HALF, ONE) == fx(0xffffffff));
        assert!(mul_add(-EPSILON, HALF, ONE) == -EPSILON * HALF + ONE);
        assert!(mul_add(ZERO, MAX, MIN) == MIN);
        assert!(mul_add(ZERO, MIN, MAX) == MAX);
    }

    #[test]
    fn test_mul_add_survives_intermediate_overflow() {
        // 3.6e9 - 2^31 = 1452516352
        assert!(mul_add(BIG, BIG, MIN) == fx(0x5693a40000000000));
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_mul_add_overflow_panics() {
        mul_add(black_box(ONE), MAX, EPSILON);
    }

    #[test]
    fn test_mul_sub_values() {
        assert!(mul_sub(TWO, THREE, ONE) == fx(0x500000000));
        assert!(mul_sub(-EPSILON, HALF, NEG_ONE) == fx(0xffffffff));
        assert!(mul_sub(BIG, BIG, MAX) == fx(0x5693a40000000001));
        assert!(mul_sub(ZERO, MAX, MAX) == MIN + EPSILON);
        assert!(mul_sub(NEG_ONE, EPSILON, MAX) == MIN);
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_mul_sub_overflow_panics() {
        mul_sub(black_box(ZERO), ZERO, MIN);
    }

    // --- lerp
    // --------------------------------------------------------------------------------------

    #[test]
    fn test_lerp_endpoints_are_exact() {
        assert!(lerp(fx(123456789), fx(-987654321), ZERO) == fx(123456789));
        assert!(lerp(fx(123456789), fx(-987654321), ONE) == fx(-987654321));
        assert!(lerp(MIN, MAX, ZERO) == MIN);
        assert!(lerp(MIN, MAX, ONE) == MAX);
        assert!(lerp(MAX, MIN, ONE) == MIN);
    }

    #[test]
    fn test_lerp_values() {
        assert!(lerp(ONE, THREE, HALF) == TWO);
        assert!(lerp(THREE, ONE, QUARTER) == fx(0x280000000));
        // Extrapolation: t is not clamped.
        assert!(lerp(ONE, THREE, TWO) == fx(0x500000000));
        assert!(lerp(ONE, THREE, NEG_ONE) == NEG_ONE);
        // Floor: 0 + (-1 ulp) / 2 = -0.5 ulp -> -1 ulp; (1 ulp) / 2 -> 0.
        assert!(lerp(ZERO, -EPSILON, HALF) == -EPSILON);
        assert!(lerp(ZERO, EPSILON, HALF) == ZERO);
        assert!(lerp(ZERO, THREE, fx(1431655765)) == fx(0xffffffff));
    }

    #[test]
    fn test_lerp_full_range_difference_does_not_overflow() {
        // b - a = 2^32 - 1 ulp does not fit a Fixed; the fused form does not care.
        assert!(lerp(MIN, MAX, HALF) == -EPSILON);
        assert!(lerp(MAX, MIN, HALF) == -EPSILON);
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_lerp_overflow_panics() {
        lerp(black_box(MIN), MAX, TWO);
    }

    // --- norm_squared
    // --------------------------------------------------------------------------------

    #[test]
    fn test_norm_squared_values() {
        assert!(norm_squared2(fx(0x180000000), fx(-0x280000000)) == fx(0x880000000));
        assert!(norm_squared3(ONE, -TWO, TWO) == fx(0x900000000));
        assert!(norm_squared4(ONE, NEG_ONE, ONE, NEG_ONE) == fx(0x400000000));
        assert!(norm_squared4(EPSILON, EPSILON, EPSILON, EPSILON) == ZERO);
        // 4 * 2^-32 (exact sum of four 2^-32 squares): the unfused form would also give 4 ulp.
        let s = fx(65536);
        assert!(norm_squared4(s, s, s, s) == fx(4));
        assert!(norm_squared2(fx(0xb50400000000), ZERO) == fx(0x7ffea81000000000));
        assert!(
            norm_squared3(
                fx(0x753000000000), fx(0x753000000000), fx(0x465000000000),
            ) == fx(0x7e99ab0000000000),
        );
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_norm_squared3_overflow_panics() {
        // 30000^2 + 30000^2 + 19000^2 = 2.161e9 > 2^31
        norm_squared3(black_box(fx(0x753000000000)), fx(0x753000000000), fx(0x4a3800000000));
    }

    // --- norm
    // ----------------------------------------------------------------------------------------

    #[test]
    fn test_norm_pythagorean_values() {
        assert!(norm2(THREE, fx(-0x400000000)) == fx(0x500000000));
        assert!(norm3(ONE, -TWO, TWO) == THREE);
        assert!(norm4(ONE, NEG_ONE, ONE, NEG_ONE) == TWO);
        assert!(norm2(ZERO, ZERO) == ZERO);
        assert!(norm3(ZERO, ZERO, ZERO) == ZERO);
        assert!(norm4(ZERO, ZERO, ZERO, ZERO) == ZERO);
    }

    #[test]
    fn test_norm_is_exactly_floored() {
        assert!(norm2(ONE, ONE) == sqrt(TWO));
        assert!(norm2(EPSILON, EPSILON) == EPSILON);
        assert!(norm2(EPSILON, ZERO) == EPSILON);
        assert!(norm3(EPSILON, EPSILON, EPSILON) == EPSILON);
        assert!(norm4(EPSILON, EPSILON, EPSILON, EPSILON) == fx(2));
    }

    #[test]
    fn test_norm_of_world_scale_vector() {
        // |(1e6, -1e6, 1e6)| = 1732050.80756..: the squares (1e12) are far outside the range.
        let a = fx(0xf424000000000);
        assert!(norm3(a, -a, a) == fx(7439101573518717));
    }

    #[test]
    fn test_norm_boundaries() {
        assert!(norm2(MAX, ZERO) == MAX);
        assert!(norm2(ZERO, -MAX) == MAX);
        let h = fx(0x3fffffffffffffff);
        assert!(norm4(h, h, h, h) == fx(0x7ffffffffffffffe));
        // Largest equal components with |v| <= MAX.
        let d = fx(6521908912666391106);
        assert!(norm2(d, d) == MAX);
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_norm2_min_panics() {
        norm2(black_box(MIN), ZERO);
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_norm2_overflow_panics() {
        norm2(black_box(MAX), MAX);
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_norm3_overflow_panics() {
        norm3(black_box(MAX), MAX, MAX);
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_norm4_largest_accumulation_panics() {
        // The sum of squares is 2^128: one past the u128 range of the integer square root.
        norm4(black_box(MIN), MIN, MIN, MIN);
    }

    // --- gas benchmarks (net = raw - baseline of the group) --------------------------------------
    // `unfused` = the same value written with operators (several roundings and checks): what the
    // fused kernel saves. Results are exact here, so both forms agree.

    #[test]
    #[inline(never)]
    fn bench_sqr__baseline() {
        let _a = black_box(fx(-0x240000000));
        let e = black_box(fx(0x510000000));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sqr__fixed() {
        let a = black_box(fx(-0x240000000));
        let e = black_box(fx(0x510000000));
        assert!(sqr(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sqr__unfused() {
        let a = black_box(fx(-0x240000000));
        let e = black_box(fx(0x510000000));
        assert!(a * a == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sum_prod2__baseline() {
        let _a0 = black_box(fx(0x180000000));
        let _b0 = black_box(fx(-0x480000000));
        let _a1 = black_box(fx(-0x240000000));
        let _b1 = black_box(fx(0x40000000));
        let e = black_box(fx(-0x750000000));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sum_prod2__fixed() {
        let a0 = black_box(fx(0x180000000));
        let b0 = black_box(fx(-0x480000000));
        let a1 = black_box(fx(-0x240000000));
        let b1 = black_box(fx(0x40000000));
        let e = black_box(fx(-0x750000000));
        assert!(sum_prod2(a0, b0, a1, b1) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sum_prod2__unfused() {
        let a0 = black_box(fx(0x180000000));
        let b0 = black_box(fx(-0x480000000));
        let a1 = black_box(fx(-0x240000000));
        let b1 = black_box(fx(0x40000000));
        let e = black_box(fx(-0x750000000));
        assert!(a0 * b0 + a1 * b1 == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sum_prod3__baseline() {
        let _a0 = black_box(fx(0x180000000));
        let _b0 = black_box(fx(-0x480000000));
        let _a1 = black_box(fx(-0x240000000));
        let _b1 = black_box(fx(0x40000000));
        let _a2 = black_box(fx(0x3c0000000));
        let _b2 = black_box(fx(0x200000000));
        let e = black_box(fx(0x30000000));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sum_prod3__fixed() {
        let a0 = black_box(fx(0x180000000));
        let b0 = black_box(fx(-0x480000000));
        let a1 = black_box(fx(-0x240000000));
        let b1 = black_box(fx(0x40000000));
        let a2 = black_box(fx(0x3c0000000));
        let b2 = black_box(fx(0x200000000));
        let e = black_box(fx(0x30000000));
        assert!(sum_prod3(a0, b0, a1, b1, a2, b2) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sum_prod3__unfused() {
        let a0 = black_box(fx(0x180000000));
        let b0 = black_box(fx(-0x480000000));
        let a1 = black_box(fx(-0x240000000));
        let b1 = black_box(fx(0x40000000));
        let a2 = black_box(fx(0x3c0000000));
        let b2 = black_box(fx(0x200000000));
        let e = black_box(fx(0x30000000));
        assert!(a0 * b0 + a1 * b1 + a2 * b2 == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sum_prod4__baseline() {
        let _a0 = black_box(fx(0x180000000));
        let _b0 = black_box(fx(-0x480000000));
        let _a1 = black_box(fx(-0x240000000));
        let _b1 = black_box(fx(0x40000000));
        let _a2 = black_box(fx(0x3c0000000));
        let _b2 = black_box(fx(0x200000000));
        let _a3 = black_box(fx(-0x80000000));
        let _b3 = black_box(fx(0x720000000));
        let e = black_box(fx(-0x360000000));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sum_prod4__fixed() {
        let a0 = black_box(fx(0x180000000));
        let b0 = black_box(fx(-0x480000000));
        let a1 = black_box(fx(-0x240000000));
        let b1 = black_box(fx(0x40000000));
        let a2 = black_box(fx(0x3c0000000));
        let b2 = black_box(fx(0x200000000));
        let a3 = black_box(fx(-0x80000000));
        let b3 = black_box(fx(0x720000000));
        let e = black_box(fx(-0x360000000));
        assert!(sum_prod4(a0, b0, a1, b1, a2, b2, a3, b3) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sum_prod4__unfused() {
        let a0 = black_box(fx(0x180000000));
        let b0 = black_box(fx(-0x480000000));
        let a1 = black_box(fx(-0x240000000));
        let b1 = black_box(fx(0x40000000));
        let a2 = black_box(fx(0x3c0000000));
        let b2 = black_box(fx(0x200000000));
        let a3 = black_box(fx(-0x80000000));
        let b3 = black_box(fx(0x720000000));
        let e = black_box(fx(-0x360000000));
        assert!(a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3 == e);
    }

    #[test]
    #[inline(never)]
    fn bench_diff_prod__baseline() {
        let _a = black_box(fx(0x180000000));
        let _b = black_box(fx(-0x480000000));
        let _c = black_box(fx(-0x240000000));
        let _d = black_box(fx(0x40000000));
        let e = black_box(fx(-0x630000000));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_diff_prod__fixed() {
        let a = black_box(fx(0x180000000));
        let b = black_box(fx(-0x480000000));
        let c = black_box(fx(-0x240000000));
        let d = black_box(fx(0x40000000));
        let e = black_box(fx(-0x630000000));
        assert!(diff_prod(a, b, c, d) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_diff_prod__unfused() {
        let a = black_box(fx(0x180000000));
        let b = black_box(fx(-0x480000000));
        let c = black_box(fx(-0x240000000));
        let d = black_box(fx(0x40000000));
        let e = black_box(fx(-0x630000000));
        assert!(a * b - c * d == e);
    }

    #[test]
    #[inline(never)]
    fn bench_mul_add__baseline() {
        let _a = black_box(fx(0x180000000));
        let _b = black_box(fx(-0x480000000));
        let _c = black_box(fx(-0x240000000));
        let e = black_box(fx(-0x900000000));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_mul_add__fixed() {
        let a = black_box(fx(0x180000000));
        let b = black_box(fx(-0x480000000));
        let c = black_box(fx(-0x240000000));
        let e = black_box(fx(-0x900000000));
        assert!(mul_add(a, b, c) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_mul_add__unfused() {
        let a = black_box(fx(0x180000000));
        let b = black_box(fx(-0x480000000));
        let c = black_box(fx(-0x240000000));
        let e = black_box(fx(-0x900000000));
        assert!(a * b + c == e);
    }

    #[test]
    #[inline(never)]
    fn bench_mul_sub__baseline() {
        let _a = black_box(fx(0x180000000));
        let _b = black_box(fx(-0x480000000));
        let _c = black_box(fx(-0x240000000));
        let e = black_box(fx(-0x480000000));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_mul_sub__fixed() {
        let a = black_box(fx(0x180000000));
        let b = black_box(fx(-0x480000000));
        let c = black_box(fx(-0x240000000));
        let e = black_box(fx(-0x480000000));
        assert!(mul_sub(a, b, c) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_mul_sub__unfused() {
        let a = black_box(fx(0x180000000));
        let b = black_box(fx(-0x480000000));
        let c = black_box(fx(-0x240000000));
        let e = black_box(fx(-0x480000000));
        assert!(a * b - c == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lerp__baseline() {
        let _a = black_box(fx(0x180000000));
        let _b = black_box(fx(-0x480000000));
        let _t = black_box(fx(0x40000000));
        let e = black_box(fx(0x0));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lerp__fixed() {
        let a = black_box(fx(0x180000000));
        let b = black_box(fx(-0x480000000));
        let t = black_box(fx(0x40000000));
        let e = black_box(fx(0x0));
        assert!(lerp(a, b, t) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lerp__unfused() {
        let a = black_box(fx(0x180000000));
        let b = black_box(fx(-0x480000000));
        let t = black_box(fx(0x40000000));
        let e = black_box(fx(0x0));
        assert!(a + (b - a) * t == e);
    }

    #[test]
    #[inline(never)]
    fn bench_norm_squared2__baseline() {
        let _x = black_box(fx(0x180000000));
        let _y = black_box(fx(-0x240000000));
        let e = black_box(fx(0x750000000));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_norm_squared2__fixed() {
        let x = black_box(fx(0x180000000));
        let y = black_box(fx(-0x240000000));
        let e = black_box(fx(0x750000000));
        assert!(norm_squared2(x, y) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_norm_squared2__unfused() {
        let x = black_box(fx(0x180000000));
        let y = black_box(fx(-0x240000000));
        let e = black_box(fx(0x750000000));
        assert!(x * x + y * y == e);
    }

    #[test]
    #[inline(never)]
    fn bench_norm_squared3__baseline() {
        let _x = black_box(fx(0x180000000));
        let _y = black_box(fx(-0x240000000));
        let _z = black_box(fx(0x3c0000000));
        let e = black_box(fx(0x1560000000));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_norm_squared3__fixed() {
        let x = black_box(fx(0x180000000));
        let y = black_box(fx(-0x240000000));
        let z = black_box(fx(0x3c0000000));
        let e = black_box(fx(0x1560000000));
        assert!(norm_squared3(x, y, z) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_norm_squared3__unfused() {
        let x = black_box(fx(0x180000000));
        let y = black_box(fx(-0x240000000));
        let z = black_box(fx(0x3c0000000));
        let e = black_box(fx(0x1560000000));
        assert!(x * x + y * y + z * z == e);
    }

    #[test]
    #[inline(never)]
    fn bench_norm_squared4__baseline() {
        let _x = black_box(fx(0x180000000));
        let _y = black_box(fx(-0x240000000));
        let _z = black_box(fx(0x3c0000000));
        let _w = black_box(fx(-0x80000000));
        let e = black_box(fx(0x15a0000000));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_norm_squared4__fixed() {
        let x = black_box(fx(0x180000000));
        let y = black_box(fx(-0x240000000));
        let z = black_box(fx(0x3c0000000));
        let w = black_box(fx(-0x80000000));
        let e = black_box(fx(0x15a0000000));
        assert!(norm_squared4(x, y, z, w) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_norm2__baseline() {
        let _x = black_box(fx(0x180000000));
        let _y = black_box(fx(-0x240000000));
        let e = black_box(fx(0x2b4440e69));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_norm2__fixed() {
        let x = black_box(fx(0x180000000));
        let y = black_box(fx(-0x240000000));
        let e = black_box(fx(0x2b4440e69));
        assert!(norm2(x, y) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_norm2__unfused() {
        let x = black_box(fx(0x180000000));
        let y = black_box(fx(-0x240000000));
        let e = black_box(fx(0x2b4440e69));
        assert!(sqrt(x * x + y * y) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_norm3__baseline() {
        let _x = black_box(fx(0x180000000));
        let _y = black_box(fx(-0x240000000));
        let _z = black_box(fx(0x3c0000000));
        let e = black_box(fx(0x49f9146ee));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_norm3__fixed() {
        let x = black_box(fx(0x180000000));
        let y = black_box(fx(-0x240000000));
        let z = black_box(fx(0x3c0000000));
        let e = black_box(fx(0x49f9146ee));
        assert!(norm3(x, y, z) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_norm3__unfused() {
        let x = black_box(fx(0x180000000));
        let y = black_box(fx(-0x240000000));
        let z = black_box(fx(0x3c0000000));
        let e = black_box(fx(0x49f9146ee));
        assert!(sqrt(x * x + y * y + z * z) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_norm4__baseline() {
        let _x = black_box(fx(0x180000000));
        let _y = black_box(fx(-0x240000000));
        let _z = black_box(fx(0x3c0000000));
        let _w = black_box(fx(-0x80000000));
        let e = black_box(fx(0x4a6780446));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_norm4__fixed() {
        let x = black_box(fx(0x180000000));
        let y = black_box(fx(-0x240000000));
        let z = black_box(fx(0x3c0000000));
        let w = black_box(fx(-0x80000000));
        let e = black_box(fx(0x4a6780446));
        assert!(norm4(x, y, z, w) == e);
    }
}
