//! `Wide`: the explicit unscaled accumulator for sums of more than 4 products.

use super::kernels;
use super::types::Fixed;

/// Exact accumulator of unscaled products (scale 2^64), e.g. for the 6-term rows of `Matrix6`
/// or dynamic dot products: accumulate with `add_prod` / `sub_prod` / `add` / `sub`, then round
/// and check ONCE with `rescale` (or `sqrt` for a norm, `mul_scalar` for a triple product).
///
/// Representation: a `felt252` holding the signed sum (negatives as `P - |x|`). Accumulating
/// costs one felt multiplication and one felt addition per product, without any range check;
/// `rescale` costs the same as a plain `Fixed` multiplication's.
///
/// Safety (no silent wrap-around): the field is private and the only operations that return a
/// `Wide` add ONE term of magnitude `<= 2^126` each (there is deliberately no `Wide + Wide`, no
/// `Wide`-returning scaling, no `Serde`). After `n` operations the magnitude is `<= n * 2^126`;
/// aliasing modulo `P ~ 2^251` would need `n >= 2^124` operations, which no execution can
/// perform. The terminal operations then range-check the exact value, so an out-of-range result
/// always panics:
/// - `rescale` accepts the exact sum in `[-2^95, 2^95)`, `sqrt` in `[0, 2^126)`;
/// - `mul_scalar(s)` multiplies the felt by `|s| <= 2^63` first: `|sum * s| <= n * 2^189`, still
///   exact (no alias modulo `P > 2^251`) for any `n <= 2^61`, a bound no execution can reach
///   either. It accepts the exact product iff `sum * s` is in `[-2^127, 2^127)`, the interval
///   whose floor by `2^64` fits `i64`; anything else lands outside `[0, 2^128)` after the bias
///   (below `2^250 + 2^127` if positive, above `P - 2^250 + 2^127` if negative) and panics.
#[derive(Copy, Drop)]
pub struct Wide {
    sum: felt252,
}

#[generate_trait]
pub impl WideImpl of WideTrait {
    /// The empty sum.
    #[inline(always)]
    fn zero() -> Wide {
        Wide { sum: 0 }
    }

    /// The exact product `a * b`.
    #[inline(always)]
    fn from_prod(a: Fixed, b: Fixed) -> Wide {
        Wide { sum: kernels::wide_prod(a.raw, b.raw) }
    }

    /// The value `c`, aligned with the products.
    #[inline(always)]
    fn from_fixed(c: Fixed) -> Wide {
        Wide { sum: kernels::wide_from(c.raw) }
    }

    /// `self + a * b`, exact.
    #[inline(always)]
    fn add_prod(self: Wide, a: Fixed, b: Fixed) -> Wide {
        Wide { sum: self.sum + kernels::wide_prod(a.raw, b.raw) }
    }

    /// `self - a * b`, exact.
    #[inline(always)]
    fn sub_prod(self: Wide, a: Fixed, b: Fixed) -> Wide {
        Wide { sum: self.sum - kernels::wide_prod(a.raw, b.raw) }
    }

    /// `self + c`, exact.
    #[inline(always)]
    fn add(self: Wide, c: Fixed) -> Wide {
        Wide { sum: self.sum + kernels::wide_from(c.raw) }
    }

    /// `self - c`, exact.
    #[inline(always)]
    fn sub(self: Wide, c: Fixed) -> Wide {
        Wide { sum: self.sum - kernels::wide_from(c.raw) }
    }

    /// `floor(self)` as a `Fixed`: the single rounding and the single overflow check of the
    /// whole accumulation. Panics with `errors::OVERFLOW`.
    #[inline(always)]
    fn rescale(self: Wide) -> Fixed {
        Fixed { raw: kernels::rescale(self.sum) }
    }

    /// `floor(self * s)` as a `Fixed`: the exact accumulated value times a scalar, rounded ONCE.
    /// Terminal: it consumes the accumulator and returns a `Fixed` (no `Wide` is ever scaled).
    ///
    /// Use case: exact triple products `(a * b - c * d) * e` with a single rounding, e.g. the
    /// cofactor expansions of 4x4 / 6x6 determinants, where `diff_prod(a, b, c, d) * e` would
    /// round twice (and `self.rescale() * s` too). Also covers accumulators far beyond the `Fixed`
    /// range that come back in range after the multiplication by a small scalar: only the final
    /// result is checked (see the safety paragraph of `Wide`). Panics with `errors::OVERFLOW` iff
    /// the floored result does not fit `Fixed`. No upstream equivalent (upstream floats round at
    /// every operation).
    #[inline(always)]
    fn mul_scalar(self: Wide, s: Fixed) -> Fixed {
        Fixed { raw: kernels::wide_mul_rescale(self.sum, s.raw) }
    }

    /// `floor(sqrt(self))`: integer square root of the unscaled sum (no rescale), for norms of
    /// long vectors. Panics with `errors::SQRT_OF_NEGATIVE` / `errors::OVERFLOW`.
    #[inline(always)]
    fn sqrt(self: Wide) -> Fixed {
        Fixed { raw: kernels::wide_sqrt(self.sum) }
    }
}

#[cfg(test)]
mod tests {
    use nalgebra_testing::black_box;
    use crate::fixed::convert::from_raw as fx;
    use crate::fixed::fused;
    use crate::fixed::types::{EPSILON, Fixed, HALF, MAX, MIN, NEG_ONE, ONE, TWO, ZERO};
    use super::{Wide, WideTrait};

    const THREE: Fixed = Fixed { raw: 0x300000000 };
    const EIGHTH: Fixed = Fixed { raw: 0x20000000 };

    #[test]
    fn test_zero_rescales_to_zero() {
        assert!(WideTrait::zero().rescale() == ZERO);
        assert!(WideTrait::zero().sqrt() == ZERO);
    }

    #[test]
    fn test_from_prod_is_mul() {
        let a = fx(-0x380000000);
        let b = fx(0x240000000);
        assert!(WideTrait::from_prod(a, b).rescale() == a * b);
        assert!(WideTrait::from_prod(-EPSILON, HALF).rescale() == -EPSILON);
    }

    #[test]
    fn test_from_fixed_roundtrip() {
        assert!(WideTrait::from_fixed(MAX).rescale() == MAX);
        assert!(WideTrait::from_fixed(MIN).rescale() == MIN);
        assert!(WideTrait::from_fixed(-EPSILON).rescale() == -EPSILON);
    }

    #[test]
    fn test_add_prod_matches_fused_kernels() {
        let (a0, b0, a1, b1) = (
            fx(0x180000000), fx(-0x480000000), fx(-0x240000000), fx(0x40000000),
        );
        let (a2, b2, a3, b3) = (fx(0x3c0000000), TWO, -HALF, fx(0x720000000));
        let w = WideTrait::zero().add_prod(a0, b0).add_prod(a1, b1);
        assert!(w.rescale() == fused::sum_prod2(a0, b0, a1, b1));
        let w = w.add_prod(a2, b2);
        assert!(w.rescale() == fused::sum_prod3(a0, b0, a1, b1, a2, b2));
        let w = w.add_prod(a3, b3);
        assert!(w.rescale() == fused::sum_prod4(a0, b0, a1, b1, a2, b2, a3, b3));
        assert!(w.rescale() == fx(-0x360000000));
    }

    #[test]
    fn test_mixed_accumulation() {
        // 2 * 3 - 0.5 * 2 + 1 - 3 = 3
        let w = WideTrait::from_prod(TWO, THREE).sub_prod(HALF, TWO).add(ONE).sub(THREE);
        assert!(w.rescale() == THREE);
        assert!(WideTrait::zero().sub_prod(TWO, THREE).rescale() == fx(-0x600000000));
        assert!(WideTrait::zero().sub(MAX).rescale() == MIN + EPSILON);
    }

    #[test]
    fn test_rescale_rounds_once_over_eight_products() {
        // 8 * (1 ulp * 1/8) = 1 ulp exactly; any per-product rounding would give 0.
        let mut w = WideTrait::zero();
        for _ in 0..8_u8 {
            w = w.add_prod(EPSILON, EIGHTH);
        }
        assert!(w.rescale() == EPSILON);
        // -(7/8) ulp floors to -1 ulp.
        let mut w = WideTrait::zero();
        for _ in 0..7_u8 {
            w = w.sub_prod(EPSILON, EIGHTH);
        }
        assert!(w.rescale() == -EPSILON);
    }

    #[test]
    fn test_accumulation_far_beyond_the_fixed_range() {
        // 64 * MIN^2 = 2^132, then back to 1: the intermediate never wraps nor panics.
        let mut w = WideTrait::from_fixed(ONE);
        for _ in 0..64_u8 {
            w = w.add_prod(MIN, MIN);
        }
        for _ in 0..64_u8 {
            w = w.sub_prod(MIN, MIN);
        }
        assert!(w.rescale() == ONE);
    }

    #[test]
    fn test_rescale_boundaries() {
        assert!(WideTrait::from_fixed(MAX).add(MIN).rescale() == -EPSILON);
        // MAX + 1 ulp - 1 ulp, in the wide domain.
        assert!(WideTrait::from_fixed(MAX).add(EPSILON).sub(EPSILON).rescale() == MAX);
        // MAX + 0.5 ulp floors to MAX.
        assert!(WideTrait::from_fixed(MAX).add_prod(EPSILON, HALF).rescale() == MAX);
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_rescale_overflow_panics() {
        black_box(WideTrait::from_fixed(MAX)).add(EPSILON).rescale();
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_rescale_underflow_panics() {
        // MIN - 0.5 ulp floors to MIN - 1 ulp.
        black_box(WideTrait::from_fixed(MIN)).sub_prod(EPSILON, HALF).rescale();
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_rescale_eight_full_range_products_panics() {
        let mut w: Wide = black_box(WideTrait::zero());
        for _ in 0..8_u8 {
            w = w.add_prod(MIN, MIN);
        }
        w.rescale();
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_rescale_large_negative_panics() {
        let mut w: Wide = black_box(WideTrait::zero());
        for _ in 0..8_u8 {
            w = w.add_prod(MIN, MAX);
        }
        w.rescale();
    }

    #[test]
    fn test_wide_mul_scalar_single_product_times_one_is_mul() {
        let (a, b) = (fx(-0x380000000), fx(0x240000001));
        assert!(WideTrait::from_prod(a, b).mul_scalar(ONE) == a * b);
        assert!(WideTrait::from_prod(-EPSILON, HALF).mul_scalar(ONE) == -EPSILON);
        assert!(WideTrait::from_prod(MAX, EPSILON).mul_scalar(ONE) == MAX * EPSILON);
        assert!(WideTrait::from_fixed(MIN).mul_scalar(ONE) == MIN);
        assert!(WideTrait::from_fixed(MAX).mul_scalar(ONE) == MAX);
    }

    #[test]
    fn test_wide_mul_scalar_rounds_once() {
        // (0.5 ulp) * 3 = 1.5 ulp floors to 1 ulp; rescaling first floors 0.5 ulp to 0.
        let w = WideTrait::from_prod(EPSILON, HALF);
        assert!(w.mul_scalar(THREE) == EPSILON);
        assert!(w.rescale() * THREE == ZERO);
        // (-0.5 ulp) * 3 = -1.5 ulp floors to -2 ulp; rescaling first gives -3 ulp.
        let w = WideTrait::zero().sub_prod(EPSILON, HALF);
        assert!(w.mul_scalar(THREE) == fx(-2));
        assert!(w.rescale() * THREE == fx(-3));
        // Triple product: one ulp away from `diff_prod(a, b, c, d) * e` (Python model).
        let (a, b, c, d) = (fx(0x1a2b3c4d5), fx(-0x2345678ab), fx(0x3456789), fx(0x7fedcba98));
        let e = fx(-0x2c3d4e5f7);
        assert!(WideTrait::from_prod(a, b).sub_prod(c, d).mul_scalar(e) == fx(0xa40659288));
        assert!(fused::diff_prod(a, b, c, d) * e == fx(0xa40659289));
    }

    #[test]
    fn test_wide_mul_scalar_far_beyond_the_fixed_range() {
        // MIN^2 = 2^126 (unscaled), times 1 ulp: 2^62 raw.
        assert!(WideTrait::from_prod(MIN, MIN).mul_scalar(EPSILON) == fx(0x4000000000000000));
        // 2 * MIN^2 = 2^127, times -1 ulp: exactly MIN.
        let w = WideTrait::from_prod(MIN, MIN).add_prod(MIN, MIN);
        assert!(w.mul_scalar(-EPSILON) == MIN);
        // 2^127 - 1, times 1 ulp: exactly MAX.
        assert!(w.sub_prod(EPSILON, EPSILON).mul_scalar(EPSILON) == MAX);
        // 64 * MIN^2 = 2^132 times 0.
        let mut w = WideTrait::zero();
        for _ in 0..64_u8 {
            w = w.add_prod(MIN, MIN);
        }
        assert!(w.mul_scalar(ZERO) == ZERO);
        assert!(WideTrait::zero().mul_scalar(MIN) == ZERO);
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_wide_mul_scalar_overflow_panics() {
        // 2^127 * 1 ulp = MAX + 1 ulp.
        black_box(WideTrait::from_prod(MIN, MIN)).add_prod(MIN, MIN).mul_scalar(EPSILON);
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_wide_mul_scalar_underflow_panics() {
        // -(2^127 + 1) * 1 ulp floors to MIN - 1 ulp.
        let w = black_box(WideTrait::from_prod(MIN, MIN))
            .add_prod(MIN, MIN)
            .add_prod(EPSILON, EPSILON);
        w.mul_scalar(-EPSILON);
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_wide_mul_scalar_large_accumulator_panics() {
        // 64 * MIN^2 = 2^132 times -1 ulp: far below MIN, well inside the aliasing-free domain.
        let mut w: Wide = black_box(WideTrait::zero());
        for _ in 0..64_u8 {
            w = w.add_prod(MIN, MIN);
        }
        w.mul_scalar(-EPSILON);
    }

    #[test]
    fn test_sqrt_is_norm_of_long_vectors() {
        // |(1, 2, 3, 4, 5, 6)| = sqrt(91) = 9.539392..
        let mut w = WideTrait::zero();
        for i in 1..7_i32 {
            let x: Fixed = i.into();
            w = w.add_prod(x, x);
        }
        assert!(w.sqrt() == fx(40971376724)); // isqrt(91 * 2^64), from the Python model
        // |(2, 2, 2, 2, 2, 2, 2, 2, 2)| = 6
        let mut w = WideTrait::zero();
        for _ in 0..9_u8 {
            w = w.add_prod(-TWO, -TWO);
        }
        assert!(w.sqrt() == fx(0x600000000));
        assert!(WideTrait::from_prod(MAX, MAX).sqrt() == MAX);
        assert!(WideTrait::from_fixed(TWO).sqrt() == crate::fixed::types::SQRT_2);
    }

    #[test]
    #[should_panic(expected: 'simba: sqrt of negative')]
    fn test_sqrt_negative_panics() {
        black_box(WideTrait::zero()).sub_prod(EPSILON, EPSILON).sqrt();
    }

    #[test]
    #[should_panic(expected: 'simba: sqrt of negative')]
    fn test_sqrt_large_negative_panics() {
        black_box(WideTrait::from_prod(MIN, MAX)).add_prod(MIN, MAX).add_prod(MIN, MAX).sqrt();
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_sqrt_overflow_panics() {
        black_box(WideTrait::from_prod(MIN, MIN)).sqrt();
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_sqrt_beyond_u128_panics() {
        let mut w: Wide = black_box(WideTrait::zero());
        for _ in 0..6_u8 {
            w = w.add_prod(MIN, MIN);
        }
        w.sqrt();
    }

    #[test]
    fn test_sub_and_neg_one_products() {
        assert!(WideTrait::zero().add_prod(NEG_ONE, NEG_ONE).sub(ONE).rescale() == ZERO);
    }

    // --- gas benchmarks (net = raw - baseline of the group) --------------------------------------
    // Typed-accumulator and chunked candidates: `bench_sum_prod8__alt_*` in
    // `fixed::kernels::alternatives`.

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
    fn bench_sum_prod4__wide() {
        let a0 = black_box(fx(0x180000000));
        let b0 = black_box(fx(-0x480000000));
        let a1 = black_box(fx(-0x240000000));
        let b1 = black_box(fx(0x40000000));
        let a2 = black_box(fx(0x3c0000000));
        let b2 = black_box(fx(0x200000000));
        let a3 = black_box(fx(-0x80000000));
        let b3 = black_box(fx(0x720000000));
        let e = black_box(fx(-0x360000000));
        assert!(
            WideTrait::zero()
                .add_prod(a0, b0)
                .add_prod(a1, b1)
                .add_prod(a2, b2)
                .add_prod(a3, b3)
                .rescale() == e,
        );
    }

    #[test]
    #[inline(never)]
    fn bench_sum_prod4__wide_from_prod() {
        let a0 = black_box(fx(0x180000000));
        let b0 = black_box(fx(-0x480000000));
        let a1 = black_box(fx(-0x240000000));
        let b1 = black_box(fx(0x40000000));
        let a2 = black_box(fx(0x3c0000000));
        let b2 = black_box(fx(0x200000000));
        let a3 = black_box(fx(-0x80000000));
        let b3 = black_box(fx(0x720000000));
        let e = black_box(fx(-0x360000000));
        assert!(
            WideTrait::from_prod(a0, b0)
                .add_prod(a1, b1)
                .add_prod(a2, b2)
                .add_prod(a3, b3)
                .rescale() == e,
        );
    }

    #[test]
    #[inline(never)]
    fn bench_sum_prod4__fused_kernel() {
        let a0 = black_box(fx(0x180000000));
        let b0 = black_box(fx(-0x480000000));
        let a1 = black_box(fx(-0x240000000));
        let b1 = black_box(fx(0x40000000));
        let a2 = black_box(fx(0x3c0000000));
        let b2 = black_box(fx(0x200000000));
        let a3 = black_box(fx(-0x80000000));
        let b3 = black_box(fx(0x720000000));
        let e = black_box(fx(-0x360000000));
        assert!(fused::sum_prod4(a0, b0, a1, b1, a2, b2, a3, b3) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sum_prod6__baseline() {
        let _a0 = black_box(fx(0x180000000));
        let _b0 = black_box(fx(-0x480000000));
        let _a1 = black_box(fx(-0x240000000));
        let _b1 = black_box(fx(0x40000000));
        let _a2 = black_box(fx(0x3c0000000));
        let _b2 = black_box(fx(0x200000000));
        let _a3 = black_box(fx(-0x80000000));
        let _b3 = black_box(fx(0x720000000));
        let _a4 = black_box(fx(0x20000000));
        let _b4 = black_box(fx(-0x300000000));
        let _a5 = black_box(fx(0x680000000));
        let _b5 = black_box(fx(0x1c0000000));
        let e = black_box(fx(0x7a0000000));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sum_prod6__wide() {
        let a0 = black_box(fx(0x180000000));
        let b0 = black_box(fx(-0x480000000));
        let a1 = black_box(fx(-0x240000000));
        let b1 = black_box(fx(0x40000000));
        let a2 = black_box(fx(0x3c0000000));
        let b2 = black_box(fx(0x200000000));
        let a3 = black_box(fx(-0x80000000));
        let b3 = black_box(fx(0x720000000));
        let a4 = black_box(fx(0x20000000));
        let b4 = black_box(fx(-0x300000000));
        let a5 = black_box(fx(0x680000000));
        let b5 = black_box(fx(0x1c0000000));
        let e = black_box(fx(0x7a0000000));
        assert!(
            WideTrait::zero()
                .add_prod(a0, b0)
                .add_prod(a1, b1)
                .add_prod(a2, b2)
                .add_prod(a3, b3)
                .add_prod(a4, b4)
                .add_prod(a5, b5)
                .rescale() == e,
        );
    }

    #[test]
    #[inline(never)]
    fn bench_sum_prod6__wide_from_prod() {
        let a0 = black_box(fx(0x180000000));
        let b0 = black_box(fx(-0x480000000));
        let a1 = black_box(fx(-0x240000000));
        let b1 = black_box(fx(0x40000000));
        let a2 = black_box(fx(0x3c0000000));
        let b2 = black_box(fx(0x200000000));
        let a3 = black_box(fx(-0x80000000));
        let b3 = black_box(fx(0x720000000));
        let a4 = black_box(fx(0x20000000));
        let b4 = black_box(fx(-0x300000000));
        let a5 = black_box(fx(0x680000000));
        let b5 = black_box(fx(0x1c0000000));
        let e = black_box(fx(0x7a0000000));
        assert!(
            WideTrait::from_prod(a0, b0)
                .add_prod(a1, b1)
                .add_prod(a2, b2)
                .add_prod(a3, b3)
                .add_prod(a4, b4)
                .add_prod(a5, b5)
                .rescale() == e,
        );
    }

    #[test]
    #[inline(never)]
    fn bench_sum_prod8__baseline() {
        let _a0 = black_box(fx(0x180000000));
        let _b0 = black_box(fx(-0x480000000));
        let _a1 = black_box(fx(-0x240000000));
        let _b1 = black_box(fx(0x40000000));
        let _a2 = black_box(fx(0x3c0000000));
        let _b2 = black_box(fx(0x200000000));
        let _a3 = black_box(fx(-0x80000000));
        let _b3 = black_box(fx(0x720000000));
        let _a4 = black_box(fx(0x20000000));
        let _b4 = black_box(fx(-0x300000000));
        let _a5 = black_box(fx(0x680000000));
        let _b5 = black_box(fx(0x1c0000000));
        let _a6 = black_box(fx(-0x740000000));
        let _b6 = black_box(fx(0x80000000));
        let _a7 = black_box(fx(0x200000000));
        let _b7 = black_box(fx(-0x880000000));
        let e = black_box(fx(-0xd00000000));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sum_prod8__wide() {
        let a0 = black_box(fx(0x180000000));
        let b0 = black_box(fx(-0x480000000));
        let a1 = black_box(fx(-0x240000000));
        let b1 = black_box(fx(0x40000000));
        let a2 = black_box(fx(0x3c0000000));
        let b2 = black_box(fx(0x200000000));
        let a3 = black_box(fx(-0x80000000));
        let b3 = black_box(fx(0x720000000));
        let a4 = black_box(fx(0x20000000));
        let b4 = black_box(fx(-0x300000000));
        let a5 = black_box(fx(0x680000000));
        let b5 = black_box(fx(0x1c0000000));
        let a6 = black_box(fx(-0x740000000));
        let b6 = black_box(fx(0x80000000));
        let a7 = black_box(fx(0x200000000));
        let b7 = black_box(fx(-0x880000000));
        let e = black_box(fx(-0xd00000000));
        assert!(
            WideTrait::zero()
                .add_prod(a0, b0)
                .add_prod(a1, b1)
                .add_prod(a2, b2)
                .add_prod(a3, b3)
                .add_prod(a4, b4)
                .add_prod(a5, b5)
                .add_prod(a6, b6)
                .add_prod(a7, b7)
                .rescale() == e,
        );
    }

    #[test]
    #[inline(never)]
    fn bench_sum_prod8__wide_from_prod() {
        let a0 = black_box(fx(0x180000000));
        let b0 = black_box(fx(-0x480000000));
        let a1 = black_box(fx(-0x240000000));
        let b1 = black_box(fx(0x40000000));
        let a2 = black_box(fx(0x3c0000000));
        let b2 = black_box(fx(0x200000000));
        let a3 = black_box(fx(-0x80000000));
        let b3 = black_box(fx(0x720000000));
        let a4 = black_box(fx(0x20000000));
        let b4 = black_box(fx(-0x300000000));
        let a5 = black_box(fx(0x680000000));
        let b5 = black_box(fx(0x1c0000000));
        let a6 = black_box(fx(-0x740000000));
        let b6 = black_box(fx(0x80000000));
        let a7 = black_box(fx(0x200000000));
        let b7 = black_box(fx(-0x880000000));
        let e = black_box(fx(-0xd00000000));
        assert!(
            WideTrait::from_prod(a0, b0)
                .add_prod(a1, b1)
                .add_prod(a2, b2)
                .add_prod(a3, b3)
                .add_prod(a4, b4)
                .add_prod(a5, b5)
                .add_prod(a6, b6)
                .add_prod(a7, b7)
                .rescale() == e,
        );
    }

    #[test]
    #[inline(never)]
    fn bench_wide_norm6__baseline() {
        let _a0 = black_box(fx(0x180000000));
        let _a1 = black_box(fx(-0x240000000));
        let _a2 = black_box(fx(0x3c0000000));
        let _a3 = black_box(fx(-0x80000000));
        let _a4 = black_box(fx(0x20000000));
        let _a5 = black_box(fx(0x680000000));
        let e = black_box(fx(0x7fe3fcef5));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_wide_norm6__wide() {
        let a0 = black_box(fx(0x180000000));
        let a1 = black_box(fx(-0x240000000));
        let a2 = black_box(fx(0x3c0000000));
        let a3 = black_box(fx(-0x80000000));
        let a4 = black_box(fx(0x20000000));
        let a5 = black_box(fx(0x680000000));
        let e = black_box(fx(0x7fe3fcef5));
        assert!(
            WideTrait::zero()
                .add_prod(a0, a0)
                .add_prod(a1, a1)
                .add_prod(a2, a2)
                .add_prod(a3, a3)
                .add_prod(a4, a4)
                .add_prod(a5, a5)
                .sqrt() == e,
        );
    }

    #[test]
    #[inline(never)]
    fn bench_wide_mixed__baseline() {
        let _a = black_box(fx(0x180000000));
        let _b = black_box(fx(-0x480000000));
        let _c = black_box(fx(-0x240000000));
        let _d = black_box(fx(0x40000000));
        let _k = black_box(fx(0x3c0000000));
        let _l = black_box(fx(-0x80000000));
        let e = black_box(fx(-0x1f0000000));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_wide_mixed__wide() {
        let a = black_box(fx(0x180000000));
        let b = black_box(fx(-0x480000000));
        let c = black_box(fx(-0x240000000));
        let d = black_box(fx(0x40000000));
        let k = black_box(fx(0x3c0000000));
        let l = black_box(fx(-0x80000000));
        let e = black_box(fx(-0x1f0000000));
        assert!(WideTrait::from_prod(a, b).sub_prod(c, d).add(k).sub(l).rescale() == e);
    }


    #[test]
    #[inline(never)]
    fn bench_wide_mul_scalar__baseline() {
        let _a = black_box(fx(0x1a2b3c4d5));
        let _b = black_box(fx(-0x2345678ab));
        let _s = black_box(fx(-0x2c3d4e5f7));
        let e = black_box(fx(0x9f814b225));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_wide_mul_scalar__fused() {
        let a = black_box(fx(0x1a2b3c4d5));
        let b = black_box(fx(-0x2345678ab));
        let s = black_box(fx(-0x2c3d4e5f7));
        let e = black_box(fx(0x9f814b225));
        assert!(WideTrait::from_prod(a, b).mul_scalar(s) == e);
    }

    /// Two roundings: 2 ulp away here.
    #[test]
    #[inline(never)]
    fn bench_wide_mul_scalar__alt_rescale_then_mul() {
        let a = black_box(fx(0x1a2b3c4d5));
        let b = black_box(fx(-0x2345678ab));
        let s = black_box(fx(-0x2c3d4e5f7));
        let e = black_box(fx(0x9f814b227));
        assert!(WideTrait::from_prod(a, b).rescale() * s == e);
    }

    #[test]
    #[inline(never)]
    fn bench_triple_product__baseline() {
        let _a = black_box(fx(0x1a2b3c4d5));
        let _b = black_box(fx(-0x2345678ab));
        let _c = black_box(fx(0x3456789));
        let _d = black_box(fx(0x7fedcba98));
        let _s = black_box(fx(-0x2c3d4e5f7));
        let e = black_box(fx(0xa40659288));
        assert!(e == e);
    }

    /// `(a * b - c * d) * s`, one rounding.
    #[test]
    #[inline(never)]
    fn bench_triple_product__wide_mul_scalar() {
        let a = black_box(fx(0x1a2b3c4d5));
        let b = black_box(fx(-0x2345678ab));
        let c = black_box(fx(0x3456789));
        let d = black_box(fx(0x7fedcba98));
        let s = black_box(fx(-0x2c3d4e5f7));
        let e = black_box(fx(0xa40659288));
        assert!(WideTrait::from_prod(a, b).sub_prod(c, d).mul_scalar(s) == e);
    }

    /// Two roundings: 1 ulp away here.
    #[test]
    #[inline(never)]
    fn bench_triple_product__alt_diff_prod_then_mul() {
        let a = black_box(fx(0x1a2b3c4d5));
        let b = black_box(fx(-0x2345678ab));
        let c = black_box(fx(0x3456789));
        let d = black_box(fx(0x7fedcba98));
        let s = black_box(fx(-0x2c3d4e5f7));
        let e = black_box(fx(0xa40659289));
        assert!(fused::diff_prod(a, b, c, d) * s == e);
    }
}
