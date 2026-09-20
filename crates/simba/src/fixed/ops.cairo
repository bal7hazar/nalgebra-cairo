//! Operators of `Fixed`: `+ - * / %`, unary `-`, compound assignments and ordering.

use core::ops::{AddAssign, DivAssign, MulAssign, RemAssign, SubAssign};
use super::kernels;
use super::types::Fixed;

/// `a + b`, exact. Panics with `errors::OVERFLOW`. Upstream: `Add for f64`.
pub impl FixedAdd of Add<Fixed> {
    #[inline(always)]
    fn add(lhs: Fixed, rhs: Fixed) -> Fixed {
        Fixed { raw: kernels::add(lhs.raw, rhs.raw) }
    }
}

/// `a - b`, exact. Panics with `errors::OVERFLOW`. Upstream: `Sub for f64`.
pub impl FixedSub of Sub<Fixed> {
    #[inline(always)]
    fn sub(lhs: Fixed, rhs: Fixed) -> Fixed {
        Fixed { raw: kernels::sub(lhs.raw, rhs.raw) }
    }
}

/// `-a`, exact. Panics with `errors::OVERFLOW` for `MIN`. Upstream: `Neg for f64`.
pub impl FixedNeg of Neg<Fixed> {
    #[inline(always)]
    fn neg(a: Fixed) -> Fixed {
        Fixed { raw: kernels::neg(a.raw) }
    }
}

/// `floor(a * b)` on the exact product (one rounding, toward -inf, whatever the signs).
/// Panics with `errors::OVERFLOW`. Upstream: `Mul for f64`.
///
/// Never write `a * b + c * d`: use the fused kernels (`sum_prod2`, `mul_add`, ...), which round
/// once and tolerate intermediate overflow.
pub impl FixedMul of Mul<Fixed> {
    #[inline(always)]
    fn mul(lhs: Fixed, rhs: Fixed) -> Fixed {
        Fixed { raw: kernels::mul(lhs.raw, rhs.raw) }
    }
}

/// `floor(a / b)` on the exact quotient: FLOOR division (toward -inf), consistent with `Mul`:
/// `-1 / 3 = -0.33333333349` (raw -1431655766), where truncation would give raw -1431655765.
/// Panics with `errors::DIVISION_BY_ZERO` / `errors::OVERFLOW`. Upstream: `Div for f64`.
///
/// Division is the most expensive operator: store reciprocals (`recip`) where upstream divides
/// repeatedly.
pub impl FixedDiv of Div<Fixed> {
    #[inline(always)]
    fn div(lhs: Fixed, rhs: Fixed) -> Fixed {
        Fixed { raw: kernels::div(lhs.raw, rhs.raw) }
    }
}

/// FLOORED modulo `a - b * floor(a / b)`, exact: the result is zero or has the sign of the
/// DIVISOR and `|a % b| < |b|`. So `x % ONE` is `x - floor(x)` and `angle % TAU` lies in
/// `[0, TAU)`. Cannot overflow. Panics with `errors::DIVISION_BY_ZERO`.
///
/// Deviation from upstream: Rust's `%` on floats is the TRUNCATED remainder (sign of the
/// dividend, `-7 % 3 = -1`); here `-7 % 3 = 2`. Floored is both consistent with the floor
/// rounding of the whole library and twice cheaper than the native truncated `i64` remainder.
pub impl FixedRem of Rem<Fixed> {
    #[inline(always)]
    fn rem(lhs: Fixed, rhs: Fixed) -> Fixed {
        Fixed { raw: kernels::rem(lhs.raw, rhs.raw) }
    }
}

pub impl FixedAddAssign of AddAssign<Fixed, Fixed> {
    #[inline(always)]
    fn add_assign(ref self: Fixed, rhs: Fixed) {
        self = self + rhs;
    }
}

pub impl FixedSubAssign of SubAssign<Fixed, Fixed> {
    #[inline(always)]
    fn sub_assign(ref self: Fixed, rhs: Fixed) {
        self = self - rhs;
    }
}

pub impl FixedMulAssign of MulAssign<Fixed, Fixed> {
    #[inline(always)]
    fn mul_assign(ref self: Fixed, rhs: Fixed) {
        self = self * rhs;
    }
}

pub impl FixedDivAssign of DivAssign<Fixed, Fixed> {
    #[inline(always)]
    fn div_assign(ref self: Fixed, rhs: Fixed) {
        self = self / rhs;
    }
}

pub impl FixedRemAssign of RemAssign<Fixed, Fixed> {
    #[inline(always)]
    fn rem_assign(ref self: Fixed, rhs: Fixed) {
        self = self % rhs;
    }
}

/// Total order of the represented values (native `i64` comparison).
pub impl FixedPartialOrd of PartialOrd<Fixed> {
    #[inline(always)]
    fn lt(lhs: Fixed, rhs: Fixed) -> bool {
        lhs.raw < rhs.raw
    }
    #[inline(always)]
    fn ge(lhs: Fixed, rhs: Fixed) -> bool {
        lhs.raw >= rhs.raw
    }
    #[inline(always)]
    fn gt(lhs: Fixed, rhs: Fixed) -> bool {
        lhs.raw > rhs.raw
    }
    #[inline(always)]
    fn le(lhs: Fixed, rhs: Fixed) -> bool {
        lhs.raw <= rhs.raw
    }
}

#[cfg(test)]
mod tests {
    use nalgebra_testing::black_box;
    use crate::fixed::convert::from_raw as fx;
    use crate::fixed::types::{EPSILON, Fixed, HALF, MAX, MIN, NEG_ONE, ONE, TWO, ZERO};

    const THREE: Fixed = Fixed { raw: 0x300000000 };
    const SEVEN: Fixed = Fixed { raw: 0x700000000 };

    // --- add / sub / neg ---------------------------------------------------------------------

    #[test]
    fn test_add_sign_combinations() {
        assert!(fx(0x380000000) + fx(0x240000000) == fx(0x5c0000000));
        assert!(fx(0x380000000) + fx(-0x240000000) == fx(0x140000000));
        assert!(fx(-0x380000000) + fx(0x240000000) == fx(-0x140000000));
        assert!(fx(-0x380000000) + fx(-0x240000000) == fx(-0x5c0000000));
    }

    #[test]
    fn test_add_boundaries() {
        assert!(MAX + ZERO == MAX);
        assert!(MIN + ZERO == MIN);
        assert!(MAX + MIN == -EPSILON);
        assert!(MAX - EPSILON + EPSILON == MAX);
        assert!(MIN + EPSILON - EPSILON == MIN);
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_add_overflow_panics() {
        let _ = black_box(MAX) + EPSILON;
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_add_underflow_panics() {
        let _ = black_box(MIN) + -EPSILON;
    }

    #[test]
    fn test_sub_sign_combinations() {
        assert!(fx(0x380000000) - fx(0x240000000) == fx(0x140000000));
        assert!(fx(0x380000000) - fx(-0x240000000) == fx(0x5c0000000));
        assert!(fx(-0x380000000) - fx(0x240000000) == fx(-0x5c0000000));
        assert!(fx(-0x380000000) - fx(-0x240000000) == fx(-0x140000000));
    }

    #[test]
    fn test_sub_boundaries() {
        assert!(MAX - MAX == ZERO);
        assert!(MIN - MIN == ZERO);
        assert!(ZERO - MAX == MIN + EPSILON);
        assert!(-EPSILON - MAX == MIN);
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_sub_overflow_panics() {
        let _ = black_box(MAX) - -EPSILON;
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_sub_underflow_panics() {
        let _ = black_box(MIN) - EPSILON;
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_sub_zero_minus_min_panics() {
        let _ = black_box(ZERO) - MIN;
    }

    #[test]
    fn test_neg_values() {
        assert!(-ONE == NEG_ONE);
        assert!(-NEG_ONE == ONE);
        assert!(-ZERO == ZERO);
        assert!(-MAX == MIN + EPSILON);
        assert!(-(MIN + EPSILON) == MAX);
        assert!(-EPSILON == fx(-1));
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_neg_min_panics() {
        let _ = -black_box(MIN);
    }

    // --- mul -----------------------------------------------------------------------------------

    #[test]
    fn test_mul_sign_combinations() {
        assert!(fx(0x380000000) * fx(0x240000000) == fx(0x7e0000000));
        assert!(fx(0x380000000) * fx(-0x240000000) == fx(-0x7e0000000));
        assert!(fx(-0x380000000) * fx(0x240000000) == fx(-0x7e0000000));
        assert!(fx(-0x380000000) * fx(-0x240000000) == fx(0x7e0000000));
    }

    #[test]
    fn test_mul_identities() {
        assert!(MAX * ONE == MAX);
        assert!(MIN * ONE == MIN);
        assert!(MAX * NEG_ONE == MIN + EPSILON);
        assert!(MAX * ZERO == ZERO);
        assert!(MIN * ZERO == ZERO);
        assert!(MIN * HALF == fx(-0x4000000000000000));
    }

    #[test]
    fn test_mul_rounds_toward_negative_infinity() {
        // (+-1 ulp) * 0.5 = +-0.5 ulp: floor gives 0 and -1 ulp (truncation would give 0 and 0).
        assert!(EPSILON * HALF == ZERO);
        assert!(-EPSILON * HALF == -EPSILON);
        assert!(EPSILON * -HALF == -EPSILON);
        assert!(-EPSILON * -HALF == ZERO);
        // 2^-64 floors to 0, -2^-64 floors to -1 ulp.
        assert!(EPSILON * EPSILON == ZERO);
        assert!(EPSILON * -EPSILON == -EPSILON);
        // -1.5 * 1.5 = -2.25 is exact: no adjustment.
        assert!(fx(0x180000000) * fx(-0x180000000) == fx(-0x240000000));
    }

    #[test]
    fn test_mul_largest_results() {
        // 46340^2 = 2147395600 fits, 65536 * -32768 = -2^31 = MIN fits exactly.
        assert!(fx(0xb50400000000) * fx(0xb50400000000) == fx(0x7ffea81000000000));
        assert!(fx(0x1000000000000) * fx(-0x800000000000) == MIN);
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_mul_overflow_panics() {
        // 46341^2 = 2147488281 > 2^31 - 1
        let _ = black_box(fx(0xb50500000000)) * fx(0xb50500000000);
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_mul_positive_limit_panics() {
        // 65536 * 32768 = 2^31: one past MAX.
        let _ = black_box(fx(0x1000000000000)) * fx(0x800000000000);
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_mul_min_by_neg_one_panics() {
        let _ = black_box(MIN) * NEG_ONE;
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_mul_min_by_min_panics() {
        let _ = black_box(MIN) * MIN;
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_mul_max_by_max_panics() {
        let _ = black_box(MAX) * MAX;
    }

    // --- div -----------------------------------------------------------------------------------

    #[test]
    fn test_div_exact_sign_combinations() {
        assert!(SEVEN / TWO == fx(0x380000000));
        assert!(-SEVEN / TWO == fx(-0x380000000));
        assert!(SEVEN / -TWO == fx(-0x380000000));
        assert!(-SEVEN / -TWO == fx(0x380000000));
        assert!(fx(-0x600000000) / THREE == -TWO);
    }

    #[test]
    fn test_div_rounds_toward_negative_infinity() {
        // 1/3 = 0.333..: floor 1431655765; -1/3: floor -1431655766 (truncation: -1431655765).
        assert!(ONE / THREE == fx(1431655765));
        assert!(NEG_ONE / THREE == fx(-1431655766));
        assert!(ONE / -THREE == fx(-1431655766));
        assert!(NEG_ONE / -THREE == fx(1431655765));
    }

    #[test]
    fn test_div_boundaries() {
        assert!(MAX / ONE == MAX);
        assert!(MIN / ONE == MIN);
        assert!(MAX / NEG_ONE == MIN + EPSILON);
        assert!((MIN + EPSILON) / NEG_ONE == MAX);
        assert!(MIN / MIN == ONE);
        assert!(MAX / MAX == ONE);
        assert!(MIN / MAX == fx(-4294967297));
        assert!(MAX / MIN == NEG_ONE);
        assert!(MIN / TWO == fx(-0x4000000000000000));
        assert!(ZERO / MIN == ZERO);
        assert!(ZERO / EPSILON == ZERO);
        // Tiny quotients: +-2^-32 / 2^31 floors to 0 or -1 ulp.
        assert!(EPSILON / MAX == ZERO);
        assert!(-EPSILON / MAX == -EPSILON);
        assert!(EPSILON / MIN == -EPSILON);
        assert!(-EPSILON / MIN == ZERO);
    }

    #[test]
    #[should_panic(expected: 'simba: division by zero')]
    fn test_div_by_zero_panics() {
        let _ = black_box(ONE) / ZERO;
    }

    #[test]
    #[should_panic(expected: 'simba: division by zero')]
    fn test_div_zero_by_zero_panics() {
        let _ = black_box(ZERO) / ZERO;
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_div_overflow_panics() {
        let _ = black_box(MAX) / HALF;
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_div_min_by_neg_one_panics() {
        let _ = black_box(MIN) / NEG_ONE;
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_div_min_by_neg_half_panics() {
        let _ = black_box(MIN) / -HALF;
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_div_by_epsilon_panics() {
        let _ = black_box(MAX) / EPSILON;
    }

    // --- rem -----------------------------------------------------------------------------------

    #[test]
    fn test_rem_is_floored_modulo() {
        // Sign of the divisor (Rust's `%` would give 1, -1, 1, -1).
        assert!(SEVEN % THREE == ONE);
        assert!(-SEVEN % THREE == TWO);
        assert!(SEVEN % -THREE == -TWO);
        assert!(-SEVEN % -THREE == NEG_ONE);
        assert!(fx(-0x600000000) % THREE == ZERO);
        assert!(fx(0x780000000) % TWO == fx(0x180000000));
        assert!(fx(-0x780000000) % TWO == HALF);
    }

    #[test]
    fn test_rem_by_one_is_floor_fraction() {
        assert!(fx(5) % ONE == fx(5));
        assert!(fx(-5) % ONE == fx(4294967291));
        assert!(fx(-5) % NEG_ONE == fx(-5));
        assert!(fx(-0x240000000) % ONE == fx(0xc0000000));
    }

    #[test]
    fn test_rem_boundaries() {
        assert!(MIN % MAX == fx(9223372036854775806));
        assert!(MAX % MIN == -EPSILON);
        assert!(MIN % -EPSILON == ZERO);
        assert!(MIN % EPSILON == ZERO);
        assert!(MAX % EPSILON == ZERO);
        assert!(MIN % MIN == ZERO);
        assert!(MAX % MAX == ZERO);
        assert!(ZERO % MIN == ZERO);
    }

    #[test]
    #[should_panic(expected: 'simba: division by zero')]
    fn test_rem_by_zero_panics() {
        let _ = black_box(ONE) % ZERO;
    }

    // --- compound assignments and ordering -------------------------------------------------------

    #[test]
    fn test_assign_operators() {
        let mut x = SEVEN;
        x += ONE;
        assert!(x == fx(0x800000000));
        x -= TWO;
        assert!(x == fx(0x600000000));
        x *= HALF;
        assert!(x == THREE);
        x /= TWO;
        assert!(x == fx(0x180000000));
        x %= ONE;
        assert!(x == HALF);
    }

    #[test]
    fn test_partial_ord_sign_combinations() {
        assert!(ONE < TWO);
        assert!(NEG_ONE < ONE);
        assert!(-TWO < NEG_ONE);
        assert!(!(ONE < ONE));
        assert!(ONE <= ONE);
        assert!(NEG_ONE <= ZERO);
        assert!(!(TWO <= ONE));
        assert!(TWO > ONE);
        assert!(ZERO > -EPSILON);
        assert!(!(ONE > ONE));
        assert!(ONE >= ONE);
        assert!(EPSILON >= ZERO);
        assert!(!(NEG_ONE >= ZERO));
    }

    #[test]
    fn test_partial_ord_boundaries() {
        assert!(MIN < MAX);
        assert!(MIN <= MIN);
        assert!(MAX >= MAX);
        assert!(MAX > MIN);
        assert!(!(MAX < MIN));
        assert!(MIN < MIN + EPSILON);
    }

    // --- gas benchmarks (net = raw - baseline of the group) --------------------------------------
    // Losing candidates: `bench_<op>__alt_*` in `fixed::kernels::alternatives`.

    #[test]
    #[inline(never)]
    fn bench_add__baseline() {
        let _a = black_box(fx(0x380000000));
        let _b = black_box(fx(0x240000000));
        let e = black_box(fx(0x5c0000000));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_add__fixed_pp() {
        let a = black_box(fx(0x380000000));
        let b = black_box(fx(0x240000000));
        let e = black_box(fx(0x5c0000000));
        assert!(a + b == e);
    }

    #[test]
    #[inline(never)]
    fn bench_add__fixed_pn() {
        let a = black_box(fx(0x380000000));
        let b = black_box(fx(-0x240000000));
        let e = black_box(fx(0x140000000));
        assert!(a + b == e);
    }

    #[test]
    #[inline(never)]
    fn bench_add__fixed_nn() {
        let a = black_box(fx(-0x240000000));
        let b = black_box(fx(-0xc0000000));
        let e = black_box(fx(-0x300000000));
        assert!(a + b == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sub__baseline() {
        let _a = black_box(fx(0x380000000));
        let _b = black_box(fx(0x240000000));
        let e = black_box(fx(0x140000000));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sub__fixed_pp() {
        let a = black_box(fx(0x380000000));
        let b = black_box(fx(0x240000000));
        let e = black_box(fx(0x140000000));
        assert!(a - b == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sub__fixed_pn() {
        let a = black_box(fx(0x380000000));
        let b = black_box(fx(-0x240000000));
        let e = black_box(fx(0x5c0000000));
        assert!(a - b == e);
    }

    #[test]
    #[inline(never)]
    fn bench_neg__baseline() {
        let _a = black_box(fx(0x380000000));
        let e = black_box(fx(-0x380000000));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_neg__fixed_p() {
        let a = black_box(fx(0x380000000));
        let e = black_box(fx(-0x380000000));
        assert!(-a == e);
    }

    #[test]
    #[inline(never)]
    fn bench_neg__fixed_n() {
        let a = black_box(fx(-0x240000000));
        let e = black_box(fx(0x240000000));
        assert!(-a == e);
    }

    #[test]
    #[inline(never)]
    fn bench_mul__baseline() {
        let _a = black_box(fx(0x380000000));
        let _b = black_box(fx(0x240000000));
        let e = black_box(fx(0x7e0000000));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_mul__fixed_pp() {
        let a = black_box(fx(0x380000000));
        let b = black_box(fx(0x240000000));
        let e = black_box(fx(0x7e0000000));
        assert!(a * b == e);
    }

    #[test]
    #[inline(never)]
    fn bench_mul__fixed_pn() {
        let a = black_box(fx(0x380000000));
        let b = black_box(fx(-0x240000000));
        let e = black_box(fx(-0x7e0000000));
        assert!(a * b == e);
    }

    #[test]
    #[inline(never)]
    fn bench_mul__fixed_nn() {
        let a = black_box(fx(-0x240000000));
        let b = black_box(fx(-0xc0000000));
        let e = black_box(fx(0x1b0000000));
        assert!(a * b == e);
    }

    #[test]
    #[inline(never)]
    fn bench_div__baseline() {
        let _a = black_box(fx(0x100000000));
        let _b = black_box(fx(0x300000000));
        let e = black_box(fx(0x55555555));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_div__fixed_pp() {
        let a = black_box(fx(0x100000000));
        let b = black_box(fx(0x300000000));
        let e = black_box(fx(0x55555555));
        assert!(a / b == e);
    }

    #[test]
    #[inline(never)]
    fn bench_div__fixed_np() {
        let a = black_box(fx(-0x100000000));
        let b = black_box(fx(0x300000000));
        let e = black_box(fx(-0x55555556));
        assert!(a / b == e);
    }

    #[test]
    #[inline(never)]
    fn bench_div__fixed_pn() {
        let a = black_box(fx(0x100000000));
        let b = black_box(fx(-0x300000000));
        let e = black_box(fx(-0x55555556));
        assert!(a / b == e);
    }

    #[test]
    #[inline(never)]
    fn bench_div__fixed_nn() {
        let a = black_box(fx(-0x100000000));
        let b = black_box(fx(-0x300000000));
        let e = black_box(fx(0x55555555));
        assert!(a / b == e);
    }

    #[test]
    #[inline(never)]
    fn bench_rem__baseline() {
        let _a = black_box(fx(0x700000000));
        let _b = black_box(fx(0x300000000));
        let e = black_box(fx(0x100000000));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_rem__fixed_pp() {
        let a = black_box(fx(0x700000000));
        let b = black_box(fx(0x300000000));
        let e = black_box(fx(0x100000000));
        assert!(a % b == e);
    }

    #[test]
    #[inline(never)]
    fn bench_rem__fixed_np() {
        let a = black_box(fx(-0x700000000));
        let b = black_box(fx(0x300000000));
        let e = black_box(fx(0x200000000));
        assert!(a % b == e);
    }

    #[test]
    #[inline(never)]
    fn bench_rem__fixed_pn() {
        let a = black_box(fx(0x700000000));
        let b = black_box(fx(-0x300000000));
        let e = black_box(fx(-0x200000000));
        assert!(a % b == e);
    }

    #[test]
    #[inline(never)]
    fn bench_rem__fixed_nn() {
        let a = black_box(fx(-0x700000000));
        let b = black_box(fx(-0x300000000));
        let e = black_box(fx(-0x100000000));
        assert!(a % b == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lt__baseline() {
        let _a = black_box(fx(0x380000000));
        let _b = black_box(fx(-0x240000000));
        let e = black_box(false);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lt__fixed_pn() {
        let a = black_box(fx(0x380000000));
        let b = black_box(fx(-0x240000000));
        let e = black_box(false);
        assert!((a < b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lt__fixed_np() {
        let a = black_box(fx(-0x240000000));
        let b = black_box(fx(0x380000000));
        let e = black_box(true);
        assert!((a < b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lt__fixed_pp() {
        let a = black_box(fx(0x240000000));
        let b = black_box(fx(0x380000000));
        let e = black_box(true);
        assert!((a < b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_le__baseline() {
        let _a = black_box(fx(0x380000000));
        let _b = black_box(fx(-0x240000000));
        let e = black_box(false);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_le__fixed_pn() {
        let a = black_box(fx(0x380000000));
        let b = black_box(fx(-0x240000000));
        let e = black_box(false);
        assert!((a <= b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_le__fixed_np() {
        let a = black_box(fx(-0x240000000));
        let b = black_box(fx(0x380000000));
        let e = black_box(true);
        assert!((a <= b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_le__fixed_pp() {
        let a = black_box(fx(0x240000000));
        let b = black_box(fx(0x380000000));
        let e = black_box(true);
        assert!((a <= b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_gt__baseline() {
        let _a = black_box(fx(0x380000000));
        let _b = black_box(fx(-0x240000000));
        let e = black_box(true);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_gt__fixed_pn() {
        let a = black_box(fx(0x380000000));
        let b = black_box(fx(-0x240000000));
        let e = black_box(true);
        assert!((a > b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_gt__fixed_np() {
        let a = black_box(fx(-0x240000000));
        let b = black_box(fx(0x380000000));
        let e = black_box(false);
        assert!((a > b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_gt__fixed_pp() {
        let a = black_box(fx(0x240000000));
        let b = black_box(fx(0x380000000));
        let e = black_box(false);
        assert!((a > b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_ge__baseline() {
        let _a = black_box(fx(0x380000000));
        let _b = black_box(fx(-0x240000000));
        let e = black_box(true);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_ge__fixed_pn() {
        let a = black_box(fx(0x380000000));
        let b = black_box(fx(-0x240000000));
        let e = black_box(true);
        assert!((a >= b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_ge__fixed_np() {
        let a = black_box(fx(-0x240000000));
        let b = black_box(fx(0x380000000));
        let e = black_box(false);
        assert!((a >= b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_ge__fixed_pp() {
        let a = black_box(fx(0x240000000));
        let b = black_box(fx(0x380000000));
        let e = black_box(false);
        assert!((a >= b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_eq__baseline() {
        let _a = black_box(fx(0x380000000));
        let _b = black_box(fx(-0x240000000));
        let e = black_box(false);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_eq__fixed_pn() {
        let a = black_box(fx(0x380000000));
        let b = black_box(fx(-0x240000000));
        let e = black_box(false);
        assert!((a == b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_eq__fixed_np() {
        let a = black_box(fx(-0x240000000));
        let b = black_box(fx(0x380000000));
        let e = black_box(false);
        assert!((a == b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_eq__fixed_pp() {
        let a = black_box(fx(0x240000000));
        let b = black_box(fx(0x380000000));
        let e = black_box(false);
        assert!((a == b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_add_assign__baseline() {
        let _a = black_box(fx(0x380000000));
        let _b = black_box(fx(-0x240000000));
        let e = black_box(fx(0x140000000));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_add_assign__fixed_pn() {
        let a = black_box(fx(0x380000000));
        let b = black_box(fx(-0x240000000));
        let e = black_box(fx(0x140000000));
        let mut a = a;
        a += b;
        assert!(a == e);
    }

    #[test]
    #[inline(never)]
    fn bench_mul_assign__baseline() {
        let _a = black_box(fx(0x380000000));
        let _b = black_box(fx(-0x240000000));
        let e = black_box(fx(-0x7e0000000));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_mul_assign__fixed_pn() {
        let a = black_box(fx(0x380000000));
        let b = black_box(fx(-0x240000000));
        let e = black_box(fx(-0x7e0000000));
        let mut a = a;
        a *= b;
        assert!(a == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sub_assign__baseline() {
        let _a = black_box(fx(0x380000000));
        let _b = black_box(fx(-0x240000000));
        let e = black_box(fx(0x5c0000000));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sub_assign__fixed_pn() {
        let a = black_box(fx(0x380000000));
        let b = black_box(fx(-0x240000000));
        let e = black_box(fx(0x5c0000000));
        let mut a = a;
        a -= b;
        assert!(a == e);
    }

    #[test]
    #[inline(never)]
    fn bench_div_assign__baseline() {
        let _a = black_box(fx(0x100000000));
        let _b = black_box(fx(-0x300000000));
        let e = black_box(fx(-0x55555556));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_div_assign__fixed_pn() {
        let a = black_box(fx(0x100000000));
        let b = black_box(fx(-0x300000000));
        let e = black_box(fx(-0x55555556));
        let mut a = a;
        a /= b;
        assert!(a == e);
    }

    #[test]
    #[inline(never)]
    fn bench_rem_assign__baseline() {
        let _a = black_box(fx(0x700000000));
        let _b = black_box(fx(-0x300000000));
        let e = black_box(fx(-0x200000000));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_rem_assign__fixed_pn() {
        let a = black_box(fx(0x700000000));
        let b = black_box(fx(-0x300000000));
        let e = black_box(fx(-0x200000000));
        let mut a = a;
        a %= b;
        assert!(a == e);
    }
}
