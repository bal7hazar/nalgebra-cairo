//! Classic integer algorithms written in the three competing styles (math / bitwise / loop).
//!
//! Groups: `msb_u64`, `popcount_u64`, `abs_i64`, `minmax_u64`, `sqrt_u64`, `sqrt_u128`,
//! `pow_u64`.

#[feature("bounded-int-utils")]
use core::internal::bounded_int::{self, BoundedInt, ConstrainHelper, MulHelper, UnitInt, upcast};
use core::num::traits::{Pow, Sqrt, WideMul};
use crate::lookup::pow2_const_array;
use crate::msb_tree::msb_if_tree;

// ------------------------------------------------------------------------------------ msb

/// Loop: shift right until zero.
pub fn msb_loop(x: u64) -> u32 {
    let mut x = x / 2;
    let mut r = 0;
    while x != 0 {
        x /= 2;
        r += 1;
    }
    r
}

/// Binary search, math style: compare, then divide to "shift".
pub fn msb_binary_div(x: u64) -> u32 {
    let mut x = x;
    let mut r = 0;
    if x >= 0x100000000 {
        x /= 0x100000000;
        r += 32;
    }
    if x >= 0x10000 {
        x /= 0x10000;
        r += 16;
    }
    if x >= 0x100 {
        x /= 0x100;
        r += 8;
    }
    if x >= 0x10 {
        x /= 0x10;
        r += 4;
    }
    if x >= 0x4 {
        x /= 0x4;
        r += 2;
    }
    if x >= 0x2 {
        r += 1;
    }
    r
}

/// Binary search, bitwise style: test the upper half with a mask, then divide to "shift".
pub fn msb_binary_bitwise(x: u64) -> u32 {
    let mut x = x;
    let mut r = 0;
    if x & 0xffffffff00000000 != 0 {
        x /= 0x100000000;
        r += 32;
    }
    if x & 0xffff0000 != 0 {
        x /= 0x10000;
        r += 16;
    }
    if x & 0xff00 != 0 {
        x /= 0x100;
        r += 8;
    }
    if x & 0xf0 != 0 {
        x /= 0x10;
        r += 4;
    }
    if x & 0xc != 0 {
        x /= 0x4;
        r += 2;
    }
    if x & 0x2 != 0 {
        r += 1;
    }
    r
}

// ------------------------------------------------------------------------------------ popcount

pub fn popcount_loop(x: u64) -> u32 {
    let mut x = x;
    let mut count = 0;
    while x != 0 {
        let (q, r) = DivRem::div_rem(x, 2);
        if r == 1 {
            count += 1;
        }
        x = q;
    }
    count
}

/// SWAR popcount: bitwise masks, shifts emulated with divisions.
pub fn popcount_swar(x: u64) -> u32 {
    let x = x - ((x / 2) & 0x5555555555555555);
    let x = (x & 0x3333333333333333) + ((x / 4) & 0x3333333333333333);
    let x = (x + x / 16) & 0x0f0f0f0f0f0f0f0f;
    let folded: u128 = x.wide_mul(0x0101010101010101) / 0x100000000000000 % 0x100;
    folded.try_into().unwrap()
}

const NIBBLE_BITS: [u32; 16] = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4];

/// Math popcount: 16 x (DivRem by 16 + table lookup), unrolled.
pub fn popcount_nibble_table(x: u64) -> u32 {
    let table = NIBBLE_BITS.span();
    let mut count = 0;
    let mut x = x;
    for _ in 0..16_u32 {
        let (q, r) = DivRem::div_rem(x, 16);
        count += *table[r.try_into().unwrap()];
        x = q;
    }
    count
}

pub fn popcount_nibble_table_unrolled(x: u64) -> u32 {
    let table = NIBBLE_BITS.span();
    let (x, n0) = DivRem::div_rem(x, 16);
    let (x, n1) = DivRem::div_rem(x, 16);
    let (x, n2) = DivRem::div_rem(x, 16);
    let (x, n3) = DivRem::div_rem(x, 16);
    let (x, n4) = DivRem::div_rem(x, 16);
    let (x, n5) = DivRem::div_rem(x, 16);
    let (x, n6) = DivRem::div_rem(x, 16);
    let (x, n7) = DivRem::div_rem(x, 16);
    let (x, n8) = DivRem::div_rem(x, 16);
    let (x, n9) = DivRem::div_rem(x, 16);
    let (x, n10) = DivRem::div_rem(x, 16);
    let (x, n11) = DivRem::div_rem(x, 16);
    let (x, n12) = DivRem::div_rem(x, 16);
    let (x, n13) = DivRem::div_rem(x, 16);
    let (n15, n14) = DivRem::div_rem(x, 16);
    *table[n0.try_into().unwrap()]
        + *table[n1.try_into().unwrap()]
        + *table[n2.try_into().unwrap()]
        + *table[n3.try_into().unwrap()]
        + *table[n4.try_into().unwrap()]
        + *table[n5.try_into().unwrap()]
        + *table[n6.try_into().unwrap()]
        + *table[n7.try_into().unwrap()]
        + *table[n8.try_into().unwrap()]
        + *table[n9.try_into().unwrap()]
        + *table[n10.try_into().unwrap()]
        + *table[n11.try_into().unwrap()]
        + *table[n12.try_into().unwrap()]
        + *table[n13.try_into().unwrap()]
        + *table[n14.try_into().unwrap()]
        + *table[n15.try_into().unwrap()]
}

// ------------------------------------------------------------------------------------ abs / sign

pub fn abs_branch(x: i64) -> i64 {
    if x < 0 {
        -x
    } else {
        x
    }
}

impl ConstrainI64 of ConstrainHelper<i64, 0> {
    type LowT = BoundedInt<-0x8000000000000000, -1>;
    type HighT = BoundedInt<0, 0x7fffffffffffffff>;
}
impl NegateI64Low of MulHelper<BoundedInt<-0x8000000000000000, -1>, UnitInt<-1>> {
    type Result = BoundedInt<1, 0x8000000000000000>;
}

/// `|x|` as `u64` (total: `i64::MIN` is representable), bounded-int constrain + negate.
pub fn abs_bounded(x: i64) -> u64 {
    match bounded_int::constrain::<i64, 0, ConstrainI64>(x) {
        Ok(neg) => upcast(bounded_int::mul::<_, UnitInt<-1>, NegateI64Low>(neg, -1)),
        Err(pos) => upcast(pos),
    }
}

/// `|x|` as `u64` through felt252: branch on the comparison, negate in the field, narrow.
pub fn abs_felt(x: i64) -> u64 {
    let f: felt252 = x.into();
    if x < 0 {
        (-f).try_into().unwrap()
    } else {
        f.try_into().unwrap()
    }
}

pub fn sign_branch(x: i64) -> i64 {
    if x < 0 {
        -1
    } else if x == 0 {
        0
    } else {
        1
    }
}

pub fn neg(x: i64) -> i64 {
    -x
}

/// Negation as `0 - x`.
pub fn neg_sub(x: i64) -> i64 {
    0 - x
}

// ------------------------------------------------------------------------------------ min / max

pub fn min_branch(a: u64, b: u64) -> u64 {
    if a < b {
        a
    } else {
        b
    }
}

pub fn min_corelib(a: u64, b: u64) -> u64 {
    core::cmp::min(a, b)
}

/// The comparison libfunc directly: a single branching libfunc, no bool materialised.
pub fn min_overflowing_sub(a: u64, b: u64) -> u64 {
    #[feature("corelib-internal-use")]
    match core::integer::u64_overflowing_sub(a, b) {
        Ok(_) => b,
        Err(_) => a,
    }
}

/// "Branchless" arithmetic select: `b + lt * (a - b)` in felt252, then narrow.
pub fn min_arith_felt(a: u64, b: u64) -> u64 {
    let lt: felt252 = (a < b).into();
    let fa: felt252 = a.into();
    let fb: felt252 = b.into();
    (fb + lt * (fa - fb)).try_into().unwrap()
}

pub fn clamp_branch(x: u64, lo: u64, hi: u64) -> u64 {
    if x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    }
}

pub fn clamp_min_max(x: u64, lo: u64, hi: u64) -> u64 {
    core::cmp::min(core::cmp::max(x, lo), hi)
}

pub fn min_branch_i64(a: i64, b: i64) -> i64 {
    if a < b {
        a
    } else {
        b
    }
}

// ------------------------------------------------------------------------------------ sqrt

pub fn sqrt_u64_corelib(n: u64) -> u64 {
    n.sqrt().into()
}

/// Textbook Newton (Heron) iteration, starting from `n / 2 + 1`.
pub fn sqrt_u64_newton_loop(n: u64) -> u64 {
    if n < 2 {
        return n;
    }
    let mut x = n / 2 + 1;
    let mut y = (x + n / x) / 2;
    while y < x {
        x = y;
        y = (x + n / x) / 2;
    }
    x
}

/// Newton with an msb-based seed (`2^(msb/2)`, within a factor 2 of the root) and a fixed number
/// of unrolled iterations (precision doubles each time: 1, 2, 4, 8, 16, 32 bits), then the
/// classic `min(x, n / x)` correction.
pub fn sqrt_u64_newton_unrolled(n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    let x = pow2_const_array(msb_if_tree(n) / 2);
    let x = (x + n / x) / 2;
    let x = (x + n / x) / 2;
    let x = (x + n / x) / 2;
    let x = (x + n / x) / 2;
    let x = (x + n / x) / 2;
    let x = (x + n / x) / 2;
    let y = n / x;
    if x < y {
        x
    } else {
        y
    }
}

pub fn sqrt_u128_corelib(n: u128) -> u128 {
    n.sqrt().into()
}

pub fn sqrt_u128_newton_loop(n: u128) -> u128 {
    if n < 2 {
        return n;
    }
    let mut x = n / 2 + 1;
    let mut y = (x + n / x) / 2;
    while y < x {
        x = y;
        y = (x + n / x) / 2;
    }
    x
}

// ------------------------------------------------------------------------------------ pow

pub fn pow_loop(base: u64, exp: u32) -> u64 {
    let mut r = 1;
    let mut i = exp;
    while i != 0 {
        r *= base;
        i -= 1;
    }
    r
}

pub fn pow_square_multiply(base: u64, exp: u32) -> u64 {
    let mut r = 1;
    let mut b = base;
    let mut e = exp;
    loop {
        let (q, bit) = DivRem::div_rem(e, 2);
        if bit == 1 {
            r *= b;
        }
        if q == 0 {
            break;
        }
        b *= b;
        e = q;
    }
    r
}

pub fn pow_corelib(base: u64, exp: u32) -> u64 {
    base.pow(exp)
}

const POW10: [u64; 20] = [
    1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000, 10000000000,
    100000000000, 1000000000000, 10000000000000, 100000000000000, 1000000000000000,
    10000000000000000, 100000000000000000, 1000000000000000000, 10000000000000000000,
];

pub fn pow10_lookup(exp: u32) -> u64 {
    *POW10.span()[exp]
}

#[cfg(test)]
mod tests {
    use harness::black_box;
    use super::*;

    const X: u64 = 0x0000123456789abc; // msb = 44, popcount = 22

    #[test]
    #[inline(never)]
    fn bench_msb_u64__baseline() {
        let x = black_box(X);
        assert!(x == X);
    }

    #[test]
    #[inline(never)]
    fn bench_msb_u64__loop_div2() {
        let x = black_box(X);
        assert!(x == X);
        assert!(msb_loop(x) == 44);
    }

    #[test]
    #[inline(never)]
    fn bench_msb_u64__binary_search_cmp_div() {
        let x = black_box(X);
        assert!(x == X);
        assert!(msb_binary_div(x) == 44);
    }

    #[test]
    #[inline(never)]
    fn bench_msb_u64__binary_search_mask_div() {
        let x = black_box(X);
        assert!(x == X);
        assert!(msb_binary_bitwise(x) == 44);
    }

    #[test]
    #[inline(never)]
    fn bench_msb_u64__comparison_tree() {
        let x = black_box(X);
        assert!(x == X);
        assert!(msb_if_tree(x) == 44);
    }

    #[test]
    fn test_msb_variants_agree() {
        let mut x: u64 = 1;
        for k in 0..64_u32 {
            assert!(msb_loop(x) == k);
            assert!(msb_binary_div(x) == k);
            assert!(msb_binary_bitwise(x) == k);
            assert!(msb_if_tree(x) == k);
            let y = x + x / 3;
            assert!(msb_binary_div(y) == k);
            assert!(msb_if_tree(y) == k);
            if k != 63 {
                x *= 2;
            }
        }
    }

    #[test]
    #[inline(never)]
    fn bench_popcount_u64__baseline() {
        let x = black_box(X);
        assert!(x == X);
    }

    #[test]
    #[inline(never)]
    fn bench_popcount_u64__loop_divrem2() {
        let x = black_box(X);
        assert!(x == X);
        assert!(popcount_loop(x) == 22);
    }

    #[test]
    #[inline(never)]
    fn bench_popcount_u64__swar_bitwise() {
        let x = black_box(X);
        assert!(x == X);
        assert!(popcount_swar(x) == 22);
    }

    #[test]
    #[inline(never)]
    fn bench_popcount_u64__nibble_table_loop() {
        let x = black_box(X);
        assert!(x == X);
        assert!(popcount_nibble_table(x) == 22);
    }

    #[test]
    #[inline(never)]
    fn bench_popcount_u64__nibble_table_unrolled() {
        let x = black_box(X);
        assert!(x == X);
        assert!(popcount_nibble_table_unrolled(x) == 22);
    }

    #[test]
    fn test_popcount_variants_agree() {
        for x in array![0_u64, 1, 0xff, X, 0xffffffffffffffff, 0x8000000000000001] {
            let expected = popcount_loop(x);
            assert!(popcount_swar(x) == expected);
            assert!(popcount_nibble_table(x) == expected);
            assert!(popcount_nibble_table_unrolled(x) == expected);
        }
    }

    const NEG: i64 = -0x123456789;

    #[test]
    #[inline(never)]
    fn bench_abs_i64__baseline() {
        let x = black_box(NEG);
        assert!(x == NEG);
    }

    #[test]
    #[inline(never)]
    fn bench_abs_i64__abs_if_lt_zero_neg() {
        assert!(abs_branch(black_box(NEG)) == 0x123456789);
    }

    #[test]
    #[inline(never)]
    fn bench_abs_i64__abs_bounded_constrain_to_u64() {
        assert!(abs_bounded(black_box(NEG)) == 0x123456789);
    }

    #[test]
    #[inline(never)]
    fn bench_abs_i64__abs_via_felt252_to_u64() {
        assert!(abs_felt(black_box(NEG)) == 0x123456789);
    }

    #[test]
    #[inline(never)]
    fn bench_abs_i64__sign_branches() {
        assert!(sign_branch(black_box(NEG)) == -1);
    }

    #[test]
    #[inline(never)]
    fn bench_abs_i64__neg_operator() {
        assert!(neg(black_box(NEG)) == 0x123456789);
    }

    #[test]
    #[inline(never)]
    fn bench_abs_i64__neg_zero_minus_x() {
        assert!(neg_sub(black_box(NEG)) == 0x123456789);
    }

    #[test]
    #[inline(never)]
    fn bench_minmax_u64__baseline() {
        let (a, b, c): (u64, u64, u64) = (black_box(50), black_box(70), black_box(10));
        let _ = (b, c);
        assert!(a == 50);
    }

    #[test]
    #[inline(never)]
    fn bench_minmax_u64__min_if() {
        let (a, b, c): (u64, u64, u64) = (black_box(50), black_box(70), black_box(10));
        let _ = c;
        assert!(min_branch(a, b) == 50);
    }

    #[test]
    #[inline(never)]
    fn bench_minmax_u64__min_corelib() {
        let (a, b, c): (u64, u64, u64) = (black_box(50), black_box(70), black_box(10));
        let _ = c;
        assert!(min_corelib(a, b) == 50);
    }

    #[test]
    #[inline(never)]
    fn bench_minmax_u64__min_overflowing_sub_match() {
        let (a, b, c): (u64, u64, u64) = (black_box(50), black_box(70), black_box(10));
        let _ = c;
        assert!(min_overflowing_sub(a, b) == 50);
    }

    #[test]
    #[inline(never)]
    fn bench_minmax_u64__min_branchless_felt() {
        let (a, b, c): (u64, u64, u64) = (black_box(50), black_box(70), black_box(10));
        let _ = c;
        assert!(min_arith_felt(a, b) == 50);
    }

    #[test]
    #[inline(never)]
    fn bench_minmax_u64__clamp_if_chain() {
        let (a, b, c): (u64, u64, u64) = (black_box(50), black_box(70), black_box(10));
        assert!(clamp_branch(a, c, b) == 50);
    }

    #[test]
    #[inline(never)]
    fn bench_minmax_u64__clamp_corelib_min_max() {
        let (a, b, c): (u64, u64, u64) = (black_box(50), black_box(70), black_box(10));
        assert!(clamp_min_max(a, c, b) == 50);
    }

    #[test]
    #[inline(never)]
    fn bench_minmax_i64__baseline() {
        let (a, b): (i64, i64) = (black_box(-50), black_box(70));
        let _ = b;
        assert!(a == -50);
    }

    #[test]
    #[inline(never)]
    fn bench_minmax_i64__min_if() {
        let (a, b): (i64, i64) = (black_box(-50), black_box(70));
        assert!(min_branch_i64(a, b) == -50);
    }

    // isqrt(0x0123456789abcdef) = 0x11111111
    const N: u64 = 0x0123456789abcdef;

    #[test]
    #[inline(never)]
    fn bench_sqrt_u64__baseline() {
        let n = black_box(N);
        assert!(n == N);
    }

    #[test]
    #[inline(never)]
    fn bench_sqrt_u64__corelib() {
        assert!(sqrt_u64_corelib(black_box(N)) == 0x11111111);
    }

    #[test]
    #[inline(never)]
    fn bench_sqrt_u64__newton_loop() {
        assert!(sqrt_u64_newton_loop(black_box(N)) == 0x11111111);
    }

    #[test]
    #[inline(never)]
    fn bench_sqrt_u64__newton_unrolled_msb_seed() {
        assert!(sqrt_u64_newton_unrolled(black_box(N)) == 0x11111111);
    }

    #[test]
    fn test_sqrt_variants_agree() {
        let samples = array![
            0_u64, 1, 2, 3, 4, 8, 9, 15, 16, 17, 99, 100, 101, 0xffff, 0x10000, 0xfffe0001,
            0xfffe0000, 0xffffffff, 0x100000000, N, 0xfffffffe00000001, 0xfffffffe00000000,
            0xffffffffffffffff, 0x4000000000000000, 0x3fffffffffffffff,
        ];
        for n in samples {
            let expected = sqrt_u64_corelib(n);
            assert!(sqrt_u64_newton_loop(n) == expected, "loop {n}");
            assert!(sqrt_u64_newton_unrolled(n) == expected, "unrolled {n}");
        }
    }

    const N128: u128 = 0x0123456789abcdef0123456789abcdef;

    #[test]
    #[inline(never)]
    fn bench_sqrt_u128__baseline() {
        let n = black_box(N128);
        assert!(n == N128);
    }

    #[test]
    #[inline(never)]
    fn bench_sqrt_u128__corelib() {
        assert!(sqrt_u128_corelib(black_box(N128)) == 0x1111111111111109);
    }

    #[test]
    #[inline(never)]
    fn bench_sqrt_u128__newton_loop() {
        assert!(sqrt_u128_newton_loop(black_box(N128)) == 0x1111111111111109);
    }

    #[test]
    #[inline(never)]
    fn bench_pow_u64__baseline() {
        let (b, e): (u64, u32) = (black_box(10), black_box(15));
        let _ = e;
        assert!(b == 10);
    }

    #[test]
    #[inline(never)]
    fn bench_pow_u64__loop_15_mul() {
        let (b, e): (u64, u32) = (black_box(10), black_box(15));
        assert!(pow_loop(b, e) == 1000000000000000);
    }

    #[test]
    #[inline(never)]
    fn bench_pow_u64__square_and_multiply_loop() {
        let (b, e): (u64, u32) = (black_box(10), black_box(15));
        assert!(pow_square_multiply(b, e) == 1000000000000000);
    }

    #[test]
    #[inline(never)]
    fn bench_pow_u64__corelib_pow() {
        let (b, e): (u64, u32) = (black_box(10), black_box(15));
        assert!(pow_corelib(b, e) == 1000000000000000);
    }

    #[test]
    #[inline(never)]
    fn bench_pow_u64__const_table_lookup() {
        let (b, e): (u64, u32) = (black_box(10), black_box(15));
        let _ = b;
        assert!(pow10_lookup(e) == 1000000000000000);
    }
}
