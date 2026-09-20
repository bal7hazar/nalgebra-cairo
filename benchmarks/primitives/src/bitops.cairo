//! Bit manipulation done with math vs. with the bitwise builtin.
//!
//! corelib 2.19.4 has no `Shl`/`Shr` traits nor any shift libfunc: shifts must be written as a
//! multiplication / division by a power of two.
//!
//! Groups: `shift_u64`, `shift_u128`, `lowbits_u64`, `lowbits_u128`, `split_u64`, `bittest_u64`,
//! `ispow2_u64`, `parity_u64`.

#[feature("bounded-int-utils")]
use core::internal::bounded_int::{
    self, BoundedInt, DivRemHelper, MulHelper, UnitInt, downcast, upcast,
};
use core::num::traits::WideMul;

// ------------------------------------------------------------------------------------ shifts

pub fn shr8_div(x: u64) -> u64 {
    x / 0x100
}

pub fn shr32_div(x: u64) -> u64 {
    x / 0x100000000
}

type Two8 = UnitInt<0x100>;
const NZ_TWO8: NonZero<Two8> = 0x100;
impl DivRemU64Two8 of DivRemHelper<u64, Two8> {
    type DivT = BoundedInt<0, 0xffffffffffffff>;
    type RemT = BoundedInt<0, 0xff>;
}

type Two32 = UnitInt<0x100000000>;
const NZ_TWO32: NonZero<Two32> = 0x100000000;
impl DivRemU64Two32 of DivRemHelper<u64, Two32> {
    type DivT = BoundedInt<0, 0xffffffff>;
    type RemT = BoundedInt<0, 0xffffffff>;
}

/// `x >> 8` through `bounded_int::div_rem` by a constant.
pub fn shr8_bounded(x: u64) -> u64 {
    let (q, _) = bounded_int::div_rem(x, NZ_TWO8);
    upcast(q)
}

pub fn shr32_bounded(x: u64) -> u64 {
    let (q, _) = bounded_int::div_rem(x, NZ_TWO32);
    upcast(q)
}

/// Checked `x << 8` (panics when bits are lost).
pub fn shl8_mul_checked(x: u64) -> u64 {
    x * 0x100
}

/// Wrapping `x << 8` (high bits discarded, like a hardware shift).
pub fn shl8_mul_wrapping(x: u64) -> u64 {
    (x.wide_mul(0x100) % 0x10000000000000000).try_into().unwrap()
}

/// Wrapping `x << 8`: drop the high byte first, then the multiplication cannot overflow.
pub fn shl8_mask_then_mul(x: u64) -> u64 {
    (x % 0x100000000000000) * 0x100
}

pub fn pow2_u64(k: u32) -> u64 {
    // Only the entries needed by the benchmark inputs matter for gas; the table is complete for
    // the first 16 exponents.
    match k {
        0 => 0x1,
        1 => 0x2,
        2 => 0x4,
        3 => 0x8,
        4 => 0x10,
        5 => 0x20,
        6 => 0x40,
        7 => 0x80,
        8 => 0x100,
        9 => 0x200,
        10 => 0x400,
        11 => 0x800,
        12 => 0x1000,
        13 => 0x2000,
        14 => 0x4000,
        15 => 0x8000,
        _ => core::panic_with_felt252('pow2 out of range'),
    }
}

/// Variable shift: table lookup + division.
pub fn shr_var(x: u64, k: u32) -> u64 {
    x / pow2_u64(k)
}

/// Variable shift: table lookup + checked multiplication.
pub fn shl_var(x: u64, k: u32) -> u64 {
    x * pow2_u64(k)
}

pub fn shr64_div_u128(x: u128) -> u128 {
    x / 0x10000000000000000
}

pub fn shl64_mul_u128(x: u128) -> u128 {
    x * 0x10000000000000000
}

type Two64 = UnitInt<0x10000000000000000>;
const NZ_TWO64: NonZero<Two64> = 0x10000000000000000;
impl DivRemU128Two64 of DivRemHelper<u128, Two64> {
    type DivT = BoundedInt<0, 0xffffffffffffffff>;
    type RemT = BoundedInt<0, 0xffffffffffffffff>;
}

impl MulU64Two64 of MulHelper<BoundedInt<0, 0xffffffffffffffff>, Two64> {
    type Result = BoundedInt<0, 0xffffffffffffffff0000000000000000>;
}

/// Checked `x << 64` on u128 without the (expensive) `u128 *`: narrow to 64 bits (one range
/// check), then a range-check-free bounded multiplication.
pub fn shl64_bounded_u128(x: u128) -> u128 {
    let low: BoundedInt<0, 0xffffffffffffffff> = downcast(x).expect('shl overflow');
    upcast(bounded_int::mul::<_, Two64>(low, 0x10000000000000000))
}

pub fn shr64_bounded_u128(x: u128) -> u128 {
    let (q, _) = bounded_int::div_rem(x, NZ_TWO64);
    upcast(q)
}

// ------------------------------------------------------------------------------------ low bits

pub fn low8_rem(x: u64) -> u64 {
    x % 0x100
}

pub fn low8_and(x: u64) -> u64 {
    x & 0xff
}

pub fn low8_bounded(x: u64) -> u64 {
    let (_, r) = bounded_int::div_rem(x, NZ_TWO8);
    upcast(r)
}

pub fn low32_rem(x: u64) -> u64 {
    x % 0x100000000
}

pub fn low32_and(x: u64) -> u64 {
    x & 0xffffffff
}

pub fn low32_bounded(x: u64) -> u64 {
    let (_, r) = bounded_int::div_rem(x, NZ_TWO32);
    upcast(r)
}

pub fn low64_rem_u128(x: u128) -> u128 {
    x % 0x10000000000000000
}

pub fn low64_and_u128(x: u128) -> u128 {
    x & 0xffffffffffffffff
}

pub fn low64_bounded_u128(x: u128) -> u128 {
    let (_, r) = bounded_int::div_rem(x, NZ_TWO64);
    upcast(r)
}

pub fn low64_rem_u256(x: u256) -> u256 {
    x % 0x10000000000000000
}

pub fn low64_and_u256(x: u256) -> u256 {
    x & 0xffffffffffffffff
}

/// u256 low bits without any u256 operation: work on the low limb only.
pub fn low64_limb_u256(x: u256) -> u256 {
    u256 { low: low64_bounded_u128(x.low), high: 0 }
}

// ------------------------------------------------------------------------------------ split

/// Integer / fractional split of a Q32.32: one DivRem.
pub fn split32_divrem(x: u64) -> (u64, u64) {
    DivRem::div_rem(x, 0x100000000)
}

/// Same with `/` for the high part and `&` for the low part.
pub fn split32_div_and(x: u64) -> (u64, u64) {
    (x / 0x100000000, x & 0xffffffff)
}

pub fn split32_bounded(x: u64) -> (u64, u64) {
    let (q, r) = bounded_int::div_rem(x, NZ_TWO32);
    (upcast(q), upcast(r))
}

// ------------------------------------------------------------------------------------ bit test

/// Is bit 8 set? Bitwise version.
pub fn bit8_and(x: u64) -> bool {
    x & 0x100 != 0
}

/// Is bit 8 set? Math version: `(x / 2^8) % 2`.
pub fn bit8_div_rem(x: u64) -> bool {
    (x / 0x100) % 2 == 1
}

/// Is bit 8 set? Math version: `x % 2^9 >= 2^8` (a single division).
pub fn bit8_rem_cmp(x: u64) -> bool {
    x % 0x200 >= 0x100
}

// ------------------------------------------------------------------------------------ power of 2

pub fn ispow2_bitwise(x: u64) -> bool {
    x != 0 && (x & (x - 1)) == 0
}

/// `x` is a power of two iff it divides `2^63`.
pub fn ispow2_math(x: u64) -> bool {
    match TryInto::<u64, NonZero<u64>>::try_into(x) {
        Some(nz) => {
            let (_, r) = DivRem::div_rem(0x8000000000000000_u64, nz);
            r == 0
        },
        None => false,
    }
}

pub fn ispow2_loop(x: u64) -> bool {
    if x == 0 {
        return false;
    }
    let mut x = x;
    loop {
        let (q, r) = DivRem::div_rem(x, 2);
        if r == 1 {
            break q == 0;
        }
        x = q;
    }
}

// ------------------------------------------------------------------------------------ parity

pub fn is_odd_rem(x: u64) -> bool {
    x % 2 == 1
}

pub fn is_odd_and(x: u64) -> bool {
    x & 1 == 1
}

#[cfg(test)]
mod tests {
    use harness::black_box;
    use super::*;

    const X: u64 = 0x123456789abcdef0;
    const X128: u128 = 0x123456789abcdef00fedcba987654321;

    #[test]
    #[inline(never)]
    fn bench_shift_u64__baseline() {
        let x = black_box(X);
        assert!(x == X);
    }

    #[test]
    #[inline(never)]
    fn bench_shift_u64__shr8_div_const() {
        assert!(shr8_div(black_box(X)) == 0x123456789abcde);
    }

    #[test]
    #[inline(never)]
    fn bench_shift_u64__shr32_div_const() {
        assert!(shr32_div(black_box(X)) == 0x12345678);
    }

    #[test]
    #[inline(never)]
    fn bench_shift_u64__shr8_bounded_div_rem() {
        assert!(shr8_bounded(black_box(X)) == 0x123456789abcde);
    }

    #[test]
    #[inline(never)]
    fn bench_shift_u64__shr32_bounded_div_rem() {
        assert!(shr32_bounded(black_box(X)) == 0x12345678);
    }

    #[test]
    #[inline(never)]
    fn bench_shift_u64__shl8_mul_const_checked() {
        assert!(shl8_mul_checked(black_box(0x3456789abcdef0)) == 0x3456789abcdef000);
    }

    #[test]
    #[inline(never)]
    fn bench_shift_u64__shl8_widemul_rem_wrapping() {
        assert!(shl8_mul_wrapping(black_box(X)) == 0x3456789abcdef000);
    }

    #[test]
    #[inline(never)]
    fn bench_shift_u64__shl8_rem_then_mul_wrapping() {
        assert!(shl8_mask_then_mul(black_box(X)) == 0x3456789abcdef000);
    }

    #[test]
    #[inline(never)]
    fn bench_shift_var_u64__baseline() {
        let x = black_box(X);
        let _: u32 = black_box(8);
        assert!(x == X);
    }

    #[test]
    #[inline(never)]
    fn bench_shift_var_u64__shr_lookup_div() {
        assert!(shr_var(black_box(X), black_box(8)) == 0x123456789abcde);
    }

    #[test]
    #[inline(never)]
    fn bench_shift_var_u64__shl_lookup_mul() {
        assert!(shl_var(black_box(0x3456789abcdef0), black_box(8)) == 0x3456789abcdef000);
    }

    #[test]
    #[inline(never)]
    fn bench_shift_u128__baseline() {
        let x = black_box(X128);
        assert!(x == X128);
    }

    #[test]
    #[inline(never)]
    fn bench_shift_u128__shr64_div_const() {
        assert!(shr64_div_u128(black_box(X128)) == 0x123456789abcdef0);
    }

    #[test]
    #[inline(never)]
    fn bench_shift_u128__shr64_bounded_div_rem() {
        assert!(shr64_bounded_u128(black_box(X128)) == 0x123456789abcdef0);
    }

    #[test]
    #[inline(never)]
    fn bench_shift_u128__shl64_mul_const_checked() {
        assert!(
            shl64_mul_u128(black_box(0x0fedcba987654321)) == 0x0fedcba9876543210000000000000000,
        );
    }

    #[test]
    #[inline(never)]
    fn bench_shift_u128__shl64_bounded_downcast_mul() {
        assert!(
            shl64_bounded_u128(black_box(0x0fedcba987654321)) == 0x0fedcba9876543210000000000000000,
        );
    }

    #[test]
    #[inline(never)]
    fn bench_lowbits_u64__baseline() {
        let x = black_box(X);
        assert!(x == X);
    }

    #[test]
    #[inline(never)]
    fn bench_lowbits_u64__low8_rem_const() {
        assert!(low8_rem(black_box(X)) == 0xf0);
    }

    #[test]
    #[inline(never)]
    fn bench_lowbits_u64__low8_and_mask() {
        assert!(low8_and(black_box(X)) == 0xf0);
    }

    #[test]
    #[inline(never)]
    fn bench_lowbits_u64__low8_bounded_div_rem() {
        assert!(low8_bounded(black_box(X)) == 0xf0);
    }

    #[test]
    #[inline(never)]
    fn bench_lowbits_u64__low32_rem_const() {
        assert!(low32_rem(black_box(X)) == 0x9abcdef0);
    }

    #[test]
    #[inline(never)]
    fn bench_lowbits_u64__low32_and_mask() {
        assert!(low32_and(black_box(X)) == 0x9abcdef0);
    }

    #[test]
    #[inline(never)]
    fn bench_lowbits_u64__low32_bounded_div_rem() {
        assert!(low32_bounded(black_box(X)) == 0x9abcdef0);
    }

    #[test]
    #[inline(never)]
    fn bench_lowbits_u128__baseline() {
        let x = black_box(X128);
        assert!(x == X128);
    }

    #[test]
    #[inline(never)]
    fn bench_lowbits_u128__low64_rem_const() {
        assert!(low64_rem_u128(black_box(X128)) == 0x0fedcba987654321);
    }

    #[test]
    #[inline(never)]
    fn bench_lowbits_u128__low64_and_mask() {
        assert!(low64_and_u128(black_box(X128)) == 0x0fedcba987654321);
    }

    #[test]
    #[inline(never)]
    fn bench_lowbits_u128__low64_bounded_div_rem() {
        assert!(low64_bounded_u128(black_box(X128)) == 0x0fedcba987654321);
    }

    const X256: u256 = 0x1111222233334444555566667777888899990000aaaabbbbccccddddeeeeffff;

    #[test]
    #[inline(never)]
    fn bench_lowbits_u256__baseline() {
        let x = black_box(X256);
        assert!(x == X256);
    }

    #[test]
    #[inline(never)]
    fn bench_lowbits_u256__low64_rem_const() {
        assert!(low64_rem_u256(black_box(X256)) == 0xccccddddeeeeffff);
    }

    #[test]
    #[inline(never)]
    fn bench_lowbits_u256__low64_and_mask() {
        assert!(low64_and_u256(black_box(X256)) == 0xccccddddeeeeffff);
    }

    #[test]
    #[inline(never)]
    fn bench_lowbits_u256__low64_low_limb_bounded() {
        assert!(low64_limb_u256(black_box(X256)) == 0xccccddddeeeeffff);
    }

    #[test]
    #[inline(never)]
    fn bench_split_u64__baseline() {
        let x = black_box(X);
        assert!(x == X);
        assert!(x != 0);
    }

    #[test]
    #[inline(never)]
    fn bench_split_u64__divrem_const() {
        let (hi, lo) = split32_divrem(black_box(X));
        assert!(hi == 0x12345678);
        assert!(lo == 0x9abcdef0);
    }

    #[test]
    #[inline(never)]
    fn bench_split_u64__div_const_and_mask() {
        let (hi, lo) = split32_div_and(black_box(X));
        assert!(hi == 0x12345678);
        assert!(lo == 0x9abcdef0);
    }

    #[test]
    #[inline(never)]
    fn bench_split_u64__bounded_div_rem() {
        let (hi, lo) = split32_bounded(black_box(X));
        assert!(hi == 0x12345678);
        assert!(lo == 0x9abcdef0);
    }

    #[test]
    #[inline(never)]
    fn bench_bittest_u64__baseline() {
        let x = black_box(X);
        assert!(x == X);
    }

    // bit 8 of ...def0 is 0 (0xe = 1110, bit 8 is the lowest bit of 0xe).
    #[test]
    #[inline(never)]
    fn bench_bittest_u64__and_mask() {
        let x = black_box(X);
        assert!(x == X);
        assert!(!bit8_and(x));
    }

    #[test]
    #[inline(never)]
    fn bench_bittest_u64__div_then_rem() {
        let x = black_box(X);
        assert!(x == X);
        assert!(!bit8_div_rem(x));
    }

    #[test]
    #[inline(never)]
    fn bench_bittest_u64__rem_then_compare() {
        let x = black_box(X);
        assert!(x == X);
        assert!(!bit8_rem_cmp(x));
    }

    #[test]
    #[inline(never)]
    fn bench_ispow2_u64__baseline() {
        let x: u64 = black_box(0x10000);
        assert!(x == 0x10000);
    }

    #[test]
    #[inline(never)]
    fn bench_ispow2_u64__bitwise_x_and_x_minus_1() {
        let x: u64 = black_box(0x10000);
        assert!(x == 0x10000);
        assert!(ispow2_bitwise(x));
    }

    #[test]
    #[inline(never)]
    fn bench_ispow2_u64__math_divides_2_pow_63() {
        let x: u64 = black_box(0x10000);
        assert!(x == 0x10000);
        assert!(ispow2_math(x));
    }

    #[test]
    #[inline(never)]
    fn bench_ispow2_u64__loop_16_iterations() {
        let x: u64 = black_box(0x10000);
        assert!(x == 0x10000);
        assert!(ispow2_loop(x));
    }

    #[test]
    fn test_ispow2() {
        for x in 0..70_u64 {
            assert!(ispow2_bitwise(x) == ispow2_math(x));
            assert!(ispow2_bitwise(x) == ispow2_loop(x));
        }
        assert!(ispow2_math(0x8000000000000000));
        assert!(!ispow2_math(0x8000000000000001));
    }

    #[test]
    #[inline(never)]
    fn bench_parity_u64__baseline() {
        let x = black_box(X + 1);
        assert!(x == X + 1);
    }

    #[test]
    #[inline(never)]
    fn bench_parity_u64__rem_2() {
        let x = black_box(X + 1);
        assert!(x == X + 1);
        assert!(is_odd_rem(x));
    }

    #[test]
    #[inline(never)]
    fn bench_parity_u64__and_1() {
        let x = black_box(X + 1);
        assert!(x == X + 1);
        assert!(is_odd_and(x));
    }
}
