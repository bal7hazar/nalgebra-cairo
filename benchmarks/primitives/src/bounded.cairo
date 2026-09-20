//! `BoundedInt` experiments (`core::internal::bounded_int`, behind the `bounded-int-utils`
//! feature): can range-tracked arithmetic remove range checks from fixed-point kernels?
//!
//! Groups:
//! - `fpmul_u64`: unsigned Q32.32 multiplication `(a * b) >> 32` on `u64`.
//! - `fpmul_i64`: signed Q32.32 multiplication on `i64` (native `i128` path, sign-magnitude,
//!   bounded-int truncating and bounded-int floor/offset variants).
//! - `fpmul_u128`: unsigned Q64.64 multiplication on `u128`.
//! - `sum4_u64`: `a + b + c + d` with per-operation checks vs. a single final check.
//! - `dot3_u64`, `dot3_i64`: Q32.32 dot product, accumulating wide products before rescaling.

#[feature("bounded-int-utils")]
use core::internal::bounded_int::{
    self, AddHelper, BoundedInt, ConstrainHelper, DivRemHelper, MulHelper, SubHelper, UnitInt,
    downcast, upcast,
};
use core::num::traits::WideMul;

// ---------------------------------------------------------------------------------------------
// Unsigned Q32.32 on u64
// ---------------------------------------------------------------------------------------------

const ONE_U128: u128 = 0x100000000;
const NZ_ONE_U128: NonZero<u128> = 0x100000000;

/// Native: `u64_wide_mul` (no range check) + `u128 /` by a constant + checked narrowing.
pub fn fpmul_u64_native(a: u64, b: u64) -> u64 {
    (a.wide_mul(b) / ONE_U128).try_into().unwrap()
}

/// Native, going through `u128 *` (what naive code does).
pub fn fpmul_u64_naive_u128(a: u64, b: u64) -> u64 {
    let a: u128 = a.into();
    let b: u128 = b.into();
    (a * b / ONE_U128).try_into().unwrap()
}

/// Native, going through `u256` (what very naive code does).
pub fn fpmul_u64_naive_u256(a: u64, b: u64) -> u64 {
    let a: u256 = a.into();
    let b: u256 = b.into();
    (a * b / 0x100000000_u256).try_into().unwrap()
}

/// Through felt252: multiply in the field (1 step), convert back to u128, divide.
pub fn fpmul_u64_felt(a: u64, b: u64) -> u64 {
    let p: felt252 = a.into() * b.into();
    let p: u128 = p.try_into().unwrap();
    (p / ONE_U128).try_into().unwrap()
}

type U64Sq = BoundedInt<0, 0xfffffffffffffffe0000000000000001>;
type U64SqShr32 = BoundedInt<0, 0xfffffffffffffffe00000000>;
type U32Rem = BoundedInt<0, 0xffffffff>;
type Two32 = UnitInt<0x100000000>;
const NZ_TWO32: NonZero<Two32> = 0x100000000;

impl MulU64U64 of MulHelper<u64, u64> {
    type Result = U64Sq;
}
impl DivRemU64SqTwo32 of DivRemHelper<U64Sq, Two32> {
    type DivT = U64SqShr32;
    type RemT = U32Rem;
}

/// BoundedInt: exact-range product, div_rem by the constant `2^32`, one checked narrowing.
pub fn fpmul_u64_bounded(a: u64, b: u64) -> u64 {
    let p: U64Sq = bounded_int::mul(a, b);
    let (q, _r) = bounded_int::div_rem(p, NZ_TWO32);
    downcast(q).expect('fpmul overflow')
}

// ---------------------------------------------------------------------------------------------
// Signed Q32.32 on i64
// ---------------------------------------------------------------------------------------------

const ONE_I128: i128 = 0x100000000;

/// Native: `i64_wide_mul` + `i128 /` by a constant (truncating) + checked narrowing.
pub fn fpmul_i64_native(a: i64, b: i64) -> i64 {
    (a.wide_mul(b) / ONE_I128).try_into().unwrap()
}

/// Sign-magnitude representation: what a `struct { mag: u64, sign: bool }` fixed-point pays.
#[derive(Copy, Drop, PartialEq)]
pub struct SignMag {
    pub mag: u64,
    pub sign: bool,
}

pub fn fpmul_signmag(a: SignMag, b: SignMag) -> SignMag {
    SignMag { mag: fpmul_u64_native(a.mag, b.mag), sign: a.sign != b.sign }
}

pub fn fpmul_signmag_bounded(a: SignMag, b: SignMag) -> SignMag {
    SignMag { mag: fpmul_u64_bounded(a.mag, b.mag), sign: a.sign != b.sign }
}

type I64Sq = BoundedInt<-0x3fffffffffffffff8000000000000000, 0x40000000000000000000000000000000>;
type I64SqNeg = BoundedInt<-0x3fffffffffffffff8000000000000000, -1>;
type I64SqPos = BoundedInt<0, 0x40000000000000000000000000000000>;
type I64SqNegated = BoundedInt<1, 0x3fffffffffffffff8000000000000000>;
type MinusOne = UnitInt<-1>;

impl MulI64I64 of MulHelper<i64, i64> {
    type Result = I64Sq;
}
impl ConstrainI64Sq of ConstrainHelper<I64Sq, 0> {
    type LowT = I64SqNeg;
    type HighT = I64SqPos;
}
impl NegateI64SqNeg of MulHelper<I64SqNeg, MinusOne> {
    type Result = I64SqNegated;
}
impl DivRemI64SqPos of DivRemHelper<I64SqPos, Two32> {
    type DivT = BoundedInt<0, 0x400000000000000000000000>;
    type RemT = U32Rem;
}
impl DivRemI64SqNegated of DivRemHelper<I64SqNegated, Two32> {
    type DivT = BoundedInt<0, 0x3fffffffffffffff80000000>;
    type RemT = U32Rem;
}
impl NegateQuotient of MulHelper<BoundedInt<0, 0x3fffffffffffffff80000000>, MinusOne> {
    type Result = BoundedInt<-0x3fffffffffffffff80000000, 0>;
}

/// BoundedInt, truncating (same semantics as `i128 /`): split on the sign of the product.
pub fn fpmul_i64_bounded_trunc(a: i64, b: i64) -> i64 {
    let p: I64Sq = bounded_int::mul(a, b);
    match bounded_int::constrain::<I64Sq, 0>(p) {
        Ok(neg) => {
            let m: I64SqNegated = bounded_int::mul::<_, MinusOne>(neg, -1);
            let (q, _r) = bounded_int::div_rem(m, NZ_TWO32);
            let q = bounded_int::mul::<_, MinusOne>(q, -1);
            downcast(q).expect('fpmul overflow')
        },
        Err(pos) => {
            let (q, _r) = bounded_int::div_rem(pos, NZ_TWO32);
            downcast(q).expect('fpmul overflow')
        },
    }
}

type Offset = UnitInt<0x40000000000000000000000000000000>;
type I64SqOffset = BoundedInt<0x8000000000000000, 0x80000000000000000000000000000000>;
type I64SqOffsetShr32 = BoundedInt<0x80000000, 0x800000000000000000000000>;
type OffsetShr32 = UnitInt<0x400000000000000000000000>;

impl AddOffset of AddHelper<I64Sq, Offset> {
    type Result = I64SqOffset;
}
impl DivRemI64SqOffset of DivRemHelper<I64SqOffset, Two32> {
    type DivT = I64SqOffsetShr32;
    type RemT = U32Rem;
}
impl SubOffset of SubHelper<I64SqOffsetShr32, OffsetShr32> {
    type Result = BoundedInt<-0x3fffffffffffffff80000000, 0x400000000000000000000000>;
}

/// BoundedInt, branchless floor semantics (arithmetic shift right): bias the product by `2^126`
/// so that it is non-negative, div_rem by `2^32`, remove the bias, one checked narrowing.
pub fn fpmul_i64_bounded_floor(a: i64, b: i64) -> i64 {
    let p: I64Sq = bounded_int::mul(a, b);
    let biased: I64SqOffset = bounded_int::add::<_, Offset>(p, 0x40000000000000000000000000000000);
    let (q, _r) = bounded_int::div_rem(biased, NZ_TWO32);
    let q = bounded_int::sub::<_, OffsetShr32>(q, 0x400000000000000000000000);
    downcast(q).expect('fpmul overflow')
}

// ---------------------------------------------------------------------------------------------
// Unsigned Q64.64 on u128
// ---------------------------------------------------------------------------------------------

/// Naive: widen to u256, multiply, divide, narrow.
pub fn fpmul_u128_naive_u256(a: u128, b: u128) -> u128 {
    (a.wide_mul(b) / 0x10000000000000000_u256).try_into().unwrap()
}

/// Limbs: `u128_wide_mul` gives (high, low); result = high * 2^64 + low / 2^64.
pub fn fpmul_u128_limbs(a: u128, b: u128) -> u128 {
    let p = a.wide_mul(b);
    let high: u64 = p.high.try_into().expect('fpmul overflow');
    high.into() * 0x10000000000000000_u128 + p.low / 0x10000000000000000_u128
}

type U64B = BoundedInt<0, 0xffffffffffffffff>;
type Two64 = UnitInt<0x10000000000000000>;
const NZ_TWO64: NonZero<Two64> = 0x10000000000000000;

impl DivRemU128Two64 of DivRemHelper<u128, Two64> {
    type DivT = U64B;
    type RemT = U64B;
}
impl MulU64Two64 of MulHelper<U64B, Two64> {
    type Result = BoundedInt<0, 0xffffffffffffffff0000000000000000>;
}
impl AddHighLow of AddHelper<BoundedInt<0, 0xffffffffffffffff0000000000000000>, U64B> {
    type Result = BoundedInt<0, 0xffffffffffffffffffffffffffffffff>;
}

/// Limbs + BoundedInt: the recombination `high * 2^64 + (low >> 64)` is range-check free.
pub fn fpmul_u128_limbs_bounded(a: u128, b: u128) -> u128 {
    let p = a.wide_mul(b);
    let high: U64B = downcast(p.high).expect('fpmul overflow');
    let (low_high, _) = bounded_int::div_rem(p.low, NZ_TWO64);
    let shifted = bounded_int::mul::<_, Two64>(high, 0x10000000000000000);
    upcast(bounded_int::add(shifted, low_high))
}

// ---------------------------------------------------------------------------------------------
// Deferred overflow check
// ---------------------------------------------------------------------------------------------

pub fn sum4_native(a: u64, b: u64, c: u64, d: u64) -> u64 {
    a + b + c + d
}

pub fn sum4_felt(a: u64, b: u64, c: u64, d: u64) -> u64 {
    let s: felt252 = a.into() + b.into() + c.into() + d.into();
    s.try_into().unwrap()
}

pub fn sum4_u128(a: u64, b: u64, c: u64, d: u64) -> u64 {
    let s: u128 = a.into() + b.into() + c.into() + d.into();
    s.try_into().unwrap()
}

type Sum2 = BoundedInt<0, 0x1fffffffffffffffe>;
type Sum4 = BoundedInt<0, 0x3fffffffffffffffc>;
impl AddU64U64 of AddHelper<u64, u64> {
    type Result = Sum2;
}
impl AddSum2Sum2 of AddHelper<Sum2, Sum2> {
    type Result = Sum4;
}

pub fn sum4_bounded(a: u64, b: u64, c: u64, d: u64) -> u64 {
    let ab: Sum2 = bounded_int::add(a, b);
    let cd: Sum2 = bounded_int::add(c, d);
    let s: Sum4 = bounded_int::add(ab, cd);
    downcast(s).expect('sum overflow')
}

// ---------------------------------------------------------------------------------------------
// Dot product of two 3-vectors (Q32.32): accumulate wide products, rescale once
// ---------------------------------------------------------------------------------------------

/// Native unsigned: three `u64_wide_mul`, two checked `u128 +`, one `u128 /`, one narrowing.
pub fn dot3_u64_native(a: [u64; 3], b: [u64; 3]) -> u64 {
    let [ax, ay, az] = a;
    let [bx, by, bz] = b;
    ((ax.wide_mul(bx) + ay.wide_mul(by) + az.wide_mul(bz)) / ONE_U128).try_into().unwrap()
}

/// Naive unsigned: three full fixed-point multiplications, two checked `u64 +`.
pub fn dot3_u64_naive(a: [u64; 3], b: [u64; 3]) -> u64 {
    let [ax, ay, az] = a;
    let [bx, by, bz] = b;
    fpmul_u64_native(ax, bx) + fpmul_u64_native(ay, by) + fpmul_u64_native(az, bz)
}

type U64Sq2 = BoundedInt<0, 0x1fffffffffffffffc0000000000000002>;
type U64Sq3 = BoundedInt<0, 0x2fffffffffffffffa0000000000000003>;
impl AddSqSq of AddHelper<U64Sq, U64Sq> {
    type Result = U64Sq2;
}
impl AddSq2Sq of AddHelper<U64Sq2, U64Sq> {
    type Result = U64Sq3;
}
impl DivRemU64Sq3 of DivRemHelper<U64Sq3, Two32> {
    type DivT = BoundedInt<0, 0x2fffffffffffffffa00000000>;
    type RemT = U32Rem;
}

/// BoundedInt unsigned: products and sums are range-check free, one div_rem, one narrowing.
pub fn dot3_u64_bounded(a: [u64; 3], b: [u64; 3]) -> u64 {
    let [ax, ay, az] = a;
    let [bx, by, bz] = b;
    let xx: U64Sq = bounded_int::mul(ax, bx);
    let yy: U64Sq = bounded_int::mul(ay, by);
    let zz: U64Sq = bounded_int::mul(az, bz);
    let sum: U64Sq3 = bounded_int::add(bounded_int::add(xx, yy), zz);
    let (q, _r) = bounded_int::div_rem(sum, NZ_TWO32);
    downcast(q).expect('dot overflow')
}

/// Native signed: three `i64_wide_mul`, two checked `i128 +`, one `i128 /`, one narrowing.
pub fn dot3_i64_native(a: [i64; 3], b: [i64; 3]) -> i64 {
    let [ax, ay, az] = a;
    let [bx, by, bz] = b;
    ((ax.wide_mul(bx) + ay.wide_mul(by) + az.wide_mul(bz)) / ONE_I128).try_into().unwrap()
}

/// Naive signed: three full fixed-point multiplications, two checked `i64 +`.
pub fn dot3_i64_naive(a: [i64; 3], b: [i64; 3]) -> i64 {
    let [ax, ay, az] = a;
    let [bx, by, bz] = b;
    fpmul_i64_native(ax, bx) + fpmul_i64_native(ay, by) + fpmul_i64_native(az, bz)
}

type I64Sq2 = BoundedInt<-0x7fffffffffffffff0000000000000000, 0x80000000000000000000000000000000>;
type I64Sq3 = BoundedInt<-0xbffffffffffffffe8000000000000000, 0xc0000000000000000000000000000000>;
type Offset3 = UnitInt<0xc0000000000000000000000000000000>;
type I64Sq3Offset = BoundedInt<0x18000000000000000, 0x180000000000000000000000000000000>;
type I64Sq3OffsetShr32 = BoundedInt<0x180000000, 0x1800000000000000000000000>;
type Offset3Shr32 = UnitInt<0xc00000000000000000000000>;
impl AddISqISq of AddHelper<I64Sq, I64Sq> {
    type Result = I64Sq2;
}
impl AddISq2ISq of AddHelper<I64Sq2, I64Sq> {
    type Result = I64Sq3;
}
impl AddOffset3 of AddHelper<I64Sq3, Offset3> {
    type Result = I64Sq3Offset;
}
impl DivRemI64Sq3Offset of DivRemHelper<I64Sq3Offset, Two32> {
    type DivT = I64Sq3OffsetShr32;
    type RemT = U32Rem;
}
impl SubOffset3 of SubHelper<I64Sq3OffsetShr32, Offset3Shr32> {
    type Result = BoundedInt<-0xbffffffffffffffe80000000, 0xc00000000000000000000000>;
}

/// BoundedInt signed (floor semantics): everything but the final narrowing is range-check free
/// except the single div_rem.
pub fn dot3_i64_bounded(a: [i64; 3], b: [i64; 3]) -> i64 {
    let [ax, ay, az] = a;
    let [bx, by, bz] = b;
    let xx: I64Sq = bounded_int::mul(ax, bx);
    let yy: I64Sq = bounded_int::mul(ay, by);
    let zz: I64Sq = bounded_int::mul(az, bz);
    let sum: I64Sq3 = bounded_int::add(bounded_int::add(xx, yy), zz);
    let biased: I64Sq3Offset = bounded_int::add::<
        _, Offset3,
    >(sum, 0xc0000000000000000000000000000000);
    let (q, _r) = bounded_int::div_rem(biased, NZ_TWO32);
    let q = bounded_int::sub::<_, Offset3Shr32>(q, 0xc00000000000000000000000);
    downcast(q).expect('dot overflow')
}

#[cfg(test)]
mod tests {
    use harness::black_box;
    use super::*;

    // 3.5 * 2.25 = 7.875 in Q32.32
    const A: u64 = 0x380000000;
    const B: u64 = 0x240000000;
    const AB: u64 = 0x7e0000000;

    #[test]
    #[inline(never)]
    fn bench_fpmul_u64__baseline() {
        let a = black_box(A);
        let _ = black_box(B);
        assert!(a == A);
    }

    #[test]
    #[inline(never)]
    fn bench_fpmul_u64__native_widemul_div() {
        assert!(fpmul_u64_native(black_box(A), black_box(B)) == AB);
    }

    #[test]
    #[inline(never)]
    fn bench_fpmul_u64__naive_u128_mul_div() {
        assert!(fpmul_u64_naive_u128(black_box(A), black_box(B)) == AB);
    }

    #[test]
    #[inline(never)]
    fn bench_fpmul_u64__naive_u256_mul_div() {
        assert!(fpmul_u64_naive_u256(black_box(A), black_box(B)) == AB);
    }

    #[test]
    #[inline(never)]
    fn bench_fpmul_u64__felt_mul_div() {
        assert!(fpmul_u64_felt(black_box(A), black_box(B)) == AB);
    }

    #[test]
    #[inline(never)]
    fn bench_fpmul_u64__bounded() {
        assert!(fpmul_u64_bounded(black_box(A), black_box(B)) == AB);
    }

    // -3.5 * 2.25 = -7.875
    const NA: i64 = -0x380000000;
    const SB: i64 = 0x240000000;
    const NAB: i64 = -0x7e0000000;

    #[test]
    #[inline(never)]
    fn bench_fpmul_i64__baseline() {
        let a = black_box(NA);
        let _ = black_box(SB);
        assert!(a == NA);
    }

    #[test]
    #[inline(never)]
    fn bench_fpmul_i64__native_widemul_i128_div() {
        assert!(fpmul_i64_native(black_box(NA), black_box(SB)) == NAB);
    }

    #[test]
    #[inline(never)]
    fn bench_fpmul_i64__native_widemul_i128_div_pos() {
        assert!(fpmul_i64_native(black_box(SB), black_box(SB)) == 0x510000000);
    }

    #[test]
    #[inline(never)]
    fn bench_fpmul_i64__bounded_trunc() {
        assert!(fpmul_i64_bounded_trunc(black_box(NA), black_box(SB)) == NAB);
    }

    #[test]
    #[inline(never)]
    fn bench_fpmul_i64__bounded_trunc_pos() {
        assert!(fpmul_i64_bounded_trunc(black_box(SB), black_box(SB)) == 0x510000000);
    }

    #[test]
    #[inline(never)]
    fn bench_fpmul_i64__bounded_floor() {
        assert!(fpmul_i64_bounded_floor(black_box(NA), black_box(SB)) == NAB);
    }

    #[test]
    #[inline(never)]
    fn bench_fpmul_i64__bounded_floor_pos() {
        assert!(fpmul_i64_bounded_floor(black_box(SB), black_box(SB)) == 0x510000000);
    }

    #[test]
    fn test_fpmul_i64_semantics() {
        // -1 ulp * 0.5: truncation gives 0, floor gives -1 ulp.
        assert!(fpmul_i64_native(-1, 0x80000000) == 0);
        assert!(fpmul_i64_bounded_trunc(-1, 0x80000000) == 0);
        assert!(fpmul_i64_bounded_floor(-1, 0x80000000) == -1);
        // extremes
        let max: i64 = 0x7fffffffffffffff;
        assert!(fpmul_i64_bounded_floor(max, 0x100000000) == max);
        assert!(fpmul_i64_bounded_floor(-max - 1, 0x100000000) == -max - 1);
        assert!(fpmul_i64_bounded_trunc(-max - 1, 0x100000000) == -max - 1);
        assert!(fpmul_i64_bounded_trunc(-max, -0x100000000) == max);
    }

    #[test]
    #[should_panic(expected: 'fpmul overflow')]
    fn test_fpmul_i64_floor_overflow() {
        fpmul_i64_bounded_floor(black_box(0x7fffffffffffffff), 0x200000000);
    }

    #[test]
    #[inline(never)]
    fn bench_fpmul_signmag__baseline() {
        let a = black_box(SignMag { mag: A, sign: true });
        let _ = black_box(SignMag { mag: B, sign: false });
        assert!(a == SignMag { mag: A, sign: true });
    }

    #[test]
    #[inline(never)]
    fn bench_fpmul_signmag__native() {
        let a = black_box(SignMag { mag: A, sign: true });
        let b = black_box(SignMag { mag: B, sign: false });
        assert!(fpmul_signmag(a, b) == SignMag { mag: AB, sign: true });
    }

    #[test]
    #[inline(never)]
    fn bench_fpmul_signmag__bounded() {
        let a = black_box(SignMag { mag: A, sign: true });
        let b = black_box(SignMag { mag: B, sign: false });
        assert!(fpmul_signmag_bounded(a, b) == SignMag { mag: AB, sign: true });
    }

    // Q64.64: 3.5 * 2.25 = 7.875
    const A128: u128 = 0x38000000000000000;
    const B128: u128 = 0x24000000000000000;
    const AB128: u128 = 0x7e000000000000000;

    #[test]
    #[inline(never)]
    fn bench_fpmul_u128__baseline() {
        let a = black_box(A128);
        let _ = black_box(B128);
        assert!(a == A128);
    }

    #[test]
    #[inline(never)]
    fn bench_fpmul_u128__naive_u256_div() {
        assert!(fpmul_u128_naive_u256(black_box(A128), black_box(B128)) == AB128);
    }

    #[test]
    #[inline(never)]
    fn bench_fpmul_u128__limbs() {
        assert!(fpmul_u128_limbs(black_box(A128), black_box(B128)) == AB128);
    }

    #[test]
    #[inline(never)]
    fn bench_fpmul_u128__limbs_bounded() {
        assert!(fpmul_u128_limbs_bounded(black_box(A128), black_box(B128)) == AB128);
    }

    #[test]
    #[inline(never)]
    fn bench_sum4_u64__baseline() {
        let a: u64 = black_box(1_000_000_000_000);
        let _: u64 = black_box(2_000_000_000_000);
        let _: u64 = black_box(3_000_000_000_000);
        let _: u64 = black_box(4_000_000_000_000);
        assert!(a == 1_000_000_000_000);
    }

    #[test]
    #[inline(never)]
    fn bench_sum4_u64__native() {
        let r = sum4_native(
            black_box(1_000_000_000_000),
            black_box(2_000_000_000_000),
            black_box(3_000_000_000_000),
            black_box(4_000_000_000_000),
        );
        assert!(r == 10_000_000_000_000);
    }

    #[test]
    #[inline(never)]
    fn bench_sum4_u64__via_felt252() {
        let r = sum4_felt(
            black_box(1_000_000_000_000),
            black_box(2_000_000_000_000),
            black_box(3_000_000_000_000),
            black_box(4_000_000_000_000),
        );
        assert!(r == 10_000_000_000_000);
    }

    #[test]
    #[inline(never)]
    fn bench_sum4_u64__via_u128() {
        let r = sum4_u128(
            black_box(1_000_000_000_000),
            black_box(2_000_000_000_000),
            black_box(3_000_000_000_000),
            black_box(4_000_000_000_000),
        );
        assert!(r == 10_000_000_000_000);
    }

    #[test]
    #[inline(never)]
    fn bench_sum4_u64__bounded() {
        let r = sum4_bounded(
            black_box(1_000_000_000_000),
            black_box(2_000_000_000_000),
            black_box(3_000_000_000_000),
            black_box(4_000_000_000_000),
        );
        assert!(r == 10_000_000_000_000);
    }

    // (1.5, -2, 0.5) . (2, 0.25, -4) = 3 - 0.5 - 2 = 0.5 ; unsigned: (1.5, 2, 0.5) . (2, 0.25, 4) =
    // 5.5
    #[test]
    #[inline(never)]
    fn bench_dot3_u64__baseline() {
        let a: [u64; 3] = black_box([0x180000000, 0x200000000, 0x80000000]);
        let b: [u64; 3] = black_box([0x200000000, 0x40000000, 0x400000000]);
        let [x, _, _] = a;
        let _ = b;
        assert!(x == 0x180000000);
    }

    #[test]
    #[inline(never)]
    fn bench_dot3_u64__naive_3_fpmul() {
        let a: [u64; 3] = black_box([0x180000000, 0x200000000, 0x80000000]);
        let b: [u64; 3] = black_box([0x200000000, 0x40000000, 0x400000000]);
        assert!(dot3_u64_naive(a, b) == 0x580000000);
    }

    #[test]
    #[inline(never)]
    fn bench_dot3_u64__native_wide_acc() {
        let a: [u64; 3] = black_box([0x180000000, 0x200000000, 0x80000000]);
        let b: [u64; 3] = black_box([0x200000000, 0x40000000, 0x400000000]);
        assert!(dot3_u64_native(a, b) == 0x580000000);
    }

    #[test]
    #[inline(never)]
    fn bench_dot3_u64__bounded() {
        let a: [u64; 3] = black_box([0x180000000, 0x200000000, 0x80000000]);
        let b: [u64; 3] = black_box([0x200000000, 0x40000000, 0x400000000]);
        assert!(dot3_u64_bounded(a, b) == 0x580000000);
    }

    #[test]
    #[inline(never)]
    fn bench_dot3_i64__baseline() {
        let a: [i64; 3] = black_box([0x180000000, -0x200000000, 0x80000000]);
        let b: [i64; 3] = black_box([0x200000000, 0x40000000, -0x400000000]);
        let [x, _, _] = a;
        let _ = b;
        assert!(x == 0x180000000);
    }

    #[test]
    #[inline(never)]
    fn bench_dot3_i64__naive_3_fpmul() {
        let a: [i64; 3] = black_box([0x180000000, -0x200000000, 0x80000000]);
        let b: [i64; 3] = black_box([0x200000000, 0x40000000, -0x400000000]);
        assert!(dot3_i64_naive(a, b) == 0x80000000);
    }

    #[test]
    #[inline(never)]
    fn bench_dot3_i64__native_wide_acc() {
        let a: [i64; 3] = black_box([0x180000000, -0x200000000, 0x80000000]);
        let b: [i64; 3] = black_box([0x200000000, 0x40000000, -0x400000000]);
        assert!(dot3_i64_native(a, b) == 0x80000000);
    }

    #[test]
    #[inline(never)]
    fn bench_dot3_i64__bounded_floor() {
        let a: [i64; 3] = black_box([0x180000000, -0x200000000, 0x80000000]);
        let b: [i64; 3] = black_box([0x200000000, 0x40000000, -0x400000000]);
        assert!(dot3_i64_bounded(a, b) == 0x80000000);
    }

    #[test]
    fn test_dot3_bounded_extremes() {
        let max: i64 = 0x7fffffffffffffff;
        let min: i64 = -max - 1;
        let one: i64 = 0x100000000;
        assert!(dot3_i64_bounded([max, 0, 0], [one, 0, 0]) == max);
        assert!(dot3_i64_bounded([min, 0, 0], [one, 0, 0]) == min);
        // -1 ulp * 0.5 summed three times: floor(-1.5 ulp) = -2 ulp (native truncation gives -1).
        assert!(dot3_i64_bounded([-1, -1, -1], [0x80000000, 0x80000000, 0x80000000]) == -2);
        assert!(dot3_i64_native([-1, -1, -1], [0x80000000, 0x80000000, 0x80000000]) == -1);
    }

    #[test]
    #[should_panic(expected: 'dot overflow')]
    fn test_dot3_bounded_overflow() {
        let min: i64 = -0x7fffffffffffffff - 1;
        dot3_i64_bounded(black_box([min, min, min]), [min, min, min]);
    }
}
