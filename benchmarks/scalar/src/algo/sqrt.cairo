//! sqrt / inverse sqrt / normalize alternatives for `BI64` (Q32.32).

#[feature("bounded-int-utils")]
use core::internal::bounded_int::{
    self, AddHelper, BoundedInt, DivRemHelper, MulHelper, SubHelper, UnitInt, downcast,
};
use core::num::traits::{Sqrt, WideMul};
use crate::algo::seeds;
use crate::bi64::{self, BI64};

/// 1. corelib integer sqrt (`u128_sqrt` libfunc: the prover guesses, the program verifies).
#[inline(always)]
pub fn sqrt_corelib(a: BI64) -> BI64 {
    bi64::sqrt(a)
}

#[inline(always)]
fn widen(a: BI64) -> u128 {
    let m: u64 = a.raw.try_into().expect('sqrt of negative');
    m.wide_mul(0x100000000_u64)
}

/// 2. Textbook integer Newton loop, seeded with n/2 + 1, runs until it stops decreasing.
pub fn sqrt_newton_loop(a: BI64) -> BI64 {
    let n = widen(a);
    if n == 0 {
        return a;
    }
    let mut x: u128 = n / 2 + 1;
    loop {
        let y = (x + n / x) / 2;
        if y >= x {
            break;
        }
        x = y;
    }
    let r: u64 = x.try_into().unwrap();
    BI64 { raw: r.try_into().unwrap() }
}

/// 3. Newton seeded with a power of two within 2x of the root (compare tree on the magnitude,
/// no bitwise op), 6 unrolled iterations + final correction.
pub fn sqrt_newton_seeded(a: BI64) -> BI64 {
    let m: u64 = a.raw.try_into().expect('sqrt of negative');
    if m == 0 {
        return a;
    }
    let n: u128 = m.wide_mul(0x100000000_u64);
    let x: u128 = seeds::sqrt_seed(m).into();
    let x = (x + n / x) / 2;
    let x = (x + n / x) / 2;
    let x = (x + n / x) / 2;
    let x = (x + n / x) / 2;
    let x = (x + n / x) / 2;
    let x = (x + n / x) / 2;
    let x = if x * x > n {
        x - 1
    } else {
        x
    };
    let r: u64 = x.try_into().unwrap();
    BI64 { raw: r.try_into().unwrap() }
}

const ONE_AND_HALF: BI64 = BI64 { raw: 0x180000000 };
const HALF: BI64 = BI64 { raw: 0x80000000 };

/// Multiplication-only Newton for 1/sqrt(a): y <- y * (1.5 - (a/2) * y * y), 6 iterations from a
/// power-of-two seed. Approximate (precision degrades for large `a`).
pub fn inv_sqrt_newton(a: BI64) -> BI64 {
    let m: u64 = a.raw.try_into().expect('sqrt of negative');
    let half_a = a * HALF;
    let y = BI64 { raw: seeds::rsqrt_seed(m).try_into().unwrap() };
    let y = y * (ONE_AND_HALF - half_a * y * y);
    let y = y * (ONE_AND_HALF - half_a * y * y);
    let y = y * (ONE_AND_HALF - half_a * y * y);
    let y = y * (ONE_AND_HALF - half_a * y * y);
    let y = y * (ONE_AND_HALF - half_a * y * y);
    y * (ONE_AND_HALF - half_a * y * y)
}

/// 4. sqrt(a) = a * (1/sqrt(a)).
pub fn sqrt_via_inv_sqrt(a: BI64) -> BI64 {
    a * inv_sqrt_newton(a)
}

/// 1/sqrt(a) as ONE / sqrt(a) with the generic signed division.
#[inline(always)]
pub fn inv_sqrt_div(a: BI64) -> BI64 {
    BI64 { raw: bi64::ONE } / bi64::sqrt(a)
}

const TWO64: felt252 = 0x10000000000000000;
const TWO96: felt252 = 0x1000000000000000000000000;
type Inv64 = BoundedInt<1, 0x10000000000000000>;
type Inv96 = BoundedInt<0x100000000, 0x1000000000000000000000000>;
impl DivTwo64U64 of DivRemHelper<UnitInt<TWO64>, u64> {
    type DivT = Inv64;
    type RemT = BoundedInt<0, 0xfffffffffffffffe>;
}
impl DivTwo96U64 of DivRemHelper<UnitInt<TWO96>, u64> {
    type DivT = Inv96;
    type RemT = BoundedInt<0, 0xfffffffffffffffe>;
}

/// 1/s for a positive raw `s`: 2^64 / s as one unsigned bounded div_rem (no sign handling).
#[inline(always)]
fn inv_raw(s: u64) -> BI64 {
    let s: NonZero<u64> = s.try_into().expect('division by zero');
    let (q, _r) = bounded_int::div_rem::<UnitInt<TWO64>, u64>(0x10000000000000000, s);
    BI64 { raw: downcast(q).expect('BI64 inv overflow') }
}

/// 1/sqrt(a) = 2^64 / isqrt(a << 32).
#[inline(always)]
pub fn inv_sqrt_direct(a: BI64) -> BI64 {
    let s: u64 = widen(a).sqrt();
    inv_raw(s)
}

// --- normalize ---------------------------------------------------------------------------------

/// v / |v| with three floor divisions (exact to 1 ulp).
#[inline(always)]
pub fn normalize3_div(x: BI64, y: BI64, z: BI64) -> (BI64, BI64, BI64) {
    let len = bi64::length3(x, y, z);
    (x / len, y / len, z / len)
}

/// v * (1/|v|), Q32.32 inverse: cheap but the inverse only has log2(2^32/|v|) significant bits.
#[inline(always)]
pub fn normalize3_inv(x: BI64, y: BI64, z: BI64) -> (BI64, BI64, BI64) {
    let len = bi64::length3(x, y, z);
    let inv = inv_raw(len.raw.try_into().unwrap());
    (x * inv, y * inv, z * inv)
}

type WideInvProd =
    BoundedInt<
        -0x8000000000000000000000000000000000000000, 0x7fffffffffffffff000000000000000000000000,
    >;
const TWO160: felt252 = 0x10000000000000000000000000000000000000000;
type WideInvOff =
    BoundedInt<
        0x8000000000000000000000000000000000000000, 0x17fffffffffffffff000000000000000000000000,
    >;
type WideInvQuotOff = BoundedInt<0x800000000000000000000000, 0x17fffffffffffffff00000000>;
type WideInvQuot = BoundedInt<-0x800000000000000000000000, 0x7fffffffffffffff00000000>;
impl MulI64Inv96 of MulHelper<i64, Inv96> {
    type Result = WideInvProd;
}
impl AddWideInvOff of AddHelper<WideInvProd, UnitInt<TWO160>> {
    type Result = WideInvOff;
}
impl DivWideInvOff of DivRemHelper<WideInvOff, UnitInt<TWO64>> {
    type DivT = WideInvQuotOff;
    type RemT = BoundedInt<0, 0xffffffffffffffff>;
}
impl SubWideInvQuotOff of SubHelper<WideInvQuotOff, UnitInt<TWO96>> {
    type Result = WideInvQuot;
}

#[inline(always)]
fn mul_inv96(x: BI64, inv: Inv96) -> BI64 {
    let p: WideInvProd = bounded_int::mul::<i64, Inv96>(x.raw, inv);
    let p: WideInvOff = bounded_int::add::<
        WideInvProd, UnitInt<TWO160>,
    >(p, 0x10000000000000000000000000000000000000000);
    let (q, _r) = bounded_int::div_rem::<WideInvOff, UnitInt<TWO64>>(p, 0x10000000000000000);
    let q: WideInvQuot = bounded_int::sub::<
        WideInvQuotOff, UnitInt<TWO96>,
    >(q, 0x1000000000000000000000000);
    BI64 { raw: downcast(q).expect('BI64 overflow') }
}

/// v * (2^96/|v|) >> 64: one unsigned division, inverse kept with 64 fractional bits, so the
/// result is accurate to ~1 ulp whatever the length.
#[inline(always)]
pub fn normalize3_wide_inv(x: BI64, y: BI64, z: BI64) -> (BI64, BI64, BI64) {
    let len = bi64::length3(x, y, z);
    let s: u64 = len.raw.try_into().unwrap();
    let s: NonZero<u64> = s.try_into().expect('division by zero');
    let (inv, _r) = bounded_int::div_rem::<UnitInt<TWO96>, u64>(0x1000000000000000000000000, s);
    (mul_inv96(x, inv), mul_inv96(y, inv), mul_inv96(z, inv))
}
