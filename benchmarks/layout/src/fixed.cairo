//! Minimal Q32.32 scalar used by every layout experiment, plus the `Scalar` glue that lets the
//! same generic benchmark run with `Fixed`, `i64` and `felt252` (to separate layout cost from
//! scalar cost).

use core::num::traits::Sqrt;

pub const ONE: i64 = 0x100000000;
pub const ONE_WIDE: i128 = 0x100000000;
pub const HALF: i64 = 0x80000000;

#[derive(Copy, Drop, Debug)]
pub struct Fixed {
    pub raw: i64,
}

/// Wide (unrescaled, Q64.64-ish) intermediate used by the lazy-rescale experiments.
#[derive(Copy, Drop, Debug)]
pub struct Wide {
    pub raw: i128,
}

#[generate_trait]
pub impl FixedImpl of FixedTrait {
    #[inline(always)]
    fn from_int(n: i64) -> Fixed {
        Fixed { raw: n * ONE }
    }

    #[inline(always)]
    fn from_raw(raw: i64) -> Fixed {
        Fixed { raw }
    }

    /// Product kept at 2^64 scale: no division, no narrowing.
    #[inline(always)]
    fn wide_mul(self: Fixed, rhs: Fixed) -> Wide {
        Wide { raw: self.raw.into() * rhs.raw.into() }
    }

    /// Square root: sqrt(raw * 2^32) computed on the widened unsigned value.
    fn sqrt(self: Fixed) -> Fixed {
        assert(self.raw >= 0, 'sqrt of negative');
        let wide: u128 = self.raw.try_into().unwrap();
        let root: u64 = (wide * 0x100000000).sqrt();
        Fixed { raw: root.try_into().unwrap() }
    }
}

#[generate_trait]
pub impl WideImpl of WideTrait {
    /// Single rescale + narrowing of an accumulated wide value.
    #[inline(always)]
    fn rescale(self: Wide) -> Fixed {
        Fixed { raw: (self.raw / ONE_WIDE).try_into().expect('fixed overflow') }
    }
}

pub impl WideAdd of Add<Wide> {
    #[inline(always)]
    fn add(lhs: Wide, rhs: Wide) -> Wide {
        Wide { raw: lhs.raw + rhs.raw }
    }
}

pub impl WideSub of Sub<Wide> {
    #[inline(always)]
    fn sub(lhs: Wide, rhs: Wide) -> Wide {
        Wide { raw: lhs.raw - rhs.raw }
    }
}

pub impl FixedAdd of Add<Fixed> {
    #[inline(always)]
    fn add(lhs: Fixed, rhs: Fixed) -> Fixed {
        Fixed { raw: lhs.raw + rhs.raw }
    }
}

pub impl FixedSub of Sub<Fixed> {
    #[inline(always)]
    fn sub(lhs: Fixed, rhs: Fixed) -> Fixed {
        Fixed { raw: lhs.raw - rhs.raw }
    }
}

pub impl FixedMul of Mul<Fixed> {
    fn mul(lhs: Fixed, rhs: Fixed) -> Fixed {
        let wide: i128 = lhs.raw.into() * rhs.raw.into();
        Fixed { raw: (wide / ONE_WIDE).try_into().expect('fixed overflow') }
    }
}

pub impl FixedDiv of Div<Fixed> {
    fn div(lhs: Fixed, rhs: Fixed) -> Fixed {
        let wide: i128 = lhs.raw.into() * ONE_WIDE;
        Fixed { raw: (wide / rhs.raw.into()).try_into().expect('fixed overflow') }
    }
}

pub impl FixedNeg of Neg<Fixed> {
    #[inline(always)]
    fn neg(a: Fixed) -> Fixed {
        Fixed { raw: -a.raw }
    }
}

pub impl FixedPartialEq of PartialEq<Fixed> {
    #[inline(always)]
    fn eq(lhs: @Fixed, rhs: @Fixed) -> bool {
        *lhs.raw == *rhs.raw
    }
}

/// Glue so one generic benchmark body can run on several scalar types.
pub trait Scalar<T> {
    /// Integer -> scalar (the benchmark inputs are small integers).
    fn from_int(n: i64) -> T;
}

pub impl FixedScalar of Scalar<Fixed> {
    #[inline(always)]
    fn from_int(n: i64) -> Fixed {
        Fixed { raw: n * ONE }
    }
}

pub impl I64Scalar of Scalar<i64> {
    #[inline(always)]
    fn from_int(n: i64) -> i64 {
        n
    }
}

pub impl Felt252Scalar of Scalar<felt252> {
    #[inline(always)]
    fn from_int(n: i64) -> felt252 {
        n.into()
    }
}

/// Opaque scalar input: the conversion is const-folded, the value is not.
#[inline]
pub fn input<T, +Scalar<T>, +Drop<T>>(n: i64) -> T {
    harness::black_box(Scalar::<T>::from_int(n))
}

#[cfg(test)]
mod tests {
    use super::{Fixed, FixedTrait, HALF, ONE, WideTrait};

    #[test]
    fn fixed_arithmetic() {
        let a = FixedTrait::from_raw(3 * HALF); // 1.5
        let b = FixedTrait::from_int(-4);
        assert!(a * b == FixedTrait::from_int(-6));
        assert!(a + b == FixedTrait::from_raw(-5 * HALF));
        assert!(a - b == FixedTrait::from_raw(11 * HALF));
        assert!(-a == FixedTrait::from_raw(-3 * HALF));
        assert!(FixedTrait::from_int(-6) / b == a);
        assert!(FixedTrait::from_int(9).sqrt() == FixedTrait::from_int(3));
        assert!(a.wide_mul(b).rescale() == Fixed { raw: -6 * ONE });
    }
}

/// Field division, only so that `felt252` can run the generic `inverse` benchmarks.
pub impl Felt252Div of Div<felt252> {
    #[inline(always)]
    fn div(lhs: felt252, rhs: felt252) -> felt252 {
        core::felt252_div(lhs, rhs.try_into().expect('division by zero'))
    }
}
