//! The `Fixed` type and its constants.

use core::fmt::{Display, Error, Formatter};
use core::num::traits::{Bounded, One, Zero};
// Re-exported HERE on purpose: impls visible in the module of a type are found without any import
// (`crates/simba/tests/test_public_api.cairo`), so `use simba::fixed::Fixed` is enough downstream.
pub use super::convert::{
    I128TryIntoFixed, I16IntoFixed, I32IntoFixed, I64TryIntoFixed, I8IntoFixed, U128TryIntoFixed,
    U16IntoFixed, U32TryIntoFixed, U64TryIntoFixed, U8IntoFixed,
};
pub use super::ops::{
    FixedAdd, FixedAddAssign, FixedDiv, FixedDivAssign, FixedMul, FixedMulAssign, FixedNeg,
    FixedPartialOrd, FixedRem, FixedRemAssign, FixedSub, FixedSubAssign,
};

/// Q32.32 fixed-point number: `value = raw / 2^32`, stored in a native `i64` (one felt).
///
/// Range `[-2^31, 2^31)` (about ±2.147e9), resolution `2^-32` (about 2.33e-10). Unique zero.
/// Numeric specification (DESIGN D2): every operation rounds toward -inf (floor) exactly once
/// per output scalar, and panics with `simba::errors::OVERFLOW` when the result does not fit;
/// nothing ever wraps. The same inputs give bit-identical outputs forever.
///
/// Upstream equivalent: the `f32`/`f64` scalars of nalgebra (`simba::scalar::RealField`).
#[derive(Copy, Drop, PartialEq, Serde, Default, Hash, Debug)]
pub struct Fixed {
    /// `value * 2^32`.
    pub raw: i64,
}

/// Number of fractional bits.
pub const FRAC_BITS: u32 = 32;

// Constants are floor-rounded to Q32.32 by `tools/fixed_model/fixed_model.py` (`constants()`),
// and pinned by `fixed::tests_generated::test_constants_match_model`.

/// 0.
pub const ZERO: Fixed = Fixed { raw: 0 };
/// 1.
pub const ONE: Fixed = Fixed { raw: 0x100000000 };
/// -1.
pub const NEG_ONE: Fixed = Fixed { raw: -0x100000000 };
/// 2.
pub const TWO: Fixed = Fixed { raw: 0x200000000 };
/// 1/2.
pub const HALF: Fixed = Fixed { raw: 0x80000000 };
/// Smallest positive value, 2^-32: one raw unit (1 ulp). Not a float-style machine epsilon.
pub const EPSILON: Fixed = Fixed { raw: 1 };
/// Smallest value, -2^31.
pub const MIN: Fixed = Fixed { raw: -0x8000000000000000 };
/// Largest value, 2^31 - 2^-32.
pub const MAX: Fixed = Fixed { raw: 0x7fffffffffffffff };
/// π, floored (3.14159265346...).
pub const PI: Fixed = Fixed { raw: 13493037704 };
/// 2π, floored.
pub const TAU: Fixed = Fixed { raw: 26986075409 };
/// π/2, floored.
pub const FRAC_PI_2: Fixed = Fixed { raw: 6746518852 };
/// π/3, floored.
pub const FRAC_PI_3: Fixed = Fixed { raw: 4497679234 };
/// π/4, floored.
pub const FRAC_PI_4: Fixed = Fixed { raw: 3373259426 };
/// π/6, floored.
pub const FRAC_PI_6: Fixed = Fixed { raw: 2248839617 };
/// 1/π, floored.
pub const FRAC_1_PI: Fixed = Fixed { raw: 1367130551 };
/// Euler's number e, floored.
pub const E: Fixed = Fixed { raw: 11674931554 };
/// ln 2, floored.
pub const LN_2: Fixed = Fixed { raw: 2977044471 };
/// ln 10, floored.
pub const LN_10: Fixed = Fixed { raw: 9889527670 };
/// √2, floored.
pub const SQRT_2: Fixed = Fixed { raw: 6074000999 };
/// 1/√2, floored.
pub const FRAC_1_SQRT_2: Fixed = Fixed { raw: 3037000499 };

pub impl FixedZero of Zero<Fixed> {
    #[inline(always)]
    fn zero() -> Fixed {
        ZERO
    }
    #[inline(always)]
    fn is_zero(self: @Fixed) -> bool {
        *self.raw == 0
    }
    #[inline(always)]
    fn is_non_zero(self: @Fixed) -> bool {
        *self.raw != 0
    }
}

pub impl FixedOne of One<Fixed> {
    #[inline(always)]
    fn one() -> Fixed {
        ONE
    }
    #[inline(always)]
    fn is_one(self: @Fixed) -> bool {
        *self.raw == 0x100000000
    }
    #[inline(always)]
    fn is_non_one(self: @Fixed) -> bool {
        *self.raw != 0x100000000
    }
}

pub impl FixedBounded of Bounded<Fixed> {
    const MIN: Fixed = MIN;
    const MAX: Fixed = MAX;
}

/// Decimal rendering for tests and debugging (not gas-sensitive): sign, integer part and the
/// 10 first decimals of the exact value, truncated (`-3.5000000000`).
pub impl FixedDisplay of Display<Fixed> {
    fn fmt(self: @Fixed, ref f: Formatter) -> Result<(), Error> {
        let raw: i128 = (*self.raw).into();
        let mag: u128 = if raw < 0 {
            f.buffer.append(@"-");
            (-raw).try_into().unwrap()
        } else {
            raw.try_into().unwrap()
        };
        let int_part = mag / 0x100000000;
        let frac = (mag % 0x100000000) * 10000000000 / 0x100000000;
        Display::fmt(@int_part, ref f)?;
        f.buffer.append(@".");
        let mut pad: u128 = 1000000000;
        while pad > frac && pad > 1 {
            f.buffer.append(@"0");
            pad /= 10;
        }
        Display::fmt(@frac, ref f)
    }
}

#[cfg(test)]
mod tests {
    use core::hash::{HashStateExTrait, HashStateTrait};
    use core::num::traits::{Bounded, One, Zero};
    use core::poseidon::PoseidonTrait;
    use nalgebra_testing::black_box;
    use super::{EPSILON, Fixed, HALF, MAX, MIN, NEG_ONE, ONE, TWO, ZERO};

    #[test]
    fn test_default_is_zero() {
        let d: Fixed = Default::default();
        assert!(d == ZERO);
    }

    #[test]
    fn test_constants_integer_values() {
        assert!(ONE.raw == 4294967296);
        assert!(NEG_ONE.raw == -ONE.raw);
        assert!(TWO.raw == 2 * ONE.raw);
        assert!(HALF.raw * 2 == ONE.raw);
        assert!(EPSILON.raw == 1);
        assert!(MIN.raw == Bounded::<i64>::MIN);
        assert!(MAX.raw == Bounded::<i64>::MAX);
    }

    #[test]
    fn test_zero_one_bounded_traits() {
        assert!(Zero::<Fixed>::zero() == ZERO);
        assert!(ZERO.is_zero());
        assert!(!ZERO.is_non_zero());
        assert!(EPSILON.is_non_zero());
        assert!(One::<Fixed>::one() == ONE);
        assert!(ONE.is_one());
        assert!(TWO.is_non_one());
        assert!(Bounded::<Fixed>::MIN == MIN);
        assert!(Bounded::<Fixed>::MAX == MAX);
    }

    #[test]
    fn test_partial_eq_unique_zero() {
        assert!(Fixed { raw: 0 } == ZERO);
        assert!(Fixed { raw: 1 } != ZERO);
        assert!(Fixed { raw: -1 } != Fixed { raw: 1 });
    }

    #[test]
    fn test_serde_roundtrip() {
        let mut out: Array<felt252> = array![];
        Fixed { raw: -0x380000000 }.serialize(ref out);
        MAX.serialize(ref out);
        assert!(out.len() == 2);
        let mut span = out.span();
        assert!(Serde::<Fixed>::deserialize(ref span) == Some(Fixed { raw: -0x380000000 }));
        assert!(Serde::<Fixed>::deserialize(ref span) == Some(MAX));
    }

    #[test]
    fn test_serde_rejects_out_of_range() {
        let mut span = array![0x8000000000000000].span();
        assert!(Serde::<Fixed>::deserialize(ref span).is_none());
    }

    #[test]
    fn test_hash_depends_on_value() {
        let h1 = PoseidonTrait::new().update_with(ONE).finalize();
        let h2 = PoseidonTrait::new().update_with(ONE).finalize();
        let h3 = PoseidonTrait::new().update_with(TWO).finalize();
        assert!(h1 == h2);
        assert!(h1 != h3);
    }

    #[test]
    fn test_display_decimal() {
        assert!(format!("{}", Fixed { raw: -0x380000000 }) == "-3.5000000000");
        assert!(format!("{}", ZERO) == "0.0000000000");
        assert!(format!("{}", Fixed { raw: 0x100000000 + 0x4000000 }) == "1.0156250000");
        assert!(format!("{}", EPSILON) == "0.0000000002");
        assert!(format!("{}", MIN) == "-2147483648.0000000000");
        assert!(format!("{}", MAX) == "2147483647.9999999997");
    }

    #[test]
    fn test_debug_shows_raw() {
        assert!(format!("{:?}", ONE) == "Fixed { raw: 4294967296 }");
    }

    // --- gas benchmarks (net = raw - baseline of the group) --------------------------------------

    #[test]
    #[inline(never)]
    fn bench_is_zero__baseline() {
        let _a = black_box(ONE);
        let e = black_box(false);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_is_zero__fixed_is_zero() {
        let a = black_box(ONE);
        let e = black_box(false);
        assert!(a.is_zero() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_is_zero__fixed_is_non_zero() {
        let a = black_box(ZERO);
        let e = black_box(false);
        assert!(a.is_non_zero() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_is_zero__fixed_is_one() {
        let a = black_box(TWO);
        let e = black_box(false);
        assert!(a.is_one() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_is_zero__fixed_is_non_one() {
        let a = black_box(ONE);
        let e = black_box(false);
        assert!(a.is_non_one() == e);
    }
}
