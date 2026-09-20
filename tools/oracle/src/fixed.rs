//! Q32.32 number model: `value = raw / 2^32`, `raw: i64`, rounding = floor.

/// Number of fractional bits of the Cairo `Fixed` scalar.
pub const FRAC_BITS: u32 = 32;
/// `2^32` as an `f64` (the raw value of `1.0`).
pub const ONE: f64 = (1u64 << FRAC_BITS) as f64;
/// Largest input magnitude (raw) the oracle ever generates: `2^53`, so `raw as f64` is exact.
pub const MAX_EXACT_RAW: i64 = 1 << 53;
/// Below this scaled magnitude the floor of an `f64` result is checked for ambiguity.
const AMBIGUITY_CHECK_LIMIT: f64 = 35_184_372_088_832.0; // 2^45

/// Exact conversion of a raw Q32.32 value to `f64` (exact because `|raw| <= 2^53`).
pub fn to_f64(raw: i64) -> f64 {
    assert!(
        raw.abs() <= MAX_EXACT_RAW,
        "raw input {raw} is not exactly representable in f64"
    );
    raw as f64 / ONE
}

/// Floor-quantisation of an `f64` to raw Q32.32. `None` when not finite or out of the `i64` range.
pub fn quantize(x: f64) -> Option<i64> {
    if !x.is_finite() {
        return None;
    }
    let s = (x * ONE).floor();
    // 2^63 is exactly representable; the valid range is [-2^63, 2^63).
    if !(-9_223_372_036_854_775_808.0..9_223_372_036_854_775_808.0).contains(&s) {
        return None;
    }
    Some(s as i64)
}

/// True when `floor(x * 2^32)` could change under a last-bit difference of the `f64` result
/// (different libm, different evaluation order). Such cases are resampled so that the committed
/// vectors are reproducible. Above `2^45` raw the check is skipped (an `f64` no longer carries
/// enough fractional bits); the tolerance covers the oracle's own error there.
pub fn is_ambiguous(x: f64) -> bool {
    let s = x * ONE;
    if s.abs() >= AMBIGUITY_CHECK_LIMIT {
        return false;
    }
    let frac = s - s.floor();
    if frac == 0.0 {
        // Exactly representable (copies, negations, sums): nothing to disambiguate.
        return false;
    }
    let margin = (s.abs() / 281_474_976_710_656.0).max(1.0 / 1024.0); // max(2^-10, |s| * 2^-48)
    frac < margin || frac > 1.0 - margin
}

/// Floor of an exact wide integer to raw Q32.32: `shift = 32` for a product of two raws
/// (`2^64` scale), `64` for a product of three.
pub fn rescale_exact(wide: i128, shift: u32) -> Option<i64> {
    i64::try_from(wide >> shift).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_round_trip_is_exact() {
        for raw in [
            0i64,
            1,
            -1,
            4_294_967_296,
            -4_294_967_297,
            123_456_789_012_345,
            -(1 << 52) + 1,
            (1 << 53) - 1,
        ] {
            assert_eq!(quantize(to_f64(raw)), Some(raw));
        }
    }

    #[test]
    fn test_quantize_floors_toward_minus_infinity() {
        assert_eq!(quantize(0.5 / ONE), Some(0));
        assert_eq!(quantize(-0.5 / ONE), Some(-1));
        assert_eq!(quantize(-1.0), Some(-4_294_967_296));
        assert_eq!(quantize(1.75), Some(7_516_192_768));
    }

    #[test]
    fn test_quantize_rejects_out_of_range() {
        assert_eq!(quantize(2_147_483_648.0), None);
        assert_eq!(quantize(-2_147_483_648.0), Some(i64::MIN));
        assert_eq!(quantize(f64::NAN), None);
        assert_eq!(quantize(f64::INFINITY), None);
    }

    #[test]
    fn test_rescale_exact_floors() {
        assert_eq!(rescale_exact(-1, 32), Some(-1));
        assert_eq!(rescale_exact((3i128 << 32) + 5, 32), Some(3));
        assert_eq!(rescale_exact(-(3i128 << 32) - 5, 32), Some(-4));
        assert_eq!(rescale_exact(-(3i128 << 64) - 5, 64), Some(-4));
        assert_eq!(rescale_exact(i128::MAX, 32), None);
    }

    #[test]
    fn test_is_ambiguous_near_integers_only() {
        assert!(!is_ambiguous(3.0 / ONE));
        assert!(is_ambiguous((3.0 - 1e-6) / ONE));
        assert!(is_ambiguous((3.0 + 1e-6) / ONE));
        assert!(!is_ambiguous(3.5 / ONE));
    }
}
