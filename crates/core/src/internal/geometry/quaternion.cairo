//! In `nalgebra_core`: the impl `ApproxEqImpl`. This module is split over packages; the other parts
//! are in `nalgebra_geometry3`.
//!
//! Internal, no stability promise: the crate-private items of `geometry::quaternion` that the
//! packages above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use simba::scalar::Real;

/// Crate-internal scalar forms of upstream's `approx::RelativeEq` / `approx::UlpsEq`, shared by
/// the `relative_eq` / `ulps_eq` of `Quaternion`, `UnitQuaternion` and `UnitComplex` (tolerances
/// in ulp, DESIGN D3). See `QuaternionTrait::relative_eq` / `ulps_eq` for the semantics.
#[generate_trait]
pub impl ApproxEqImpl<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Sub<T>, +Mul<T>, +PartialEq<T>,
> of ApproxEqTrait<T> {
    /// `|a - b| <= epsilon` ulp, or same signs and `|a - b| <= max(|a|, |b|) · max_relative`.
    #[inline(always)]
    fn relative_eq(a: T, b: T, epsilon: u64, max_relative: T) -> bool {
        if R::abs_diff_eq(a, b, epsilon) {
            return true;
        }
        if R::is_sign_negative(a) != R::is_sign_negative(b) {
            return false;
        }
        // `|a - b| <= limit`, with `Real` comparisons only (no `PartialOrd` needed by callers).
        let limit = R::max(R::abs(a), R::abs(b)) * max_relative;
        R::max(R::abs(a - b), limit) == limit
    }

    /// `|a - b| <= epsilon` ulp, or same signs and `|a - b| <= max_ulps` ulp.
    #[inline(always)]
    fn ulps_eq(a: T, b: T, epsilon: u64, max_ulps: u32) -> bool {
        if R::abs_diff_eq(a, b, epsilon) {
            return true;
        }
        R::is_sign_negative(a) == R::is_sign_negative(b) && R::abs_diff_eq(a, b, max_ulps.into())
    }
}
