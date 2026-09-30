//! Internal, no stability promise: the crate-private items of `linalg::givens` that the packages
//! above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use crate::linalg::givens::GivensRotation;

/// The field reads of the `GivensRotate` / `GivensRotateRows` impls, without the bounds of
/// `GivensRotationTrait::c` / `s` (crate-private: the impls of the shapes of dimension 5 and 6 live
/// in the packages above `nalgebra_core`, out of reach of the private fields, docs/SPLIT.md
/// §12.3).
/// `#[inline(always)]`: the same Sierra as `self.c` / `self.s` written in place.
#[generate_trait]
pub impl GivensRotationInternalImpl<T, +Copy<T>, +Drop<T>> of GivensRotationInternalTrait<T> {
    #[inline(always)]
    fn c(self: GivensRotation<T>) -> T {
        self.c
    }

    #[inline(always)]
    fn s(self: GivensRotation<T>) -> T {
        self.s
    }
}
