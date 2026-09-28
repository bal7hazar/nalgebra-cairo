//! Internal, no stability promise: the crate-private items of `geometry::rotation3` that the
//! packages above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use simba::scalar::Real;
use crate::geometry::rotation3::{Rotation3, Rotation3Trait};

/// Crate-internal by-value forms of the in-place `renormalize` (WP 8.0: the
/// public methods are upstream's `&mut self` ones), for the tests and the value-style call sites.
#[generate_trait]
pub impl Rotation3InternalImpl<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of Rotation3InternalTrait<T> {
    /// `self` renormalized exactly (`Rotation3Trait::renormalize`), by value.
    #[inline(always)]
    fn renormalized(self: Rotation3<T>) -> Rotation3<T> {
        let mut r = self;
        Rotation3Trait::renormalize(ref r);
        r
    }
}
