//! Internal, no stability promise: the crate-private items of `base::unit` that the packages
//! above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use simba::scalar::Real;
use crate::base::unit::{Normed, Unit, UnitTrait};

/// Crate-internal by-value forms of the in-place `renormalize` / `renormalize_fast` (WP 8.0: the
/// public methods are upstream's `&mut self` ones), for the tests and the value-style call sites.
#[generate_trait]
pub impl UnitInternalImpl<
    V,
    T,
    impl N: Normed<V, T>,
    impl R: Real<T>,
    +Add<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialOrd<T>,
    +Copy<V>,
    +Drop<V>,
    +Copy<T>,
    +Drop<T>,
> of UnitInternalTrait<V, T> {
    /// `self` renormalized exactly (`UnitTrait::renormalize`), by value.
    #[inline(always)]
    fn renormalized(self: Unit<V>) -> Unit<V> {
        let mut u = self;
        let _ = UnitTrait::renormalize(ref u);
        u
    }

    /// `self` renormalized by one Newton step (`UnitTrait::renormalize_fast`), by value.
    #[inline(always)]
    fn renormalized_fast(self: Unit<V>) -> Unit<V> {
        let mut u = self;
        UnitTrait::renormalize_fast(ref u);
        u
    }
}
