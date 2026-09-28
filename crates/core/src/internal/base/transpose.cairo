//! Internal, no stability promise: the crate-private items of `base::transpose` that the packages
//! above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

/// The transpose of a shape (the operand of the `_tr` forms of `blas`, and of the `tr_` / `ad_`
/// forms of `solve`): struct moves only, free once inlined.
pub trait BlasTranspose<M> {
    type Output;
    fn tr(self: M) -> Self::Output;
}
