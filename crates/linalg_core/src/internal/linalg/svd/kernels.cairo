//! Internal, no stability promise: the crate-private items of `linalg::svd::kernels` that the
//! packages above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use simba::scalar::Real;

/// The singular-value filters shared by the SVDs of every static shape (crate-internal).
#[generate_trait]
pub impl SvdRightImpl<
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
> of SvdRightTrait<T> {
    /// `1 / s` when `s > eps`, `0` otherwise: upstream's `pseudo_inverse` filter.
    #[inline(always)]
    fn inverted(s: T, eps: T) -> T {
        if s > eps {
            s.recip()
        } else {
            R::zero()
        }
    }
    /// `y / s` when `s > eps`, `0` otherwise: upstream's `solve` filter.
    #[inline(always)]
    fn divided(y: T, s: T, eps: T) -> T {
        if s > eps {
            R::div(y, s)
        } else {
            R::zero()
        }
    }
}
