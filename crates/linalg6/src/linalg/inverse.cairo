//! In-place inversion of the static square matrices (upstream `SquareMatrix::try_inverse_mut`,
//! `src/linalg/inverse.rs`), on the sizes that have an inverse: the closed forms of
//! `Matrix2/3/4::try_inverse` and the LU inverse of `Matrix6` (`Matrix6LuTrait::try_inverse`).

use nalgebra_shapes6::base::matrix6::Matrix6;
use nalgebra_static6_wide::linalg::lu::lu6::Matrix6LuTrait;
use simba::scalar::Real;

/// `SquareMatrix::try_inverse_mut` on `Matrix6<T>`.
#[generate_trait]
pub impl Matrix6InverseImpl<
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
> of Matrix6InverseTrait<T> {
    /// Overwrites `self` with its inverse and returns `true`, or returns `false` and leaves
    /// `self` unchanged when it is not invertible: `Matrix6LuTrait::try_inverse` (bit-identical,
    /// the same criterion). Upstream: `SquareMatrix::try_inverse_mut` (which leaves `self`
    /// unchanged on failure for the closed-form sizes and partially overwritten otherwise).
    #[inline(always)]
    fn try_inverse_mut(ref self: Matrix6<T>) -> bool {
        match Matrix6LuTrait::try_inverse(self) {
            Option::Some(m) => {
                self = m;
                true
            },
            Option::None => false,
        }
    }
}
