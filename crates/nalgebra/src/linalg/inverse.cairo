//! In-place inversion of the static square matrices (upstream `SquareMatrix::try_inverse_mut`,
//! `src/linalg/inverse.rs`), on the sizes that have an inverse: the closed forms of
//! `Matrix2/3/4::try_inverse` and the LU inverse of `Matrix6` (`Matrix6LuTrait::try_inverse`).

use simba::scalar::Real;
use crate::base::matrix2::{Matrix2, Matrix2Trait};
use crate::base::matrix3::{Matrix3, Matrix3Trait};
use crate::base::matrix4::{Matrix4, Matrix4Trait};
use crate::base::matrix6::Matrix6;
use crate::linalg::lu::Matrix6LuTrait;

/// `SquareMatrix::try_inverse_mut` on `Matrix2<T>`.
#[generate_trait]
pub impl Matrix2InverseImpl<
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
> of Matrix2InverseTrait<T> {
    /// Overwrites `self` with its inverse and returns `true`, or returns `false` and leaves
    /// `self` unchanged when it is not invertible: `Matrix2Trait::try_inverse` (bit-identical, the
    /// same criterion). Upstream: `SquareMatrix::try_inverse_mut` (which leaves `self` unchanged
    /// on failure for the closed-form sizes and partially overwritten otherwise).
    #[inline(always)]
    fn try_inverse_mut(ref self: Matrix2<T>) -> bool {
        match Matrix2Trait::try_inverse(self) {
            Option::Some(m) => {
                self = m;
                true
            },
            Option::None => false,
        }
    }
}

/// `SquareMatrix::try_inverse_mut` on `Matrix3<T>`.
#[generate_trait]
pub impl Matrix3InverseImpl<
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
> of Matrix3InverseTrait<T> {
    /// Overwrites `self` with its inverse and returns `true`, or returns `false` and leaves
    /// `self` unchanged when it is not invertible: `Matrix3Trait::try_inverse` (bit-identical, the
    /// same criterion). Upstream: `SquareMatrix::try_inverse_mut` (which leaves `self` unchanged
    /// on failure for the closed-form sizes and partially overwritten otherwise).
    #[inline(always)]
    fn try_inverse_mut(ref self: Matrix3<T>) -> bool {
        match Matrix3Trait::try_inverse(self) {
            Option::Some(m) => {
                self = m;
                true
            },
            Option::None => false,
        }
    }
}

/// `SquareMatrix::try_inverse_mut` on `Matrix4<T>`.
#[generate_trait]
pub impl Matrix4InverseImpl<
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
> of Matrix4InverseTrait<T> {
    /// Overwrites `self` with its inverse and returns `true`, or returns `false` and leaves
    /// `self` unchanged when it is not invertible: `Matrix4Trait::try_inverse` (bit-identical, the
    /// same criterion). Upstream: `SquareMatrix::try_inverse_mut` (which leaves `self` unchanged
    /// on failure for the closed-form sizes and partially overwritten otherwise).
    #[inline(always)]
    fn try_inverse_mut(ref self: Matrix4<T>) -> bool {
        match Matrix4Trait::try_inverse(self) {
            Option::Some(m) => {
                self = m;
                true
            },
            Option::None => false,
        }
    }
}

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
