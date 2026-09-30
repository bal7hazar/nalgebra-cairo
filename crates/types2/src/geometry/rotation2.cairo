//! In `nalgebra_types2`: the `struct` `Rotation2`; its 5 impls, among them `Rotation2Mul`,
//! `Rotation2Div`, `Rotation2Default` and 2 more; the sub-module `errors`. This module is split
//! over packages; the other parts are in `nalgebra_geometry2`.
//!
//! `Rotation2`: a 2D rotation stored as a 2x2 orthogonal matrix (upstream
//! `nalgebra::Rotation2`, which is `Rotation<T, 2>`).
//!
//! - `Rotation2Trait` / `Rotation2Impl`: construction, accessors, composition, transforms,
//!   conversions and renormalization — everything that is algebraic, hence available for any
//!   `simba::scalar::Real` scalar;
//! - `Rotation2AngleTrait` / `Rotation2AngleImpl`: the operations that go through an angle
//!   (`new`, `angle`, `powf`, `scaled_rotation_between`), which additionally need
//!   `simba::scalar::Transcendental`;
//! - `a * b` (composition of two rotations) and the conversions from / to `UnitComplex<T>`: their
//!   impls live in this module, where the compiler finds them without any import.
//!
//! `Rotation2` and `UnitComplex` hold the same information: the matrix is
//! `[[cos θ, -sin θ], [sin θ, cos θ]]`, whose first column IS the complex `(re, im)`. Measured
//! (Sierra gas, net of the group baseline), the matrix form costs **2.6x more per composition**
//! (10 260 against 4 000) and twice the storage, for transforms that are bit-identical and cost
//! exactly the same (4 000 either way, `bench_rotation2_transform_point__alt_unit_complex`). It
//! exists because upstream has it, because it composes with the `Matrix2` world without a
//! conversion, and because a `Rotation2` is what `to_homogeneous` and the matrix decompositions
//! consume. **Prefer `UnitComplex` in hot code**; converting costs 800 gas one way and 2 400 the
//! other, so even a single composition pays for the round trip.
//!
//! Numeric contract (AGENTS.md): every sum of products goes through a fused `Real` kernel (one
//! floor rounding and one overflow check per output scalar); nothing wraps silently.

use core::num::traits::One;
use core::ops::Index;
use simba::scalar::Real;
use crate::base::matrix2::Matrix2;

/// A 2D rotation of angle `θ`, stored as the orthogonal matrix
/// `[[cos θ, -sin θ], [sin θ, cos θ]]` (`matrix.m11 = cos θ`, `matrix.m21 = sin θ`).
///
/// Nothing enforces the invariant `Rᵀ R = I`: build with `new` / `from_matrix` /
/// `rotation_between` / a `UnitComplex` conversion, or with `from_matrix_unchecked` when the
/// matrix is known to be a rotation.
#[derive(Copy, Drop, PartialEq, Serde, Debug, Hash)]
pub struct Rotation2<T> {
    pub matrix: Matrix2<T>,
}

/// `a * b`: the composition of two rotations (turn by `b`, then by `a`), the product of their
/// matrices. FOUR fused kernels, each output component floored once, bit for bit what upstream's
/// matrix product gives.
///
/// The two extra products buy bit-exactness: composing through the complex form (two kernels,
/// 4 260 gas against 9 950, `bench_rotation2_mul__alt_complex`) gives `m12` as `-floor(sin)`
/// where the matrix product gives `floor(-sin)`, which differ by 1 ulp whenever the exact value
/// is not an integer — and the oracle (`rotation2_mul`, tolerance 0) expects upstream's floor.
/// Composing in `UnitComplex` form costs 4 000 gas and is exact there, which is what hot code
/// should do. Upstream: `Mul`.
pub impl Rotation2Mul<T, impl R: Real<T>, +Copy<T>, +Drop<T>> of Mul<Rotation2<T>> {
    fn mul(lhs: Rotation2<T>, rhs: Rotation2<T>) -> Rotation2<T> {
        let (a, b) = (lhs.matrix, rhs.matrix);
        Rotation2 {
            matrix: Matrix2 {
                m11: R::sum_prod2(a.m11, b.m11, a.m12, b.m21),
                m21: R::sum_prod2(a.m21, b.m11, a.m22, b.m21),
                m12: R::sum_prod2(a.m11, b.m12, a.m12, b.m22),
                m22: R::sum_prod2(a.m21, b.m12, a.m22, b.m22),
            },
        }
    }
}

/// `a / b = a * b⁻¹`: the product with the transpose (four fused kernels, one rounding per
/// entry). Upstream: `Div<Rotation2>`.
pub impl Rotation2Div<T, impl R: Real<T>, +Copy<T>, +Drop<T>> of Div<Rotation2<T>> {
    fn div(lhs: Rotation2<T>, rhs: Rotation2<T>) -> Rotation2<T> {
        let (a, b) = (lhs.matrix, rhs.matrix);
        Rotation2 {
            matrix: Matrix2 {
                m11: R::sum_prod2(a.m11, b.m11, a.m12, b.m12),
                m21: R::sum_prod2(a.m21, b.m11, a.m22, b.m12),
                m12: R::sum_prod2(a.m11, b.m21, a.m12, b.m22),
                m22: R::sum_prod2(a.m21, b.m21, a.m22, b.m22),
            },
        }
    }
}

/// `Default::default()`: the identity rotation. Upstream: `Default for Rotation2`.
pub impl Rotation2Default<T, impl R: Real<T>, +Drop<T>> of Default<Rotation2<T>> {
    #[inline(always)]
    fn default() -> Rotation2<T> {
        Rotation2 {
            matrix: Matrix2 { m11: R::one(), m21: R::zero(), m12: R::zero(), m22: R::one() },
        }
    }
}

/// `One::one()`: the identity rotation; `is_one` compares with the identity matrix exactly.
/// Upstream: `num::One for Rotation2`.
pub impl Rotation2One<T, impl R: Real<T>, +PartialEq<T>, +Copy<T>, +Drop<T>> of One<Rotation2<T>> {
    #[inline(always)]
    fn one() -> Rotation2<T> {
        Rotation2Default::<T>::default()
    }

    #[inline(always)]
    fn is_one(self: @Rotation2<T>) -> bool {
        *self == Rotation2Default::<T>::default()
    }

    #[inline(always)]
    fn is_non_one(self: @Rotation2<T>) -> bool {
        !Self::is_one(self)
    }
}

/// `r[(i, j)]`: the entry of row `i` and column `j` of the matrix. Panics with
/// `nalgebra: index out of bounds` for `i > 1` or `j > 1`. Upstream: `Index<(usize, usize)> for
/// Rotation`.
pub impl Rotation2Index<T, +Copy<T>, +Drop<T>> of Index<Rotation2<T>, (usize, usize)> {
    type Target = T;

    fn index(ref self: Rotation2<T>, index: (usize, usize)) -> T {
        let m = self.matrix;
        match index {
            (0, 0) => m.m11,
            (0, 1) => m.m12,
            (1, 0) => m.m21,
            (1, 1) => m.m22,
            _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
        }
    }
}

/// Panic messages of `Rotation2` (stable API).
pub mod errors {
    /// `r[(i, j)]` with `i > 1` or `j > 1`.
    pub const INDEX_OUT_OF_BOUNDS: felt252 = 'nalgebra: index out of bounds';
    /// `renormalize` of a matrix whose first column is zero (upstream's result is `NaN`).
    pub const ZERO_COLUMN: felt252 = 'nalgebra: zero column (NaN)';
}
