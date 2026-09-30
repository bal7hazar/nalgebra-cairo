//! In `nalgebra_types3`: the `struct` `Rotation3`; its 5 impls, among them `Rotation3Mul`,
//! `Rotation3Div`, `Rotation3Default` and 2 more; the sub-module `errors`. This module is split
//! over packages; the other parts are in `nalgebra_geometry3`.
//!
//! `Rotation3`: a 3D rotation stored as a 3x3 orthonormal matrix (upstream `nalgebra::Rotation3`).
//!
//! - `Rotation3Trait` / `Rotation3Impl`: constructors, transforms, composition, conversions to and
//!   from `UnitQuaternion`, renormalisation — everything that needs no trigonometry;
//! - `Rotation3AngleTrait` / `Rotation3AngleImpl`: axis-angle and Euler-angle constructors and
//!   extractors, which additionally need `simba::scalar::Transcendental`;
//! - the operator `*` (composition) lives in this module, where the compiler finds it without any
//!   import.
//!
//! The matrix form is the right representation when several vectors are transformed with the same
//! rotation: `transform_vector` is one `Matrix3 * Vector3` (9 products, 6 650 gas) against 15
//! products (23 230) for `UnitQuaternion::transform_vector`, and building the matrix from a
//! quaternion costs 23 530 — so the break-even is TWO vectors per step (measured,
//! `bench_unit_quaternion_transform_vector_x2__*`). Composition is the other way round: 23 310 for
//! a matrix product against 11 860 for a Hamilton product, and a matrix composed repeatedly drifts
//! out of orthonormality, which `renormalize` (upstream's closed-form polar factor, about 530 000
//! gas) has to fix.
//!
//! The invariant (orthonormal, determinant +1) is a CONTRACT, like upstream's
//! `from_matrix_unchecked`: nothing checks it, and `inverse` is implemented as the transpose.
//!
//! Numeric contract (AGENTS.md): every sum of products goes through a fused `Real` kernel (one
//! floor rounding and one overflow check per output scalar); nothing wraps silently.

use core::num::traits::One;
use core::ops::Index;
use nalgebra_core::internal::base::transpose::BlasTranspose;
use simba::scalar::Real;
use crate::base::matrix3::Matrix3;

/// A 3D rotation as an orthonormal 3x3 matrix of determinant +1.
#[derive(Copy, Drop, PartialEq, Serde, Debug, Hash)]
pub struct Rotation3<T> {
    pub matrix: Matrix3<T>,
}

/// The composition of two rotations: the product of the matrices, `lhs` applied last (27 products,
/// each entry one fused `sum_prod3`). Repeated composition drifts out of orthonormality:
/// `renormalize` restores it. Upstream: `Mul`.
pub impl Rotation3Mul<
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
> of Mul<Rotation3<T>> {
    #[inline(always)]
    fn mul(lhs: Rotation3<T>, rhs: Rotation3<T>) -> Rotation3<T> {
        Rotation3 { matrix: lhs.matrix * rhs.matrix }
    }
}

/// `a / b = a * b⁻¹`: the product with the transpose (27 products, one rounding per entry).
/// Upstream: `Div<Rotation3>`.
pub impl Rotation3Div<
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
> of Div<Rotation3<T>> {
    #[inline(always)]
    fn div(lhs: Rotation3<T>, rhs: Rotation3<T>) -> Rotation3<T> {
        Rotation3 { matrix: lhs.matrix * BlasTranspose::tr(rhs.matrix) }
    }
}

/// `Default::default()`: the identity rotation. Upstream: `Default for Rotation3`.
pub impl Rotation3Default<T, impl R: Real<T>, +Copy<T>, +Drop<T>> of Default<Rotation3<T>> {
    #[inline(always)]
    fn default() -> Rotation3<T> {
        let (o, l) = (R::zero(), R::one());
        Rotation3 {
            matrix: Matrix3 {
                m11: l, m21: o, m31: o, m12: o, m22: l, m32: o, m13: o, m23: o, m33: l,
            },
        }
    }
}

/// `One::one()`: the identity rotation; `is_one` compares with the identity matrix exactly.
/// Upstream: `num::One for Rotation3`.
pub impl Rotation3One<T, impl R: Real<T>, +PartialEq<T>, +Copy<T>, +Drop<T>> of One<Rotation3<T>> {
    #[inline(always)]
    fn one() -> Rotation3<T> {
        Rotation3Default::<T>::default()
    }

    #[inline(always)]
    fn is_one(self: @Rotation3<T>) -> bool {
        *self == Rotation3Default::<T>::default()
    }

    #[inline(always)]
    fn is_non_one(self: @Rotation3<T>) -> bool {
        !Self::is_one(self)
    }
}

/// `r[(i, j)]`: the entry of row `i` and column `j` of the matrix. Panics with
/// `nalgebra: index out of bounds` for `i > 2` or `j > 2`. Upstream: `Index<(usize, usize)> for
/// Rotation`.
pub impl Rotation3Index<T, +Copy<T>, +Drop<T>> of Index<Rotation3<T>, (usize, usize)> {
    type Target = T;

    fn index(ref self: Rotation3<T>, index: (usize, usize)) -> T {
        let m = self.matrix;
        let (i, j) = index;
        let row = match i {
            0 => (m.m11, m.m12, m.m13),
            1 => (m.m21, m.m22, m.m23),
            2 => (m.m31, m.m32, m.m33),
            _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
        };
        let (a, b, c) = row;
        match j {
            0 => a,
            1 => b,
            2 => c,
            _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
        }
    }
}

/// Panic messages of `Rotation3` (stable API).
pub mod errors {
    /// `r[(i, j)]` with `i > 2` or `j > 2`.
    pub const INDEX_OUT_OF_BOUNDS: felt252 = 'nalgebra: index out of bounds';
    /// `euler_angles_ordered` with a first axis not orthogonal to the other two (upstream's
    /// `assert_relative_eq!(.., epsilon = 1e-6)`).
    pub const AXES_NOT_ORTHOGONAL: felt252 = 'nalgebra: axes not orthogonal';
}
