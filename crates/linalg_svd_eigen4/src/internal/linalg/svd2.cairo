//! Internal, no stability promise: the crate-private items of `linalg::svd2` that the packages
//! above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use nalgebra_core::base::matrix2::Matrix2;
use nalgebra_core::base::matrix_mul::MatrixMul;
use nalgebra_core::base::matrix_tr_mul::MatrixTrMul;
use nalgebra_core::base::vector2::Vector2;
use nalgebra_core::internal::base::sym_matrix2::{SymMatrix2, SymMatrix2Trait};
use nalgebra_static3::base::matrix2::Matrix2Trait;
use nalgebra_static3::internal::base::matrix2::Matrix2InternalTrait;
use simba::scalar::Real;
use crate::internal::linalg::symmetric_eigen2::SymmetricEigen2InternalTrait;

/// Crate-internal kernels of `Svd2<T>` (WP 8.0: the public API is strictly upstream's): the Gram
/// matrix `MᵀM` as a `SymMatrix2` (the input of the eigen decomposition).
#[generate_trait]
pub impl Svd2InternalImpl<
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
> of Svd2InternalTrait<T> {
    /// The left singular vectors of `new` from the sorted `w_i = M v_i` and `σ_1`: `u_1 = w_1 /
    /// σ_1` (or `e_1` when `σ_1 = 0`), `u_2 = ± perp(u_1)` (see `new`). Skipped (`compute_u =
    /// false`) they cost no Cairo step; the Sierra gas of the snapshots charges a branch at its
    /// costliest path whatever is executed (docs/BENCHMARK.md).
    #[inline(always)]
    fn left(w1: Vector2<T>, w2: Vector2<T>, s1: T) -> Matrix2<T> {
        let u1 = if s1 == R::zero() {
            Vector2 { x: R::one(), y: R::zero() }
        } else {
            Vector2 { x: R::div(w1.x, s1), y: R::div(w1.y, s1) }
        };
        // `<perp(u1), w2>` with perp(u1) = (-u1.y, u1.x): one rounding, and its sign orients
        // u2.
        let along = R::diff_prod(u1.x, w2.y, u1.y, w2.x);
        let u2 = if along.is_sign_negative() {
            Vector2 { x: u1.y, y: -u1.x }
        } else {
            Vector2 { x: -u1.y, y: u1.x }
        };
        Matrix2Trait::from_columns(u1, u2)
    }

    /// `MᵀM` as a symmetric matrix: 3 fused kernels instead of 8 products, bit-identical to the
    /// upper triangle of `m.transpose() * m` and to `m.transpose().mul_transpose()` — which
    /// would reuse `base` instead of repeating the kernel, and costs the moves of the transpose
    /// (measured at 2 760 gas in 3x3, `bench_svd3_gram__*`). Panics on overflow.
    /// Upstream: `m.tr_mul(&m)`.
    #[inline(always)]
    fn gram(m: Matrix2<T>) -> SymMatrix2<T> {
        SymMatrix2 {
            m11: R::norm_squared2(m.m11, m.m21),
            m12: R::sum_prod2(m.m11, m.m12, m.m21, m.m22),
            m22: R::norm_squared2(m.m12, m.m22),
        }
    }
}
