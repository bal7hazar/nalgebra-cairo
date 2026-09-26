//! Operators, indexing, products and conversions of the dynamic matrices (DESIGN D5).
//!
//! Element-wise operators panic with `nalgebra: dimension mismatch` on different shapes (upstream:
//! "Matrix addition/subtraction dimensions mismatch."), products when the inner dimensions differ
//! (upstream: "Matrix multiplication dimensions mismatch."), and the indexing of a component out
//! of the shape with `nalgebra: index out of bounds` (upstream: "Matrix index out of bounds.").

use core::ops::IndexView;
use simba::scalar::Real;
use super::super::errors;
use super::super::matrix_index::MatrixIndex;
use super::super::matrix_mul::MatrixMul;
use super::dmatrix::DMatrix;
use super::dvector::{DVector, check_len};
use super::kernels::DynKernels;
use super::row_dvector::RowDVector;

/// Panics with `nalgebra: dimension mismatch` unless both matrices have the same shape.
fn check_shape<T>(a: @DMatrix<T>, b: @DMatrix<T>) {
    if a.nrows != b.nrows || a.ncols != b.ncols {
        core::panic_with_felt252(errors::DIMENSION_MISMATCH)
    }
}

/// The component at the linear (column-major) index `k`, `None` out of bounds.
fn get_linear<T, +Copy<T>>(data: Span<T>, k: usize) -> Option<T> {
    match data.get(k) {
        Some(x) => Some(*x.unbox()),
        None => None,
    }
}

/// The component at the linear index `k`; panics with `nalgebra: index out of bounds`.
fn at_linear<T, +Copy<T>>(data: Span<T>, k: usize) -> T {
    match data.get(k) {
        Some(x) => *x.unbox(),
        None => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
    }
}

// --- DMatrix --------------------------------------------------------------------------------------

/// `a + b` component-wise. Upstream: `Add<Matrix> for Matrix`.
pub impl DMatrixAdd<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
> of Add<DMatrix<T>> {
    fn add(lhs: DMatrix<T>, rhs: DMatrix<T>) -> DMatrix<T> {
        check_shape(@lhs, @rhs);
        DMatrix { data: DynKernels::add(lhs.data, rhs.data), nrows: lhs.nrows, ncols: lhs.ncols }
    }
}

/// `a - b` component-wise. Upstream: `Sub<Matrix> for Matrix`.
pub impl DMatrixSub<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
> of Sub<DMatrix<T>> {
    fn sub(lhs: DMatrix<T>, rhs: DMatrix<T>) -> DMatrix<T> {
        check_shape(@lhs, @rhs);
        DMatrix { data: DynKernels::sub(lhs.data, rhs.data), nrows: lhs.nrows, ncols: lhs.ncols }
    }
}

/// `-a`. Upstream: `Neg for Matrix`.
pub impl DMatrixNeg<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
> of Neg<DMatrix<T>> {
    fn neg(a: DMatrix<T>) -> DMatrix<T> {
        DMatrix { data: DynKernels::neg(a.data), nrows: a.nrows, ncols: a.ncols }
    }
}

/// `a * b`, every component ONE fused sum of products (one floor rounding). Panics with
/// `nalgebra: dimension mismatch` unless `a.ncols == b.nrows`. Upstream: `Mul<Matrix> for Matrix`.
pub impl DMatrixMul<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
> of Mul<DMatrix<T>> {
    fn mul(lhs: DMatrix<T>, rhs: DMatrix<T>) -> DMatrix<T> {
        if lhs.ncols != rhs.nrows {
            core::panic_with_felt252(errors::DIMENSION_MISMATCH)
        }
        DMatrix {
            data: DynKernels::mul(lhs.data, lhs.nrows, lhs.ncols, rhs.data, rhs.ncols),
            nrows: lhs.nrows,
            ncols: rhs.ncols,
        }
    }
}

/// `m[k]`: the component at the column-major index `k`. Upstream: `Index<usize>`.
pub impl DMatrixIndexLinear<T, +Copy<T>, +Drop<T>> of IndexView<DMatrix<T>, usize> {
    type Target = T;
    #[inline]
    fn index(self: @DMatrix<T>, index: usize) -> T {
        at_linear(*self.data, index)
    }
}

/// `m[(i, j)]`: the component at row `i`, column `j`. Upstream: `Index<(usize, usize)>`.
pub impl DMatrixIndexPair<T, +Copy<T>, +Drop<T>> of IndexView<DMatrix<T>, (usize, usize)> {
    type Target = T;
    #[inline]
    fn index(self: @DMatrix<T>, index: (usize, usize)) -> T {
        let (i, j) = index;
        if i >= *self.nrows || j >= *self.ncols {
            core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS)
        }
        at_linear(*self.data, i + j * *self.nrows)
    }
}

/// `m.get(k)` / `m.index(k)`: the component at the column-major index `k`. Upstream:
/// `Matrix::get` / `Matrix::index` with a `usize`.
pub impl DMatrixMatrixIndexLinear<T, +Copy<T>, +Drop<T>> of MatrixIndex<DMatrix<T>, usize> {
    type Output = T;
    #[inline]
    fn get(self: DMatrix<T>, index: usize) -> Option<T> {
        get_linear(self.data, index)
    }
    #[inline]
    fn index(self: DMatrix<T>, index: usize) -> T {
        at_linear(self.data, index)
    }
}

/// `m.get((i, j))` / `m.index((i, j))`. Upstream: `Matrix::get` / `Matrix::index` with a
/// `(usize, usize)`.
pub impl DMatrixMatrixIndexPair<
    T, +Copy<T>, +Drop<T>,
> of MatrixIndex<DMatrix<T>, (usize, usize)> {
    type Output = T;
    #[inline]
    fn get(self: DMatrix<T>, index: (usize, usize)) -> Option<T> {
        let (i, j) = index;
        if i >= self.nrows || j >= self.ncols {
            return None;
        }
        get_linear(self.data, i + j * self.nrows)
    }
    #[inline]
    fn index(self: DMatrix<T>, index: (usize, usize)) -> T {
        let (i, j) = index;
        if i >= self.nrows || j >= self.ncols {
            core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS)
        }
        at_linear(self.data, i + j * self.nrows)
    }
}

/// `a * b` for conformable dynamic matrices (`a * b`). Upstream: `Mul<Matrix> for Matrix`.
pub impl DMatrixMulDMatrix<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
> of MatrixMul<DMatrix<T>, DMatrix<T>> {
    type Output = DMatrix<T>;
    #[inline]
    fn mul_mat(self: DMatrix<T>, rhs: DMatrix<T>) -> DMatrix<T> {
        DMatrixMul::mul(self, rhs)
    }
}

/// `m * v`, a `DVector` of `m.nrows` fused sums of products. Panics with `nalgebra: dimension
/// mismatch` unless `v.len() == m.ncols`. Upstream: `Mul<Vector> for Matrix`.
pub impl DMatrixMulDVector<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
> of MatrixMul<DMatrix<T>, DVector<T>> {
    type Output = DVector<T>;
    #[inline]
    fn mul_mat(self: DMatrix<T>, rhs: DVector<T>) -> DVector<T> {
        if self.ncols != rhs.data.len() {
            core::panic_with_felt252(errors::DIMENSION_MISMATCH)
        }
        DVector { data: DynKernels::mul(self.data, self.nrows, self.ncols, rhs.data, 1) }
    }
}

/// `r * m`, a `RowDVector` of `m.ncols` fused sums of products. Panics with `nalgebra: dimension
/// mismatch` unless `r.len() == m.nrows`. Upstream: `Mul<Matrix> for RowVector`.
pub impl RowDVectorMulDMatrix<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
> of MatrixMul<RowDVector<T>, DMatrix<T>> {
    type Output = RowDVector<T>;
    #[inline]
    fn mul_mat(self: RowDVector<T>, rhs: DMatrix<T>) -> RowDVector<T> {
        if self.data.len() != rhs.nrows {
            core::panic_with_felt252(errors::DIMENSION_MISMATCH)
        }
        RowDVector { data: DynKernels::mul(self.data, 1, rhs.nrows, rhs.data, rhs.ncols) }
    }
}

/// `v * r`, the outer product (`v.len() x r.len()`, each component one floored product).
/// Upstream: `Mul<RowVector> for Vector`.
pub impl DVectorMulRowDVector<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
> of MatrixMul<DVector<T>, RowDVector<T>> {
    type Output = DMatrix<T>;
    #[inline]
    fn mul_mat(self: DVector<T>, rhs: RowDVector<T>) -> DMatrix<T> {
        let m = self.data.len();
        let n = rhs.data.len();
        DMatrix { data: DynKernels::mul(self.data, m, 1, rhs.data, n), nrows: m, ncols: n }
    }
}

// --- DVector / RowDVector -------------------------------------------------------------------------

/// `a + b` component-wise. Upstream: `Add<Matrix> for Matrix`.
pub impl DVectorAdd<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
> of Add<DVector<T>> {
    fn add(lhs: DVector<T>, rhs: DVector<T>) -> DVector<T> {
        check_len(lhs.data, rhs.data);
        DVector { data: DynKernels::add(lhs.data, rhs.data) }
    }
}

/// `a - b` component-wise. Upstream: `Sub<Matrix> for Matrix`.
pub impl DVectorSub<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
> of Sub<DVector<T>> {
    fn sub(lhs: DVector<T>, rhs: DVector<T>) -> DVector<T> {
        check_len(lhs.data, rhs.data);
        DVector { data: DynKernels::sub(lhs.data, rhs.data) }
    }
}

/// `-a`. Upstream: `Neg for Matrix`.
pub impl DVectorNeg<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
> of Neg<DVector<T>> {
    fn neg(a: DVector<T>) -> DVector<T> {
        DVector { data: DynKernels::neg(a.data) }
    }
}

/// `v[i]`. Upstream: `Index<usize>`.
pub impl DVectorIndexLinear<T, +Copy<T>, +Drop<T>> of IndexView<DVector<T>, usize> {
    type Target = T;
    #[inline]
    fn index(self: @DVector<T>, index: usize) -> T {
        at_linear(*self.data, index)
    }
}

/// `v.get(i)` / `v.index(i)`. Upstream: `Matrix::get` / `Matrix::index` with a `usize`.
pub impl DVectorMatrixIndexLinear<T, +Copy<T>, +Drop<T>> of MatrixIndex<DVector<T>, usize> {
    type Output = T;
    #[inline]
    fn get(self: DVector<T>, index: usize) -> Option<T> {
        get_linear(self.data, index)
    }
    #[inline]
    fn index(self: DVector<T>, index: usize) -> T {
        at_linear(self.data, index)
    }
}

/// `a + b` component-wise. Upstream: `Add<Matrix> for Matrix`.
pub impl RowDVectorAdd<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
> of Add<RowDVector<T>> {
    fn add(lhs: RowDVector<T>, rhs: RowDVector<T>) -> RowDVector<T> {
        check_len(lhs.data, rhs.data);
        RowDVector { data: DynKernels::add(lhs.data, rhs.data) }
    }
}

/// `a - b` component-wise. Upstream: `Sub<Matrix> for Matrix`.
pub impl RowDVectorSub<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
> of Sub<RowDVector<T>> {
    fn sub(lhs: RowDVector<T>, rhs: RowDVector<T>) -> RowDVector<T> {
        check_len(lhs.data, rhs.data);
        RowDVector { data: DynKernels::sub(lhs.data, rhs.data) }
    }
}

/// `-a`. Upstream: `Neg for Matrix`.
pub impl RowDVectorNeg<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
> of Neg<RowDVector<T>> {
    fn neg(a: RowDVector<T>) -> RowDVector<T> {
        RowDVector { data: DynKernels::neg(a.data) }
    }
}

/// `r[j]`. Upstream: `Index<usize>`.
pub impl RowDVectorIndexLinear<T, +Copy<T>, +Drop<T>> of IndexView<RowDVector<T>, usize> {
    type Target = T;
    #[inline]
    fn index(self: @RowDVector<T>, index: usize) -> T {
        at_linear(*self.data, index)
    }
}

/// `r.get(j)` / `r.index(j)`. Upstream: `Matrix::get` / `Matrix::index` with a `usize`.
pub impl RowDVectorMatrixIndexLinear<
    T, +Copy<T>, +Drop<T>,
> of MatrixIndex<RowDVector<T>, usize> {
    type Output = T;
    #[inline]
    fn get(self: RowDVector<T>, index: usize) -> Option<T> {
        get_linear(self.data, index)
    }
    #[inline]
    fn index(self: RowDVector<T>, index: usize) -> T {
        at_linear(self.data, index)
    }
}

// --- conversions ----------------------------------------------------------------------------------

/// A `DVector` as the `len x 1` `DMatrix` (no copy). Upstream: `From<Matrix>` (a change of
/// storage type, `OVector<T, Dyn>` to `OMatrix<T, Dyn, Dyn>`).
pub impl DVectorIntoDMatrix<T> of Into<DVector<T>, DMatrix<T>> {
    #[inline(always)]
    fn into(self: DVector<T>) -> DMatrix<T> {
        DMatrix { data: self.data, nrows: self.data.len(), ncols: 1 }
    }
}

/// A `RowDVector` as the `1 x len` `DMatrix` (no copy). Upstream: `From<Matrix>`.
pub impl RowDVectorIntoDMatrix<T> of Into<RowDVector<T>, DMatrix<T>> {
    #[inline(always)]
    fn into(self: RowDVector<T>) -> DMatrix<T> {
        DMatrix { data: self.data, nrows: 1, ncols: self.data.len() }
    }
}

/// The `DVector` of the values of an `Array` (no copy). Upstream: `From<Vec<T>> for DVector<T>`.
pub impl ArrayIntoDVector<T, +Drop<T>> of Into<Array<T>, DVector<T>> {
    #[inline(always)]
    fn into(self: Array<T>) -> DVector<T> {
        DVector { data: self.span() }
    }
}

/// The `RowDVector` of the values of an `Array` (no copy). Upstream: `From<Vec<T>> for
/// RowDVector<T>`.
pub impl ArrayIntoRowDVector<T, +Drop<T>> of Into<Array<T>, RowDVector<T>> {
    #[inline(always)]
    fn into(self: Array<T>) -> RowDVector<T> {
        RowDVector { data: self.span() }
    }
}
