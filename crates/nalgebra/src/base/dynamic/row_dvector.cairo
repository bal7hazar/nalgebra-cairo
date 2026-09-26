//! `RowDVector<T>`: a row vector whose length is a runtime value (upstream `RowDVector`,
//! `RowOVector<T, Dyn>`, alias `Matrix1xX`), DESIGN D5.

use simba::scalar::Real;
use super::super::errors;
use super::dmatrix::DMatrix;
use super::kernels::DynKernels;
use super::dvector::{DVector, check_len};

/// A dynamically sized row vector (upstream `RowDVector<T>`): its components in one `Span<T>`.
/// `Copy` (upstream `Clone`); `Serde` writes the length-prefixed components.
#[derive(Copy, Drop, PartialEq, Serde, Debug)]
pub struct RowDVector<T> {
    pub(crate) data: Span<T>,
}

/// Upstream alias `Matrix1xX`: the same type as `RowDVector` (one Cairo struct per shape, like
/// `Matrix1x3` / `RowVector3`).
pub type Matrix1xX<T> = RowDVector<T>;

/// Methods of `RowDVector<T>` for any `Real` scalar.
#[generate_trait]
pub impl RowDVectorImpl<
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
> of RowDVectorTrait<T> {
    // --- construction ----------------------------------------------------------------------

    /// The zero row vector of length `ncols`. Upstream: `RowDVector::zeros(ncols)`.
    #[inline]
    fn zeros(ncols: usize) -> RowDVector<T> {
        RowDVector { data: DynKernels::filled(ncols, R::zero()).span() }
    }

    /// The row vector of length `ncols` whose components all equal `elem`. Upstream:
    /// `RowDVector::from_element(ncols, elem)`.
    #[inline]
    fn from_element(ncols: usize, elem: T) -> RowDVector<T> {
        RowDVector { data: DynKernels::filled(ncols, elem).span() }
    }

    /// Alias of `from_element`. Upstream: `RowDVector::repeat(ncols, elem)`.
    #[inline]
    fn repeat(ncols: usize, elem: T) -> RowDVector<T> {
        Self::from_element(ncols, elem)
    }

    /// The row vector `(1, 0, .., 0)` of length `ncols` (the `1 x ncols` identity). Upstream:
    /// `RowDVector::identity(ncols)`.
    #[inline]
    fn identity(ncols: usize) -> RowDVector<T> {
        Self::from_diagonal_element(ncols, R::one())
    }

    /// The row vector `(elt, 0, .., 0)` of length `ncols` (empty when `ncols == 0`). Upstream:
    /// `RowDVector::from_diagonal_element(ncols, elt)`.
    fn from_diagonal_element(ncols: usize, elt: T) -> RowDVector<T> {
        if ncols == 0 {
            return RowDVector { data: array![].span() };
        }
        let mut out: Array<T> = array![elt];
        DynKernels::append_n(ref out, ncols - 1, R::zero());
        RowDVector { data: out.span() }
    }

    /// The row vector of length `ncols` whose first component is `elts[0]` (when given), zeros
    /// elsewhere. Panics with `nalgebra: diagonal too long` when `elts` has more values than the
    /// `1 x ncols` diagonal. Upstream: `RowDVector::from_partial_diagonal(ncols, &[T])`.
    #[inline]
    fn from_partial_diagonal(ncols: usize, elts: Span<T>) -> RowDVector<T> {
        RowDVector { data: DynKernels::partial_diagonal(1, ncols, elts) }
    }

    /// The row vector of the values of `data`, without copy. Upstream:
    /// `RowDVector::from_vec(Vec<T>)`.
    #[inline(always)]
    fn from_vec(data: Array<T>) -> RowDVector<T> {
        RowDVector { data: data.span() }
    }

    /// The row vector of the values of `data`, without copy. Upstream:
    /// `RowDVector::from_column_slice(&[T])`.
    #[inline(always)]
    fn from_column_slice(data: Span<T>) -> RowDVector<T> {
        RowDVector { data }
    }

    /// The row vector of the values of `data` (a row: both orders agree).
    /// Upstream: `RowDVector::from_row_slice(&[T])`.
    #[inline(always)]
    fn from_row_slice(data: Span<T>) -> RowDVector<T> {
        RowDVector { data }
    }

    /// The row vector of length `ncols` of the values of `data`. Panics with `nalgebra: wrong
    /// slice length` unless `data.len() == ncols`. Upstream: `RowDVector::from_iterator(ncols,
    /// iter)` (a `Span` here).
    #[inline]
    fn from_iterator(ncols: usize, data: Span<T>) -> RowDVector<T> {
        if data.len() != ncols {
            core::panic_with_felt252(errors::SLICE_LENGTH)
        }
        RowDVector { data }
    }

    /// `from_iterator` (the orders agree on a row). Upstream:
    /// `RowDVector::from_row_iterator(ncols, iter)`.
    #[inline]
    fn from_row_iterator(ncols: usize, data: Span<T>) -> RowDVector<T> {
        Self::from_iterator(ncols, data)
    }

    /// The row vector of length `ncols` whose component `j` is `f(0, j)` (upstream's `(row,
    /// column)` closure). `f`'s output converts `Into<T>`. Upstream: `RowDVector::from_fn(ncols,
    /// f)`.
    #[cfg(feature: 'closures')]
    fn from_fn<
        F,
        +Drop<F>,
        impl Func: core::ops::Fn<F, (usize, usize)>,
        +Into<Func::Output, T>,
        +Drop<Func::Output>,
    >(
        ncols: usize, f: F,
    ) -> RowDVector<T> {
        let mut out: Array<T> = array![];
        let mut j: usize = 0;
        while j != ncols {
            out.append(f(0, j).into());
            j += 1;
        }
        RowDVector { data: out.span() }
    }

    // --- properties ----------------------------------------------------------------------------

    /// `1`. Upstream: `nrows`.
    #[inline(always)]
    fn nrows(self: RowDVector<T>) -> usize {
        1
    }

    /// The number of columns (the length). Upstream: `ncols`.
    #[inline(always)]
    fn ncols(self: RowDVector<T>) -> usize {
        self.data.len()
    }

    /// `(1, len)`. Upstream: `shape`.
    #[inline(always)]
    fn shape(self: RowDVector<T>) -> (usize, usize) {
        (1, self.data.len())
    }

    /// The number of components. Upstream: `len`.
    #[inline(always)]
    fn len(self: RowDVector<T>) -> usize {
        self.data.len()
    }

    /// Whether the vector has no component. Upstream: `is_empty`.
    #[inline(always)]
    fn is_empty(self: RowDVector<T>) -> bool {
        self.data.is_empty()
    }

    /// The components. Upstream: `as_slice` (a `Span` here).
    #[inline(always)]
    fn as_slice(self: RowDVector<T>) -> Span<T> {
        self.data
    }

    // --- arithmetic ------------------------------------------------------------------------------

    /// The transpose, a `DVector` of the same components (no copy). Upstream: `transpose`.
    #[inline(always)]
    fn transpose(self: RowDVector<T>) -> DVector<T> {
        DVector { data: self.data }
    }

    /// `self * k`, each component floored once. Upstream: `scale`.
    #[inline]
    fn scale(self: RowDVector<T>, k: T) -> RowDVector<T> {
        RowDVector { data: DynKernels::scale(self.data, k) }
    }

    /// `self / k`, each component divided with round-to-nearest. Upstream: `unscale`.
    #[inline]
    fn unscale(self: RowDVector<T>, k: T) -> RowDVector<T> {
        RowDVector { data: DynKernels::unscale(self.data, k) }
    }

    /// The component-wise product. Panics with `nalgebra: dimension mismatch` on different
    /// lengths. Upstream: `component_mul`.
    #[inline]
    fn component_mul(self: RowDVector<T>, rhs: RowDVector<T>) -> RowDVector<T> {
        check_len(self.data, rhs.data);
        RowDVector { data: DynKernels::component_mul(self.data, rhs.data) }
    }

    /// `sum self[i] * rhs[i]`, accumulated exactly and floored once. Panics with `nalgebra:
    /// dimension mismatch` on different lengths. Upstream: `dot`.
    #[inline]
    fn dot(self: RowDVector<T>, rhs: RowDVector<T>) -> T {
        check_len(self.data, rhs.data);
        DynKernels::dot(self.data, rhs.data)
    }

    /// The squared Euclidean norm, accumulated exactly and floored once. Upstream:
    /// `norm_squared`.
    #[inline]
    fn norm_squared(self: RowDVector<T>) -> T {
        DynKernels::norm_squared(self.data)
    }

    /// The Euclidean norm: square root of the UNSCALED exact sum of squares, floored once.
    /// Upstream: `norm`.
    #[inline]
    fn norm(self: RowDVector<T>) -> T {
        DynKernels::norm(self.data)
    }

    /// Whether both vectors have the same length and every pair of components is within `ulps`
    /// raw units. Upstream: `approx::AbsDiffEq::abs_diff_eq` (ulp, DESIGN D3).
    #[inline]
    fn abs_diff_eq(self: RowDVector<T>, other: RowDVector<T>, ulps: u64) -> bool {
        self.data.len() == other.data.len()
            && DynKernels::abs_diff_eq(self.data, other.data, ulps)
    }

    // --- edition ---------------------------------------------------------------------------------

    /// `self` with a component `val` inserted at index `i` (`i <= len`). Panics with `nalgebra:
    /// index out of bounds` otherwise. Upstream: `insert_column`.
    #[inline]
    fn insert_column(self: RowDVector<T>, i: usize, val: T) -> RowDVector<T> {
        Self::insert_columns(self, i, 1, val)
    }

    /// `self` with `n` components `val` inserted at index `i` (`i <= len`). Upstream:
    /// `insert_columns`.
    #[inline]
    fn insert_columns(self: RowDVector<T>, i: usize, n: usize, val: T) -> RowDVector<T> {
        RowDVector { data: DynKernels::insert_columns(self.data, 1, self.data.len(), i, n, val) }
    }

    /// `self` with `D` components `val` inserted at index `i`. Upstream:
    /// `insert_fixed_columns::<D>(i, val)`.
    #[inline]
    fn insert_fixed_columns<const D: usize>(self: RowDVector<T>, i: usize, val: T) -> RowDVector<T> {
        Self::insert_columns(self, i, D, val)
    }

    /// `self` without its component `i`. Panics with `nalgebra: index out of bounds` when `i >=
    /// len`. Upstream: `remove_column`.
    #[inline]
    fn remove_column(self: RowDVector<T>, i: usize) -> RowDVector<T> {
        Self::remove_columns(self, i, 1)
    }

    /// `self` without the components `i .. i + n`. Panics with `nalgebra: index out of bounds`
    /// when `i + n > len`. Upstream: `remove_columns`.
    #[inline]
    fn remove_columns(self: RowDVector<T>, i: usize, n: usize) -> RowDVector<T> {
        RowDVector { data: DynKernels::remove_columns(self.data, 1, self.data.len(), i, n) }
    }

    /// `self` without the components `i .. i + D`. Upstream: `remove_fixed_columns::<D>(i)`.
    #[inline]
    fn remove_fixed_columns<const D: usize>(self: RowDVector<T>, i: usize) -> RowDVector<T> {
        Self::remove_columns(self, i, D)
    }

    /// `self` without the components whose index is in `indices` (repeated or out of range
    /// indices are ignored). Upstream: `remove_columns_at` (`&[usize]`).
    #[inline]
    fn remove_columns_at(self: RowDVector<T>, indices: Span<usize>) -> RowDVector<T> {
        let (data, _) = DynKernels::remove_columns_at(self.data, 1, self.data.len(), indices);
        RowDVector { data }
    }

    /// `self` with `new_ncols` components: the first `min(len, new_ncols)` kept, the new ones
    /// `val`. Upstream: `resize_horizontally`.
    #[inline]
    fn resize_horizontally(self: RowDVector<T>, new_ncols: usize, val: T) -> RowDVector<T> {
        RowDVector { data: DynKernels::resize(self.data, 1, self.data.len(), 1, new_ncols, val) }
    }

    /// `self = self.resize_horizontally(new_ncols, val)`. Upstream: `resize_horizontally_mut`.
    #[inline]
    fn resize_horizontally_mut(ref self: RowDVector<T>, new_ncols: usize, val: T) {
        self = Self::resize_horizontally(self, new_ncols, val);
    }

    /// `self` resized to a `new_nrows x new_ncols` `DMatrix`, the common block kept, the new
    /// components `val`. Upstream: `resize`.
    #[inline]
    fn resize(self: RowDVector<T>, new_nrows: usize, new_ncols: usize, val: T) -> DMatrix<T> {
        DMatrix {
            data: DynKernels::resize(self.data, 1, self.data.len(), new_nrows, new_ncols, val),
            nrows: new_nrows,
            ncols: new_ncols,
        }
    }
}
