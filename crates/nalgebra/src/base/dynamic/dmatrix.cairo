//! `DMatrix<T>`: a matrix whose dimensions are runtime values (upstream `DMatrix`, `OMatrix<T,
//! Dyn, Dyn>`), and the ten partially dynamic aliases `Matrix2xX..Matrix6xX`,
//! `MatrixXx2..MatrixXx6` (DESIGN D5).
//!
//! Storage: the components in COLUMN-major order (upstream's `VecStorage`) in one `Span<T>`, with
//! the two dimensions. A `Span` is a view of a write-once array, so `DMatrix` is `Copy` (upstream
//! `Clone`, like every Cairo value type here) and every operation builds a new span; the `_mut`
//! forms (`resize_mut`...) take `ref self` and replace it. Loops are allowed (the sizes are
//! runtime values); every sum of products is ONE exact accumulation floored once
//! (`Real::Wide`), and the products of square matrices up to 6x6 dispatch to the static kernels
//! (bit-identical, measured cheaper).

use simba::scalar::Real;
use super::super::errors;
use super::dvector::DVector;
use super::kernels::DynKernels;
use super::row_dvector::RowDVector;

/// A dynamically sized matrix (upstream `DMatrix<T>`): `nrows x ncols` components stored in
/// column-major order. Built by the constructors of `DMatrixTrait` (`zeros(nrows, ncols)`,
/// `from_vec(nrows, ncols, data)`...) or converted from a static shape (`m.into()`). `Serde`
/// writes `data` (length-prefixed, column-major), then `nrows`, `ncols`: upstream's `VecStorage`
/// field order.
#[derive(Copy, Drop, PartialEq, Serde, Debug)]
pub struct DMatrix<T> {
    pub(crate) data: Span<T>,
    pub(crate) nrows: usize,
    pub(crate) ncols: usize,
}

/// Upstream alias `Matrix2xX` (2 rows, a dynamic number of columns): a `DMatrix` whose row count
/// is 2 by construction (the static edition forms of the 2-row shapes return it). Cairo has no
/// const generics: the fixed dimension is an invariant of the value, not of the type (DESIGN D5).
pub type Matrix2xX<T> = DMatrix<T>;
/// Upstream alias `Matrix3xX`: a `DMatrix` with 3 rows (DESIGN D5).
pub type Matrix3xX<T> = DMatrix<T>;
/// Upstream alias `Matrix4xX`: a `DMatrix` with 4 rows (DESIGN D5).
pub type Matrix4xX<T> = DMatrix<T>;
/// Upstream alias `Matrix5xX`: a `DMatrix` with 5 rows (DESIGN D5).
pub type Matrix5xX<T> = DMatrix<T>;
/// Upstream alias `Matrix6xX`: a `DMatrix` with 6 rows (DESIGN D5).
pub type Matrix6xX<T> = DMatrix<T>;
/// Upstream alias `MatrixXx2` (a dynamic number of rows, 2 columns): a `DMatrix` with 2 columns
/// (DESIGN D5).
pub type MatrixXx2<T> = DMatrix<T>;
/// Upstream alias `MatrixXx3`: a `DMatrix` with 3 columns (DESIGN D5).
pub type MatrixXx3<T> = DMatrix<T>;
/// Upstream alias `MatrixXx4`: a `DMatrix` with 4 columns (DESIGN D5).
pub type MatrixXx4<T> = DMatrix<T>;
/// Upstream alias `MatrixXx5`: a `DMatrix` with 5 columns (DESIGN D5).
pub type MatrixXx5<T> = DMatrix<T>;
/// Upstream alias `MatrixXx6`: a `DMatrix` with 6 columns (DESIGN D5).
pub type MatrixXx6<T> = DMatrix<T>;

/// Methods of `DMatrix<T>` for any `Real` scalar.
#[generate_trait]
pub impl DMatrixImpl<
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
> of DMatrixTrait<T> {
    // --- construction ----------------------------------------------------------------------

    /// The `nrows x ncols` zero matrix. Upstream: `DMatrix::zeros(nrows, ncols)`.
    #[inline]
    fn zeros(nrows: usize, ncols: usize) -> DMatrix<T> {
        DMatrix { data: DynKernels::filled(nrows * ncols, R::zero()).span(), nrows, ncols }
    }

    /// The `nrows x ncols` matrix whose components all equal `elem`. Upstream:
    /// `DMatrix::from_element(nrows, ncols, elem)`.
    #[inline]
    fn from_element(nrows: usize, ncols: usize, elem: T) -> DMatrix<T> {
        DMatrix { data: DynKernels::filled(nrows * ncols, elem).span(), nrows, ncols }
    }

    /// Alias of `from_element`. Upstream: `DMatrix::repeat(nrows, ncols, elem)`.
    #[inline]
    fn repeat(nrows: usize, ncols: usize, elem: T) -> DMatrix<T> {
        Self::from_element(nrows, ncols, elem)
    }

    /// The `nrows x ncols` matrix with ones on the diagonal, zeros elsewhere (rectangular
    /// allowed). Upstream: `DMatrix::identity(nrows, ncols)`.
    #[inline]
    fn identity(nrows: usize, ncols: usize) -> DMatrix<T> {
        Self::from_diagonal_element(nrows, ncols, R::one())
    }

    /// The `nrows x ncols` matrix with `elt` on the diagonal, zeros elsewhere. Upstream:
    /// `DMatrix::from_diagonal_element(nrows, ncols, elt)`.
    fn from_diagonal_element(nrows: usize, ncols: usize, elt: T) -> DMatrix<T> {
        let min = if nrows < ncols {
            nrows
        } else {
            ncols
        };
        let diag = DynKernels::filled(min, elt).span();
        DMatrix { data: DynKernels::partial_diagonal(nrows, ncols, diag), nrows, ncols }
    }

    /// The `nrows x ncols` matrix whose first `elts.len()` diagonal components are `elts`, zeros
    /// elsewhere. Panics with `nalgebra: diagonal too long` when `elts` is longer than the
    /// diagonal (upstream: "Too many diagonal elements provided."). Upstream:
    /// `DMatrix::from_partial_diagonal(nrows, ncols, &[T])`.
    #[inline]
    fn from_partial_diagonal(nrows: usize, ncols: usize, elts: Span<T>) -> DMatrix<T> {
        DMatrix { data: DynKernels::partial_diagonal(nrows, ncols, elts), nrows, ncols }
    }

    /// The `nrows x ncols` matrix of `data` in COLUMN-major order, without copy. Panics with
    /// `nalgebra: wrong slice length` unless `data.len() == nrows * ncols` (upstream: "Data
    /// storage buffer dimension mismatch."). Upstream: `DMatrix::from_vec(nrows, ncols, Vec<T>)`.
    #[inline]
    fn from_vec(nrows: usize, ncols: usize, data: Array<T>) -> DMatrix<T> {
        Self::from_column_slice(nrows, ncols, data.span())
    }

    /// The `nrows x ncols` matrix of the values of `data` in COLUMN-major order. Panics with
    /// `nalgebra: wrong slice length` unless `data.len() == nrows * ncols`. Upstream:
    /// `DMatrix::from_iterator(nrows, ncols, iter)` (an iterator, which upstream reads to its
    /// `nrows * ncols` first items; a `Span` of exactly that length here).
    #[inline]
    fn from_iterator(nrows: usize, ncols: usize, data: Span<T>) -> DMatrix<T> {
        Self::from_column_slice(nrows, ncols, data)
    }

    /// The `nrows x ncols` matrix of the values of `data` in ROW-major order. Panics with
    /// `nalgebra: wrong slice length` unless `data.len() == nrows * ncols`. Upstream:
    /// `DMatrix::from_row_iterator(nrows, ncols, iter)` (a `Span` here).
    #[inline]
    fn from_row_iterator(nrows: usize, ncols: usize, data: Span<T>) -> DMatrix<T> {
        Self::from_row_slice(nrows, ncols, data)
    }

    /// The `nrows x ncols` matrix of `data` in COLUMN-major order, without copy. Panics with
    /// `nalgebra: wrong slice length` unless `data.len() == nrows * ncols`. Upstream:
    /// `DMatrix::from_column_slice(nrows, ncols, &[T])`.
    #[inline]
    fn from_column_slice(nrows: usize, ncols: usize, data: Span<T>) -> DMatrix<T> {
        if data.len() != nrows * ncols {
            core::panic_with_felt252(errors::SLICE_LENGTH)
        }
        DMatrix { data, nrows, ncols }
    }

    /// The `nrows x ncols` matrix of `data` in ROW-major order (one transposing copy). Panics
    /// with `nalgebra: wrong slice length` unless `data.len() == nrows * ncols`. Upstream:
    /// `DMatrix::from_row_slice(nrows, ncols, &[T])`.
    #[inline]
    fn from_row_slice(nrows: usize, ncols: usize, data: Span<T>) -> DMatrix<T> {
        if data.len() != nrows * ncols {
            core::panic_with_felt252(errors::SLICE_LENGTH)
        }
        DMatrix { data: DynKernels::from_row_major(data, nrows, ncols), nrows, ncols }
    }

    /// The `nrows x ncols` matrix whose component `(i, j)` is `f(i, j)`, `f` being called in
    /// column-major order like upstream. `f` is any closure or `Fn` value of `(usize, usize)`
    /// whose output converts `Into<T>` (the identity included). Upstream:
    /// `DMatrix::from_fn(nrows, ncols, f)`.
    #[cfg(feature: 'closures')]
    fn from_fn<
        F,
        +Drop<F>,
        impl Func: core::ops::Fn<F, (usize, usize)>,
        +Into<Func::Output, T>,
        +Drop<Func::Output>,
    >(
        nrows: usize, ncols: usize, f: F,
    ) -> DMatrix<T> {
        let mut out: Array<T> = array![];
        let mut j: usize = 0;
        while j != ncols {
            let mut i: usize = 0;
            while i != nrows {
                out.append(f(i, j).into());
                i += 1;
            }
            j += 1;
        }
        DMatrix { data: out.span(), nrows, ncols }
    }

    // --- properties ----------------------------------------------------------------------------

    /// The number of rows. Upstream: `nrows`.
    #[inline(always)]
    fn nrows(self: DMatrix<T>) -> usize {
        self.nrows
    }

    /// The number of columns. Upstream: `ncols`.
    #[inline(always)]
    fn ncols(self: DMatrix<T>) -> usize {
        self.ncols
    }

    /// `(nrows, ncols)`. Upstream: `shape`.
    #[inline(always)]
    fn shape(self: DMatrix<T>) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    /// The number of components, `nrows * ncols`. Upstream: `len`.
    #[inline(always)]
    fn len(self: DMatrix<T>) -> usize {
        self.data.len()
    }

    /// Whether the matrix has no component (a zero dimension). Upstream: `is_empty`.
    #[inline(always)]
    fn is_empty(self: DMatrix<T>) -> bool {
        self.data.is_empty()
    }

    /// Whether `nrows == ncols`. Upstream: `is_square`.
    #[inline(always)]
    fn is_square(self: DMatrix<T>) -> bool {
        self.nrows == self.ncols
    }

    /// The components in column-major order (upstream's storage order). Upstream: `as_slice`
    /// (`&[T]`; a `Span`, a view of the same data, here).
    #[inline(always)]
    fn as_slice(self: DMatrix<T>) -> Span<T> {
        self.data
    }

    /// Column `j` as a `DVector` (a view of the same data, no copy). Panics with `nalgebra: index
    /// out of bounds` when `j >= ncols`. Upstream: `column` (a view).
    #[inline]
    fn column(self: DMatrix<T>, j: usize) -> DVector<T> {
        if j >= self.ncols {
            core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS)
        }
        DVector { data: self.data.slice(j * self.nrows, self.nrows) }
    }

    /// Row `i` as a `RowDVector` (a strided copy). Panics with `nalgebra: index out of bounds`
    /// when `i >= nrows`. Upstream: `row` (a view).
    fn row(self: DMatrix<T>, i: usize) -> RowDVector<T> {
        if i >= self.nrows {
            core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS)
        }
        let mut out: Array<T> = array![];
        let mut k = i;
        let len = self.data.len();
        while k < len {
            out.append(*self.data[k]);
            k += self.nrows;
        }
        RowDVector { data: out.span() }
    }

    // --- arithmetic ------------------------------------------------------------------------------

    /// The transpose (a copy). Upstream: `transpose`.
    #[inline]
    fn transpose(self: DMatrix<T>) -> DMatrix<T> {
        DMatrix {
            data: DynKernels::transpose(self.data, self.nrows, self.ncols),
            nrows: self.ncols,
            ncols: self.nrows,
        }
    }

    /// `self * k`, each component floored once. Upstream: `scale` (`self * k`).
    #[inline]
    fn scale(self: DMatrix<T>, k: T) -> DMatrix<T> {
        DMatrix { data: DynKernels::scale(self.data, k), nrows: self.nrows, ncols: self.ncols }
    }

    /// `self / k`, each component divided with round-to-nearest (`Real::div`). Panics on a zero
    /// `k`. Upstream: `unscale` (`self / k`).
    #[inline]
    fn unscale(self: DMatrix<T>, k: T) -> DMatrix<T> {
        DMatrix { data: DynKernels::unscale(self.data, k), nrows: self.nrows, ncols: self.ncols }
    }

    /// The component-wise product, each component floored once. Panics with `nalgebra: dimension
    /// mismatch` on different shapes. Upstream: `component_mul`.
    #[inline]
    fn component_mul(self: DMatrix<T>, rhs: DMatrix<T>) -> DMatrix<T> {
        check_same_shape(self, rhs);
        DMatrix {
            data: DynKernels::component_mul(self.data, rhs.data),
            nrows: self.nrows,
            ncols: self.ncols,
        }
    }

    /// The Frobenius dot product `sum self[k] * rhs[k]`, accumulated exactly and floored once.
    /// Panics with `nalgebra: dimension mismatch` on different shapes. Upstream: `dot`.
    #[inline]
    fn dot(self: DMatrix<T>, rhs: DMatrix<T>) -> T {
        check_same_shape(self, rhs);
        DynKernels::dot(self.data, rhs.data)
    }

    /// The squared Frobenius norm, accumulated exactly and floored once. Upstream:
    /// `norm_squared`.
    #[inline]
    fn norm_squared(self: DMatrix<T>) -> T {
        DynKernels::norm_squared(self.data)
    }

    /// The Frobenius norm: square root of the UNSCALED exact sum of squares, floored once (only
    /// the result must fit). Upstream: `norm`.
    #[inline]
    fn norm(self: DMatrix<T>) -> T {
        DynKernels::norm(self.data)
    }

    /// Whether both matrices have the same shape and every pair of components is within `ulps`
    /// raw units. Upstream: `approx::AbsDiffEq::abs_diff_eq`, the tolerance in ulp (DESIGN D3).
    #[inline]
    fn abs_diff_eq(self: DMatrix<T>, other: DMatrix<T>, ulps: u64) -> bool {
        self.nrows == other.nrows
            && self.ncols == other.ncols
            && DynKernels::abs_diff_eq(self.data, other.data, ulps)
    }

    // --- edition ---------------------------------------------------------------------------------

    /// `self` with a column of `val` inserted at index `i` (`i <= ncols`). Panics with `nalgebra:
    /// index out of bounds` otherwise (upstream: "Column insertion index out of range.").
    /// Upstream: `insert_column`.
    #[inline]
    fn insert_column(self: DMatrix<T>, i: usize, val: T) -> DMatrix<T> {
        Self::insert_columns(self, i, 1, val)
    }

    /// `self` with `n` columns of `val` inserted at index `i` (`i <= ncols`). Panics with
    /// `nalgebra: index out of bounds` otherwise. Upstream: `insert_columns`.
    #[inline]
    fn insert_columns(self: DMatrix<T>, i: usize, n: usize, val: T) -> DMatrix<T> {
        DMatrix {
            data: DynKernels::insert_columns(self.data, self.nrows, self.ncols, i, n, val),
            nrows: self.nrows,
            ncols: self.ncols + n,
        }
    }

    /// `self` with `D` columns of `val` inserted at index `i`. Upstream:
    /// `insert_fixed_columns::<D>(i, val)` (a const generic, the same here).
    #[inline]
    fn insert_fixed_columns<const D: usize>(self: DMatrix<T>, i: usize, val: T) -> DMatrix<T> {
        Self::insert_columns(self, i, D, val)
    }

    /// `self` with a row of `val` inserted at index `i` (`i <= nrows`). Panics with `nalgebra:
    /// index out of bounds` otherwise (upstream: "Row insertion index out of range."). Upstream:
    /// `insert_row`.
    #[inline]
    fn insert_row(self: DMatrix<T>, i: usize, val: T) -> DMatrix<T> {
        Self::insert_rows(self, i, 1, val)
    }

    /// `self` with `n` rows of `val` inserted at index `i` (`i <= nrows`). Panics with
    /// `nalgebra: index out of bounds` otherwise. Upstream: `insert_rows`.
    #[inline]
    fn insert_rows(self: DMatrix<T>, i: usize, n: usize, val: T) -> DMatrix<T> {
        DMatrix {
            data: DynKernels::insert_rows(self.data, self.nrows, self.ncols, i, n, val),
            nrows: self.nrows + n,
            ncols: self.ncols,
        }
    }

    /// `self` with `D` rows of `val` inserted at index `i`. Upstream:
    /// `insert_fixed_rows::<D>(i, val)`.
    #[inline]
    fn insert_fixed_rows<const D: usize>(self: DMatrix<T>, i: usize, val: T) -> DMatrix<T> {
        Self::insert_rows(self, i, D, val)
    }

    /// `self` without its column `i`. Panics with `nalgebra: index out of bounds` when `i >=
    /// ncols` (upstream: "Column index out of range."). Upstream: `remove_column`.
    #[inline]
    fn remove_column(self: DMatrix<T>, i: usize) -> DMatrix<T> {
        Self::remove_columns(self, i, 1)
    }

    /// `self` without the columns `i .. i + n`. Panics with `nalgebra: index out of bounds` when
    /// `i + n > ncols`. Upstream: `remove_columns`.
    #[inline]
    fn remove_columns(self: DMatrix<T>, i: usize, n: usize) -> DMatrix<T> {
        DMatrix {
            data: DynKernels::remove_columns(self.data, self.nrows, self.ncols, i, n),
            nrows: self.nrows,
            ncols: self.ncols - n,
        }
    }

    /// `self` without the columns `i .. i + D`. Upstream: `remove_fixed_columns::<D>(i)`.
    #[inline]
    fn remove_fixed_columns<const D: usize>(self: DMatrix<T>, i: usize) -> DMatrix<T> {
        Self::remove_columns(self, i, D)
    }

    /// `self` without the columns whose index is in `indices` (any order; repeated or out of
    /// range indices are ignored, like upstream's `contains` test). Upstream: `remove_columns_at`
    /// (`&[usize]`).
    #[inline]
    fn remove_columns_at(self: DMatrix<T>, indices: Span<usize>) -> DMatrix<T> {
        let (data, ncols) = DynKernels::remove_columns_at(
            self.data, self.nrows, self.ncols, indices,
        );
        DMatrix { data, nrows: self.nrows, ncols }
    }

    /// `self` without its row `i`. Panics with `nalgebra: index out of bounds` when `i >=
    /// nrows` (upstream: "Row index out of range."). Upstream: `remove_row`.
    #[inline]
    fn remove_row(self: DMatrix<T>, i: usize) -> DMatrix<T> {
        Self::remove_rows(self, i, 1)
    }

    /// `self` without the rows `i .. i + n`. Panics with `nalgebra: index out of bounds` when
    /// `i + n > nrows`. Upstream: `remove_rows`.
    #[inline]
    fn remove_rows(self: DMatrix<T>, i: usize, n: usize) -> DMatrix<T> {
        DMatrix {
            data: DynKernels::remove_rows(self.data, self.nrows, self.ncols, i, n),
            nrows: self.nrows - n,
            ncols: self.ncols,
        }
    }

    /// `self` without the rows `i .. i + D`. Upstream: `remove_fixed_rows::<D>(i)`.
    #[inline]
    fn remove_fixed_rows<const D: usize>(self: DMatrix<T>, i: usize) -> DMatrix<T> {
        Self::remove_rows(self, i, D)
    }

    /// `self` without the rows whose index is in `indices` (any order; repeated or out of range
    /// indices are ignored). Upstream: `remove_rows_at` (`&[usize]`).
    #[inline]
    fn remove_rows_at(self: DMatrix<T>, indices: Span<usize>) -> DMatrix<T> {
        let (data, nrows) = DynKernels::remove_rows_at(self.data, self.nrows, self.ncols, indices);
        DMatrix { data, nrows, ncols: self.ncols }
    }

    /// `self` resized to `new_nrows x new_ncols`: `result[(i, j)] == self[(i, j)]` on the common
    /// block, the new components are `val`. Upstream: `resize`.
    #[inline]
    fn resize(self: DMatrix<T>, new_nrows: usize, new_ncols: usize, val: T) -> DMatrix<T> {
        DMatrix {
            data: DynKernels::resize(
                self.data, self.nrows, self.ncols, new_nrows, new_ncols, val,
            ),
            nrows: new_nrows,
            ncols: new_ncols,
        }
    }

    /// `self` with `new_nrows` rows (same columns), the new components `val`. Upstream:
    /// `resize_vertically`.
    #[inline]
    fn resize_vertically(self: DMatrix<T>, new_nrows: usize, val: T) -> DMatrix<T> {
        Self::resize(self, new_nrows, self.ncols, val)
    }

    /// `self` with `new_ncols` columns (same rows), the new components `val`. Upstream:
    /// `resize_horizontally`.
    #[inline]
    fn resize_horizontally(self: DMatrix<T>, new_ncols: usize, val: T) -> DMatrix<T> {
        Self::resize(self, self.nrows, new_ncols, val)
    }

    /// `self = self.resize(new_nrows, new_ncols, val)`. Upstream: `resize_mut`.
    #[inline]
    fn resize_mut(ref self: DMatrix<T>, new_nrows: usize, new_ncols: usize, val: T) {
        self = Self::resize(self, new_nrows, new_ncols, val);
    }

    /// `self = self.resize_vertically(new_nrows, val)`. Upstream: `resize_vertically_mut`.
    #[inline]
    fn resize_vertically_mut(ref self: DMatrix<T>, new_nrows: usize, val: T) {
        self = Self::resize(self, new_nrows, self.ncols, val);
    }

    /// `self = self.resize_horizontally(new_ncols, val)`. Upstream: `resize_horizontally_mut`.
    #[inline]
    fn resize_horizontally_mut(ref self: DMatrix<T>, new_ncols: usize, val: T) {
        self = Self::resize(self, self.nrows, new_ncols, val);
    }

    /// The row vector of `f(column)` for every column. Upstream: `compress_rows` (`f` takes a
    /// column view; a `DVector` view of the same data here).
    #[cfg(feature: 'closures')]
    fn compress_rows<
        F,
        +Drop<F>,
        impl Func: core::ops::Fn<F, (DVector<T>,)>,
        +Into<Func::Output, T>,
        +Drop<Func::Output>,
    >(
        self: DMatrix<T>, f: F,
    ) -> RowDVector<T> {
        RowDVector { data: compress_columns_of(self, f) }
    }

    /// The column vector of `f(column)` for every column (`compress_rows` transposed).
    /// Upstream: `compress_rows_tr`.
    #[cfg(feature: 'closures')]
    fn compress_rows_tr<
        F,
        +Drop<F>,
        impl Func: core::ops::Fn<F, (DVector<T>,)>,
        +Into<Func::Output, T>,
        +Drop<Func::Output>,
    >(
        self: DMatrix<T>, f: F,
    ) -> DVector<T> {
        DVector { data: compress_columns_of(self, f) }
    }

    /// `init` folded with every column: `acc = f(acc, column)` from the first column to the last.
    /// Upstream: `compress_columns(init, f)` (`f` mutates `&mut acc`; Cairo closures take and
    /// return it by value).
    #[cfg(feature: 'closures')]
    fn compress_columns<
        F,
        +Drop<F>,
        impl Func: core::ops::Fn<F, (DVector<T>, DVector<T>)>,
        +Into<Func::Output, DVector<T>>,
        +Drop<Func::Output>,
    >(
        self: DMatrix<T>, init: DVector<T>, f: F,
    ) -> DVector<T> {
        if init.data.len() != self.nrows {
            core::panic_with_felt252(errors::DIMENSION_MISMATCH)
        }
        let mut acc = init;
        let mut j: usize = 0;
        while j != self.ncols {
            acc = f(acc, DVector { data: self.data.slice(j * self.nrows, self.nrows) }).into();
            j += 1;
        }
        acc
    }
}

/// `f(column)` for every column of `m`.
#[cfg(feature: 'closures')]
fn compress_columns_of<
    T,
    +Copy<T>,
    +Drop<T>,
    F,
    +Drop<F>,
    impl Func: core::ops::Fn<F, (DVector<T>,)>,
    +Into<Func::Output, T>,
    +Drop<Func::Output>,
>(
    m: DMatrix<T>, f: F,
) -> Span<T> {
    let mut out: Array<T> = array![];
    let mut j: usize = 0;
    while j != m.ncols {
        out.append(f(DVector { data: m.data.slice(j * m.nrows, m.nrows) }).into());
        j += 1;
    }
    out.span()
}

/// Panics with `nalgebra: dimension mismatch` unless `a` and `b` have the same shape.
fn check_same_shape<T>(a: DMatrix<T>, b: DMatrix<T>) {
    if a.nrows != b.nrows || a.ncols != b.ncols {
        core::panic_with_felt252(errors::DIMENSION_MISMATCH)
    }
}
