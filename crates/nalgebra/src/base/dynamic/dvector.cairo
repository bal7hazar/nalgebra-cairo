//! `DVector<T>`: a column vector whose length is a runtime value (upstream `DVector`,
//! `OVector<T, Dyn>`, alias `MatrixXx1`), DESIGN D5.

use core::ops::IndexView;
use simba::scalar::Real;
use super::dmatrix::DMatrix;
use super::kernels::{DynKernels, at_linear, get_linear};
use super::row_dvector::RowDVector;
use super::super::errors;
use super::super::matrix1::Matrix1;
use super::super::matrix_index::MatrixIndex;
use super::super::matrix_mul::MatrixMul;
use super::super::vector2::Vector2;
use super::super::vector3::Vector3;
use super::super::vector4::Vector4;
use super::super::vector5::Vector5;
use super::super::vector6::Vector6;

/// A dynamically sized column vector (upstream `DVector<T>`): its components in one `Span<T>`.
/// `Copy` (upstream `Clone`); `Serde` writes the length-prefixed components.
#[derive(Copy, Drop, PartialEq, Serde, Debug)]
pub struct DVector<T> {
    pub(crate) data: Span<T>,
}

/// Upstream alias `MatrixXx1`: the same type as `DVector` (one Cairo struct per shape, like
/// `Matrix3x1` / `Vector3`).
pub type MatrixXx1<T> = DVector<T>;

/// Methods of `DVector<T>` for any `Real` scalar.
#[generate_trait]
pub impl DVectorImpl<
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
> of DVectorTrait<T> {
    // --- construction ----------------------------------------------------------------------

    /// The zero vector of length `nrows`. Upstream: `DVector::zeros(nrows)`.
    #[inline]
    fn zeros(nrows: usize) -> DVector<T> {
        DVector { data: DynKernels::filled(nrows, R::zero()).span() }
    }

    /// The vector of length `nrows` whose components all equal `elem`. Upstream:
    /// `DVector::from_element(nrows, elem)`.
    #[inline]
    fn from_element(nrows: usize, elem: T) -> DVector<T> {
        DVector { data: DynKernels::filled(nrows, elem).span() }
    }

    /// Alias of `from_element`. Upstream: `DVector::repeat(nrows, elem)`.
    #[inline]
    fn repeat(nrows: usize, elem: T) -> DVector<T> {
        Self::from_element(nrows, elem)
    }

    /// The vector `(1, 0, .., 0)` of length `nrows` (the `nrows x 1` identity). Upstream:
    /// `DVector::identity(nrows)`.
    #[inline]
    fn identity(nrows: usize) -> DVector<T> {
        Self::from_diagonal_element(nrows, R::one())
    }

    /// The vector `(elt, 0, .., 0)` of length `nrows` (empty when `nrows == 0`). Upstream:
    /// `DVector::from_diagonal_element(nrows, elt)`.
    fn from_diagonal_element(nrows: usize, elt: T) -> DVector<T> {
        if nrows == 0 {
            return DVector { data: array![].span() };
        }
        let mut out: Array<T> = array![elt];
        DynKernels::append_n(ref out, nrows - 1, R::zero());
        DVector { data: out.span() }
    }

    /// The vector of length `nrows` whose first component is `elts[0]` (when given), zeros
    /// elsewhere. Panics with `nalgebra: diagonal too long` when `elts` has more values than the
    /// `nrows x 1` diagonal. Upstream: `DVector::from_partial_diagonal(nrows, &[T])`.
    #[inline]
    fn from_partial_diagonal(nrows: usize, elts: Span<T>) -> DVector<T> {
        DVector { data: DynKernels::partial_diagonal(nrows, 1, elts) }
    }

    /// The vector of the values of `data`, without copy. Upstream: `DVector::from_vec(Vec<T>)`.
    #[inline(always)]
    fn from_vec(data: Array<T>) -> DVector<T> {
        DVector { data: data.span() }
    }

    /// The vector of the values of `data`, without copy. Upstream:
    /// `DVector::from_column_slice(&[T])`.
    #[inline(always)]
    fn from_column_slice(data: Span<T>) -> DVector<T> {
        DVector { data }
    }

    /// The vector of the values of `data` (a column: row-major and column-major orders agree).
    /// Upstream: `DVector::from_row_slice(&[T])`.
    #[inline(always)]
    fn from_row_slice(data: Span<T>) -> DVector<T> {
        DVector { data }
    }

    /// The vector of length `nrows` of the values of `data`. Panics with `nalgebra: wrong slice
    /// length` unless `data.len() == nrows`. Upstream: `DVector::from_iterator(nrows, iter)` (a
    /// `Span` here).
    #[inline]
    fn from_iterator(nrows: usize, data: Span<T>) -> DVector<T> {
        if data.len() != nrows {
            core::panic_with_felt252(errors::SLICE_LENGTH)
        }
        DVector { data }
    }

    /// `from_iterator` (the orders agree on a column). Upstream:
    /// `DVector::from_row_iterator(nrows, iter)`.
    #[inline]
    fn from_row_iterator(nrows: usize, data: Span<T>) -> DVector<T> {
        Self::from_iterator(nrows, data)
    }

    /// The vector of length `nrows` whose component `i` is `f(i, 0)` (upstream's `(row, column)`
    /// closure). `f`'s output converts `Into<T>`. Upstream: `DVector::from_fn(nrows, f)`.
    #[cfg(feature: 'closures')]
    fn from_fn<
        F,
        +Drop<F>,
        impl Func: core::ops::Fn<F, (usize, usize)>,
        +Into<Func::Output, T>,
        +Drop<Func::Output>,
    >(
        nrows: usize, f: F,
    ) -> DVector<T> {
        let mut out: Array<T> = array![];
        let mut i: usize = 0;
        while i != nrows {
            out.append(f(i, 0).into());
            i += 1;
        }
        DVector { data: out.span() }
    }

    // --- properties ----------------------------------------------------------------------------

    /// The number of rows (the length). Upstream: `nrows`.
    #[inline(always)]
    fn nrows(self: DVector<T>) -> usize {
        self.data.len()
    }

    /// `1`. Upstream: `ncols`.
    #[inline(always)]
    fn ncols(self: DVector<T>) -> usize {
        1
    }

    /// `(len, 1)`. Upstream: `shape`.
    #[inline(always)]
    fn shape(self: DVector<T>) -> (usize, usize) {
        (self.data.len(), 1)
    }

    /// The number of components. Upstream: `len`.
    #[inline(always)]
    fn len(self: DVector<T>) -> usize {
        self.data.len()
    }

    /// Whether the vector has no component. Upstream: `is_empty`.
    #[inline(always)]
    fn is_empty(self: DVector<T>) -> bool {
        self.data.is_empty()
    }

    /// The components. Upstream: `as_slice` (a `Span` here).
    #[inline(always)]
    fn as_slice(self: DVector<T>) -> Span<T> {
        self.data
    }

    // --- arithmetic ------------------------------------------------------------------------------

    /// The transpose, a `RowDVector` of the same components (no copy). Upstream: `transpose`.
    #[inline(always)]
    fn transpose(self: DVector<T>) -> RowDVector<T> {
        RowDVector { data: self.data }
    }

    /// `self * k`, each component floored once. Upstream: `scale`.
    #[inline]
    fn scale(self: DVector<T>, k: T) -> DVector<T> {
        DVector { data: DynKernels::scale(self.data, k) }
    }

    /// `self / k`, each component divided with round-to-nearest. Upstream: `unscale`.
    #[inline]
    fn unscale(self: DVector<T>, k: T) -> DVector<T> {
        DVector { data: DynKernels::unscale(self.data, k) }
    }

    /// The component-wise product. Panics with `nalgebra: dimension mismatch` on different
    /// lengths. Upstream: `component_mul`.
    #[inline]
    fn component_mul(self: DVector<T>, rhs: DVector<T>) -> DVector<T> {
        check_len(self.data, rhs.data);
        DVector { data: DynKernels::component_mul(self.data, rhs.data) }
    }

    /// `sum self[i] * rhs[i]`, accumulated exactly and floored once. Panics with `nalgebra:
    /// dimension mismatch` on different lengths. Upstream: `dot`.
    #[inline]
    fn dot(self: DVector<T>, rhs: DVector<T>) -> T {
        check_len(self.data, rhs.data);
        DynKernels::dot(self.data, rhs.data)
    }

    /// The squared Euclidean norm, accumulated exactly and floored once. Upstream:
    /// `norm_squared`.
    #[inline]
    fn norm_squared(self: DVector<T>) -> T {
        DynKernels::norm_squared(self.data)
    }

    /// The Euclidean norm: square root of the UNSCALED exact sum of squares, floored once.
    /// Upstream: `norm`.
    #[inline]
    fn norm(self: DVector<T>) -> T {
        DynKernels::norm(self.data)
    }

    /// Whether both vectors have the same length and every pair of components is within `ulps`
    /// raw units. Upstream: `approx::AbsDiffEq::abs_diff_eq` (ulp, DESIGN D3).
    #[inline]
    fn abs_diff_eq(self: DVector<T>, other: DVector<T>, ulps: u64) -> bool {
        self.data.len() == other.data.len() && DynKernels::abs_diff_eq(self.data, other.data, ulps)
    }

    // --- edition ---------------------------------------------------------------------------------

    /// `self` with a component `val` inserted at index `i` (`i <= len`). Panics with `nalgebra:
    /// index out of bounds` otherwise. Upstream: `insert_row`.
    #[inline]
    fn insert_row(self: DVector<T>, i: usize, val: T) -> DVector<T> {
        Self::insert_rows(self, i, 1, val)
    }

    /// `self` with `n` components `val` inserted at index `i` (`i <= len`). Upstream:
    /// `insert_rows`.
    #[inline]
    fn insert_rows(self: DVector<T>, i: usize, n: usize, val: T) -> DVector<T> {
        DVector { data: DynKernels::insert_columns(self.data, 1, self.data.len(), i, n, val) }
    }

    /// `self` with `D` components `val` inserted at index `i`. Upstream:
    /// `insert_fixed_rows::<D>(i, val)`.
    #[inline]
    fn insert_fixed_rows<const D: usize>(self: DVector<T>, i: usize, val: T) -> DVector<T> {
        Self::insert_rows(self, i, D, val)
    }

    /// `self` without its component `i`. Panics with `nalgebra: index out of bounds` when `i >=
    /// len`. Upstream: `remove_row`.
    #[inline]
    fn remove_row(self: DVector<T>, i: usize) -> DVector<T> {
        Self::remove_rows(self, i, 1)
    }

    /// `self` without the components `i .. i + n`. Panics with `nalgebra: index out of bounds`
    /// when `i + n > len`. Upstream: `remove_rows`.
    #[inline]
    fn remove_rows(self: DVector<T>, i: usize, n: usize) -> DVector<T> {
        DVector { data: DynKernels::remove_columns(self.data, 1, self.data.len(), i, n) }
    }

    /// `self` without the components `i .. i + D`. Upstream: `remove_fixed_rows::<D>(i)`.
    #[inline]
    fn remove_fixed_rows<const D: usize>(self: DVector<T>, i: usize) -> DVector<T> {
        Self::remove_rows(self, i, D)
    }

    /// `self` without the components whose index is in `indices` (repeated or out of range
    /// indices are ignored). Upstream: `remove_rows_at` (`&[usize]`).
    #[inline]
    fn remove_rows_at(self: DVector<T>, indices: Span<usize>) -> DVector<T> {
        let (data, _) = DynKernels::remove_columns_at(self.data, 1, self.data.len(), indices);
        DVector { data }
    }

    /// `self` with `new_nrows` components: the first `min(len, new_nrows)` kept, the new ones
    /// `val`. Upstream: `resize_vertically`.
    #[inline]
    fn resize_vertically(self: DVector<T>, new_nrows: usize, val: T) -> DVector<T> {
        DVector { data: DynKernels::resize(self.data, 1, self.data.len(), 1, new_nrows, val) }
    }

    /// `self = self.resize_vertically(new_nrows, val)`. Upstream: `resize_vertically_mut`.
    #[inline]
    fn resize_vertically_mut(ref self: DVector<T>, new_nrows: usize, val: T) {
        self = Self::resize_vertically(self, new_nrows, val);
    }

    /// `self` resized to a `new_nrows x new_ncols` `DMatrix`, the common block kept, the new
    /// components `val`. Upstream: `resize`.
    #[inline]
    fn resize(self: DVector<T>, new_nrows: usize, new_ncols: usize, val: T) -> DMatrix<T> {
        DMatrix {
            data: DynKernels::resize(self.data, self.data.len(), 1, new_nrows, new_ncols, val),
            nrows: new_nrows,
            ncols: new_ncols,
        }
    }
}

/// Panics with `nalgebra: dimension mismatch` unless `a` and `b` have the same length.
pub(crate) fn check_len<T>(a: Span<T>, b: Span<T>) {
    if a.len() != b.len() {
        core::panic_with_felt252(errors::DIMENSION_MISMATCH)
    }
}

// --- operators, indexing, products, conversions --------------------------------------------------

/// Upstream: `Mul<RowVector> for Vector`.
pub impl DVectorMulRowDVector<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +Sub<T>, +Mul<T>, +Neg<T>,
> of MatrixMul<DVector<T>, RowDVector<T>> {
    type Output = DMatrix<T>;
    #[inline]
    fn mul_mat(self: DVector<T>, rhs: RowDVector<T>) -> DMatrix<T> {
        let m = self.data.len();
        let n = rhs.data.len();
        DMatrix { data: DynKernels::mul(self.data, m, 1, rhs.data, n), nrows: m, ncols: n }
    }
}

/// `a + b` component-wise. Upstream: `Add<Matrix> for Matrix`.
pub impl DVectorAdd<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +Sub<T>, +Mul<T>, +Neg<T>,
> of Add<DVector<T>> {
    fn add(lhs: DVector<T>, rhs: DVector<T>) -> DVector<T> {
        check_len(lhs.data, rhs.data);
        DVector { data: DynKernels::add(lhs.data, rhs.data) }
    }
}

/// `a - b` component-wise. Upstream: `Sub<Matrix> for Matrix`.
pub impl DVectorSub<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +Sub<T>, +Mul<T>, +Neg<T>,
> of Sub<DVector<T>> {
    fn sub(lhs: DVector<T>, rhs: DVector<T>) -> DVector<T> {
        check_len(lhs.data, rhs.data);
        DVector { data: DynKernels::sub(lhs.data, rhs.data) }
    }
}

/// `-a`. Upstream: `Neg for Matrix`.
pub impl DVectorNeg<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +Sub<T>, +Mul<T>, +Neg<T>,
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

/// storage type, `OVector<T, Dyn>` to `OMatrix<T, Dyn, Dyn>`).
pub impl DVectorIntoDMatrix<T> of Into<DVector<T>, DMatrix<T>> {
    #[inline(always)]
    fn into(self: DVector<T>) -> DMatrix<T> {
        DMatrix { data: self.data, nrows: self.data.len(), ncols: 1 }
    }
}

/// The `DVector` of the values of an `Array` (no copy). Upstream: `From<Vec<T>> for DVector<T>`.
pub impl ArrayIntoDVector<T, +Drop<T>> of Into<Array<T>, DVector<T>> {
    #[inline(always)]
    fn into(self: Array<T>) -> DVector<T> {
        DVector { data: self.span() }
    }
}

// --- conversions from the static shapes (generated by tools/shapegen/dynamic.py) -----------------
// shapegen: begin
/// The `DVector` of the 1 components of a `Matrix1`. Upstream: `From<Matrix> for DVector`.
pub impl Matrix1IntoDVector<T, +Copy<T>, +Drop<T>> of Into<Matrix1<T>, DVector<T>> {
    #[inline]
    fn into(self: Matrix1<T>) -> DVector<T> {
        DVector { data: array![self.x].span() }
    }
}

/// The `DVector` of the 2 components of a `Vector2`. Upstream: `From<Matrix> for DVector`.
pub impl Vector2IntoDVector<T, +Copy<T>, +Drop<T>> of Into<Vector2<T>, DVector<T>> {
    #[inline]
    fn into(self: Vector2<T>) -> DVector<T> {
        DVector { data: array![self.x, self.y].span() }
    }
}

/// The `DVector` of the 3 components of a `Vector3`. Upstream: `From<Matrix> for DVector`.
pub impl Vector3IntoDVector<T, +Copy<T>, +Drop<T>> of Into<Vector3<T>, DVector<T>> {
    #[inline]
    fn into(self: Vector3<T>) -> DVector<T> {
        DVector { data: array![self.x, self.y, self.z].span() }
    }
}

/// The `DVector` of the 4 components of a `Vector4`. Upstream: `From<Matrix> for DVector`.
pub impl Vector4IntoDVector<T, +Copy<T>, +Drop<T>> of Into<Vector4<T>, DVector<T>> {
    #[inline]
    fn into(self: Vector4<T>) -> DVector<T> {
        DVector { data: array![self.x, self.y, self.z, self.w].span() }
    }
}

/// The `DVector` of the 5 components of a `Vector5`. Upstream: `From<Matrix> for DVector`.
pub impl Vector5IntoDVector<T, +Copy<T>, +Drop<T>> of Into<Vector5<T>, DVector<T>> {
    #[inline]
    fn into(self: Vector5<T>) -> DVector<T> {
        DVector { data: array![self.x, self.y, self.z, self.w, self.a].span() }
    }
}

/// The `DVector` of the 6 components of a `Vector6`. Upstream: `From<Matrix> for DVector`.
pub impl Vector6IntoDVector<T, +Copy<T>, +Drop<T>> of Into<Vector6<T>, DVector<T>> {
    #[inline]
    fn into(self: Vector6<T>) -> DVector<T> {
        DVector { data: array![self.x, self.y, self.z, self.w, self.a, self.b].span() }
    }
}
// shapegen: end
