//! `CsMatrix<T>`: a compressed sparse column matrix (upstream's legacy
//! `nalgebra::sparse::CsMatrix`, `CsMatrix<T, Dyn, Dyn, CsVecStorage<T, Dyn, Dyn>>`), its methods,
//! operators and conversions.
//!
//! Storage (upstream's `CsVecStorage`): `nrows`, `ncols`, the column pointers `p`, the row
//! indices `i` and the values `vals`, the entries of column `j` being `i[p[j]..p[j + 1]]` /
//! `vals[p[j]..p[j + 1]]`. Deviation: `p` keeps the `ncols + 1` pointers of the usual CSC layout
//! (upstream stores `ncols` of them and ends the last column at `len()`); the `p()` accessor
//! returns upstream's `ncols` first ones. Every constructor (`from_triplet`, `From<Matrix>`, the
//! operations) produces SORTED, deduplicated columns, the invariant the kernels rely on (as
//! upstream's). The spans are views of write-once arrays: `CsMatrix` is `Copy` (upstream
//! `Clone`) and every operation builds new spans (`sparse/cs_utils.cairo`: sorts of packed keys
//! and gathers stand for upstream's scatters).

use simba::scalar::Real;
use crate::base::dynamic::{DMatrix, DVector};
use crate::base::errors;
use crate::base::matrix1::Matrix1;
use crate::base::matrix2::Matrix2;
use crate::base::matrix2x3::Matrix2x3;
use crate::base::matrix2x4::Matrix2x4;
use crate::base::matrix2x5::Matrix2x5;
use crate::base::matrix2x6::Matrix2x6;
use crate::base::matrix3::Matrix3;
use crate::base::matrix3x2::Matrix3x2;
use crate::base::matrix3x4::Matrix3x4;
use crate::base::matrix3x5::Matrix3x5;
use crate::base::matrix3x6::Matrix3x6;
use crate::base::matrix4::Matrix4;
use crate::base::matrix4x2::Matrix4x2;
use crate::base::matrix4x3::Matrix4x3;
use crate::base::matrix4x5::Matrix4x5;
use crate::base::matrix4x6::Matrix4x6;
use crate::base::matrix5::Matrix5;
use crate::base::matrix5x2::Matrix5x2;
use crate::base::matrix5x3::Matrix5x3;
use crate::base::matrix5x4::Matrix5x4;
use crate::base::matrix5x6::Matrix5x6;
use crate::base::matrix6::Matrix6;
use crate::base::matrix6x2::Matrix6x2;
use crate::base::matrix6x3::Matrix6x3;
use crate::base::matrix6x4::Matrix6x4;
use crate::base::matrix6x5::Matrix6x5;
use crate::base::row_vector2::RowVector2;
use crate::base::row_vector3::RowVector3;
use crate::base::row_vector4::RowVector4;
use crate::base::row_vector5::RowVector5;
use crate::base::row_vector6::RowVector6;
use crate::base::vector2::Vector2;
use crate::base::vector3::Vector3;
use crate::base::vector4::Vector4;
use crate::base::vector5::Vector5;
use crate::base::vector6::Vector6;
use super::cs_kernels::CsKernels;
use super::cs_utils::{gather, natural_runs, sort_runs, transpose_pattern};
use super::errors as sparse_errors;

/// A compressed sparse column matrix (upstream `CsMatrix<T>`): `nrows x ncols`, its non-zero
/// entries stored column by column, rows ascending. Built by `CsMatrixTrait::from_triplet`, from a
/// dense matrix (`m.into()`), or by the operations. `PartialEq` compares the storage (shape,
/// pointers, indices, values), like upstream's derived one.
#[derive(Copy, Drop, PartialEq, Debug)]
pub struct CsMatrix<T> {
    pub(crate) nrows: usize,
    pub(crate) ncols: usize,
    pub(crate) p: Span<usize>,
    pub(crate) i: Span<usize>,
    pub(crate) vals: Span<T>,
}

/// Upstream alias `CsVector<T>` (`CsMatrix<T, Dyn, U1>`): a `CsMatrix` with one column by
/// construction (the right-hand side and result of `solve_lower_triangular_cs`, the `x` of
/// `axpy_cs`). Cairo has no const generics: the column count is an invariant of the value.
pub type CsVector<T> = CsMatrix<T>;

/// Methods of `CsMatrix<T>` for any `Real` scalar.
#[generate_trait]
pub impl CsMatrixImpl<
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
> of CsMatrixTrait<T> {
    /// The `nrows x ncols` matrix of the triplets `(irows[k], icols[k], vals[k])`: entries sorted
    /// by column then row, duplicates SUMMED (exact additions), explicit zeros kept. Panics with
    /// `nalgebra: triplet lengths` unless the three spans have the same length, with `nalgebra:
    /// index out of bounds` for a row `>= nrows` or a column `>= ncols`. Upstream:
    /// `CsMatrix::from_triplet(nrows, ncols, &irows, &icols, &vals)`. Deviation: upstream 0.35's
    /// `sort_with_workspace` overwrites duplicates before `dedup` sums them (two triplets
    /// `(0, 0, 1)`, `(0, 0, 2)` give 4 there); the sum intended by its `dedup` is what is
    /// computed here (3).
    fn from_triplet(
        nrows: usize, ncols: usize, irows: Span<usize>, icols: Span<usize>, vals: Span<T>,
    ) -> CsMatrix<T> {
        let n = vals.len();
        if irows.len() != n || icols.len() != n {
            core::panic_with_felt252(sparse_errors::TRIPLET_LENGTHS);
        }
        // Keys `(col * nrows + row) * 2^32 + k`: unique, sorted by column, row, then position.
        let mut keys: Array<u128> = array![];
        let mut rr = irows;
        let mut cc = icols;
        let nrows_w: u128 = nrows.into();
        let mut k: usize = 0;
        while let Some(r) = rr.pop_front() {
            let r = *r;
            let c = *cc.pop_front().unwrap();
            if r >= nrows || c >= ncols {
                core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS);
            }
            keys.append((c.into() * nrows_w + r.into()) * 0x100000000 + k.into());
            k += 1;
        }
        let keys = keys.span();
        let mut sorted = sort_runs(keys, natural_runs(keys));
        let mut p: Array<usize> = array![0];
        let mut out_i: Array<usize> = array![];
        let mut out_v: Array<T> = array![];
        let mut col: usize = 0;
        let mut count: usize = 0;
        let mut have = false;
        let mut cur_r: usize = 0;
        let mut cur_c: usize = 0;
        let mut cur_v: T = R::zero();
        while let Some(key) = sorted.pop_front() {
            let (_, pos) = DivRem::div_rem(*key, 0x100000000);
            let pos: usize = pos.try_into().unwrap();
            let r = *irows.at(pos);
            let c = *icols.at(pos);
            let v = *vals.at(pos);
            if have && r == cur_r && c == cur_c {
                cur_v = cur_v + v;
            } else {
                if have {
                    while col != cur_c {
                        p.append(count);
                        col += 1;
                    }
                    out_i.append(cur_r);
                    out_v.append(cur_v);
                    count += 1;
                }
                have = true;
                cur_r = r;
                cur_c = c;
                cur_v = v;
            }
        }
        if have {
            while col != cur_c {
                p.append(count);
                col += 1;
            }
            out_i.append(cur_r);
            out_v.append(cur_v);
            count += 1;
        }
        while col != ncols {
            p.append(count);
            col += 1;
        }
        CsMatrix { nrows, ncols, p: p.span(), i: out_i.span(), vals: out_v.span() }
    }

    /// The number of stored entries (explicit zeros included). Upstream: `CsMatrix::len`.
    #[inline(always)]
    fn len(self: @CsMatrix<T>) -> usize {
        (*self.vals).len()
    }

    /// The number of rows. Upstream: `CsMatrix::nrows`.
    #[inline(always)]
    fn nrows(self: @CsMatrix<T>) -> usize {
        *self.nrows
    }

    /// The number of columns. Upstream: `CsMatrix::ncols`.
    #[inline(always)]
    fn ncols(self: @CsMatrix<T>) -> usize {
        *self.ncols
    }

    /// `(nrows, ncols)`. Upstream: `CsMatrix::shape`.
    #[inline(always)]
    fn shape(self: @CsMatrix<T>) -> (usize, usize) {
        (*self.nrows, *self.ncols)
    }

    /// Whether `nrows == ncols`. Upstream: `CsMatrix::is_square`.
    #[inline(always)]
    fn is_square(self: @CsMatrix<T>) -> bool {
        *self.nrows == *self.ncols
    }

    /// Whether the row indices of every column are strictly ascending (always true for the
    /// matrices this module builds). Upstream: `CsMatrix::is_sorted`.
    fn is_sorted(self: @CsMatrix<T>) -> bool {
        let mut pp = *self.p;
        let mut ii = *self.i;
        let mut start = *pp.pop_front().unwrap();
        let mut sorted = true;
        while let Some(e) = pp.pop_front() {
            let e = *e;
            if e != start {
                let mut prev = *ii.pop_front().unwrap();
                let mut q = start + 1;
                while q != e {
                    let r = *ii.pop_front().unwrap();
                    if r <= prev {
                        sorted = false;
                    }
                    prev = r;
                    q += 1;
                }
            }
            start = e;
        }
        sorted
    }

    /// The transpose (`ncols x nrows`, columns sorted). Upstream: `CsMatrix::transpose` (a
    /// counting sort by row; here a merge of the sorted columns, then a gather of the values).
    fn transpose(self: @CsMatrix<T>) -> CsMatrix<T> {
        let (tp, ti, src) = transpose_pattern(*self.nrows, *self.p, *self.i);
        CsMatrix {
            nrows: *self.ncols, ncols: *self.nrows, p: tp, i: ti, vals: gather(*self.vals, src),
        }
    }

    /// The column pointers, `ncols` of them (column `j` starts at `p()[j]`, the last one ends at
    /// `len()`). Upstream: `CsVecStorage::p` (`m.data.p()`; the storage is the matrix here).
    #[inline(always)]
    fn p(self: @CsMatrix<T>) -> Span<usize> {
        (*self.p).slice(0, *self.ncols)
    }

    /// The row index of every stored entry, column by column. Upstream: `CsVecStorage::i`
    /// (`m.data.i()`).
    #[inline(always)]
    fn i(self: @CsMatrix<T>) -> Span<usize> {
        *self.i
    }

    /// The stored values, column by column (the order `CsCholesky::decompose_*` expects).
    /// Upstream: `CsVecStorage::values` (`m.data.values()`).
    #[inline(always)]
    fn values(self: @CsMatrix<T>) -> Span<T> {
        *self.vals
    }

    /// Every stored value times `k` (one floor per value; the pattern is kept, zeros included).
    /// Upstream: `CsMatrix * T` (`impl Mul<T>`; Cairo's `Mul` is homogeneous, heterogeneous
    /// operators are named methods, DESIGN D4).
    fn scale(self: CsMatrix<T>, k: T) -> CsMatrix<T> {
        let mut out: Array<T> = array![];
        let mut vv = self.vals;
        while let Some(x) = vv.pop_front() {
            out.append(*x * k);
        }
        CsMatrix { vals: out.span(), ..self }
    }
}

/// `a + b`: the union of the patterns, the common entries added (exactly), sums that cancel kept
/// as explicit zeros (like upstream's). Panics with `nalgebra: dimension mismatch` unless the
/// shapes are equal. Upstream: `&a + &b` (`impl Add<CsMatrix>`).
pub impl CsMatrixAdd<
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
> of Add<CsMatrix<T>> {
    fn add(lhs: CsMatrix<T>, rhs: CsMatrix<T>) -> CsMatrix<T> {
        CsKernels::add(lhs, rhs)
    }
}

/// `a * b` (`a.ncols() == b.nrows()`, else panics with `nalgebra: dimension mismatch`): every
/// output entry is ONE exact sum of products floored once (upstream accumulates rounded
/// products), zero results dropped from the pattern like upstream's. Upstream: `&a * &b` (`impl
/// Mul<CsMatrix>`; a scatter per column there, a row-by-column gather on the pattern of the
/// column here).
pub impl CsMatrixMul<
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
> of Mul<CsMatrix<T>> {
    fn mul(lhs: CsMatrix<T>, rhs: CsMatrix<T>) -> CsMatrix<T> {
        CsKernels::mul(lhs, rhs)
    }
}

/// The dense `DMatrix` of a sparse matrix (absent entries are zeros). Upstream: `From<CsMatrix>
/// for OMatrix`.
pub impl CsMatrixIntoDMatrix<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<CsMatrix<T>, DMatrix<T>> {
    fn into(self: CsMatrix<T>) -> DMatrix<T> {
        DMatrix { data: CsKernels::dense_data(self), nrows: self.nrows, ncols: self.ncols }
    }
}

/// The sparse matrix of the non-zero components of a `DMatrix` (column by column). Upstream:
/// `From<Matrix> for CsMatrix`.
pub impl DMatrixIntoCsMatrix<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<DMatrix<T>, CsMatrix<T>> {
    fn into(self: DMatrix<T>) -> CsMatrix<T> {
        CsKernels::from_dense(self.data, self.nrows, self.ncols)
    }
}

/// The dense `DVector` of a one-column sparse matrix (a `CsVector`); panics with `nalgebra:
/// dimension mismatch` unless `ncols == 1`. Upstream: `From<CsMatrix> for OMatrix` (`DVector`
/// is `OMatrix<T, Dyn, U1>`).
pub impl CsMatrixIntoDVector<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<CsMatrix<T>, DVector<T>> {
    fn into(self: CsMatrix<T>) -> DVector<T> {
        check_shape(@self, self.nrows, 1);
        DVector { data: CsKernels::dense_data(self) }
    }
}

/// The one-column sparse matrix (a `CsVector`) of the non-zero components of a `DVector`.
/// Upstream: `From<Matrix> for CsMatrix`.
pub impl DVectorIntoCsMatrix<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<DVector<T>, CsMatrix<T>> {
    fn into(self: DVector<T>) -> CsMatrix<T> {
        CsKernels::from_dense(self.data, self.data.len(), 1)
    }
}

/// Panics with `nalgebra: dimension mismatch` unless `m` is `nrows x ncols`.
#[inline(always)]
fn check_shape<T>(m: @CsMatrix<T>, nrows: usize, ncols: usize) {
    if *m.nrows != nrows || *m.ncols != ncols {
        core::panic_with_felt252(errors::DIMENSION_MISMATCH);
    }
}

/// The column-major components of `m`, checked to be `nrows x ncols`, as one fixed-size array.
fn dense_of<T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>>(
    m: CsMatrix<T>, nrows: usize, ncols: usize,
) -> Span<T> {
    check_shape(@m, nrows, ncols);
    CsKernels::dense_data(m)
}

// The static shapes (upstream's generic `From<CsMatrix<T, R, C>> for OMatrix<T, R, C>` and
// `From<Matrix<T, R, C>> for CsMatrix<T, R, C>` at every static `R`, `C`): a dimension check and
// one fixed-size array; the other way through the `DMatrix` conversion.
// cs-static: begin
/// The `1x1` `Matrix1` of a sparse matrix; panics with `nalgebra: dimension mismatch` unless it
/// is `1x1`. Upstream: `From<CsMatrix> for OMatrix`.
pub impl CsMatrixIntoMatrix1<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<CsMatrix<T>, Matrix1<T>> {
    fn into(self: CsMatrix<T>) -> Matrix1<T> {
        let boxed: @Box<[T; 1]> = dense_of(self, 1, 1).try_into().unwrap();
        let [a0] = boxed.unbox();
        Matrix1 { x: a0 }
    }
}

/// The sparse matrix of the non-zero components of a `Matrix1`. Upstream: `From<Matrix> for
/// CsMatrix`.
pub impl Matrix1IntoCsMatrix<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<Matrix1<T>, CsMatrix<T>> {
    fn into(self: Matrix1<T>) -> CsMatrix<T> {
        let d: DMatrix<T> = self.into();
        CsKernels::from_dense(d.data, 1, 1)
    }
}

/// The `1x2` `RowVector2` of a sparse matrix; panics with `nalgebra: dimension mismatch` unless it
/// is `1x2`. Upstream: `From<CsMatrix> for OMatrix`.
pub impl CsMatrixIntoRowVector2<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<CsMatrix<T>, RowVector2<T>> {
    fn into(self: CsMatrix<T>) -> RowVector2<T> {
        let boxed: @Box<[T; 2]> = dense_of(self, 1, 2).try_into().unwrap();
        let [a0, a1] = boxed.unbox();
        RowVector2 { x: a0, y: a1 }
    }
}

/// The sparse matrix of the non-zero components of a `RowVector2`. Upstream: `From<Matrix> for
/// CsMatrix`.
pub impl RowVector2IntoCsMatrix<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<RowVector2<T>, CsMatrix<T>> {
    fn into(self: RowVector2<T>) -> CsMatrix<T> {
        let d: DMatrix<T> = self.into();
        CsKernels::from_dense(d.data, 1, 2)
    }
}

/// The `1x3` `RowVector3` of a sparse matrix; panics with `nalgebra: dimension mismatch` unless it
/// is `1x3`. Upstream: `From<CsMatrix> for OMatrix`.
pub impl CsMatrixIntoRowVector3<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<CsMatrix<T>, RowVector3<T>> {
    fn into(self: CsMatrix<T>) -> RowVector3<T> {
        let boxed: @Box<[T; 3]> = dense_of(self, 1, 3).try_into().unwrap();
        let [a0, a1, a2] = boxed.unbox();
        RowVector3 { x: a0, y: a1, z: a2 }
    }
}

/// The sparse matrix of the non-zero components of a `RowVector3`. Upstream: `From<Matrix> for
/// CsMatrix`.
pub impl RowVector3IntoCsMatrix<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<RowVector3<T>, CsMatrix<T>> {
    fn into(self: RowVector3<T>) -> CsMatrix<T> {
        let d: DMatrix<T> = self.into();
        CsKernels::from_dense(d.data, 1, 3)
    }
}

/// The `1x4` `RowVector4` of a sparse matrix; panics with `nalgebra: dimension mismatch` unless it
/// is `1x4`. Upstream: `From<CsMatrix> for OMatrix`.
pub impl CsMatrixIntoRowVector4<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<CsMatrix<T>, RowVector4<T>> {
    fn into(self: CsMatrix<T>) -> RowVector4<T> {
        let boxed: @Box<[T; 4]> = dense_of(self, 1, 4).try_into().unwrap();
        let [a0, a1, a2, a3] = boxed.unbox();
        RowVector4 { x: a0, y: a1, z: a2, w: a3 }
    }
}

/// The sparse matrix of the non-zero components of a `RowVector4`. Upstream: `From<Matrix> for
/// CsMatrix`.
pub impl RowVector4IntoCsMatrix<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<RowVector4<T>, CsMatrix<T>> {
    fn into(self: RowVector4<T>) -> CsMatrix<T> {
        let d: DMatrix<T> = self.into();
        CsKernels::from_dense(d.data, 1, 4)
    }
}

/// The `1x5` `RowVector5` of a sparse matrix; panics with `nalgebra: dimension mismatch` unless it
/// is `1x5`. Upstream: `From<CsMatrix> for OMatrix`.
pub impl CsMatrixIntoRowVector5<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<CsMatrix<T>, RowVector5<T>> {
    fn into(self: CsMatrix<T>) -> RowVector5<T> {
        let boxed: @Box<[T; 5]> = dense_of(self, 1, 5).try_into().unwrap();
        let [a0, a1, a2, a3, a4] = boxed.unbox();
        RowVector5 { x: a0, y: a1, z: a2, w: a3, a: a4 }
    }
}

/// The sparse matrix of the non-zero components of a `RowVector5`. Upstream: `From<Matrix> for
/// CsMatrix`.
pub impl RowVector5IntoCsMatrix<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<RowVector5<T>, CsMatrix<T>> {
    fn into(self: RowVector5<T>) -> CsMatrix<T> {
        let d: DMatrix<T> = self.into();
        CsKernels::from_dense(d.data, 1, 5)
    }
}

/// The `1x6` `RowVector6` of a sparse matrix; panics with `nalgebra: dimension mismatch` unless it
/// is `1x6`. Upstream: `From<CsMatrix> for OMatrix`.
pub impl CsMatrixIntoRowVector6<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<CsMatrix<T>, RowVector6<T>> {
    fn into(self: CsMatrix<T>) -> RowVector6<T> {
        let boxed: @Box<[T; 6]> = dense_of(self, 1, 6).try_into().unwrap();
        let [a0, a1, a2, a3, a4, a5] = boxed.unbox();
        RowVector6 { x: a0, y: a1, z: a2, w: a3, a: a4, b: a5 }
    }
}

/// The sparse matrix of the non-zero components of a `RowVector6`. Upstream: `From<Matrix> for
/// CsMatrix`.
pub impl RowVector6IntoCsMatrix<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<RowVector6<T>, CsMatrix<T>> {
    fn into(self: RowVector6<T>) -> CsMatrix<T> {
        let d: DMatrix<T> = self.into();
        CsKernels::from_dense(d.data, 1, 6)
    }
}

/// The `2x1` `Vector2` of a sparse matrix; panics with `nalgebra: dimension mismatch` unless it
/// is `2x1`. Upstream: `From<CsMatrix> for OMatrix`.
pub impl CsMatrixIntoVector2<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<CsMatrix<T>, Vector2<T>> {
    fn into(self: CsMatrix<T>) -> Vector2<T> {
        let boxed: @Box<[T; 2]> = dense_of(self, 2, 1).try_into().unwrap();
        let [a0, a1] = boxed.unbox();
        Vector2 { x: a0, y: a1 }
    }
}

/// The sparse matrix of the non-zero components of a `Vector2`. Upstream: `From<Matrix> for
/// CsMatrix`.
pub impl Vector2IntoCsMatrix<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<Vector2<T>, CsMatrix<T>> {
    fn into(self: Vector2<T>) -> CsMatrix<T> {
        let d: DMatrix<T> = self.into();
        CsKernels::from_dense(d.data, 2, 1)
    }
}

/// The `2x2` `Matrix2` of a sparse matrix; panics with `nalgebra: dimension mismatch` unless it
/// is `2x2`. Upstream: `From<CsMatrix> for OMatrix`.
pub impl CsMatrixIntoMatrix2<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<CsMatrix<T>, Matrix2<T>> {
    fn into(self: CsMatrix<T>) -> Matrix2<T> {
        let boxed: @Box<[T; 4]> = dense_of(self, 2, 2).try_into().unwrap();
        let [a0, a1, a2, a3] = boxed.unbox();
        Matrix2 { m11: a0, m21: a1, m12: a2, m22: a3 }
    }
}

/// The sparse matrix of the non-zero components of a `Matrix2`. Upstream: `From<Matrix> for
/// CsMatrix`.
pub impl Matrix2IntoCsMatrix<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<Matrix2<T>, CsMatrix<T>> {
    fn into(self: Matrix2<T>) -> CsMatrix<T> {
        let d: DMatrix<T> = self.into();
        CsKernels::from_dense(d.data, 2, 2)
    }
}

/// The `2x3` `Matrix2x3` of a sparse matrix; panics with `nalgebra: dimension mismatch` unless it
/// is `2x3`. Upstream: `From<CsMatrix> for OMatrix`.
pub impl CsMatrixIntoMatrix2x3<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<CsMatrix<T>, Matrix2x3<T>> {
    fn into(self: CsMatrix<T>) -> Matrix2x3<T> {
        let boxed: @Box<[T; 6]> = dense_of(self, 2, 3).try_into().unwrap();
        let [a0, a1, a2, a3, a4, a5] = boxed.unbox();
        Matrix2x3 { m11: a0, m21: a1, m12: a2, m22: a3, m13: a4, m23: a5 }
    }
}

/// The sparse matrix of the non-zero components of a `Matrix2x3`. Upstream: `From<Matrix> for
/// CsMatrix`.
pub impl Matrix2x3IntoCsMatrix<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<Matrix2x3<T>, CsMatrix<T>> {
    fn into(self: Matrix2x3<T>) -> CsMatrix<T> {
        let d: DMatrix<T> = self.into();
        CsKernels::from_dense(d.data, 2, 3)
    }
}

/// The `2x4` `Matrix2x4` of a sparse matrix; panics with `nalgebra: dimension mismatch` unless it
/// is `2x4`. Upstream: `From<CsMatrix> for OMatrix`.
pub impl CsMatrixIntoMatrix2x4<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<CsMatrix<T>, Matrix2x4<T>> {
    fn into(self: CsMatrix<T>) -> Matrix2x4<T> {
        let boxed: @Box<[T; 8]> = dense_of(self, 2, 4).try_into().unwrap();
        let [a0, a1, a2, a3, a4, a5, a6, a7] = boxed.unbox();
        Matrix2x4 { m11: a0, m21: a1, m12: a2, m22: a3, m13: a4, m23: a5, m14: a6, m24: a7 }
    }
}

/// The sparse matrix of the non-zero components of a `Matrix2x4`. Upstream: `From<Matrix> for
/// CsMatrix`.
pub impl Matrix2x4IntoCsMatrix<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<Matrix2x4<T>, CsMatrix<T>> {
    fn into(self: Matrix2x4<T>) -> CsMatrix<T> {
        let d: DMatrix<T> = self.into();
        CsKernels::from_dense(d.data, 2, 4)
    }
}

/// The `2x5` `Matrix2x5` of a sparse matrix; panics with `nalgebra: dimension mismatch` unless it
/// is `2x5`. Upstream: `From<CsMatrix> for OMatrix`.
pub impl CsMatrixIntoMatrix2x5<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<CsMatrix<T>, Matrix2x5<T>> {
    fn into(self: CsMatrix<T>) -> Matrix2x5<T> {
        let boxed: @Box<[T; 10]> = dense_of(self, 2, 5).try_into().unwrap();
        let [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9] = boxed.unbox();
        Matrix2x5 {
            m11: a0,
            m21: a1,
            m12: a2,
            m22: a3,
            m13: a4,
            m23: a5,
            m14: a6,
            m24: a7,
            m15: a8,
            m25: a9,
        }
    }
}

/// The sparse matrix of the non-zero components of a `Matrix2x5`. Upstream: `From<Matrix> for
/// CsMatrix`.
pub impl Matrix2x5IntoCsMatrix<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<Matrix2x5<T>, CsMatrix<T>> {
    fn into(self: Matrix2x5<T>) -> CsMatrix<T> {
        let d: DMatrix<T> = self.into();
        CsKernels::from_dense(d.data, 2, 5)
    }
}

/// The `2x6` `Matrix2x6` of a sparse matrix; panics with `nalgebra: dimension mismatch` unless it
/// is `2x6`. Upstream: `From<CsMatrix> for OMatrix`.
pub impl CsMatrixIntoMatrix2x6<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<CsMatrix<T>, Matrix2x6<T>> {
    fn into(self: CsMatrix<T>) -> Matrix2x6<T> {
        let boxed: @Box<[T; 12]> = dense_of(self, 2, 6).try_into().unwrap();
        let [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11] = boxed.unbox();
        Matrix2x6 {
            m11: a0,
            m21: a1,
            m12: a2,
            m22: a3,
            m13: a4,
            m23: a5,
            m14: a6,
            m24: a7,
            m15: a8,
            m25: a9,
            m16: a10,
            m26: a11,
        }
    }
}

/// The sparse matrix of the non-zero components of a `Matrix2x6`. Upstream: `From<Matrix> for
/// CsMatrix`.
pub impl Matrix2x6IntoCsMatrix<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<Matrix2x6<T>, CsMatrix<T>> {
    fn into(self: Matrix2x6<T>) -> CsMatrix<T> {
        let d: DMatrix<T> = self.into();
        CsKernels::from_dense(d.data, 2, 6)
    }
}

/// The `3x1` `Vector3` of a sparse matrix; panics with `nalgebra: dimension mismatch` unless it
/// is `3x1`. Upstream: `From<CsMatrix> for OMatrix`.
pub impl CsMatrixIntoVector3<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<CsMatrix<T>, Vector3<T>> {
    fn into(self: CsMatrix<T>) -> Vector3<T> {
        let boxed: @Box<[T; 3]> = dense_of(self, 3, 1).try_into().unwrap();
        let [a0, a1, a2] = boxed.unbox();
        Vector3 { x: a0, y: a1, z: a2 }
    }
}

/// The sparse matrix of the non-zero components of a `Vector3`. Upstream: `From<Matrix> for
/// CsMatrix`.
pub impl Vector3IntoCsMatrix<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<Vector3<T>, CsMatrix<T>> {
    fn into(self: Vector3<T>) -> CsMatrix<T> {
        let d: DMatrix<T> = self.into();
        CsKernels::from_dense(d.data, 3, 1)
    }
}

/// The `3x2` `Matrix3x2` of a sparse matrix; panics with `nalgebra: dimension mismatch` unless it
/// is `3x2`. Upstream: `From<CsMatrix> for OMatrix`.
pub impl CsMatrixIntoMatrix3x2<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<CsMatrix<T>, Matrix3x2<T>> {
    fn into(self: CsMatrix<T>) -> Matrix3x2<T> {
        let boxed: @Box<[T; 6]> = dense_of(self, 3, 2).try_into().unwrap();
        let [a0, a1, a2, a3, a4, a5] = boxed.unbox();
        Matrix3x2 { m11: a0, m21: a1, m31: a2, m12: a3, m22: a4, m32: a5 }
    }
}

/// The sparse matrix of the non-zero components of a `Matrix3x2`. Upstream: `From<Matrix> for
/// CsMatrix`.
pub impl Matrix3x2IntoCsMatrix<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<Matrix3x2<T>, CsMatrix<T>> {
    fn into(self: Matrix3x2<T>) -> CsMatrix<T> {
        let d: DMatrix<T> = self.into();
        CsKernels::from_dense(d.data, 3, 2)
    }
}

/// The `3x3` `Matrix3` of a sparse matrix; panics with `nalgebra: dimension mismatch` unless it
/// is `3x3`. Upstream: `From<CsMatrix> for OMatrix`.
pub impl CsMatrixIntoMatrix3<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<CsMatrix<T>, Matrix3<T>> {
    fn into(self: CsMatrix<T>) -> Matrix3<T> {
        let boxed: @Box<[T; 9]> = dense_of(self, 3, 3).try_into().unwrap();
        let [a0, a1, a2, a3, a4, a5, a6, a7, a8] = boxed.unbox();
        Matrix3 { m11: a0, m21: a1, m31: a2, m12: a3, m22: a4, m32: a5, m13: a6, m23: a7, m33: a8 }
    }
}

/// The sparse matrix of the non-zero components of a `Matrix3`. Upstream: `From<Matrix> for
/// CsMatrix`.
pub impl Matrix3IntoCsMatrix<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<Matrix3<T>, CsMatrix<T>> {
    fn into(self: Matrix3<T>) -> CsMatrix<T> {
        let d: DMatrix<T> = self.into();
        CsKernels::from_dense(d.data, 3, 3)
    }
}

/// The `3x4` `Matrix3x4` of a sparse matrix; panics with `nalgebra: dimension mismatch` unless it
/// is `3x4`. Upstream: `From<CsMatrix> for OMatrix`.
pub impl CsMatrixIntoMatrix3x4<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<CsMatrix<T>, Matrix3x4<T>> {
    fn into(self: CsMatrix<T>) -> Matrix3x4<T> {
        let boxed: @Box<[T; 12]> = dense_of(self, 3, 4).try_into().unwrap();
        let [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11] = boxed.unbox();
        Matrix3x4 {
            m11: a0,
            m21: a1,
            m31: a2,
            m12: a3,
            m22: a4,
            m32: a5,
            m13: a6,
            m23: a7,
            m33: a8,
            m14: a9,
            m24: a10,
            m34: a11,
        }
    }
}

/// The sparse matrix of the non-zero components of a `Matrix3x4`. Upstream: `From<Matrix> for
/// CsMatrix`.
pub impl Matrix3x4IntoCsMatrix<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<Matrix3x4<T>, CsMatrix<T>> {
    fn into(self: Matrix3x4<T>) -> CsMatrix<T> {
        let d: DMatrix<T> = self.into();
        CsKernels::from_dense(d.data, 3, 4)
    }
}

/// The `3x5` `Matrix3x5` of a sparse matrix; panics with `nalgebra: dimension mismatch` unless it
/// is `3x5`. Upstream: `From<CsMatrix> for OMatrix`.
pub impl CsMatrixIntoMatrix3x5<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<CsMatrix<T>, Matrix3x5<T>> {
    fn into(self: CsMatrix<T>) -> Matrix3x5<T> {
        let boxed: @Box<[T; 15]> = dense_of(self, 3, 5).try_into().unwrap();
        let [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14] = boxed.unbox();
        Matrix3x5 {
            m11: a0,
            m21: a1,
            m31: a2,
            m12: a3,
            m22: a4,
            m32: a5,
            m13: a6,
            m23: a7,
            m33: a8,
            m14: a9,
            m24: a10,
            m34: a11,
            m15: a12,
            m25: a13,
            m35: a14,
        }
    }
}

/// The sparse matrix of the non-zero components of a `Matrix3x5`. Upstream: `From<Matrix> for
/// CsMatrix`.
pub impl Matrix3x5IntoCsMatrix<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<Matrix3x5<T>, CsMatrix<T>> {
    fn into(self: Matrix3x5<T>) -> CsMatrix<T> {
        let d: DMatrix<T> = self.into();
        CsKernels::from_dense(d.data, 3, 5)
    }
}

/// The `3x6` `Matrix3x6` of a sparse matrix; panics with `nalgebra: dimension mismatch` unless it
/// is `3x6`. Upstream: `From<CsMatrix> for OMatrix`.
pub impl CsMatrixIntoMatrix3x6<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<CsMatrix<T>, Matrix3x6<T>> {
    fn into(self: CsMatrix<T>) -> Matrix3x6<T> {
        let boxed: @Box<[T; 18]> = dense_of(self, 3, 6).try_into().unwrap();
        let [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17] = boxed
            .unbox();
        Matrix3x6 {
            m11: a0,
            m21: a1,
            m31: a2,
            m12: a3,
            m22: a4,
            m32: a5,
            m13: a6,
            m23: a7,
            m33: a8,
            m14: a9,
            m24: a10,
            m34: a11,
            m15: a12,
            m25: a13,
            m35: a14,
            m16: a15,
            m26: a16,
            m36: a17,
        }
    }
}

/// The sparse matrix of the non-zero components of a `Matrix3x6`. Upstream: `From<Matrix> for
/// CsMatrix`.
pub impl Matrix3x6IntoCsMatrix<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<Matrix3x6<T>, CsMatrix<T>> {
    fn into(self: Matrix3x6<T>) -> CsMatrix<T> {
        let d: DMatrix<T> = self.into();
        CsKernels::from_dense(d.data, 3, 6)
    }
}

/// The `4x1` `Vector4` of a sparse matrix; panics with `nalgebra: dimension mismatch` unless it
/// is `4x1`. Upstream: `From<CsMatrix> for OMatrix`.
pub impl CsMatrixIntoVector4<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<CsMatrix<T>, Vector4<T>> {
    fn into(self: CsMatrix<T>) -> Vector4<T> {
        let boxed: @Box<[T; 4]> = dense_of(self, 4, 1).try_into().unwrap();
        let [a0, a1, a2, a3] = boxed.unbox();
        Vector4 { x: a0, y: a1, z: a2, w: a3 }
    }
}

/// The sparse matrix of the non-zero components of a `Vector4`. Upstream: `From<Matrix> for
/// CsMatrix`.
pub impl Vector4IntoCsMatrix<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<Vector4<T>, CsMatrix<T>> {
    fn into(self: Vector4<T>) -> CsMatrix<T> {
        let d: DMatrix<T> = self.into();
        CsKernels::from_dense(d.data, 4, 1)
    }
}

/// The `4x2` `Matrix4x2` of a sparse matrix; panics with `nalgebra: dimension mismatch` unless it
/// is `4x2`. Upstream: `From<CsMatrix> for OMatrix`.
pub impl CsMatrixIntoMatrix4x2<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<CsMatrix<T>, Matrix4x2<T>> {
    fn into(self: CsMatrix<T>) -> Matrix4x2<T> {
        let boxed: @Box<[T; 8]> = dense_of(self, 4, 2).try_into().unwrap();
        let [a0, a1, a2, a3, a4, a5, a6, a7] = boxed.unbox();
        Matrix4x2 { m11: a0, m21: a1, m31: a2, m41: a3, m12: a4, m22: a5, m32: a6, m42: a7 }
    }
}

/// The sparse matrix of the non-zero components of a `Matrix4x2`. Upstream: `From<Matrix> for
/// CsMatrix`.
pub impl Matrix4x2IntoCsMatrix<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<Matrix4x2<T>, CsMatrix<T>> {
    fn into(self: Matrix4x2<T>) -> CsMatrix<T> {
        let d: DMatrix<T> = self.into();
        CsKernels::from_dense(d.data, 4, 2)
    }
}

/// The `4x3` `Matrix4x3` of a sparse matrix; panics with `nalgebra: dimension mismatch` unless it
/// is `4x3`. Upstream: `From<CsMatrix> for OMatrix`.
pub impl CsMatrixIntoMatrix4x3<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<CsMatrix<T>, Matrix4x3<T>> {
    fn into(self: CsMatrix<T>) -> Matrix4x3<T> {
        let boxed: @Box<[T; 12]> = dense_of(self, 4, 3).try_into().unwrap();
        let [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11] = boxed.unbox();
        Matrix4x3 {
            m11: a0,
            m21: a1,
            m31: a2,
            m41: a3,
            m12: a4,
            m22: a5,
            m32: a6,
            m42: a7,
            m13: a8,
            m23: a9,
            m33: a10,
            m43: a11,
        }
    }
}

/// The sparse matrix of the non-zero components of a `Matrix4x3`. Upstream: `From<Matrix> for
/// CsMatrix`.
pub impl Matrix4x3IntoCsMatrix<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<Matrix4x3<T>, CsMatrix<T>> {
    fn into(self: Matrix4x3<T>) -> CsMatrix<T> {
        let d: DMatrix<T> = self.into();
        CsKernels::from_dense(d.data, 4, 3)
    }
}

/// The `4x4` `Matrix4` of a sparse matrix; panics with `nalgebra: dimension mismatch` unless it
/// is `4x4`. Upstream: `From<CsMatrix> for OMatrix`.
pub impl CsMatrixIntoMatrix4<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<CsMatrix<T>, Matrix4<T>> {
    fn into(self: CsMatrix<T>) -> Matrix4<T> {
        let boxed: @Box<[T; 16]> = dense_of(self, 4, 4).try_into().unwrap();
        let [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15] = boxed.unbox();
        Matrix4 {
            m11: a0,
            m21: a1,
            m31: a2,
            m41: a3,
            m12: a4,
            m22: a5,
            m32: a6,
            m42: a7,
            m13: a8,
            m23: a9,
            m33: a10,
            m43: a11,
            m14: a12,
            m24: a13,
            m34: a14,
            m44: a15,
        }
    }
}

/// The sparse matrix of the non-zero components of a `Matrix4`. Upstream: `From<Matrix> for
/// CsMatrix`.
pub impl Matrix4IntoCsMatrix<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<Matrix4<T>, CsMatrix<T>> {
    fn into(self: Matrix4<T>) -> CsMatrix<T> {
        let d: DMatrix<T> = self.into();
        CsKernels::from_dense(d.data, 4, 4)
    }
}

/// The `4x5` `Matrix4x5` of a sparse matrix; panics with `nalgebra: dimension mismatch` unless it
/// is `4x5`. Upstream: `From<CsMatrix> for OMatrix`.
pub impl CsMatrixIntoMatrix4x5<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<CsMatrix<T>, Matrix4x5<T>> {
    fn into(self: CsMatrix<T>) -> Matrix4x5<T> {
        let boxed: @Box<[T; 20]> = dense_of(self, 4, 5).try_into().unwrap();
        let [
            a0,
            a1,
            a2,
            a3,
            a4,
            a5,
            a6,
            a7,
            a8,
            a9,
            a10,
            a11,
            a12,
            a13,
            a14,
            a15,
            a16,
            a17,
            a18,
            a19,
        ] =
            boxed
            .unbox();
        Matrix4x5 {
            m11: a0,
            m21: a1,
            m31: a2,
            m41: a3,
            m12: a4,
            m22: a5,
            m32: a6,
            m42: a7,
            m13: a8,
            m23: a9,
            m33: a10,
            m43: a11,
            m14: a12,
            m24: a13,
            m34: a14,
            m44: a15,
            m15: a16,
            m25: a17,
            m35: a18,
            m45: a19,
        }
    }
}

/// The sparse matrix of the non-zero components of a `Matrix4x5`. Upstream: `From<Matrix> for
/// CsMatrix`.
pub impl Matrix4x5IntoCsMatrix<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<Matrix4x5<T>, CsMatrix<T>> {
    fn into(self: Matrix4x5<T>) -> CsMatrix<T> {
        let d: DMatrix<T> = self.into();
        CsKernels::from_dense(d.data, 4, 5)
    }
}

/// The `4x6` `Matrix4x6` of a sparse matrix; panics with `nalgebra: dimension mismatch` unless it
/// is `4x6`. Upstream: `From<CsMatrix> for OMatrix`.
pub impl CsMatrixIntoMatrix4x6<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<CsMatrix<T>, Matrix4x6<T>> {
    fn into(self: CsMatrix<T>) -> Matrix4x6<T> {
        let boxed: @Box<[T; 24]> = dense_of(self, 4, 6).try_into().unwrap();
        let [
            a0,
            a1,
            a2,
            a3,
            a4,
            a5,
            a6,
            a7,
            a8,
            a9,
            a10,
            a11,
            a12,
            a13,
            a14,
            a15,
            a16,
            a17,
            a18,
            a19,
            a20,
            a21,
            a22,
            a23,
        ] =
            boxed
            .unbox();
        Matrix4x6 {
            m11: a0,
            m21: a1,
            m31: a2,
            m41: a3,
            m12: a4,
            m22: a5,
            m32: a6,
            m42: a7,
            m13: a8,
            m23: a9,
            m33: a10,
            m43: a11,
            m14: a12,
            m24: a13,
            m34: a14,
            m44: a15,
            m15: a16,
            m25: a17,
            m35: a18,
            m45: a19,
            m16: a20,
            m26: a21,
            m36: a22,
            m46: a23,
        }
    }
}

/// The sparse matrix of the non-zero components of a `Matrix4x6`. Upstream: `From<Matrix> for
/// CsMatrix`.
pub impl Matrix4x6IntoCsMatrix<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<Matrix4x6<T>, CsMatrix<T>> {
    fn into(self: Matrix4x6<T>) -> CsMatrix<T> {
        let d: DMatrix<T> = self.into();
        CsKernels::from_dense(d.data, 4, 6)
    }
}

/// The `5x1` `Vector5` of a sparse matrix; panics with `nalgebra: dimension mismatch` unless it
/// is `5x1`. Upstream: `From<CsMatrix> for OMatrix`.
pub impl CsMatrixIntoVector5<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<CsMatrix<T>, Vector5<T>> {
    fn into(self: CsMatrix<T>) -> Vector5<T> {
        let boxed: @Box<[T; 5]> = dense_of(self, 5, 1).try_into().unwrap();
        let [a0, a1, a2, a3, a4] = boxed.unbox();
        Vector5 { x: a0, y: a1, z: a2, w: a3, a: a4 }
    }
}

/// The sparse matrix of the non-zero components of a `Vector5`. Upstream: `From<Matrix> for
/// CsMatrix`.
pub impl Vector5IntoCsMatrix<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<Vector5<T>, CsMatrix<T>> {
    fn into(self: Vector5<T>) -> CsMatrix<T> {
        let d: DMatrix<T> = self.into();
        CsKernels::from_dense(d.data, 5, 1)
    }
}

/// The `5x2` `Matrix5x2` of a sparse matrix; panics with `nalgebra: dimension mismatch` unless it
/// is `5x2`. Upstream: `From<CsMatrix> for OMatrix`.
pub impl CsMatrixIntoMatrix5x2<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<CsMatrix<T>, Matrix5x2<T>> {
    fn into(self: CsMatrix<T>) -> Matrix5x2<T> {
        let boxed: @Box<[T; 10]> = dense_of(self, 5, 2).try_into().unwrap();
        let [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9] = boxed.unbox();
        Matrix5x2 {
            m11: a0,
            m21: a1,
            m31: a2,
            m41: a3,
            m51: a4,
            m12: a5,
            m22: a6,
            m32: a7,
            m42: a8,
            m52: a9,
        }
    }
}

/// The sparse matrix of the non-zero components of a `Matrix5x2`. Upstream: `From<Matrix> for
/// CsMatrix`.
pub impl Matrix5x2IntoCsMatrix<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<Matrix5x2<T>, CsMatrix<T>> {
    fn into(self: Matrix5x2<T>) -> CsMatrix<T> {
        let d: DMatrix<T> = self.into();
        CsKernels::from_dense(d.data, 5, 2)
    }
}

/// The `5x3` `Matrix5x3` of a sparse matrix; panics with `nalgebra: dimension mismatch` unless it
/// is `5x3`. Upstream: `From<CsMatrix> for OMatrix`.
pub impl CsMatrixIntoMatrix5x3<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<CsMatrix<T>, Matrix5x3<T>> {
    fn into(self: CsMatrix<T>) -> Matrix5x3<T> {
        let boxed: @Box<[T; 15]> = dense_of(self, 5, 3).try_into().unwrap();
        let [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14] = boxed.unbox();
        Matrix5x3 {
            m11: a0,
            m21: a1,
            m31: a2,
            m41: a3,
            m51: a4,
            m12: a5,
            m22: a6,
            m32: a7,
            m42: a8,
            m52: a9,
            m13: a10,
            m23: a11,
            m33: a12,
            m43: a13,
            m53: a14,
        }
    }
}

/// The sparse matrix of the non-zero components of a `Matrix5x3`. Upstream: `From<Matrix> for
/// CsMatrix`.
pub impl Matrix5x3IntoCsMatrix<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<Matrix5x3<T>, CsMatrix<T>> {
    fn into(self: Matrix5x3<T>) -> CsMatrix<T> {
        let d: DMatrix<T> = self.into();
        CsKernels::from_dense(d.data, 5, 3)
    }
}

/// The `5x4` `Matrix5x4` of a sparse matrix; panics with `nalgebra: dimension mismatch` unless it
/// is `5x4`. Upstream: `From<CsMatrix> for OMatrix`.
pub impl CsMatrixIntoMatrix5x4<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<CsMatrix<T>, Matrix5x4<T>> {
    fn into(self: CsMatrix<T>) -> Matrix5x4<T> {
        let boxed: @Box<[T; 20]> = dense_of(self, 5, 4).try_into().unwrap();
        let [
            a0,
            a1,
            a2,
            a3,
            a4,
            a5,
            a6,
            a7,
            a8,
            a9,
            a10,
            a11,
            a12,
            a13,
            a14,
            a15,
            a16,
            a17,
            a18,
            a19,
        ] =
            boxed
            .unbox();
        Matrix5x4 {
            m11: a0,
            m21: a1,
            m31: a2,
            m41: a3,
            m51: a4,
            m12: a5,
            m22: a6,
            m32: a7,
            m42: a8,
            m52: a9,
            m13: a10,
            m23: a11,
            m33: a12,
            m43: a13,
            m53: a14,
            m14: a15,
            m24: a16,
            m34: a17,
            m44: a18,
            m54: a19,
        }
    }
}

/// The sparse matrix of the non-zero components of a `Matrix5x4`. Upstream: `From<Matrix> for
/// CsMatrix`.
pub impl Matrix5x4IntoCsMatrix<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<Matrix5x4<T>, CsMatrix<T>> {
    fn into(self: Matrix5x4<T>) -> CsMatrix<T> {
        let d: DMatrix<T> = self.into();
        CsKernels::from_dense(d.data, 5, 4)
    }
}

/// The `5x5` `Matrix5` of a sparse matrix; panics with `nalgebra: dimension mismatch` unless it
/// is `5x5`. Upstream: `From<CsMatrix> for OMatrix`.
pub impl CsMatrixIntoMatrix5<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<CsMatrix<T>, Matrix5<T>> {
    fn into(self: CsMatrix<T>) -> Matrix5<T> {
        let boxed: @Box<[T; 25]> = dense_of(self, 5, 5).try_into().unwrap();
        let [
            a0,
            a1,
            a2,
            a3,
            a4,
            a5,
            a6,
            a7,
            a8,
            a9,
            a10,
            a11,
            a12,
            a13,
            a14,
            a15,
            a16,
            a17,
            a18,
            a19,
            a20,
            a21,
            a22,
            a23,
            a24,
        ] =
            boxed
            .unbox();
        Matrix5 {
            m11: a0,
            m21: a1,
            m31: a2,
            m41: a3,
            m51: a4,
            m12: a5,
            m22: a6,
            m32: a7,
            m42: a8,
            m52: a9,
            m13: a10,
            m23: a11,
            m33: a12,
            m43: a13,
            m53: a14,
            m14: a15,
            m24: a16,
            m34: a17,
            m44: a18,
            m54: a19,
            m15: a20,
            m25: a21,
            m35: a22,
            m45: a23,
            m55: a24,
        }
    }
}

/// The sparse matrix of the non-zero components of a `Matrix5`. Upstream: `From<Matrix> for
/// CsMatrix`.
pub impl Matrix5IntoCsMatrix<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<Matrix5<T>, CsMatrix<T>> {
    fn into(self: Matrix5<T>) -> CsMatrix<T> {
        let d: DMatrix<T> = self.into();
        CsKernels::from_dense(d.data, 5, 5)
    }
}

/// The `5x6` `Matrix5x6` of a sparse matrix; panics with `nalgebra: dimension mismatch` unless it
/// is `5x6`. Upstream: `From<CsMatrix> for OMatrix`.
pub impl CsMatrixIntoMatrix5x6<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<CsMatrix<T>, Matrix5x6<T>> {
    fn into(self: CsMatrix<T>) -> Matrix5x6<T> {
        let boxed: @Box<[T; 30]> = dense_of(self, 5, 6).try_into().unwrap();
        let [
            a0,
            a1,
            a2,
            a3,
            a4,
            a5,
            a6,
            a7,
            a8,
            a9,
            a10,
            a11,
            a12,
            a13,
            a14,
            a15,
            a16,
            a17,
            a18,
            a19,
            a20,
            a21,
            a22,
            a23,
            a24,
            a25,
            a26,
            a27,
            a28,
            a29,
        ] =
            boxed
            .unbox();
        Matrix5x6 {
            m11: a0,
            m21: a1,
            m31: a2,
            m41: a3,
            m51: a4,
            m12: a5,
            m22: a6,
            m32: a7,
            m42: a8,
            m52: a9,
            m13: a10,
            m23: a11,
            m33: a12,
            m43: a13,
            m53: a14,
            m14: a15,
            m24: a16,
            m34: a17,
            m44: a18,
            m54: a19,
            m15: a20,
            m25: a21,
            m35: a22,
            m45: a23,
            m55: a24,
            m16: a25,
            m26: a26,
            m36: a27,
            m46: a28,
            m56: a29,
        }
    }
}

/// The sparse matrix of the non-zero components of a `Matrix5x6`. Upstream: `From<Matrix> for
/// CsMatrix`.
pub impl Matrix5x6IntoCsMatrix<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<Matrix5x6<T>, CsMatrix<T>> {
    fn into(self: Matrix5x6<T>) -> CsMatrix<T> {
        let d: DMatrix<T> = self.into();
        CsKernels::from_dense(d.data, 5, 6)
    }
}

/// The `6x1` `Vector6` of a sparse matrix; panics with `nalgebra: dimension mismatch` unless it
/// is `6x1`. Upstream: `From<CsMatrix> for OMatrix`.
pub impl CsMatrixIntoVector6<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<CsMatrix<T>, Vector6<T>> {
    fn into(self: CsMatrix<T>) -> Vector6<T> {
        let boxed: @Box<[T; 6]> = dense_of(self, 6, 1).try_into().unwrap();
        let [a0, a1, a2, a3, a4, a5] = boxed.unbox();
        Vector6 { x: a0, y: a1, z: a2, w: a3, a: a4, b: a5 }
    }
}

/// The sparse matrix of the non-zero components of a `Vector6`. Upstream: `From<Matrix> for
/// CsMatrix`.
pub impl Vector6IntoCsMatrix<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<Vector6<T>, CsMatrix<T>> {
    fn into(self: Vector6<T>) -> CsMatrix<T> {
        let d: DMatrix<T> = self.into();
        CsKernels::from_dense(d.data, 6, 1)
    }
}

/// The `6x2` `Matrix6x2` of a sparse matrix; panics with `nalgebra: dimension mismatch` unless it
/// is `6x2`. Upstream: `From<CsMatrix> for OMatrix`.
pub impl CsMatrixIntoMatrix6x2<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<CsMatrix<T>, Matrix6x2<T>> {
    fn into(self: CsMatrix<T>) -> Matrix6x2<T> {
        let boxed: @Box<[T; 12]> = dense_of(self, 6, 2).try_into().unwrap();
        let [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11] = boxed.unbox();
        Matrix6x2 {
            m11: a0,
            m21: a1,
            m31: a2,
            m41: a3,
            m51: a4,
            m61: a5,
            m12: a6,
            m22: a7,
            m32: a8,
            m42: a9,
            m52: a10,
            m62: a11,
        }
    }
}

/// The sparse matrix of the non-zero components of a `Matrix6x2`. Upstream: `From<Matrix> for
/// CsMatrix`.
pub impl Matrix6x2IntoCsMatrix<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<Matrix6x2<T>, CsMatrix<T>> {
    fn into(self: Matrix6x2<T>) -> CsMatrix<T> {
        let d: DMatrix<T> = self.into();
        CsKernels::from_dense(d.data, 6, 2)
    }
}

/// The `6x3` `Matrix6x3` of a sparse matrix; panics with `nalgebra: dimension mismatch` unless it
/// is `6x3`. Upstream: `From<CsMatrix> for OMatrix`.
pub impl CsMatrixIntoMatrix6x3<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<CsMatrix<T>, Matrix6x3<T>> {
    fn into(self: CsMatrix<T>) -> Matrix6x3<T> {
        let boxed: @Box<[T; 18]> = dense_of(self, 6, 3).try_into().unwrap();
        let [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17] = boxed
            .unbox();
        Matrix6x3 {
            m11: a0,
            m21: a1,
            m31: a2,
            m41: a3,
            m51: a4,
            m61: a5,
            m12: a6,
            m22: a7,
            m32: a8,
            m42: a9,
            m52: a10,
            m62: a11,
            m13: a12,
            m23: a13,
            m33: a14,
            m43: a15,
            m53: a16,
            m63: a17,
        }
    }
}

/// The sparse matrix of the non-zero components of a `Matrix6x3`. Upstream: `From<Matrix> for
/// CsMatrix`.
pub impl Matrix6x3IntoCsMatrix<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<Matrix6x3<T>, CsMatrix<T>> {
    fn into(self: Matrix6x3<T>) -> CsMatrix<T> {
        let d: DMatrix<T> = self.into();
        CsKernels::from_dense(d.data, 6, 3)
    }
}

/// The `6x4` `Matrix6x4` of a sparse matrix; panics with `nalgebra: dimension mismatch` unless it
/// is `6x4`. Upstream: `From<CsMatrix> for OMatrix`.
pub impl CsMatrixIntoMatrix6x4<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<CsMatrix<T>, Matrix6x4<T>> {
    fn into(self: CsMatrix<T>) -> Matrix6x4<T> {
        let boxed: @Box<[T; 24]> = dense_of(self, 6, 4).try_into().unwrap();
        let [
            a0,
            a1,
            a2,
            a3,
            a4,
            a5,
            a6,
            a7,
            a8,
            a9,
            a10,
            a11,
            a12,
            a13,
            a14,
            a15,
            a16,
            a17,
            a18,
            a19,
            a20,
            a21,
            a22,
            a23,
        ] =
            boxed
            .unbox();
        Matrix6x4 {
            m11: a0,
            m21: a1,
            m31: a2,
            m41: a3,
            m51: a4,
            m61: a5,
            m12: a6,
            m22: a7,
            m32: a8,
            m42: a9,
            m52: a10,
            m62: a11,
            m13: a12,
            m23: a13,
            m33: a14,
            m43: a15,
            m53: a16,
            m63: a17,
            m14: a18,
            m24: a19,
            m34: a20,
            m44: a21,
            m54: a22,
            m64: a23,
        }
    }
}

/// The sparse matrix of the non-zero components of a `Matrix6x4`. Upstream: `From<Matrix> for
/// CsMatrix`.
pub impl Matrix6x4IntoCsMatrix<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<Matrix6x4<T>, CsMatrix<T>> {
    fn into(self: Matrix6x4<T>) -> CsMatrix<T> {
        let d: DMatrix<T> = self.into();
        CsKernels::from_dense(d.data, 6, 4)
    }
}

/// The `6x5` `Matrix6x5` of a sparse matrix; panics with `nalgebra: dimension mismatch` unless it
/// is `6x5`. Upstream: `From<CsMatrix> for OMatrix`.
pub impl CsMatrixIntoMatrix6x5<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<CsMatrix<T>, Matrix6x5<T>> {
    fn into(self: CsMatrix<T>) -> Matrix6x5<T> {
        let boxed: @Box<[T; 30]> = dense_of(self, 6, 5).try_into().unwrap();
        let [
            a0,
            a1,
            a2,
            a3,
            a4,
            a5,
            a6,
            a7,
            a8,
            a9,
            a10,
            a11,
            a12,
            a13,
            a14,
            a15,
            a16,
            a17,
            a18,
            a19,
            a20,
            a21,
            a22,
            a23,
            a24,
            a25,
            a26,
            a27,
            a28,
            a29,
        ] =
            boxed
            .unbox();
        Matrix6x5 {
            m11: a0,
            m21: a1,
            m31: a2,
            m41: a3,
            m51: a4,
            m61: a5,
            m12: a6,
            m22: a7,
            m32: a8,
            m42: a9,
            m52: a10,
            m62: a11,
            m13: a12,
            m23: a13,
            m33: a14,
            m43: a15,
            m53: a16,
            m63: a17,
            m14: a18,
            m24: a19,
            m34: a20,
            m44: a21,
            m54: a22,
            m64: a23,
            m15: a24,
            m25: a25,
            m35: a26,
            m45: a27,
            m55: a28,
            m65: a29,
        }
    }
}

/// The sparse matrix of the non-zero components of a `Matrix6x5`. Upstream: `From<Matrix> for
/// CsMatrix`.
pub impl Matrix6x5IntoCsMatrix<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<Matrix6x5<T>, CsMatrix<T>> {
    fn into(self: Matrix6x5<T>) -> CsMatrix<T> {
        let d: DMatrix<T> = self.into();
        CsKernels::from_dense(d.data, 6, 5)
    }
}

/// The `6x6` `Matrix6` of a sparse matrix; panics with `nalgebra: dimension mismatch` unless it
/// is `6x6`. Upstream: `From<CsMatrix> for OMatrix`.
pub impl CsMatrixIntoMatrix6<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<CsMatrix<T>, Matrix6<T>> {
    fn into(self: CsMatrix<T>) -> Matrix6<T> {
        let boxed: @Box<[T; 36]> = dense_of(self, 6, 6).try_into().unwrap();
        let [
            a0,
            a1,
            a2,
            a3,
            a4,
            a5,
            a6,
            a7,
            a8,
            a9,
            a10,
            a11,
            a12,
            a13,
            a14,
            a15,
            a16,
            a17,
            a18,
            a19,
            a20,
            a21,
            a22,
            a23,
            a24,
            a25,
            a26,
            a27,
            a28,
            a29,
            a30,
            a31,
            a32,
            a33,
            a34,
            a35,
        ] =
            boxed
            .unbox();
        Matrix6 {
            m11: a0,
            m21: a1,
            m31: a2,
            m41: a3,
            m51: a4,
            m61: a5,
            m12: a6,
            m22: a7,
            m32: a8,
            m42: a9,
            m52: a10,
            m62: a11,
            m13: a12,
            m23: a13,
            m33: a14,
            m43: a15,
            m53: a16,
            m63: a17,
            m14: a18,
            m24: a19,
            m34: a20,
            m44: a21,
            m54: a22,
            m64: a23,
            m15: a24,
            m25: a25,
            m35: a26,
            m45: a27,
            m55: a28,
            m65: a29,
            m16: a30,
            m26: a31,
            m36: a32,
            m46: a33,
            m56: a34,
            m66: a35,
        }
    }
}

/// The sparse matrix of the non-zero components of a `Matrix6`. Upstream: `From<Matrix> for
/// CsMatrix`.
pub impl Matrix6IntoCsMatrix<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of Into<Matrix6<T>, CsMatrix<T>> {
    fn into(self: Matrix6<T>) -> CsMatrix<T> {
        let d: DMatrix<T> = self.into();
        CsKernels::from_dense(d.data, 6, 6)
    }
}
// cs-static: end
