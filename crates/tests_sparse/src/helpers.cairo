//! Builders of the sparse test operands from integers or raw Q32.32 values, passed through
//! `black_box` (AGENTS.md: no constant-specialised instances), and comparisons.

use fixed::Fixed;
use nalgebra::sparse::{CsMatrix, CsMatrixTrait};
use nalgebra::{DMatrix, DMatrixTrait, DVector, DVectorTrait};
use nalgebra_testing::black_box;
use nalgebra_tests_utils::{fx, int};

/// The `Fixed` values of the integers `values`.
pub fn ints(mut values: Span<i64>) -> Span<Fixed> {
    let mut out: Array<Fixed> = array![];
    while let Some(v) = values.pop_front() {
        out.append(int(*v));
    }
    black_box(out.span())
}

/// The `Fixed` values of the raws `values`.
pub fn raws(mut values: Span<i64>) -> Span<Fixed> {
    let mut out: Array<Fixed> = array![];
    while let Some(v) = values.pop_front() {
        out.append(fx(*v));
    }
    black_box(out.span())
}

/// The `nrows x ncols` sparse matrix of the integer triplets `(row, col, value)`.
pub fn cs(nrows: usize, ncols: usize, mut triplets: Span<(usize, usize, i64)>) -> CsMatrix<Fixed> {
    let mut rows: Array<usize> = array![];
    let mut cols: Array<usize> = array![];
    let mut vals: Array<i64> = array![];
    while let Some(t) = triplets.pop_front() {
        let (r, c, v) = *t;
        rows.append(r);
        cols.append(c);
        vals.append(v);
    }
    CsMatrixTrait::from_triplet(
        black_box(nrows),
        black_box(ncols),
        black_box(rows.span()),
        black_box(cols.span()),
        ints(vals.span()),
    )
}

/// The `nrows x ncols` matrix of the integers `rows` (ROW-major).
pub fn dmi(nrows: usize, ncols: usize, rows: Span<i64>) -> DMatrix<Fixed> {
    DMatrixTrait::from_row_slice(black_box(nrows), black_box(ncols), ints(rows))
}

/// The `nrows x ncols` matrix of the raws `rows` (ROW-major).
pub fn dmr(nrows: usize, ncols: usize, rows: Span<i64>) -> DMatrix<Fixed> {
    DMatrixTrait::from_row_slice(black_box(nrows), black_box(ncols), raws(rows))
}

/// The column vector of the integers `values`.
pub fn dvi(values: Span<i64>) -> DVector<Fixed> {
    DVectorTrait::from_column_slice(ints(values))
}

/// The dense column-major components of a sparse matrix.
pub fn dense(m: CsMatrix<Fixed>) -> Span<Fixed> {
    let d: DMatrix<Fixed> = m.into();
    d.as_slice()
}

/// The ROW-major components of a dense matrix (the oracle's order).
pub fn rows_of(m: DMatrix<Fixed>) -> Span<Fixed> {
    m.transpose().as_slice()
}

/// The largest `|a - b|` in raw units between two equally long spans (`u128::MAX` when the
/// lengths differ).
pub fn max_ulp(mut a: Span<Fixed>, mut b: Span<Fixed>) -> u128 {
    if a.len() != b.len() {
        return 0xffffffffffffffffffffffffffffffff;
    }
    let mut worst: u128 = 0;
    while let Some(x) = a.pop_front() {
        let y = b.pop_front().unwrap();
        let d: i128 = (*x).raw.into() - (*y).raw.into();
        let d: u128 = if d < 0 {
            (-d).try_into().unwrap()
        } else {
            d.try_into().unwrap()
        };
        if d > worst {
            worst = d;
        }
    }
    worst
}
