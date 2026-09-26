//! Builders of the dynamic matrices from integers or raw Q32.32 values (ROW-major, the order of
//! upstream's `DMatrix::from_row_slice` and of the oracle), their operands passed through
//! `black_box` (AGENTS.md: no constant-specialised instances).

use fixed::Fixed;
use nalgebra::{DMatrix, DMatrixTrait, DVector, DVectorTrait, RowDVector, RowDVectorTrait};
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

/// The column vector of the raws `values`.
pub fn dvr(values: Span<i64>) -> DVector<Fixed> {
    DVectorTrait::from_column_slice(raws(values))
}

/// The row vector of the integers `values`.
pub fn rdvi(values: Span<i64>) -> RowDVector<Fixed> {
    RowDVectorTrait::from_column_slice(ints(values))
}

/// `0, 1, 2, ..` modulo 7 minus 3 (the benchmark operands), `n` of them.
pub fn ramp(n: usize) -> Span<Fixed> {
    let mut out: Array<Fixed> = array![];
    let mut k: usize = 0;
    while k != n {
        let v: i64 = k.into();
        out.append(int(v % 7 - 3));
        k += 1;
    }
    black_box(out.span())
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
