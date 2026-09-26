//! Oracle vectors of the dynamic matrices (`oracle_dynamic.cairo`, suite `dynamic` of
//! `tools/oracle`): products at 3x3 and 6x6 (dispatched to the static kernels), 16x16 and 4x7 by
//! 7x5 (the loops), matrix-vector products, dot product and norms of 16 components. Exact floors
//! except the norm (2 ulp). Generated with `--max-per-dist 2`.

use core::array::ToSpanTrait;
use fixed::Fixed;
use nalgebra::{DMatrix, DMatrixTrait, DVector, DVectorTrait, MatrixMul};
use crate::helpers::{max_ulp, raws};
use crate::oracle_dynamic as data;

/// The row-major raws of a `[[i64; C]; R]` oracle matrix, as one span.
fn flat<Row, +ToSpanTrait<Row, i64>, +Drop<Row>>(mut rows: Span<Row>) -> Span<i64> {
    let mut out: Array<i64> = array![];
    while let Some(row) = rows.pop_front() {
        out.append_span(ToSpanTrait::span(row));
    }
    out.span()
}

fn mat<Row, +ToSpanTrait<Row, i64>, +Drop<Row>>(
    rows: Span<Row>, nrows: usize, ncols: usize,
) -> DMatrix<Fixed> {
    DMatrixTrait::from_row_slice(nrows, ncols, raws(flat(rows)))
}

fn vec<Row, +ToSpanTrait<Row, i64>, +Drop<Row>>(rows: Span<Row>) -> DVector<Fixed> {
    DVectorTrait::from_column_slice(raws(flat(rows)))
}

#[test]
fn test_dmatrix3_mul_oracle() {
    let mut cases = data::dmatrix3_mul_cases();
    while let Some(c) = cases.pop_front() {
        let (a, b, ab, tol) = *c;
        let p = mat(a.span(), 3, 3) * mat(b.span(), 3, 3);
        assert!(max_ulp(p.as_slice(), mat(ab.span(), 3, 3).as_slice()) <= tol.into());
    }
}

#[test]
fn test_dmatrix6_mul_oracle() {
    let mut cases = data::dmatrix6_mul_cases();
    while let Some(c) = cases.pop_front() {
        let (a, b, ab, tol) = *c;
        let p = mat(a.span(), 6, 6) * mat(b.span(), 6, 6);
        assert!(max_ulp(p.as_slice(), mat(ab.span(), 6, 6).as_slice()) <= tol.into());
    }
}

#[test]
fn test_dmatrix16_mul_oracle() {
    let mut cases = data::dmatrix16_mul_cases();
    while let Some(c) = cases.pop_front() {
        let (a, b, ab, tol) = *c;
        let p = mat(a.span(), 16, 16) * mat(b.span(), 16, 16);
        assert!(max_ulp(p.as_slice(), mat(ab.span(), 16, 16).as_slice()) <= tol.into());
    }
}

#[test]
fn test_dmatrix4x7_mul_7x5_oracle() {
    let mut cases = data::dmatrix4x7_mul_7x5_cases();
    while let Some(c) = cases.pop_front() {
        let (a, b, ab, tol) = *c;
        let p = mat(a.span(), 4, 7) * mat(b.span(), 7, 5);
        assert!(p.shape() == (4, 5));
        assert!(max_ulp(p.as_slice(), mat(ab.span(), 4, 5).as_slice()) <= tol.into());
        // `mul_mat` is `*`.
        assert!(mat(a.span(), 4, 7).mul_mat(mat(b.span(), 7, 5)) == p);
    }
}

#[test]
fn test_dmatrix6_mul_vec_oracle() {
    let mut cases = data::dmatrix6_mul_vec_cases();
    while let Some(c) = cases.pop_front() {
        let (a, v, av, tol) = *c;
        let p = mat(a.span(), 6, 6).mul_mat(vec(v.span()));
        assert!(max_ulp(p.as_slice(), vec(av.span()).as_slice()) <= tol.into());
    }
}

#[test]
fn test_dmatrix16_mul_vec_oracle() {
    let mut cases = data::dmatrix16_mul_vec_cases();
    while let Some(c) = cases.pop_front() {
        let (a, v, av, tol) = *c;
        let p = mat(a.span(), 16, 16).mul_mat(vec(v.span()));
        assert!(max_ulp(p.as_slice(), vec(av.span()).as_slice()) <= tol.into());
    }
}

#[test]
fn test_dmatrix3x8_mul_vec_oracle() {
    let mut cases = data::dmatrix3x8_mul_vec_cases();
    while let Some(c) = cases.pop_front() {
        let (a, v, av, tol) = *c;
        let p = mat(a.span(), 3, 8).mul_mat(vec(v.span()));
        assert!(max_ulp(p.as_slice(), vec(av.span()).as_slice()) <= tol.into());
    }
}

#[test]
fn test_dvector16_dot_oracle() {
    let mut cases = data::dvector16_dot_cases();
    while let Some(c) = cases.pop_front() {
        let (a, b, d, tol) = *c;
        let got = vec(a.span()).dot(vec(b.span()));
        assert!(max_ulp(array![got].span(), raws(array![d].span())) <= tol.into());
    }
}

#[test]
fn test_dvector16_norms_oracle() {
    let mut cases = data::dvector16_norm_squared_cases();
    while let Some(c) = cases.pop_front() {
        let (a, n2, tol) = *c;
        let got = vec(a.span()).norm_squared();
        assert!(max_ulp(array![got].span(), raws(array![n2].span())) <= tol.into());
    }
    let mut cases = data::dvector16_norm_cases();
    while let Some(c) = cases.pop_front() {
        let (a, n, tol) = *c;
        let got = vec(a.span()).norm();
        assert!(max_ulp(array![got].span(), raws(array![n].span())) <= tol.into());
    }
}
