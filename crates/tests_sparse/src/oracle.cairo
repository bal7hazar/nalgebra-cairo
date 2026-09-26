//! Oracle vectors of the sparse module and of the convolutions (`oracle_sparse.cairo`, suite
//! `sparse` of `tools/oracle`, emitted with `--max-per-dist 2`). The sparse operands are the
//! entries of dense random inputs on fixed patterns (1D / 2D Laplacian stencils, a rectangular
//! pattern), built here like `tools/oracle/src/suites/sparse.rs` builds them; the expected
//! values are upstream's dense forms (exact floors for the products, sums and convolutions).

use core::array::ToSpanTrait;
use fixed::Fixed;
use nalgebra::sparse::{CsCholesky, CsCholeskyTrait, CsMatrix, CsMatrixSolveTrait, CsMatrixTrait};
use nalgebra::{
    Convolution, DMatrix, DMatrixTrait, DVector, DVectorTrait, Vector3, Vector3Trait, Vector6,
    Vector6Trait,
};
use nalgebra_tests_utils::fx;
use simba::scalar::Real;
use crate::helpers::{max_ulp, raws, rows_of};
use crate::oracle_sparse as data;

/// The row-major raws of a `[[i64; C]; R]` oracle matrix, as one span.
fn flat<Row, +ToSpanTrait<Row, i64>, +Drop<Row>>(mut rows: Span<Row>) -> Span<i64> {
    let mut out: Array<i64> = array![];
    while let Some(row) = rows.pop_front() {
        out.append_span(ToSpanTrait::span(row));
    }
    out.span()
}

fn abs_diff(a: usize, b: usize) -> usize {
    if a > b {
        a - b
    } else {
        b - a
    }
}

/// The `n x n` 1D-Laplacian (tridiagonal) pattern, row-major.
fn lap1d(n: usize) -> Span<(usize, usize)> {
    let mut out: Array<(usize, usize)> = array![];
    let mut i = 0;
    while i != n {
        let mut j = 0;
        while j != n {
            if abs_diff(i, j) <= 1 {
                out.append((i, j));
            }
            j += 1;
        }
        i += 1;
    }
    out.span()
}

/// The 2D-Laplacian pattern of a `g x g` grid (5-point stencil), row-major.
fn lap2d(g: usize) -> Span<(usize, usize)> {
    let n = g * g;
    let mut out: Array<(usize, usize)> = array![];
    let mut i = 0;
    while i != n {
        let mut j = 0;
        while j != n {
            if abs_diff(i / g, j / g) + abs_diff(i % g, j % g) <= 1 {
                out.append((i, j));
            }
            j += 1;
        }
        i += 1;
    }
    out.span()
}

fn lower(mut p: Span<(usize, usize)>) -> Span<(usize, usize)> {
    let mut out: Array<(usize, usize)> = array![];
    while let Some(e) = p.pop_front() {
        let (i, j) = *e;
        if i >= j {
            out.append((i, j));
        }
    }
    out.span()
}

fn edges(mut p: Span<(usize, usize)>) -> Span<(usize, usize)> {
    let mut out: Array<(usize, usize)> = array![];
    while let Some(e) = p.pop_front() {
        let (i, j) = *e;
        if i < j {
            out.append((i, j));
        }
    }
    out.span()
}

/// The sparse `nrows x ncols` matrix of the entries of the row-major raws `x` on `pattern`.
fn masked(
    x: Span<i64>, nrows: usize, ncols: usize, mut pattern: Span<(usize, usize)>,
) -> CsMatrix<Fixed> {
    let mut rows: Array<usize> = array![];
    let mut cols: Array<usize> = array![];
    let mut vals: Array<Fixed> = array![];
    while let Some(e) = pattern.pop_front() {
        let (i, j) = *e;
        rows.append(i);
        cols.append(j);
        vals.append(fx(*x[i * ncols + j]));
    }
    CsMatrixTrait::from_triplet(nrows, ncols, rows.span(), cols.span(), vals.span())
}

fn rect_a() -> Span<(usize, usize)> {
    array![
        (0, 0), (0, 3), (1, 1), (1, 6), (2, 2), (2, 4), (3, 0), (3, 5), (4, 1), (4, 3), (4, 6),
        (2, 6),
    ]
        .span()
}

fn rect_b() -> Span<(usize, usize)> {
    array![(0, 0), (1, 2), (2, 1), (3, 3), (3, 0), (4, 2), (5, 1), (6, 3), (6, 0), (2, 3)].span()
}

/// The ROW-major dense components of a sparse matrix.
fn cs_rows(m: CsMatrix<Fixed>) -> Span<Fixed> {
    let d: DMatrix<Fixed> = m.into();
    rows_of(d)
}

#[test]
fn test_cs_mul_lap1d8_oracle() {
    let mut cases = data::cs_mul_lap1d8_cases();
    while let Some(c) = cases.pop_front() {
        let (a, b, ab, tol) = *c;
        let p = masked(flat(a.span()), 8, 8, lap1d(8)) * masked(flat(b.span()), 8, 8, lap1d(8));
        assert!(max_ulp(cs_rows(p), raws(flat(ab.span()))) <= tol.into());
    }
}

#[test]
fn test_cs_mul_rect_oracle() {
    let mut cases = data::cs_mul_rect_cases();
    while let Some(c) = cases.pop_front() {
        let (a, b, ab, tol) = *c;
        let p = masked(flat(a.span()), 5, 7, rect_a()) * masked(flat(b.span()), 7, 4, rect_b());
        assert!(max_ulp(cs_rows(p), raws(flat(ab.span()))) <= tol.into());
    }
}

#[test]
fn test_cs_add_lap2d9_oracle() {
    let mut cases = data::cs_add_lap2d9_cases();
    while let Some(c) = cases.pop_front() {
        let (a, b, s, tol) = *c;
        let r = masked(flat(a.span()), 9, 9, lap2d(3)) + masked(flat(b.span()), 9, 9, lap1d(9));
        assert!(max_ulp(cs_rows(r), raws(flat(s.span()))) <= tol.into());
    }
}

#[test]
fn test_cs_solve_lower_lap2d9_oracle() {
    let mut cases = data::cs_solve_lower_lap2d9_cases();
    while let Some(c) = cases.pop_front() {
        let (a, b, x, tol) = *c;
        let l = masked(flat(a.span()), 9, 9, lower(lap2d(3)));
        let b: DVector<Fixed> = DVectorTrait::from_column_slice(raws(flat(b.span())));
        let r = l.solve_lower_triangular(b).unwrap();
        assert!(max_ulp(r.as_slice(), raws(flat(x.span()))) <= tol.into());
    }
}

#[test]
fn test_cs_tr_solve_lower_lap2d9_oracle() {
    let mut cases = data::cs_tr_solve_lower_lap2d9_cases();
    while let Some(c) = cases.pop_front() {
        let (a, b, x, tol) = *c;
        let l = masked(flat(a.span()), 9, 9, lower(lap2d(3)));
        let b: DVector<Fixed> = DVectorTrait::from_column_slice(raws(flat(b.span())));
        let r = l.tr_solve_lower_triangular(b).unwrap();
        assert!(max_ulp(r.as_slice(), raws(flat(x.span()))) <= tol.into());
    }
}

#[test]
fn test_cs_solve_lower_cs_lap2d9_oracle() {
    let mut cases = data::cs_solve_lower_cs_lap2d9_cases();
    while let Some(c) = cases.pop_front() {
        let (a, b, x, tol) = *c;
        let l = masked(flat(a.span()), 9, 9, lower(lap2d(3)));
        // The entries 0 and 4 of b only (a 9x1 row-major matrix: index i is row i).
        let b = masked(flat(b.span()), 9, 1, array![(0, 0), (4, 0)].span());
        let r: DVector<Fixed> = l.solve_lower_triangular_cs(b).unwrap().into();
        assert!(max_ulp(r.as_slice(), raws(flat(x.span()))) <= tol.into());
    }
}

/// The SPD graph Laplacian of `pattern` (`n` nodes) from the weights `w` and shifts `s`:
/// triplets whose duplicates `from_triplet` sums (see the oracle suite).
fn laplacian(
    w: Span<i64>, s: Span<i64>, n: usize, pattern: Span<(usize, usize)>,
) -> CsMatrix<Fixed> {
    let mut rows: Array<usize> = array![];
    let mut cols: Array<usize> = array![];
    let mut vals: Array<Fixed> = array![];
    let mut es = edges(pattern);
    let mut ws = w;
    while let Some(e) = es.pop_front() {
        let (i, j) = *e;
        let wk = Real::abs(fx(*ws.pop_front().unwrap()));
        rows.append_span(array![i, j, i, j].span());
        cols.append_span(array![j, i, i, j].span());
        vals.append_span(array![-wk, -wk, wk, wk].span());
    }
    let quarter = fx(0x40000000);
    let mut k = 0;
    while k != n {
        rows.append(k);
        cols.append(k);
        vals.append(Real::abs(fx(*s[k])) + quarter);
        k += 1;
    }
    CsMatrixTrait::from_triplet(n, n, rows.span(), cols.span(), vals.span())
}

fn check_cholesky(
    w: Span<i64>, s: Span<i64>, l: Span<i64>, tol: u64, n: usize, pattern: Span<(usize, usize)>,
) {
    let a = laplacian(w, s, n, pattern);
    let chol: CsCholesky<Fixed> = CsCholeskyTrait::new(@a);
    let got = chol.unwrap_l().unwrap();
    assert!(max_ulp(cs_rows(got), raws(l)) <= tol.into());
}

#[test]
fn test_cs_cholesky_lap1d16_oracle() {
    let mut cases = data::cs_cholesky_lap1d16_cases();
    while let Some(c) = cases.pop_front() {
        let (w, s, l, tol) = *c;
        check_cholesky(flat(w.span()), flat(s.span()), flat(l.span()), tol, 16, lap1d(16));
    }
}

#[test]
fn test_cs_cholesky_lap2d16_oracle() {
    let mut cases = data::cs_cholesky_lap2d16_cases();
    while let Some(c) = cases.pop_front() {
        let (w, s, l, tol) = *c;
        check_cholesky(flat(w.span()), flat(s.span()), flat(l.span()), tol, 16, lap2d(4));
    }
}

#[test]
fn test_cs_cholesky_lap2d9_oracle() {
    let mut cases = data::cs_cholesky_lap2d9_cases();
    while let Some(c) = cases.pop_front() {
        let (w, s, l, tol) = *c;
        check_cholesky(flat(w.span()), flat(s.span()), flat(l.span()), tol, 9, lap2d(3));
    }
}

fn dvec(x: Span<i64>) -> DVector<Fixed> {
    DVectorTrait::from_column_slice(raws(x))
}

#[test]
fn test_convolve_full7_3_oracle() {
    let mut cases = data::convolve_full7_3_cases();
    while let Some(c) = cases.pop_front() {
        let (v, k, r, tol) = *c;
        let got = dvec(flat(v.span())).convolve_full(dvec(flat(k.span())));
        assert!(max_ulp(got.as_slice(), raws(flat(r.span()))) <= tol.into());
    }
}

#[test]
fn test_convolve_same7_4_oracle() {
    let mut cases = data::convolve_same7_4_cases();
    while let Some(c) = cases.pop_front() {
        let (v, k, r, tol) = *c;
        let got = dvec(flat(v.span())).convolve_same(dvec(flat(k.span())));
        assert!(max_ulp(got.as_slice(), raws(flat(r.span()))) <= tol.into());
    }
}

#[test]
fn test_convolve_valid16_5_oracle() {
    let mut cases = data::convolve_valid16_5_cases();
    while let Some(c) = cases.pop_front() {
        let (v, k, r, tol) = *c;
        let got = dvec(flat(v.span())).convolve_valid(dvec(flat(k.span())));
        assert!(max_ulp(got.as_slice(), raws(flat(r.span()))) <= tol.into());
    }
}

#[test]
fn test_convolve_full6_3_static_oracle() {
    let mut cases = data::convolve_full6_3_cases();
    while let Some(c) = cases.pop_front() {
        let (v, k, r, tol) = *c;
        let v: Vector6<Fixed> = Vector6Trait::from_column_slice(raws(flat(v.span())));
        let k: Vector3<Fixed> = Vector3Trait::from_column_slice(raws(flat(k.span())));
        let got = v.convolve_full(k);
        assert!(max_ulp(got.as_slice(), raws(flat(r.span()))) <= tol.into());
    }
}

#[test]
fn test_convolve_same6_6_static_oracle() {
    let mut cases = data::convolve_same6_6_cases();
    while let Some(c) = cases.pop_front() {
        let (v, k, r, tol) = *c;
        let v: Vector6<Fixed> = Vector6Trait::from_column_slice(raws(flat(v.span())));
        let k: Vector6<Fixed> = Vector6Trait::from_column_slice(raws(flat(k.span())));
        let got: Vector6<Fixed> = v.convolve_same(k);
        let expected: Vector6<Fixed> = Vector6Trait::from_column_slice(raws(flat(r.span())));
        assert!(got.abs_diff_eq(expected, tol));
    }
}
