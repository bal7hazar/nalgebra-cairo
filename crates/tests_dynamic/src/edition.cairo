//! Edition of `DMatrix` / `DVector` / `RowDVector` (upstream `base/edition.rs`): insertion,
//! removal, resizing and the `compress_*` folds, exact values and the out-of-range panics.

use fixed::Fixed;
use nalgebra::{DMatrix, DMatrixTrait, DVector, DVectorTrait, RowDVector, RowDVectorTrait};
use nalgebra_testing::black_box;
use nalgebra_tests_utils::int;
use simba::scalar::Real;
use crate::helpers::{dmi, dvi, ints, rdvi};

/// The 3x4 matrix `[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]`.
fn m34() -> DMatrix<Fixed> {
    dmi(3, 4, array![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12].span())
}

#[test]
fn test_dmatrix_insert_columns() {
    let m = m34();
    let v = int(0);
    assert!(
        m
            .insert_columns(
                black_box(1), 2, v,
            ) == dmi(3, 6, array![1, 0, 0, 2, 3, 4, 5, 0, 0, 6, 7, 8, 9, 0, 0, 10, 11, 12].span()),
    );
    assert!(
        m
            .insert_column(
                black_box(4), v,
            ) == dmi(3, 5, array![1, 2, 3, 4, 0, 5, 6, 7, 8, 0, 9, 10, 11, 12, 0].span()),
    );
    assert!(m.insert_fixed_columns::<2>(black_box(1), v) == m.insert_columns(1, 2, v));
    assert!(m.insert_columns(black_box(0), 0, v) == m);
}

#[test]
fn test_dmatrix_insert_rows() {
    let m = m34();
    let v = int(0);
    assert!(
        m
            .insert_rows(
                black_box(1), 2, v,
            ) == dmi(
                5, 4, array![1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 5, 6, 7, 8, 9, 10, 11, 12].span(),
            ),
    );
    assert!(
        m
            .insert_row(
                black_box(3), v,
            ) == dmi(4, 4, array![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 0, 0, 0].span()),
    );
    assert!(m.insert_fixed_rows::<1>(black_box(0), v) == m.insert_row(0, v));
}

#[test]
#[should_panic(expected: 'nalgebra: index out of bounds')]
fn test_dmatrix_insert_columns_out_of_range() {
    let _m = m34().insert_columns(black_box(5), 1, int(0));
}

#[test]
#[should_panic(expected: 'nalgebra: index out of bounds')]
fn test_dmatrix_insert_rows_out_of_range() {
    let _m = m34().insert_rows(black_box(4), 1, int(0));
}

#[test]
fn test_dmatrix_remove_columns_rows() {
    let m = m34();
    assert!(m.remove_columns(black_box(1), 2) == dmi(3, 2, array![1, 4, 5, 8, 9, 12].span()));
    assert!(m.remove_column(black_box(3)) == dmi(3, 3, array![1, 2, 3, 5, 6, 7, 9, 10, 11].span()));
    assert!(m.remove_fixed_columns::<2>(black_box(1)) == m.remove_columns(1, 2));
    assert!(m.remove_rows(black_box(0), 2) == dmi(1, 4, array![9, 10, 11, 12].span()));
    assert!(m.remove_row(black_box(1)) == dmi(2, 4, array![1, 2, 3, 4, 9, 10, 11, 12].span()));
    assert!(m.remove_fixed_rows::<1>(black_box(1)) == m.remove_row(1));
    assert!(m.remove_columns(black_box(0), 4).shape() == (3, 0));
}

#[test]
#[should_panic(expected: 'nalgebra: index out of bounds')]
fn test_dmatrix_remove_columns_out_of_range() {
    let _m = m34().remove_columns(black_box(3), 2);
}

#[test]
#[should_panic(expected: 'nalgebra: index out of bounds')]
fn test_dmatrix_remove_rows_out_of_range() {
    let _m = m34().remove_row(black_box(3));
}

/// Upstream ignores repeated and out-of-range indices (`indices.contains(..)`).
#[test]
fn test_dmatrix_remove_at() {
    let m = m34();
    let idx = black_box(array![3_usize, 0, 3, 9].span());
    assert!(m.remove_columns_at(idx) == dmi(3, 2, array![2, 3, 6, 7, 10, 11].span()));
    let idx = black_box(array![2_usize, 0].span());
    assert!(m.remove_rows_at(idx) == dmi(1, 4, array![5, 6, 7, 8].span()));
    assert!(m.remove_rows_at(black_box(array![].span())) == m);
}

#[test]
fn test_dmatrix_resize() {
    let m = m34();
    let v = int(-1);
    assert!(m.resize(black_box(4), 2, v) == dmi(4, 2, array![1, 2, 5, 6, 9, 10, -1, -1].span()));
    assert!(
        m.resize(black_box(2), 5, v) == dmi(2, 5, array![1, 2, 3, 4, -1, 5, 6, 7, 8, -1].span()),
    );
    assert!(m.resize_vertically(black_box(1), v) == dmi(1, 4, array![1, 2, 3, 4].span()));
    assert!(m.resize_horizontally(black_box(1), v) == dmi(3, 1, array![1, 5, 9].span()));
    assert!(m.resize(black_box(0), 0, v).is_empty());
    let mut a = m;
    a.resize_mut(black_box(4), 2, v);
    assert!(a == m.resize(4, 2, v));
    let mut b = m;
    b.resize_vertically_mut(black_box(2), v);
    assert!(b == m.resize(2, 4, v));
    let mut c = m;
    c.resize_horizontally_mut(black_box(5), v);
    assert!(c == m.resize(3, 5, v));
}

#[test]
fn test_dmatrix_compress() {
    let m = m34();
    let sums = m.compress_rows(|c: DVector<Fixed>| -> Fixed {
        c[0] + c[1] + c[2]
    });
    assert!(sums == rdvi(array![15, 18, 21, 24].span()));
    let sums_tr = m.compress_rows_tr(|c: DVector<Fixed>| -> Fixed {
        c[0] + c[1] + c[2]
    });
    assert!(sums_tr == dvi(array![15, 18, 21, 24].span()));
    // Row sums: fold the columns.
    let init: DVector<Fixed> = DVectorTrait::zeros(3);
    let rows = m
        .compress_columns(
            init, |acc: DVector<Fixed>, c: DVector<Fixed>| -> DVector<Fixed> {
                acc + c
            },
        );
    assert!(rows == dvi(array![10, 26, 42].span()));
}

#[test]
fn test_dvector_edition() {
    let v = dvi(array![1, 2, 3].span());
    let z = int(0);
    assert!(v.insert_rows(black_box(1), 2, z) == dvi(array![1, 0, 0, 2, 3].span()));
    assert!(v.insert_row(black_box(3), z) == dvi(array![1, 2, 3, 0].span()));
    assert!(v.insert_fixed_rows::<1>(black_box(0), z) == dvi(array![0, 1, 2, 3].span()));
    assert!(v.remove_rows(black_box(0), 2) == dvi(array![3].span()));
    assert!(v.remove_row(black_box(1)) == dvi(array![1, 3].span()));
    assert!(v.remove_fixed_rows::<2>(black_box(1)) == dvi(array![1].span()));
    assert!(v.remove_rows_at(black_box(array![2_usize, 0].span())) == dvi(array![2].span()));
    assert!(v.resize_vertically(black_box(5), z) == dvi(array![1, 2, 3, 0, 0].span()));
    assert!(v.resize_vertically(black_box(2), z) == dvi(array![1, 2].span()));
    let mut w = v;
    w.resize_vertically_mut(black_box(1), z);
    assert!(w == dvi(array![1].span()));
    assert!(v.resize(black_box(2), 2, z) == dmi(2, 2, array![1, 0, 2, 0].span()));
}

#[test]
#[should_panic(expected: 'nalgebra: index out of bounds')]
fn test_dvector_remove_rows_out_of_range() {
    let _v = dvi(array![1, 2, 3].span()).remove_rows(black_box(2), 2);
}

#[test]
fn test_row_dvector_edition() {
    let r = rdvi(array![1, 2, 3].span());
    let z = int(0);
    assert!(r.insert_columns(black_box(1), 2, z) == rdvi(array![1, 0, 0, 2, 3].span()));
    assert!(r.insert_column(black_box(0), z) == rdvi(array![0, 1, 2, 3].span()));
    assert!(r.insert_fixed_columns::<1>(black_box(3), z) == rdvi(array![1, 2, 3, 0].span()));
    assert!(r.remove_columns(black_box(1), 2) == rdvi(array![1].span()));
    assert!(r.remove_column(black_box(0)) == rdvi(array![2, 3].span()));
    assert!(r.remove_fixed_columns::<1>(black_box(2)) == rdvi(array![1, 2].span()));
    assert!(r.remove_columns_at(black_box(array![1_usize].span())) == rdvi(array![1, 3].span()));
    assert!(r.resize_horizontally(black_box(4), z) == rdvi(array![1, 2, 3, 0].span()));
    let mut w = r;
    w.resize_horizontally_mut(black_box(2), z);
    assert!(w == rdvi(array![1, 2].span()));
    assert!(r.resize(black_box(2), 2, z) == dmi(2, 2, array![1, 2, 0, 0].span()));
}

#[test]
#[should_panic(expected: 'nalgebra: index out of bounds')]
fn test_row_dvector_insert_out_of_range() {
    let _r = rdvi(array![1, 2, 3].span()).insert_column(black_box(4), int(0));
}

#[test]
fn test_empty_edition() {
    let e: DMatrix<Fixed> = DMatrixTrait::zeros(black_box(0), 0);
    let g = e.insert_columns(0, 2, Real::one());
    assert!(g.shape() == (0, 2) && g.is_empty());
    let g = g.insert_rows(0, 1, Real::one());
    assert!(g == dmi(1, 2, array![1, 1].span()));
    let d: DVector<Fixed> = DVectorTrait::zeros(black_box(0));
    assert!(d.insert_row(0, int(2)) == dvi(array![2].span()));
    let r: RowDVector<Fixed> = RowDVectorTrait::zeros(black_box(0));
    assert!(r.resize_horizontally(2, int(2)) == rdvi(array![2, 2].span()));
    assert!(ints(array![].span()).len() == 0);
}
