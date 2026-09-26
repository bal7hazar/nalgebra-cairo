//! Construction, properties and conversions of `DMatrix` / `DVector` / `RowDVector` (upstream
//! `base/construction.rs`, `base/conversion.rs`, `base/properties.rs`): exact values.

use fixed::Fixed;
use nalgebra::{
    DMatrix, DMatrixTrait, DVector, DVectorTrait, Matrix1, Matrix2x3, Matrix2x3Trait, Matrix3xX,
    Matrix6, Matrix6Trait, MatrixIndex, MatrixXx1, RowDVector, RowDVectorTrait, RowVector3,
    RowVector3Trait, Vector3, Vector3Trait,
};
use nalgebra_testing::black_box;
use nalgebra_tests_utils::int;
use crate::helpers::{dmi, dvi, ints, rdvi};

#[test]
fn test_dmatrix_from_row_slice_is_column_major() {
    let m = dmi(2, 3, array![1, 2, 3, 4, 5, 6].span());
    assert!(m.as_slice() == ints(array![1, 4, 2, 5, 3, 6].span()));
    assert!(m.nrows() == 2 && m.ncols() == 3 && m.shape() == (2, 3));
    assert!(m.len() == 6 && !m.is_empty() && !m.is_square());
    assert!(m[(1, 0)] == int(4) && m[(0, 2)] == int(3) && m[3] == int(5));
}

#[test]
fn test_dmatrix_from_vec_iterator_column_slice() {
    let data = ints(array![1, 4, 2, 5, 3, 6].span());
    let expected = dmi(2, 3, array![1, 2, 3, 4, 5, 6].span());
    let mut arr: Array<Fixed> = array![];
    arr.append_span(data);
    assert!(DMatrixTrait::from_vec(2, 3, arr) == expected);
    assert!(DMatrixTrait::from_iterator(2, 3, data) == expected);
    assert!(DMatrixTrait::from_column_slice(2, 3, data) == expected);
    let rows = ints(array![1, 2, 3, 4, 5, 6].span());
    assert!(DMatrixTrait::from_row_iterator(2, 3, rows) == expected);
}

#[test]
#[should_panic(expected: 'nalgebra: wrong slice length')]
fn test_dmatrix_from_vec_wrong_length() {
    let mut arr: Array<Fixed> = array![];
    arr.append_span(ints(array![1, 2, 3, 4, 5].span()));
    let _m: DMatrix<Fixed> = DMatrixTrait::from_vec(black_box(2), 3, arr);
}

#[test]
#[should_panic(expected: 'nalgebra: wrong slice length')]
fn test_dmatrix_from_row_iterator_wrong_length() {
    let _m: DMatrix<Fixed> = DMatrixTrait::from_row_iterator(
        black_box(2), 2, ints(array![1, 2, 3].span()),
    );
}

#[test]
fn test_dmatrix_zeros_element_repeat() {
    let z: DMatrix<Fixed> = DMatrixTrait::zeros(black_box(3), 2);
    assert!(z == dmi(3, 2, array![0, 0, 0, 0, 0, 0].span()));
    let e: DMatrix<Fixed> = DMatrixTrait::from_element(black_box(2), 2, int(7));
    assert!(e == dmi(2, 2, array![7, 7, 7, 7].span()));
    assert!(DMatrixTrait::repeat(2, 2, int(7)) == e);
    let empty: DMatrix<Fixed> = DMatrixTrait::zeros(black_box(0), 3);
    assert!(empty.is_empty() && empty.len() == 0 && empty.shape() == (0, 3));
}

#[test]
fn test_dmatrix_identity_diagonal_rectangular() {
    let i: DMatrix<Fixed> = DMatrixTrait::identity(black_box(2), 3);
    assert!(i == dmi(2, 3, array![1, 0, 0, 0, 1, 0].span()));
    let d: DMatrix<Fixed> = DMatrixTrait::from_diagonal_element(black_box(3), 2, int(5));
    assert!(d == dmi(3, 2, array![5, 0, 0, 5, 0, 0].span()));
    let p: DMatrix<Fixed> = DMatrixTrait::from_partial_diagonal(
        black_box(3), 3, ints(array![2, 3].span()),
    );
    assert!(p == dmi(3, 3, array![2, 0, 0, 0, 3, 0, 0, 0, 0].span()));
    let sq: DMatrix<Fixed> = DMatrixTrait::identity(3, 3);
    assert!(sq.is_square());
}

#[test]
#[should_panic(expected: 'nalgebra: diagonal too long')]
fn test_dmatrix_from_partial_diagonal_too_long() {
    let _p: DMatrix<Fixed> = DMatrixTrait::from_partial_diagonal(
        black_box(2), 3, ints(array![1, 2, 3].span()),
    );
}

#[test]
fn test_dmatrix_from_fn_column_major_calls() {
    let m: DMatrix<Fixed> = DMatrixTrait::from_fn(
        black_box(2), 3, |i: usize, j: usize| -> Fixed {
            let v: i64 = (10 * i + j).into();
            int(v)
        },
    );
    assert!(m == dmi(2, 3, array![0, 1, 2, 10, 11, 12].span()));
}

#[test]
fn test_dmatrix_column_row_transpose() {
    let m = dmi(2, 3, array![1, 2, 3, 4, 5, 6].span());
    assert!(m.column(1) == dvi(array![2, 5].span()));
    assert!(m.row(1) == rdvi(array![4, 5, 6].span()));
    assert!(m.transpose() == dmi(3, 2, array![1, 4, 2, 5, 3, 6].span()));
}

#[test]
#[should_panic(expected: 'nalgebra: index out of bounds')]
fn test_dmatrix_index_pair_out_of_bounds() {
    let m = dmi(2, 3, array![1, 2, 3, 4, 5, 6].span());
    // (2, 0) is inside the linear range (index 2) but not a row of a 2-row matrix.
    let _x = m[(black_box(2), 0)];
}

#[test]
#[should_panic(expected: 'nalgebra: index out of bounds')]
fn test_dmatrix_index_linear_out_of_bounds() {
    let m = dmi(2, 3, array![1, 2, 3, 4, 5, 6].span());
    let _x = m[black_box(6)];
}

#[test]
fn test_dmatrix_get() {
    let m = dmi(2, 3, array![1, 2, 3, 4, 5, 6].span());
    assert!(m.get((1, 2)) == Some(int(6)) && m.get((2, 0)).is_none());
    assert!(m.get(5) == Some(int(6)) && m.get(6).is_none());
    assert!(m.index((0, 1)) == int(2) && m.index(1) == int(4));
}

#[test]
fn test_dvector_construction() {
    let z: DVector<Fixed> = DVectorTrait::zeros(black_box(3));
    assert!(z == dvi(array![0, 0, 0].span()));
    assert!(z.nrows() == 3 && z.ncols() == 1 && z.shape() == (3, 1) && z.len() == 3);
    let e: DVector<Fixed> = DVectorTrait::from_element(black_box(2), int(4));
    assert!(e == dvi(array![4, 4].span()) && DVectorTrait::repeat(2, int(4)) == e);
    let i: DVector<Fixed> = DVectorTrait::identity(black_box(3));
    assert!(i == dvi(array![1, 0, 0].span()));
    assert!(DVectorTrait::from_diagonal_element(black_box(2), int(3)) == dvi(array![3, 0].span()));
    let p: DVector<Fixed> = DVectorTrait::from_partial_diagonal(3, ints(array![9].span()));
    assert!(p == dvi(array![9, 0, 0].span()));
    let mut arr: Array<Fixed> = array![];
    arr.append_span(ints(array![1, 2, 3].span()));
    let v: MatrixXx1<Fixed> = DVectorTrait::from_vec(arr);
    assert!(v == dvi(array![1, 2, 3].span()) && v[2] == int(3) && v.get(3).is_none());
    assert!(DVectorTrait::from_iterator(3, ints(array![1, 2, 3].span())) == v);
    assert!(DVectorTrait::from_row_iterator(3, ints(array![1, 2, 3].span())) == v);
    assert!(DVectorTrait::from_row_slice(ints(array![1, 2, 3].span())) == v);
    let f: DVector<Fixed> = DVectorTrait::from_fn(
        black_box(3), |i: usize, j: usize| -> Fixed {
            let v: i64 = (10 * i + j).into();
            int(v)
        },
    );
    assert!(f == dvi(array![0, 10, 20].span()));
    let empty: DVector<Fixed> = DVectorTrait::identity(black_box(0));
    assert!(empty.is_empty());
}

#[test]
#[should_panic(expected: 'nalgebra: wrong slice length')]
fn test_dvector_from_iterator_wrong_length() {
    let _v: DVector<Fixed> = DVectorTrait::from_iterator(black_box(2), ints(array![1].span()));
}

#[test]
fn test_row_dvector_construction() {
    let z: RowDVector<Fixed> = RowDVectorTrait::zeros(black_box(3));
    assert!(z == rdvi(array![0, 0, 0].span()));
    assert!(z.nrows() == 1 && z.ncols() == 3 && z.shape() == (1, 3) && z.len() == 3);
    let i: RowDVector<Fixed> = RowDVectorTrait::identity(black_box(2));
    assert!(i == rdvi(array![1, 0].span()));
    let f: RowDVector<Fixed> = RowDVectorTrait::from_fn(
        black_box(3), |i: usize, j: usize| -> Fixed {
            let v: i64 = (10 * i + j).into();
            int(v)
        },
    );
    assert!(f == rdvi(array![0, 1, 2].span()));
    assert!(f.transpose() == dvi(array![0, 1, 2].span()));
    assert!(dvi(array![0, 1, 2].span()).transpose() == f);
    let mut arr: Array<Fixed> = array![];
    arr.append_span(ints(array![5, 6].span()));
    assert!(RowDVectorTrait::from_vec(arr) == rdvi(array![5, 6].span()));
}

#[test]
fn test_from_array() {
    let mut arr: Array<Fixed> = array![];
    arr.append_span(ints(array![1, 2].span()));
    let v: DVector<Fixed> = arr.into();
    assert!(v == dvi(array![1, 2].span()));
    let mut arr: Array<Fixed> = array![];
    arr.append_span(ints(array![1, 2].span()));
    let r: RowDVector<Fixed> = arr.into();
    assert!(r == rdvi(array![1, 2].span()));
}

#[test]
fn test_from_static_shapes() {
    let m: Matrix2x3<Fixed> = black_box(
        Matrix2x3Trait::new(int(1), int(2), int(3), int(4), int(5), int(6)),
    );
    let d: Matrix3xX<Fixed> = m.transpose().into();
    assert!(d == dmi(3, 2, array![1, 4, 2, 5, 3, 6].span()));
    let d: DMatrix<Fixed> = m.into();
    assert!(d == dmi(2, 3, array![1, 2, 3, 4, 5, 6].span()));
    let v: Vector3<Fixed> = black_box(Vector3Trait::new(int(1), int(2), int(3)));
    let dv: DVector<Fixed> = v.into();
    assert!(dv == dvi(array![1, 2, 3].span()));
    let dm: DMatrix<Fixed> = v.into();
    assert!(dm.shape() == (3, 1));
    let r: RowVector3<Fixed> = black_box(RowVector3Trait::new(int(1), int(2), int(3)));
    let dr: RowDVector<Fixed> = r.into();
    assert!(dr == rdvi(array![1, 2, 3].span()));
    let one: Matrix1<Fixed> = black_box(Matrix1 { x: int(4) });
    let a: DVector<Fixed> = one.into();
    let b: RowDVector<Fixed> = one.into();
    assert!(a.len() == 1 && b.len() == 1);
    let six: Matrix6<Fixed> = black_box(Matrix6Trait::identity());
    let d6: DMatrix<Fixed> = six.into();
    assert!(d6 == DMatrixTrait::identity(6, 6));
    // DVector / RowDVector as DMatrix.
    let c: DMatrix<Fixed> = dv.into();
    assert!(c.shape() == (3, 1) && c.as_slice() == dv.as_slice());
    let c: DMatrix<Fixed> = dr.into();
    assert!(c.shape() == (1, 3));
}

#[test]
fn test_serde_order_is_upstream_vec_storage() {
    let m = dmi(2, 1, array![1, 2].span());
    let mut out: Array<felt252> = array![];
    m.serialize(ref out);
    // data (length-prefixed, column-major), nrows, ncols.
    let one: felt252 = 0x100000000;
    assert!(out == array![2, one, 2 * one, 2, 1]);
    let mut s = out.span();
    let back: DMatrix<Fixed> = Serde::deserialize(ref s).unwrap();
    assert!(back == m);
}
