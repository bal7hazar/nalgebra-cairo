//! The members of the static shapes that come with the dynamic matrices
//! (`base/dynamic/shapes.cairo`, generated): the edition forms with a dynamic result, the
//! fixed-size edition forms (`InsertFixedColumns`...), `from_vec` / `from_iterator` /
//! `from_row_iterator`, `len`, `is_empty` and `compress_*`, on a sample of shapes (every shape
//! runs the same template).

use fixed::Fixed;
use nalgebra::{
    DMatrix, DMatrixTrait, DVector, InsertFixedColumns, InsertFixedRows, Matrix1,
    Matrix1DynamicTrait, Matrix2x3, Matrix2x3DynamicTrait, Matrix2x3Trait, Matrix3x2, Matrix3x4,
    Matrix3x4DynamicTrait, Matrix3x4Trait, Matrix3x6, Matrix3x6Trait, Matrix4x6, Matrix4x6Trait,
    Matrix5x3, Matrix5x3Trait, Matrix6, Matrix6DynamicTrait, Matrix6Trait, RemoveFixedColumns,
    RemoveFixedRows, RowDVector, RowVector2, RowVector3, RowVector3DynamicTrait, RowVector3Trait,
    Vector2, Vector2DynamicTrait, Vector3, Vector3DynamicTrait, Vector3Trait,
};
use nalgebra_testing::black_box;
use nalgebra_tests_utils::int;
use crate::helpers::{dmi, dvi, ints, rdvi};

fn m23() -> Matrix2x3<Fixed> {
    Matrix2x3Trait::from_row_slice(ints(array![1, 2, 3, 4, 5, 6].span()))
}

fn m34() -> Matrix3x4<Fixed> {
    Matrix3x4Trait::from_row_slice(ints(array![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12].span()))
}

#[test]
fn test_static_from_vec_iterator_len() {
    let mut arr: Array<Fixed> = array![];
    arr.append_span(ints(array![1, 4, 2, 5, 3, 6].span()));
    assert!(Matrix2x3DynamicTrait::from_vec(arr) == m23());
    assert!(Matrix2x3DynamicTrait::from_iterator(ints(array![1, 4, 2, 5, 3, 6].span())) == m23());
    assert!(
        Matrix2x3DynamicTrait::from_row_iterator(ints(array![1, 2, 3, 4, 5, 6].span())) == m23(),
    );
    assert!(m23().len() == 6 && !m23().is_empty());
    let one: Matrix1<Fixed> = black_box(Matrix1 { x: int(1) });
    assert!(one.len() == 1 && !one.is_empty());
    let six: Matrix6<Fixed> = black_box(Matrix6Trait::identity());
    assert!(six.len() == 36);
}

#[test]
#[should_panic(expected: 'nalgebra: wrong slice length')]
fn test_static_from_vec_wrong_length() {
    let mut arr: Array<Fixed> = array![];
    arr.append_span(ints(array![1, 2, 3].span()));
    let _v: Vector2<Fixed> = Vector2DynamicTrait::from_vec(black_box(arr));
}

#[test]
fn test_static_dynamic_edition_matrix() {
    let m = m23();
    let v = int(0);
    let wide: DMatrix<Fixed> = m.insert_columns(black_box(1), 2, v);
    assert!(wide == dmi(2, 5, array![1, 0, 0, 2, 3, 4, 0, 0, 5, 6].span()));
    let tall: DMatrix<Fixed> = m.insert_rows(black_box(2), 1, v);
    assert!(tall == dmi(3, 3, array![1, 2, 3, 4, 5, 6, 0, 0, 0].span()));
    assert!(m.remove_columns(black_box(0), 2) == dmi(2, 1, array![3, 6].span()));
    assert!(m.remove_rows(black_box(1), 1) == dmi(1, 3, array![1, 2, 3].span()));
    assert!(
        m
            .remove_columns_at(
                black_box(array![1_usize].span()),
            ) == dmi(2, 2, array![1, 3, 4, 6].span()),
    );
    assert!(
        m.remove_rows_at(black_box(array![0_usize].span())) == dmi(1, 3, array![4, 5, 6].span()),
    );
    assert!(
        m.resize_horizontally(black_box(4), v) == dmi(2, 4, array![1, 2, 3, 0, 4, 5, 6, 0].span()),
    );
    assert!(m.resize_vertically(black_box(1), v) == dmi(1, 3, array![1, 2, 3].span()));
}

#[test]
#[should_panic(expected: 'nalgebra: index out of bounds')]
fn test_static_insert_columns_out_of_range() {
    let _m = m23().insert_columns(black_box(4), 1, int(0));
}

#[test]
fn test_static_dynamic_edition_vectors() {
    let v: Vector3<Fixed> = black_box(Vector3Trait::new(int(1), int(2), int(3)));
    let z = int(0);
    // One column: the row edition gives a `DVector` (`MatrixXx1`), the column edition a
    // `Matrix3xX`.
    let r: DVector<Fixed> = v.insert_rows(black_box(0), 1, z);
    assert!(r == dvi(array![0, 1, 2, 3].span()));
    assert!(v.remove_rows_at(black_box(array![1_usize].span())) == dvi(array![1, 3].span()));
    assert!(v.resize_vertically(black_box(2), z) == dvi(array![1, 2].span()));
    let c: DMatrix<Fixed> = v.insert_columns(black_box(1), 1, z);
    assert!(c == dmi(3, 2, array![1, 0, 2, 0, 3, 0].span()));
    assert!(v.remove_columns(black_box(0), 1).shape() == (3, 0));
    let w: RowVector3<Fixed> = black_box(RowVector3Trait::new(int(1), int(2), int(3)));
    // One row: the column edition gives a `RowDVector` (`Matrix1xX`).
    let rr: RowDVector<Fixed> = w.insert_columns(black_box(3), 1, z);
    assert!(rr == rdvi(array![1, 2, 3, 0].span()));
    assert!(w.resize_horizontally(black_box(1), z) == rdvi(array![1].span()));
    let rc: DMatrix<Fixed> = w.insert_rows(black_box(0), 1, z);
    assert!(rc == dmi(2, 3, array![0, 0, 0, 1, 2, 3].span()));
    let one: Matrix1<Fixed> = black_box(Matrix1 { x: int(5) });
    let a: DVector<Fixed> = one.insert_rows(1, 1, z);
    let b: RowDVector<Fixed> = one.insert_columns(0, 1, z);
    assert!(a == dvi(array![5, 0].span()) && b == rdvi(array![0, 5].span()));
}

#[test]
fn test_static_insert_fixed() {
    let m = m34();
    let v = int(0);
    let c: Matrix3x6<Fixed> = m.insert_fixed_columns(black_box(1), v);
    let expected: Matrix3x6<Fixed> = Matrix3x6Trait::from_row_slice(
        ints(array![1, 0, 0, 2, 3, 4, 5, 0, 0, 6, 7, 8, 9, 0, 0, 10, 11, 12].span()),
    );
    assert!(c == expected);
    let at_end: Matrix3x6<Fixed> = m.insert_fixed_columns(black_box(4), v);
    let expected: Matrix3x6<Fixed> = Matrix3x6Trait::from_row_slice(
        ints(array![1, 2, 3, 4, 0, 0, 5, 6, 7, 8, 0, 0, 9, 10, 11, 12, 0, 0].span()),
    );
    assert!(at_end == expected);
    let r: Matrix5x3<Fixed> = m23().insert_fixed_rows(black_box(1), v);
    let expected: Matrix5x3<Fixed> = Matrix5x3Trait::from_row_slice(
        ints(array![1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 5, 6].span()),
    );
    // `Matrix2x3 + 3 rows` is a `Matrix5x3`.
    assert!(r == expected);
    // Agreement with the dynamic forms.
    let d: DMatrix<Fixed> = c.into();
    assert!(d == m.insert_columns(1, 2, v));
}

#[test]
fn test_static_remove_fixed() {
    let m = m34();
    let c: Matrix3x2<Fixed> = m.remove_fixed_columns(black_box(1));
    let d: DMatrix<Fixed> = c.into();
    assert!(d == dmi(3, 2, array![1, 4, 5, 8, 9, 12].span()));
    let six: Matrix6<Fixed> = Matrix6Trait::from_row_slice(crate::helpers::ramp(36));
    let r: Matrix4x6<Fixed> = six.remove_fixed_rows(black_box(4));
    let expected: Matrix4x6<Fixed> = Matrix4x6Trait::from_row_slice(crate::helpers::ramp(24));
    assert!(r == expected);
    let w: RowVector3<Fixed> = black_box(RowVector3Trait::new(int(1), int(2), int(3)));
    let rw: RowVector2<Fixed> = w.remove_fixed_columns(black_box(0));
    assert!(rw == RowVector2 { x: int(2), y: int(3) });
    let v: Vector3<Fixed> = black_box(Vector3Trait::new(int(1), int(2), int(3)));
    let one: Matrix1<Fixed> = v.remove_fixed_rows(black_box(1));
    assert!(one == Matrix1 { x: int(1) });
}

#[test]
#[should_panic(expected: 'nalgebra: index out of bounds')]
fn test_static_insert_fixed_out_of_range() {
    let _c: Matrix3x6<Fixed> = m34().insert_fixed_columns(black_box(5), int(0));
}

#[test]
#[should_panic(expected: 'nalgebra: index out of bounds')]
fn test_static_remove_fixed_out_of_range() {
    let _c: Matrix3x2<Fixed> = m34().remove_fixed_columns(black_box(3));
}

#[test]
fn test_static_compress() {
    let m = m23();
    let sums: RowVector3<Fixed> = m.compress_rows(|c: Vector2<Fixed>| -> Fixed {
        c.x + c.y
    });
    assert!(sums == RowVector3Trait::new(int(5), int(7), int(9)));
    let sums_tr: Vector3<Fixed> = m.compress_rows_tr(|c: Vector2<Fixed>| -> Fixed {
        c.x + c.y
    });
    assert!(sums_tr == Vector3Trait::new(int(5), int(7), int(9)));
    let init: Vector2<Fixed> = black_box(Vector2 { x: int(0), y: int(0) });
    let rows: Vector2<Fixed> = m
        .compress_columns(
            init, |acc: Vector2<Fixed>, c: Vector2<Fixed>| -> Vector2<Fixed> {
                acc + c
            },
        );
    assert!(rows == Vector2 { x: int(6), y: int(15) });
}
