use fixed::Fixed;
use glam::{Mat2, Vec2};
use nalgebra::UnitComplex;
use nalgebra_testing::black_box;
use crate::glam_unit_complex::*;
use super::int;

#[test]
fn test_unit_complex_into_mat2_is_the_rotation_matrix() {
    // re = 0, im = 1 is the rotation by 90 degrees: x_axis = (0, 1), y_axis = (-1, 0).
    let m: Mat2 = black_box(UnitComplex { re: int(0), im: int(1) }).into();
    assert!(
        m == Mat2 { x_axis: Vec2 { x: int(0), y: int(1) }, y_axis: Vec2 { x: int(-1), y: int(0) } },
    );
}

#[test]
fn test_mat2_into_unit_complex_reads_the_first_column() {
    // The second column is ignored; the first one is normalized: (0, 3) / 3 = (0, 1) exactly.
    let u: UnitComplex<Fixed> = black_box(
        Mat2 { x_axis: Vec2 { x: int(0), y: int(3) }, y_axis: Vec2 { x: int(7), y: int(9) } },
    )
        .into();
    assert!(u == UnitComplex { re: int(0), im: int(1) });
    let u: UnitComplex<Fixed> = black_box(
        Mat2 { x_axis: Vec2 { x: int(-2), y: int(0) }, y_axis: Vec2 { x: int(0), y: int(-2) } },
    )
        .into();
    assert!(u == UnitComplex { re: int(-1), im: int(0) });
}

#[test]
#[should_panic(expected: ('Fixed: division by zero',))]
fn test_mat2_with_zero_first_column_panics() {
    let _u: UnitComplex<Fixed> = black_box(
        Mat2 { x_axis: Vec2 { x: int(0), y: int(0) }, y_axis: Vec2 { x: int(0), y: int(1) } },
    )
        .into();
}
