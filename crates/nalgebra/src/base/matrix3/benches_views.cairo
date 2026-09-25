//! Gas of the public `column(j)` / `row(i)` (WP 8.2c) against the crate-internal
//! `column1..3` / `row1..3` of `Matrix3InternalTrait` (WP 8.0), which `linalg` calls.
//!
//! The internal helpers take their position at compile time and only move three fields; the
//! public methods take a runtime index and select the literal through ONE `match`, which costs
//! more (and `row(i)` returns a `RowVector3`, `row2()` a `Vector3`). Neither is a drop-in
//! replacement of the other, so the internal helpers stay, with this measurement.

use fixed::Fixed;
use nalgebra_testing::black_box;
use crate::base::matrix_test_utils::{int, m3i};
use crate::base::row_vector3::{RowVector3, RowVector3Trait};
use crate::base::vector3::Vector3;
use super::{Matrix3, Matrix3InternalTrait, Matrix3Trait};

fn input() -> Matrix3<Fixed> {
    m3i([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
}

#[test]
#[inline(never)]
fn bench_matrix3_column_internal__baseline() {
    let _m = black_box(input());
    let _j: usize = black_box(1);
    let e: Vector3<Fixed> = black_box(Vector3 { x: int(2), y: int(5), z: int(8) });
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_matrix3_column_internal__column2() {
    let m = black_box(input());
    let _j: usize = black_box(1);
    let e: Vector3<Fixed> = black_box(Vector3 { x: int(2), y: int(5), z: int(8) });
    assert!(m.column2() == e);
}

#[test]
#[inline(never)]
fn bench_matrix3_column_internal__column() {
    let m = black_box(input());
    let j: usize = black_box(1);
    let e: Vector3<Fixed> = black_box(Vector3 { x: int(2), y: int(5), z: int(8) });
    assert!(m.column(j) == e);
}

#[test]
#[inline(never)]
fn bench_matrix3_row_internal__baseline() {
    let _m = black_box(input());
    let _i: usize = black_box(1);
    let e: Vector3<Fixed> = black_box(Vector3 { x: int(4), y: int(5), z: int(6) });
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_matrix3_row_internal__row2() {
    let m = black_box(input());
    let _i: usize = black_box(1);
    let e: Vector3<Fixed> = black_box(Vector3 { x: int(4), y: int(5), z: int(6) });
    assert!(m.row2() == e);
}

#[test]
#[inline(never)]
fn bench_matrix3_row_internal__row() {
    let m = black_box(input());
    let i: usize = black_box(1);
    let e: Vector3<Fixed> = black_box(Vector3 { x: int(4), y: int(5), z: int(6) });
    let r: RowVector3<Fixed> = m.row(i);
    assert!(r.transpose() == e);
}
