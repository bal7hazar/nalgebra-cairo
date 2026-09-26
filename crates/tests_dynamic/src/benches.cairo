//! Gas benchmarks of the dynamic matrices at sizes 3, 6 and 16 (DESIGN D5), and of the static
//! forms that come with them. Operands are built from `black_box`ed spans in every variant of a
//! group, the baseline included, so `net` is the operation alone.

use fixed::Fixed;
use nalgebra::{
    DMatrix, DMatrixTrait, DVector, DVectorTrait, InsertFixedColumns, Matrix3x4,
    Matrix3x4DynamicTrait, Matrix3x4Trait, Matrix3x6, Matrix6, Matrix6Trait, MatrixMul,
};
use nalgebra_testing::black_box;
use nalgebra_tests_utils::int;
use crate::helpers::ramp;

fn square(n: usize) -> DMatrix<Fixed> {
    DMatrixTrait::from_column_slice(n, n, ramp(n * n))
}

fn vector(n: usize) -> DVector<Fixed> {
    DVectorTrait::from_column_slice(ramp(n))
}

// --- products ------------------------------------------------------------------------------------

#[test]
#[inline(never)]
fn bench_dmatrix_mul3__baseline() {
    let a = square(3);
    let b = square(3);
    assert!(a.len() == b.len());
}

#[test]
#[inline(never)]
fn bench_dmatrix_mul3__library() {
    let a = square(3);
    let b = square(3);
    assert!((a * b).len() == 9);
}

#[test]
#[inline(never)]
fn bench_dmatrix_mul6__baseline() {
    let a = square(6);
    let b = square(6);
    assert!(a.len() == b.len());
}

#[test]
#[inline(never)]
fn bench_dmatrix_mul6__library() {
    let a = square(6);
    let b = square(6);
    assert!((a * b).len() == 36);
}

#[test]
#[inline(never)]
fn bench_dmatrix_mul16__baseline() {
    let a = square(16);
    let b = square(16);
    assert!(a.len() == b.len());
}

#[test]
#[inline(never)]
fn bench_dmatrix_mul16__library() {
    let a = square(16);
    let b = square(16);
    assert!((a * b).len() == 256);
}

#[test]
#[inline(never)]
fn bench_dmatrix_mul_vec3__baseline() {
    let a = square(3);
    let v = vector(3);
    assert!(a.ncols() == v.len());
}

#[test]
#[inline(never)]
fn bench_dmatrix_mul_vec3__library() {
    let a = square(3);
    let v = vector(3);
    assert!(a.mul_mat(v).len() == 3);
}

#[test]
#[inline(never)]
fn bench_dmatrix_mul_vec6__baseline() {
    let a = square(6);
    let v = vector(6);
    assert!(a.ncols() == v.len());
}

#[test]
#[inline(never)]
fn bench_dmatrix_mul_vec6__library() {
    let a = square(6);
    let v = vector(6);
    assert!(a.mul_mat(v).len() == 6);
}

#[test]
#[inline(never)]
fn bench_dmatrix_mul_vec16__baseline() {
    let a = square(16);
    let v = vector(16);
    assert!(a.ncols() == v.len());
}

#[test]
#[inline(never)]
fn bench_dmatrix_mul_vec16__library() {
    let a = square(16);
    let v = vector(16);
    assert!(a.mul_mat(v).len() == 16);
}

// --- element-wise, reductions
// ---------------------------------------------------------------------

#[test]
#[inline(never)]
fn bench_dmatrix_add6__baseline() {
    let a = square(6);
    let b = square(6);
    assert!(a.len() == b.len());
}

#[test]
#[inline(never)]
fn bench_dmatrix_add6__library() {
    let a = square(6);
    let b = square(6);
    assert!((a + b).len() == 36);
}

#[test]
#[inline(never)]
fn bench_dmatrix_add16__baseline() {
    let a = square(16);
    let b = square(16);
    assert!(a.len() == b.len());
}

#[test]
#[inline(never)]
fn bench_dmatrix_add16__library() {
    let a = square(16);
    let b = square(16);
    assert!((a + b).len() == 256);
}

#[test]
#[inline(never)]
fn bench_dvector_dot3__baseline() {
    let a = vector(3);
    let b = vector(3);
    assert!(a.len() == b.len());
}

#[test]
#[inline(never)]
fn bench_dvector_dot3__library() {
    let a = vector(3);
    let b = vector(3);
    assert!(a.dot(b) == int(14));
}

#[test]
#[inline(never)]
fn bench_dvector_dot6__baseline() {
    let a = vector(6);
    let b = vector(6);
    assert!(a.len() == b.len());
}

#[test]
#[inline(never)]
fn bench_dvector_dot6__library() {
    let a = vector(6);
    let b = vector(6);
    assert!(a.dot(b) == int(19));
}

#[test]
#[inline(never)]
fn bench_dvector_dot16__baseline() {
    let a = vector(16);
    let b = vector(16);
    assert!(a.len() == b.len());
}

#[test]
#[inline(never)]
fn bench_dvector_dot16__library() {
    let a = vector(16);
    let b = vector(16);
    assert!(a.dot(b) == int(69));
}

#[test]
#[inline(never)]
fn bench_dvector_norm16__baseline() {
    let a = vector(16);
    assert!(a.len() == 16);
}

#[test]
#[inline(never)]
fn bench_dvector_norm16__library() {
    let a = vector(16);
    assert!(a.norm() > int(8));
}

#[test]
#[inline(never)]
fn bench_dmatrix_transpose6__baseline() {
    let a = square(6);
    assert!(a.len() == 36);
}

#[test]
#[inline(never)]
fn bench_dmatrix_transpose6__library() {
    let a = square(6);
    assert!(a.transpose().len() == 36);
}

#[test]
#[inline(never)]
fn bench_dmatrix_transpose16__baseline() {
    let a = square(16);
    assert!(a.len() == 256);
}

#[test]
#[inline(never)]
fn bench_dmatrix_transpose16__library() {
    let a = square(16);
    assert!(a.transpose().len() == 256);
}

#[test]
#[inline(never)]
fn bench_dmatrix_index__baseline() {
    let a = square(6);
    let (i, j) = black_box((4_usize, 3_usize));
    assert!(a.len() == 36 && i + j == 7);
}

#[test]
#[inline(never)]
fn bench_dmatrix_index__library() {
    let a = square(6);
    let (i, j) = black_box((4_usize, 3_usize));
    assert!(a[(i, j)] == int(-2));
}

// --- edition
// --------------------------------------------------------------------------------------

#[test]
#[inline(never)]
fn bench_dmatrix_insert_columns6__baseline() {
    let a = square(6);
    let (i, n) = black_box((2_usize, 2_usize));
    assert!(a.len() == 36 && i == n);
}

#[test]
#[inline(never)]
fn bench_dmatrix_insert_columns6__library() {
    let a = square(6);
    let (i, n) = black_box((2_usize, 2_usize));
    assert!(a.insert_columns(i, n, int(0)).len() == 48);
}

#[test]
#[inline(never)]
fn bench_dmatrix_insert_rows6__baseline() {
    let a = square(6);
    let (i, n) = black_box((2_usize, 2_usize));
    assert!(a.len() == 36 && i == n);
}

#[test]
#[inline(never)]
fn bench_dmatrix_insert_rows6__library() {
    let a = square(6);
    let (i, n) = black_box((2_usize, 2_usize));
    assert!(a.insert_rows(i, n, int(0)).len() == 48);
}

#[test]
#[inline(never)]
fn bench_dmatrix_remove_rows16__baseline() {
    let a = square(16);
    let (i, n) = black_box((3_usize, 2_usize));
    assert!(a.len() == 256 && i > n);
}

#[test]
#[inline(never)]
fn bench_dmatrix_remove_rows16__library() {
    let a = square(16);
    let (i, n) = black_box((3_usize, 2_usize));
    assert!(a.remove_rows(i, n).len() == 224);
}

#[test]
#[inline(never)]
fn bench_dmatrix_resize6__baseline() {
    let a = square(6);
    let (r, c) = black_box((7_usize, 8_usize));
    assert!(a.len() == 36 && r < c);
}

#[test]
#[inline(never)]
fn bench_dmatrix_resize6__library() {
    let a = square(6);
    let (r, c) = black_box((7_usize, 8_usize));
    assert!(a.resize(r, c, int(9)).len() == 56);
}

#[test]
#[inline(never)]
fn bench_dmatrix_resize16__baseline() {
    let a = square(16);
    let (r, c) = black_box((12_usize, 18_usize));
    assert!(a.len() == 256 && r < c);
}

#[test]
#[inline(never)]
fn bench_dmatrix_resize16__library() {
    let a = square(16);
    let (r, c) = black_box((12_usize, 18_usize));
    assert!(a.resize(r, c, int(9)).len() == 216);
}

// --- static shapes
// --------------------------------------------------------------------------------

fn m34() -> Matrix3x4<Fixed> {
    Matrix3x4Trait::from_column_slice(ramp(12))
}

/// `insert_fixed_columns::<2>(i, val)` written directly: one `match` on `i` selecting a
/// `Matrix3x6` literal (the per-pair alternative to the `Matrix6` canvas of the library).
fn insert2_direct(m: Matrix3x4<Fixed>, i: usize, val: Fixed) -> Matrix3x6<Fixed> {
    match i {
        0 => Matrix3x6 {
            m11: val,
            m21: val,
            m31: val,
            m12: val,
            m22: val,
            m32: val,
            m13: m.m11,
            m23: m.m21,
            m33: m.m31,
            m14: m.m12,
            m24: m.m22,
            m34: m.m32,
            m15: m.m13,
            m25: m.m23,
            m35: m.m33,
            m16: m.m14,
            m26: m.m24,
            m36: m.m34,
        },
        1 => Matrix3x6 {
            m11: m.m11,
            m21: m.m21,
            m31: m.m31,
            m12: val,
            m22: val,
            m32: val,
            m13: val,
            m23: val,
            m33: val,
            m14: m.m12,
            m24: m.m22,
            m34: m.m32,
            m15: m.m13,
            m25: m.m23,
            m35: m.m33,
            m16: m.m14,
            m26: m.m24,
            m36: m.m34,
        },
        2 => Matrix3x6 {
            m11: m.m11,
            m21: m.m21,
            m31: m.m31,
            m12: m.m12,
            m22: m.m22,
            m32: m.m32,
            m13: val,
            m23: val,
            m33: val,
            m14: val,
            m24: val,
            m34: val,
            m15: m.m13,
            m25: m.m23,
            m35: m.m33,
            m16: m.m14,
            m26: m.m24,
            m36: m.m34,
        },
        3 => Matrix3x6 {
            m11: m.m11,
            m21: m.m21,
            m31: m.m31,
            m12: m.m12,
            m22: m.m22,
            m32: m.m32,
            m13: m.m13,
            m23: m.m23,
            m33: m.m33,
            m14: val,
            m24: val,
            m34: val,
            m15: val,
            m25: val,
            m35: val,
            m16: m.m14,
            m26: m.m24,
            m36: m.m34,
        },
        4 => Matrix3x6 {
            m11: m.m11,
            m21: m.m21,
            m31: m.m31,
            m12: m.m12,
            m22: m.m22,
            m32: m.m32,
            m13: m.m13,
            m23: m.m23,
            m33: m.m33,
            m14: m.m14,
            m24: m.m24,
            m34: m.m34,
            m15: val,
            m25: val,
            m35: val,
            m16: val,
            m26: val,
            m36: val,
        },
        _ => core::panic_with_felt252('nalgebra: index out of bounds'),
    }
}

#[test]
#[inline(never)]
fn bench_matrix3x4_insert_fixed_columns__baseline() {
    let m = m34();
    let i = black_box(1_usize);
    assert!(m.m11 == int(-3) && i == 1);
}

#[test]
#[inline(never)]
fn bench_matrix3x4_insert_fixed_columns__library() {
    let m = m34();
    let i = black_box(1_usize);
    let r: Matrix3x6<Fixed> = m.insert_fixed_columns(i, int(0));
    assert!(r.m13 == int(0) && r.m14 == m.m12);
}

#[test]
#[inline(never)]
fn bench_matrix3x4_insert_fixed_columns__alt_direct() {
    let m = m34();
    let i = black_box(1_usize);
    let r = insert2_direct(m, i, int(0));
    assert!(r.m13 == int(0) && r.m14 == m.m12);
}

#[test]
fn test_insert_fixed_columns_direct_agrees() {
    let m = m34();
    let mut i: usize = 0;
    while i != 5 {
        let r: Matrix3x6<Fixed> = m.insert_fixed_columns(i, int(7));
        assert!(r == insert2_direct(m, i, int(7)));
        i += 1;
    }
}

#[test]
#[inline(never)]
fn bench_matrix3x4_insert_columns__baseline() {
    let m = m34();
    let (i, n) = black_box((1_usize, 2_usize));
    assert!(m.m11 == int(-3) && i < n);
}

#[test]
#[inline(never)]
fn bench_matrix3x4_insert_columns__library() {
    let m = m34();
    let (i, n) = black_box((1_usize, 2_usize));
    assert!(m.insert_columns(i, n, int(0)).len() == 18);
}

#[test]
#[inline(never)]
fn bench_matrix6_into_dmatrix__baseline() {
    let m: Matrix6<Fixed> = Matrix6Trait::from_column_slice(ramp(36));
    assert!(m.m11 == int(-3));
}

#[test]
#[inline(never)]
fn bench_matrix6_into_dmatrix__library() {
    let m: Matrix6<Fixed> = Matrix6Trait::from_column_slice(ramp(36));
    let d: DMatrix<Fixed> = m.into();
    assert!(d.len() == 36);
}
