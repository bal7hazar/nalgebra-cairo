//! Arithmetic of the dynamic matrices (upstream `base/ops.rs`, `base/norm.rs`,
//! `base/componentwise.rs`): exact values on integers, the products of every pair of dynamic types
//! against the static products, and the dimension-mismatch panics.

use fixed::Fixed;
use nalgebra::{
    DMatrix, DMatrixTrait, DVector, DVectorTrait, Matrix3, Matrix3Trait, Matrix4x3, Matrix4x3Trait,
    MatrixIndex, MatrixMul, RowDVector, RowDVectorTrait, Vector3, Vector3Trait,
};
use nalgebra_testing::black_box;
use nalgebra_tests_utils::{fx, int};
use simba::scalar::Real;
use crate::helpers::{dmi, dvi, ints, ramp, rdvi};

#[test]
fn test_dmatrix_add_sub_neg() {
    let a = dmi(2, 3, array![1, 2, 3, 4, 5, 6].span());
    let b = dmi(2, 3, array![6, 5, 4, 3, 2, 1].span());
    assert!(a + b == dmi(2, 3, array![7, 7, 7, 7, 7, 7].span()));
    assert!(a - b == dmi(2, 3, array![-5, -3, -1, 1, 3, 5].span()));
    assert!(-a == dmi(2, 3, array![-1, -2, -3, -4, -5, -6].span()));
    // More than one 4-chunk plus a tail.
    let c: DMatrix<Fixed> = DMatrixTrait::from_column_slice(3, 3, ramp(9));
    assert!((c + c) - c == c);
}

#[test]
#[should_panic(expected: 'nalgebra: dimension mismatch')]
fn test_dmatrix_add_mismatch() {
    let a = dmi(2, 3, array![1, 2, 3, 4, 5, 6].span());
    let b = dmi(3, 2, array![1, 2, 3, 4, 5, 6].span());
    let _c = a + b;
}

#[test]
#[should_panic(expected: 'nalgebra: dimension mismatch')]
fn test_dmatrix_mul_mismatch() {
    let a = dmi(2, 3, array![1, 2, 3, 4, 5, 6].span());
    let _c = a * a;
}

#[test]
#[should_panic(expected: 'nalgebra: dimension mismatch')]
fn test_dmatrix_mul_vec_mismatch() {
    let a = dmi(2, 3, array![1, 2, 3, 4, 5, 6].span());
    let _c = a.mul_mat(dvi(array![1, 2].span()));
}

#[test]
fn test_dmatrix_scale_unscale_component_mul() {
    let a = dmi(2, 2, array![1, -2, 3, 4].span());
    assert!(a.scale(int(3)) == dmi(2, 2, array![3, -6, 9, 12].span()));
    assert!(a.scale(int(4)).unscale(int(4)) == a);
    // 1 / 3 rounds to nearest.
    let third = a.unscale(int(3));
    assert!(third[0] == Real::div(int(1), int(3)));
    assert!(a.component_mul(a) == dmi(2, 2, array![1, 4, 9, 16].span()));
}

#[test]
fn test_dmatrix_dot_norms() {
    let a = dmi(2, 2, array![1, 2, 2, 4].span());
    assert!(a.dot(a) == int(25) && a.norm_squared() == int(25) && a.norm() == int(5));
    // Squares far beyond the Q32.32 range: only the root must fit.
    let big = dmi(1, 2, array![300000, 400000].span());
    assert!(big.norm() == int(500000));
}

#[test]
fn test_dmatrix_abs_diff_eq() {
    let a = dmi(1, 2, array![1, 2].span());
    let b: DMatrix<Fixed> = DMatrixTrait::from_row_slice(
        1, 2, black_box(array![fx(0x100000003), int(2)].span()),
    );
    assert!(a.abs_diff_eq(b, 3) && !a.abs_diff_eq(b, 2));
    assert!(!a.abs_diff_eq(a.transpose(), 0));
}

/// The products of every pair of dynamic types, against the static products (bit-identical).
#[test]
fn test_products_match_static() {
    let m3: Matrix3<Fixed> = Matrix3Trait::from_column_slice(ramp(9));
    let v3: Vector3<Fixed> = Vector3Trait::from_column_slice(ramp(3));
    let r43: Matrix4x3<Fixed> = Matrix4x3Trait::from_column_slice(ramp(12));
    let d3: DMatrix<Fixed> = m3.into();
    let dv3: DVector<Fixed> = v3.into();
    let d43: DMatrix<Fixed> = r43.into();
    // Square (dispatched), rectangular (loops), matrix-vector both ways.
    let p: DMatrix<Fixed> = (m3 * m3).into();
    assert!(d3 * d3 == p);
    let p: DMatrix<Fixed> = r43.mul_mat(m3).into();
    assert!(d43 * d3 == p && d43.mul_mat(d3) == p);
    let p: DVector<Fixed> = m3.mul_mat(v3).into();
    assert!(d3.mul_mat(dv3) == p);
    let p: DVector<Fixed> = r43.mul_mat(v3).into();
    assert!(d43.mul_mat(dv3) == p);
    // Row vector times matrix, outer product.
    let r: RowDVector<Fixed> = dv3.transpose();
    let p: DVector<Fixed> = (d3.transpose().mul_mat(dv3));
    assert!(r.mul_mat(d3).transpose() == p);
    let outer: DMatrix<Fixed> = dv3.mul_mat(r);
    let expected: DMatrix<Fixed> = v3.mul_mat(v3.transpose()).into();
    assert!(outer == expected);
    // 7x7 (the loops, square above the dispatch).
    let a7: DMatrix<Fixed> = DMatrixTrait::from_column_slice(7, 7, ramp(49));
    let id: DMatrix<Fixed> = DMatrixTrait::identity(7, 7);
    assert!(a7 * id == a7 && id * a7 == a7);
    // An empty inner dimension gives zeros.
    let e: DMatrix<Fixed> = DMatrixTrait::zeros(2, 0);
    let f: DMatrix<Fixed> = DMatrixTrait::zeros(0, 3);
    assert!(e * f == DMatrixTrait::zeros(2, 3));
}

#[test]
fn test_dvector_ops() {
    let a = dvi(array![1, 2, 3].span());
    let b = dvi(array![4, 5, 6].span());
    assert!(a + b == dvi(array![5, 7, 9].span()) && b - a == dvi(array![3, 3, 3].span()));
    assert!(-a == dvi(array![-1, -2, -3].span()));
    assert!(a.dot(b) == int(32) && a.norm_squared() == int(14));
    assert!(a.scale(int(2)) == dvi(array![2, 4, 6].span()));
    assert!(a.component_mul(b) == dvi(array![4, 10, 18].span()));
    assert!(a.abs_diff_eq(a, 0) && !a.abs_diff_eq(b, 0));
    let n = dvi(array![3, 4].span()).norm();
    assert!(
        n == int(5) && dvi(array![3, 4].span()).unscale(int(5))[1] == Real::div(int(4), int(5)),
    );
}

#[test]
#[should_panic(expected: 'nalgebra: dimension mismatch')]
fn test_dvector_dot_mismatch() {
    let _d = dvi(array![1, 2, 3].span()).dot(dvi(array![1, 2].span()));
}

#[test]
fn test_row_dvector_ops() {
    let a = rdvi(array![1, 2, 3].span());
    let b = rdvi(array![4, 5, 6].span());
    assert!(a + b == rdvi(array![5, 7, 9].span()) && b - a == rdvi(array![3, 3, 3].span()));
    assert!(-a == rdvi(array![-1, -2, -3].span()) && a.dot(b) == int(32));
    assert!(a[2] == int(3) && a.get(3).is_none() && a.index(0) == int(1));
    assert!(a.norm() == Real::sqrt(int(14)));
    assert!(ints(array![1].span()).len() == 1);
}
