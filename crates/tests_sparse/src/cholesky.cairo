//! `CsCholesky`: symbolic analysis (pattern of `L`, fill-in), numeric factorizations (left- and
//! up-looking entry points), failures, and agreement with the dense Cholesky of `linalg`.

use fixed::Fixed;
use nalgebra::sparse::{CsCholesky, CsCholeskyTrait, CsMatrix, CsMatrixTrait};
use nalgebra::{Cholesky4Trait, Matrix4, Matrix4CholeskyTrait};
use nalgebra_testing::black_box;
use crate::helpers::{cs, dense, ints, max_ulp};

/// `A = L Lᵀ` with `L = [[2, 0, 0], [1, 3, 0], [0, 2, 1]]`: `[[4, 2, 0], [2, 10, 6], [0, 6, 5]]`.
fn spd3() -> CsMatrix<Fixed> {
    cs(
        3,
        3,
        array![(0, 0, 4), (1, 0, 2), (0, 1, 2), (1, 1, 10), (2, 1, 6), (1, 2, 6), (2, 2, 5)].span(),
    )
}

#[test]
fn test_cs_cholesky_new_exact() {
    let chol = CsCholeskyTrait::new(@spd3());
    let l = chol.l().unwrap();
    assert!(l.shape() == (3, 3));
    assert!(l.p() == array![0, 2, 4].span());
    assert!(l.i() == array![0, 1, 1, 2, 2].span());
    assert!(l.values() == ints(array![2, 1, 3, 2, 1].span()));
    assert!(chol.unwrap_l().unwrap() == l);
}

#[test]
fn test_cs_cholesky_symbolic_then_numeric() {
    let a = spd3();
    let mut chol: CsCholesky<Fixed> = CsCholeskyTrait::new_symbolic(@a);
    assert!(chol.l().is_none());
    assert!(chol.decompose_up_looking(a.values()));
    assert!(chol.l().unwrap().values() == ints(array![2, 1, 3, 2, 1].span()));
    // Same pattern, values times 4: L times 2.
    assert!(chol.decompose_left_looking(a.scale(black_box(nalgebra_tests_utils::int(4))).values()));
    assert!(chol.l().unwrap().values() == ints(array![4, 2, 6, 4, 2].span()));
}

#[test]
fn test_cs_cholesky_triangles() {
    // Values [[4, 2], [6, 10]]: the up-looking entry point reads the upper triangle (2), the
    // left-looking one the lower triangle (6), like upstream.
    let a = cs(2, 2, array![(0, 0, 4), (1, 0, 6), (0, 1, 2), (1, 1, 10)].span());
    let mut chol: CsCholesky<Fixed> = CsCholeskyTrait::new_symbolic(@a);
    assert!(chol.decompose_up_looking(a.values()));
    assert!(chol.l().unwrap().values() == ints(array![2, 1, 3].span()));
    assert!(chol.decompose_left_looking(a.values()));
    assert!(chol.l().unwrap().values() == ints(array![2, 3, 1].span()));
}

#[test]
fn test_cs_cholesky_fill_in() {
    // Arrow matrix, dense first row and column: L is full lower-triangular (fill-in).
    let a = cs(
        4,
        4,
        array![
            (0, 0, 4), (1, 0, 1), (2, 0, 1), (3, 0, 1), (0, 1, 1), (0, 2, 1), (0, 3, 1), (1, 1, 4),
            (2, 2, 4), (3, 3, 4),
        ]
            .span(),
    );
    let l = CsCholeskyTrait::new(@a).unwrap_l().unwrap();
    assert!(l.len() == 10);
    assert!(l.i() == array![0, 1, 2, 3, 1, 2, 3, 2, 3, 3].span());
    // Against the dense 4x4 Cholesky.
    let dense4: Matrix4<Fixed> = a.into();
    let ld = dense4.cholesky().unwrap().l();
    let lds: CsMatrix<Fixed> = ld.into();
    assert!(max_ulp(dense(l), dense(lds)) <= 2);
}

#[test]
fn test_cs_cholesky_no_fill_in() {
    // The reversed arrow (dense last row and column) has no fill-in.
    let a = cs(
        4,
        4,
        array![
            (3, 3, 4), (0, 3, 1), (1, 3, 1), (2, 3, 1), (3, 0, 1), (3, 1, 1), (3, 2, 1), (0, 0, 4),
            (1, 1, 4), (2, 2, 4),
        ]
            .span(),
    );
    let l = CsCholeskyTrait::new(@a).unwrap_l().unwrap();
    assert!(l.len() == 7);
    assert!(l.i() == array![0, 3, 1, 3, 2, 3, 3].span());
}

#[test]
fn test_cs_cholesky_not_positive_definite() {
    let a = cs(2, 2, array![(0, 0, 1), (1, 0, 2), (0, 1, 2), (1, 1, 1)].span());
    let mut chol: CsCholesky<Fixed> = CsCholeskyTrait::new(@a);
    assert!(chol.l().is_none());
    assert!(!chol.decompose_up_looking(a.values()));
    // A later success makes L available.
    let b = cs(2, 2, array![(0, 0, 1), (1, 0, 0), (0, 1, 0), (1, 1, 1)].span());
    assert!(chol.decompose_up_looking(b.values()));
    assert!(chol.unwrap_l().is_some());
}

#[test]
#[should_panic(expected: 'nalgebra: values too short')]
fn test_cs_cholesky_values_too_short() {
    let mut chol: CsCholesky<Fixed> = CsCholeskyTrait::new_symbolic(@spd3());
    let _ = chol.decompose_left_looking(ints(array![4, 2].span()));
}

#[test]
#[should_panic(expected: 'nalgebra: matrix not square')]
fn test_cs_cholesky_not_square() {
    let _: CsCholesky<Fixed> = CsCholeskyTrait::new_symbolic(@cs(2, 3, array![].span()));
}
