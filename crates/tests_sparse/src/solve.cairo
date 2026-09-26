//! The triangular solves of `CsMatrix`: dense right-hand sides (`solve_lower_triangular`,
//! `tr_solve_lower_triangular` and their `_mut` forms), sparse ones (`solve_lower_triangular_cs`).

use fixed::Fixed;
use nalgebra::sparse::{CsMatrix, CsMatrixSolveTrait, CsMatrixTrait};
use nalgebra::{DMatrix, DMatrixTrait};
use nalgebra_testing::black_box;
use nalgebra_tests_utils::v3i;
use crate::helpers::{cs, dmi, dvi, ints};

/// `L = [[2, 0, 0], [1, 4, 0], [3, -2, 1]]` plus an entry (0, 2) above the diagonal, which every
/// solve ignores.
fn lower() -> CsMatrix<Fixed> {
    cs(
        3,
        3,
        array![(0, 0, 2), (1, 0, 1), (2, 0, 3), (1, 1, 4), (2, 1, -2), (2, 2, 1), (0, 2, 9)].span(),
    )
}

#[test]
fn test_solve_lower_triangular_vector() {
    let x = lower().solve_lower_triangular(dvi(array![2, 9, -2].span())).unwrap();
    assert!(x.shape() == (3, 1));
    assert!(x.as_slice() == ints(array![1, 2, -1].span()));
}

#[test]
fn test_solve_lower_triangular_two_columns() {
    // Right-hand sides (2, 9, -2) and (0, -4, 5), row-major.
    let b = dmi(3, 2, array![2, 0, 9, -4, -2, 5].span());
    let x = lower().solve_lower_triangular(b).unwrap();
    assert!(x == dmi(3, 2, array![1, 0, 2, -1, -1, 3].span()));
}

#[test]
fn test_solve_lower_triangular_static_rhs() {
    let x = lower().solve_lower_triangular(v3i(2, black_box(9), -2)).unwrap();
    assert!(x.as_slice() == ints(array![1, 2, -1].span()));
}

#[test]
fn test_solve_lower_triangular_no_columns() {
    let b: DMatrix<Fixed> = DMatrixTrait::zeros(3, black_box(0));
    let singular = cs(3, 3, array![].span());
    assert!(singular.solve_lower_triangular(b).is_some());
}

#[test]
fn test_solve_lower_triangular_missing_diagonal() {
    let l = cs(3, 3, array![(0, 0, 2), (1, 0, 1), (2, 2, 1)].span());
    assert!(l.solve_lower_triangular(dvi(array![1, 1, 1].span())).is_none());
    assert!(l.tr_solve_lower_triangular(dvi(array![1, 1, 1].span())).is_none());
}

#[test]
fn test_solve_lower_triangular_zero_diagonal() {
    let l = cs(2, 2, array![(0, 0, 2), (1, 1, 0)].span());
    assert!(l.solve_lower_triangular(dvi(array![1, 1].span())).is_none());
    assert!(l.tr_solve_lower_triangular(dvi(array![1, 1].span())).is_none());
}

#[test]
fn test_tr_solve_lower_triangular() {
    let x = lower().tr_solve_lower_triangular(dvi(array![1, 10, -1].span())).unwrap();
    assert!(x.as_slice() == ints(array![1, 2, -1].span()));
}

#[test]
fn test_solve_lower_triangular_mut() {
    let mut b: DMatrix<Fixed> = dvi(array![2, 9, -2].span()).into();
    assert!(lower().solve_lower_triangular_mut(ref b));
    assert!(b.as_slice() == ints(array![1, 2, -1].span()));
    let mut c: DMatrix<Fixed> = dvi(array![1, 10, -1].span()).into();
    assert!(lower().tr_solve_lower_triangular_mut(ref c));
    assert!(c.as_slice() == ints(array![1, 2, -1].span()));
}

#[test]
fn test_solve_lower_triangular_mut_failure_keeps_b() {
    let l = cs(2, 2, array![(0, 0, 2)].span());
    let mut b: DMatrix<Fixed> = dvi(array![4, 6].span()).into();
    let before = b;
    assert!(!l.solve_lower_triangular_mut(ref b));
    assert!(b == before);
    assert!(!l.tr_solve_lower_triangular_mut(ref b));
    assert!(b == before);
}

#[test]
fn test_solve_lower_triangular_rounding() {
    // 3 x = 1: one rounding to nearest.
    let l = cs(1, 1, array![(0, 0, 3)].span());
    let x = l.solve_lower_triangular(dvi(array![1].span())).unwrap();
    assert!(x.as_slice() == array![Fixed { raw: 1431655765 }].span());
}

#[test]
#[should_panic(expected: 'nalgebra: matrix not square')]
fn test_solve_lower_triangular_not_square() {
    let _ = cs(3, 2, array![].span()).solve_lower_triangular(dvi(array![1, 1, 1].span()));
}

#[test]
#[should_panic(expected: 'nalgebra: dimension mismatch')]
fn test_solve_lower_triangular_dimension_mismatch() {
    let _ = lower().solve_lower_triangular(dvi(array![1, 1].span()));
}

/// `L = I + 2 e1 e0ᵀ + e3 e2ᵀ` (4x4).
fn lower_sparse() -> CsMatrix<Fixed> {
    cs(4, 4, array![(0, 0, 1), (1, 0, 2), (1, 1, 1), (2, 2, 1), (3, 2, 1), (3, 3, 1)].span())
}

#[test]
fn test_solve_lower_triangular_cs_reach() {
    let x = lower_sparse().solve_lower_triangular_cs(cs(4, 1, array![(0, 0, 1)].span())).unwrap();
    assert!(x.shape() == (4, 1));
    assert!(x.i() == array![0, 1].span());
    assert!(x.values() == ints(array![1, -2].span()));
    let y = lower_sparse().solve_lower_triangular_cs(cs(4, 1, array![(2, 0, 3)].span())).unwrap();
    assert!(y.i() == array![2, 3].span());
    assert!(y.values() == ints(array![3, -3].span()));
}

#[test]
fn test_solve_lower_triangular_cs_keeps_zeros() {
    let x = lower_sparse().solve_lower_triangular_cs(cs(4, 1, array![(0, 0, 0)].span())).unwrap();
    assert!(x.i() == array![0, 1].span());
    assert!(x.values() == ints(array![0, 0].span()));
}

#[test]
fn test_solve_lower_triangular_cs_matches_dense() {
    let b = cs(3, 1, array![(0, 0, 2), (2, 0, -2)].span());
    let x = lower().solve_lower_triangular_cs(b).unwrap();
    let d = lower().solve_lower_triangular(dvi(array![2, 0, -2].span())).unwrap();
    let xd: DMatrix<Fixed> = x.into();
    assert!(xd == d);
}

#[test]
fn test_solve_lower_triangular_cs_missing_diagonal() {
    // Row 3 is not reached: its missing diagonal does not matter.
    let l = cs(4, 4, array![(0, 0, 1), (1, 0, 2), (1, 1, 1), (2, 2, 1)].span());
    assert!(l.solve_lower_triangular_cs(cs(4, 1, array![(0, 0, 1)].span())).is_some());
    assert!(l.solve_lower_triangular_cs(cs(4, 1, array![(3, 0, 1)].span())).is_none());
}

#[test]
#[should_panic(expected: 'nalgebra: dimension mismatch')]
fn test_solve_lower_triangular_cs_dimension_mismatch() {
    let _ = lower_sparse().solve_lower_triangular_cs(cs(3, 1, array![].span()));
}
