//! `CsMatrix`: construction (`from_triplet`, conversions), accessors, `transpose`, `+`, `*`,
//! `scale`, `PartialEq`, `cumsum`.

use fixed::Fixed;
use nalgebra::sparse::{CsMatrix, CsMatrixTrait, cumsum};
use nalgebra::{DMatrix, DMatrixTrait, DVector, Matrix2x3, Matrix3, Matrix3Trait, Vector3};
use nalgebra_testing::black_box;
use nalgebra_tests_utils::{m3i, v3i};
use crate::helpers::{cs, dense, dmi, dvi, ints};

/// The 3x4 matrix `[[1, 0, 0, -2], [0, 0, 7, 0], [5, 0, 0, 6]]` with an explicit zero at (0, 1),
/// built from unsorted triplets with a duplicate at (1, 2) (3 + 4).
fn sample() -> CsMatrix<Fixed> {
    cs(
        3,
        4,
        array![(2, 0, 5), (0, 0, 1), (1, 2, 3), (0, 3, -2), (1, 2, 4), (2, 3, 6), (0, 1, 0)].span(),
    )
}

#[test]
fn test_from_triplet_sorts_and_sums() {
    let m = sample();
    assert!(m.p() == array![0, 2, 3, 4].span());
    assert!(m.i() == array![0, 2, 0, 1, 0, 2].span());
    assert!(m.values() == ints(array![1, 5, 0, 7, -2, 6].span()));
    assert!(m.len() == 6);
    assert!(m.is_sorted());
}

#[test]
fn test_from_triplet_dense() {
    // Column-major.
    assert!(dense(sample()) == ints(array![1, 0, 5, 0, 0, 0, 0, 7, 0, -2, 0, 6].span()));
}

#[test]
fn test_from_triplet_empty() {
    let m = cs(2, 3, array![].span());
    assert!(m.len() == 0);
    assert!(m.p() == array![0, 0, 0].span());
    assert!(dense(m) == ints(array![0, 0, 0, 0, 0, 0].span()));
}

#[test]
#[should_panic(expected: 'nalgebra: index out of bounds')]
fn test_from_triplet_row_out_of_bounds() {
    let _ = cs(2, 2, array![(2, 0, 1)].span());
}

#[test]
#[should_panic(expected: 'nalgebra: index out of bounds')]
fn test_from_triplet_col_out_of_bounds() {
    let _ = cs(2, 2, array![(0, 2, 1)].span());
}

#[test]
#[should_panic(expected: 'nalgebra: triplet lengths')]
fn test_from_triplet_lengths() {
    let _: CsMatrix<Fixed> = CsMatrixTrait::from_triplet(
        2,
        2,
        black_box(array![0, 1].span()),
        black_box(array![0].span()),
        ints(array![1, 2].span()),
    );
}

#[test]
fn test_shape_accessors() {
    let m = sample();
    assert!(m.nrows() == 3);
    assert!(m.ncols() == 4);
    assert!(m.shape() == (3, 4));
    assert!(!m.is_square());
    assert!(cs(2, 2, array![].span()).is_square());
}

#[test]
fn test_transpose() {
    let t = sample().transpose();
    assert!(t.shape() == (4, 3));
    assert!(t.p() == array![0, 3, 4].span());
    assert!(t.i() == array![0, 1, 3, 2, 0, 3].span());
    assert!(t.values() == ints(array![1, 0, -2, 7, 5, 6].span()));
    assert!(t.transpose() == sample());
}

#[test]
fn test_add() {
    let b = cs(3, 4, array![(0, 0, -1), (1, 1, 2)].span());
    let s = sample() + b;
    // (0, 0) cancels to an explicit zero, (1, 1) is new.
    assert!(s.len() == 7);
    assert!(s.i() == array![0, 2, 0, 1, 1, 0, 2].span());
    assert!(dense(s) == ints(array![0, 0, 5, 0, 2, 0, 0, 7, 0, -2, 0, 6].span()));
    assert!(s.is_sorted());
}

#[test]
#[should_panic(expected: 'nalgebra: dimension mismatch')]
fn test_add_dimension_mismatch() {
    let _ = sample() + cs(3, 3, array![].span());
}

#[test]
fn test_mul() {
    let c = cs(4, 2, array![(0, 0, 2), (2, 0, 1), (3, 1, 1), (1, 1, 5)].span());
    let p = sample() * c;
    assert!(p.shape() == (3, 2));
    // [[2, -2], [7, 0], [10, 6]]: the zero (1, 1) is dropped.
    assert!(p.len() == 5);
    assert!(p.i() == array![0, 1, 2, 0, 2].span());
    assert!(dense(p) == ints(array![2, 7, 10, -2, 0, 6].span()));
}

#[test]
fn test_mul_matches_dense() {
    let a = cs(3, 3, array![(0, 0, 2), (1, 0, -1), (1, 1, 3), (2, 1, 4), (0, 2, 1)].span());
    let b = cs(3, 3, array![(0, 0, 1), (2, 0, 2), (1, 1, -2), (0, 2, 5), (2, 2, 3)].span());
    let da: DMatrix<Fixed> = a.into();
    let db: DMatrix<Fixed> = b.into();
    assert!(dense(a * b) == (da * db).as_slice());
}

#[test]
#[should_panic(expected: 'nalgebra: dimension mismatch')]
fn test_mul_dimension_mismatch() {
    let _ = sample() * sample();
}

#[test]
fn test_scale() {
    let s = sample().scale(black_box(nalgebra_tests_utils::int(3)));
    assert!(s.i() == sample().i());
    assert!(s.values() == ints(array![3, 15, 0, 21, -6, 18].span()));
}

#[test]
fn test_partial_eq() {
    let a = cs(2, 2, array![(0, 0, 1), (1, 1, 2)].span());
    let b = cs(2, 2, array![(1, 1, 2), (0, 0, 1)].span());
    let c = cs(2, 2, array![(1, 1, 3), (0, 0, 1)].span());
    assert!(a == b);
    assert!(a != c);
    assert!(a.clone() == a);
}

#[test]
fn test_from_dmatrix_roundtrip() {
    let d = dmi(2, 3, array![1, 0, 2, 0, 0, 3].span());
    let m: CsMatrix<Fixed> = d.into();
    assert!(m.p() == array![0, 1, 1].span());
    assert!(m.i() == array![0, 0, 1].span());
    assert!(m.values() == ints(array![1, 2, 3].span()));
    let back: DMatrix<Fixed> = m.into();
    assert!(back == d);
}

#[test]
fn test_from_dvector_roundtrip() {
    let v = dvi(array![0, 4, 0, -1].span());
    let m: CsMatrix<Fixed> = v.into();
    assert!(m.shape() == (4, 1));
    assert!(m.i() == array![1, 3].span());
    let back: DVector<Fixed> = m.into();
    assert!(back == v);
}

#[test]
#[should_panic(expected: 'nalgebra: dimension mismatch')]
fn test_into_dvector_not_a_column() {
    let _: DVector<Fixed> = sample().into();
}

#[test]
fn test_static_roundtrip() {
    let a = m3i(black_box([[1, 0, 2], [0, 3, 0], [4, 0, 5]]));
    let m: CsMatrix<Fixed> = a.into();
    assert!(m.len() == 5);
    let back: Matrix3<Fixed> = m.into();
    assert!(back == a);
    let v = v3i(0, black_box(7), 0);
    let mv: CsMatrix<Fixed> = v.into();
    assert!(mv.len() == 1);
    let back: Vector3<Fixed> = mv.into();
    assert!(back == v);
    let id: Matrix3<Fixed> = cs(3, 3, array![(0, 0, 1), (1, 1, 1), (2, 2, 1)].span()).into();
    assert!(id == Matrix3Trait::identity());
}

#[test]
#[should_panic(expected: 'nalgebra: dimension mismatch')]
fn test_into_static_wrong_shape() {
    let _: Matrix2x3<Fixed> = cs(3, 3, array![].span()).into();
}

#[test]
fn test_cumsum() {
    let mut a: DVector<usize> = black_box(array![2, 0, 3, 1]).into();
    let mut b: DVector<usize> = black_box(array![9, 9, 9, 9]).into();
    let total = cumsum(ref a, ref b);
    assert!(total == 6);
    let expected: DVector<usize> = array![0, 2, 2, 5].into();
    assert!(a == expected);
    assert!(b == expected);
}

#[test]
#[should_panic(expected: 'nalgebra: dimension mismatch')]
fn test_cumsum_lengths() {
    let mut a: DVector<usize> = black_box(array![1, 2]).into();
    let mut b: DVector<usize> = black_box(array![0]).into();
    let _ = cumsum(ref a, ref b);
}
