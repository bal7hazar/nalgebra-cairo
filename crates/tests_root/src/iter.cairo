//! `iter.sum()` / `iter.product()` (upstream `Sum` / `Product` for `Matrix`, `base/ops.rs`) over
//! arrays (owned items) and spans (snapshots, upstream's references) of static shapes, `DMatrix`
//! and `DVector`. Operands through `black_box`.

use fixed::Fixed;
use nalgebra::{
    DMatrix, DMatrixTrait, DVector, DVectorTrait, Matrix1, Matrix1Trait, Matrix2x3, Matrix2x3Trait,
    Matrix3, Matrix3Trait, Matrix6, Matrix6Trait, Vector2, Vector2Trait,
};
use nalgebra_testing::black_box;
use nalgebra_tests_utils::{int, m3i};

fn a3() -> Matrix3<Fixed> {
    black_box(m3i([[1, 2, 3], [4, 5, 6], [7, 8, 10]]))
}

fn b3() -> Matrix3<Fixed> {
    black_box(m3i([[0, 1, 0], [-1, 0, 2], [3, 1, 1]]))
}

fn c3() -> Matrix3<Fixed> {
    black_box(m3i([[2, 0, 0], [0, 3, 0], [1, 0, 1]]))
}

fn f(v: i64) -> Fixed {
    black_box(int(v))
}

// --- Sum ----------------------------------------------------------------------------------------

#[test]
fn test_sum_owned_matrix3() {
    let s: Matrix3<Fixed> = array![a3(), b3(), c3()].into_iter().sum();
    assert_eq!(s, a3() + b3() + c3());
}

#[test]
fn test_sum_snapshots_matrix3() {
    let items = array![a3(), b3(), c3()];
    let s: Matrix3<Fixed> = *items.span().into_iter().sum();
    assert_eq!(s, a3() + b3() + c3());
}

#[test]
fn test_sum_one_item_and_empty() {
    let one: Matrix3<Fixed> = array![a3()].into_iter().sum();
    assert_eq!(one, a3());
    let empty: Array<Matrix3<Fixed>> = array![];
    let zero: Matrix3<Fixed> = empty.into_iter().sum();
    assert_eq!(zero, Matrix3Trait::zeros());
    let no_items: Array<Matrix2x3<Fixed>> = array![];
    let zero_snapshot: Matrix2x3<Fixed> = *no_items.span().into_iter().sum();
    assert_eq!(zero_snapshot, Matrix2x3Trait::zeros());
}

#[test]
fn test_sum_other_shapes() {
    let u: Vector2<Fixed> = black_box(Vector2Trait::new(int(1), int(-2)));
    let w: Vector2<Fixed> = black_box(Vector2Trait::new(int(3), int(5)));
    let s: Vector2<Fixed> = array![u, w, u].into_iter().sum();
    assert_eq!(s, Vector2Trait::new(int(5), int(1)));
    let m: Matrix2x3<Fixed> = black_box(Matrix2x3Trait::new(f(1), f(2), f(3), f(4), f(5), f(6)));
    let t: Matrix2x3<Fixed> = *array![m, m].span().into_iter().sum();
    assert_eq!(t, m + m);
    let big: Matrix6<Fixed> = black_box(Matrix6Trait::identity());
    let s6: Matrix6<Fixed> = array![big, big, big].into_iter().sum();
    assert_eq!(s6, Matrix6Trait::from_diagonal_element(int(3)));
}

#[test]
#[should_panic]
fn test_sum_overflow() {
    let big: Matrix1<Fixed> = black_box(Matrix1Trait::new(int(0x7fffffff)));
    let _s: Matrix1<Fixed> = array![big, big].into_iter().sum();
}

// --- Product ------------------------------------------------------------------------------------

#[test]
fn test_product_owned_matrix3() {
    // corelib's blanket `Product` over `One` + `Mul`: a fold from the identity.
    let p: Matrix3<Fixed> = array![a3(), b3(), c3()].into_iter().product();
    assert_eq!(p, a3() * b3() * c3());
}

#[test]
fn test_product_snapshots_matrix3() {
    let items = array![a3(), b3(), c3()];
    let p: Matrix3<Fixed> = *items.span().into_iter().product();
    assert_eq!(p, a3() * b3() * c3());
    // Bit-identical to the owned form (identity times `a3()` is exactly `a3()`).
    assert_eq!(p, items.into_iter().product());
}

#[test]
fn test_product_empty_is_identity() {
    let empty: Array<Matrix3<Fixed>> = array![];
    let p: Matrix3<Fixed> = *empty.span().into_iter().product();
    assert_eq!(p, Matrix3Trait::identity());
    let owned: Matrix3<Fixed> = empty.into_iter().product();
    assert_eq!(owned, Matrix3Trait::identity());
}

#[test]
fn test_product_matrix1_and_matrix6() {
    let x: Matrix1<Fixed> = black_box(Matrix1Trait::new(int(3)));
    let p: Matrix1<Fixed> = *array![x, x, x].span().into_iter().product();
    assert_eq!(p, Matrix1Trait::new(int(27)));
    let d: Matrix6<Fixed> = black_box(Matrix6Trait::from_diagonal_element(int(2)));
    let q: Matrix6<Fixed> = array![d, d].into_iter().product();
    assert_eq!(q, Matrix6Trait::from_diagonal_element(int(4)));
    let r: Matrix6<Fixed> = *array![d, d].span().into_iter().product();
    assert_eq!(r, q);
}

// --- dynamic ------------------------------------------------------------------------------------

fn dm(rows: [[i64; 2]; 2]) -> DMatrix<Fixed> {
    let [[a, b], [c, d]] = rows;
    DMatrixTrait::from_row_slice(2, 2, array![f(a), f(b), f(c), f(d)].span())
}

#[test]
fn test_sum_dmatrix() {
    let a = dm([[1, 2], [3, 4]]);
    let b = dm([[-1, 0], [5, 1]]);
    let s: DMatrix<Fixed> = array![a, b, a].into_iter().sum();
    assert_eq!(s, a + b + a);
    let t: DMatrix<Fixed> = *array![a, b, a].span().into_iter().sum();
    assert_eq!(t, s);
    let one: DMatrix<Fixed> = array![b].into_iter().sum();
    assert_eq!(one, b);
}

#[test]
fn test_sum_dvector() {
    let u: DVector<Fixed> = DVectorTrait::from_column_slice(array![f(1), f(2), f(3)].span());
    let s: DVector<Fixed> = array![u, u, u].into_iter().sum();
    assert_eq!(s, DVectorTrait::from_column_slice(array![int(3), int(6), int(9)].span()));
    let t: DVector<Fixed> = *array![u, u, u].span().into_iter().sum();
    assert_eq!(t, s);
}

#[test]
#[should_panic(expected: 'nalgebra: sum of empty iterator')]
fn test_sum_dmatrix_empty() {
    let empty: Array<DMatrix<Fixed>> = array![];
    let _s: DMatrix<Fixed> = empty.into_iter().sum();
}

#[test]
#[should_panic(expected: 'nalgebra: sum of empty iterator')]
fn test_sum_dvector_empty_snapshots() {
    let empty: Array<DVector<Fixed>> = array![];
    let _s: DVector<Fixed> = *empty.span().into_iter().sum();
}

#[test]
#[should_panic(expected: 'nalgebra: dimension mismatch')]
fn test_sum_dmatrix_shape_mismatch() {
    let a = dm([[1, 2], [3, 4]]);
    let b = DMatrixTrait::from_row_slice(1, 2, array![f(1), f(2)].span());
    let _s: DMatrix<Fixed> = array![a, b].into_iter().sum();
}
