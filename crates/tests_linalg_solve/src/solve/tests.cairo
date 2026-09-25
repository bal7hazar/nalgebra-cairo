//! Tests of `MatrixSolve` (upstream `SquareMatrix::solve_*`, `src/linalg/solve.rs`): every form on
//! every (square, right-hand side with as many rows) pair, on integer designs whose solution is
//! exact (power-of-two diagonal, integer `x`, `b = t x` exact), the zero-diagonal semantics of
//! the checked forms and the division panic of the unchecked ones. The oracle vectors
//! (`oracle_solve`) check the rounding on real-valued inputs.

use fixed::Fixed;
use nalgebra::{
    Matrix1, Matrix1Trait, Matrix2, Matrix2Trait, Matrix2x3, Matrix2x3Trait, Matrix2x4,
    Matrix2x4Trait, Matrix2x5, Matrix2x5Trait, Matrix2x6, Matrix2x6Trait, Matrix3, Matrix3Trait,
    Matrix3x2, Matrix3x2Trait, Matrix3x4, Matrix3x4Trait, Matrix3x5, Matrix3x5Trait, Matrix3x6,
    Matrix3x6Trait, Matrix4, Matrix4Trait, Matrix4x2, Matrix4x2Trait, Matrix4x3, Matrix4x3Trait,
    Matrix4x5, Matrix4x5Trait, Matrix4x6, Matrix4x6Trait, Matrix5, Matrix5Trait, Matrix5x2,
    Matrix5x2Trait, Matrix5x3, Matrix5x3Trait, Matrix5x4, Matrix5x4Trait, Matrix5x6, Matrix5x6Trait,
    Matrix6, Matrix6Trait, Matrix6x2, Matrix6x2Trait, Matrix6x3, Matrix6x3Trait, Matrix6x4,
    Matrix6x4Trait, Matrix6x5, Matrix6x5Trait, MatrixSolve, RowVector2, RowVector2Trait, RowVector3,
    RowVector3Trait, RowVector4, RowVector4Trait, RowVector5, RowVector5Trait, RowVector6,
    RowVector6Trait, Vector2, Vector2Trait, Vector3, Vector3Trait, Vector4, Vector4Trait, Vector5,
    Vector5Trait, Vector6, Vector6Trait,
};
use nalgebra_tests_utils::int;

/// `Matrix1` against a `Matrix1` right-hand side: the `Option` forms recover the integer solution
/// exactly (power-of-two diagonal:
/// every quotient is exact).
#[test]
fn test_solve_matrix1_matrix1_exact() {
    let l: Matrix1<Fixed> = Matrix1Trait::new(int(2));
    let u: Matrix1<Fixed> = Matrix1Trait::new(int(2));
    let x: Matrix1<Fixed> = Matrix1Trait::new(int(-3));
    let bl: Matrix1<Fixed> = Matrix1Trait::new(int(-6));
    let bu: Matrix1<Fixed> = Matrix1Trait::new(int(-6));
    assert!(l.solve_lower_triangular(bl) == Option::Some(x));
    assert!(u.solve_upper_triangular(bu) == Option::Some(x));
    assert!(l.tr_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.tr_solve_upper_triangular(bl) == Option::Some(x));
    assert!(l.ad_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.ad_solve_upper_triangular(bl) == Option::Some(x));
}

/// `Matrix1` against a `Matrix1` right-hand side: the `_unchecked` forms, bit-identical to the
/// checked ones.
#[test]
fn test_solve_matrix1_matrix1_unchecked_exact() {
    let l: Matrix1<Fixed> = Matrix1Trait::new(int(2));
    let u: Matrix1<Fixed> = Matrix1Trait::new(int(2));
    let x: Matrix1<Fixed> = Matrix1Trait::new(int(-3));
    let bl: Matrix1<Fixed> = Matrix1Trait::new(int(-6));
    let bu: Matrix1<Fixed> = Matrix1Trait::new(int(-6));
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    assert!(l.ad_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.ad_solve_upper_triangular_unchecked(bl) == x);
}

/// `Matrix1` against a `Matrix1` right-hand side: the `_mut` forms overwrite `b` with the solution
/// and return `true`.
#[test]
fn test_solve_matrix1_matrix1_mut_exact() {
    let l: Matrix1<Fixed> = Matrix1Trait::new(int(2));
    let u: Matrix1<Fixed> = Matrix1Trait::new(int(2));
    let x: Matrix1<Fixed> = Matrix1Trait::new(int(-3));
    let bl: Matrix1<Fixed> = Matrix1Trait::new(int(-6));
    let bu: Matrix1<Fixed> = Matrix1Trait::new(int(-6));
    let mut b = bl;
    assert!(l.solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(u.solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.tr_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.tr_solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.ad_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.ad_solve_upper_triangular_mut(ref b));
    assert!(b == x);
}

/// `Matrix1` against a `Matrix1` right-hand side: the `_unchecked_mut` forms.
#[test]
fn test_solve_matrix1_matrix1_unchecked_mut_exact() {
    let l: Matrix1<Fixed> = Matrix1Trait::new(int(2));
    let u: Matrix1<Fixed> = Matrix1Trait::new(int(2));
    let x: Matrix1<Fixed> = Matrix1Trait::new(int(-3));
    let bl: Matrix1<Fixed> = Matrix1Trait::new(int(-6));
    let bu: Matrix1<Fixed> = Matrix1Trait::new(int(-6));
    let mut b = bl;
    l.solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    u.solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.tr_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.tr_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.ad_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.ad_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
}

/// `Matrix1` against a `Matrix1` right-hand side: `with_diag` solves `(strict lower triangle of l +
/// 2 I) x = bd` and leaves `2 x` in `b`, like upstream (the quotients by `diag` are never written
/// back); a zero `diag` returns `false` and leaves `b` unchanged.
#[test]
fn test_solve_matrix1_matrix1_with_diag_exact() {
    let l: Matrix1<Fixed> = Matrix1Trait::new(int(2));
    let bd: Matrix1<Fixed> = Matrix1Trait::new(int(-6));
    let x2: Matrix1<Fixed> = Matrix1Trait::new(int(-6));
    let mut b = bd;
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(b == x2);
    let mut b = bd;
    l.solve_lower_triangular_with_diag_unchecked_mut(ref b, int(2));
    assert!(b == x2);
    let mut b = bd;
    assert!(!l.solve_lower_triangular_with_diag_mut(ref b, int(0)));
    assert!(b == bd);
}

/// `Matrix1` against a `RowVector2` right-hand side: the `Option` forms recover the integer
/// solution exactly (power-of-two diagonal:
/// every quotient is exact).
#[test]
fn test_solve_matrix1_row_vector2_exact() {
    let l: Matrix1<Fixed> = Matrix1Trait::new(int(2));
    let u: Matrix1<Fixed> = Matrix1Trait::new(int(2));
    let x: RowVector2<Fixed> = RowVector2Trait::new(int(-3), int(0));
    let bl: RowVector2<Fixed> = RowVector2Trait::new(int(-6), int(0));
    let bu: RowVector2<Fixed> = RowVector2Trait::new(int(-6), int(0));
    assert!(l.solve_lower_triangular(bl) == Option::Some(x));
    assert!(u.solve_upper_triangular(bu) == Option::Some(x));
    assert!(l.tr_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.tr_solve_upper_triangular(bl) == Option::Some(x));
    assert!(l.ad_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.ad_solve_upper_triangular(bl) == Option::Some(x));
}

/// `Matrix1` against a `RowVector2` right-hand side: the `_unchecked` forms, bit-identical to the
/// checked ones.
#[test]
fn test_solve_matrix1_row_vector2_unchecked_exact() {
    let l: Matrix1<Fixed> = Matrix1Trait::new(int(2));
    let u: Matrix1<Fixed> = Matrix1Trait::new(int(2));
    let x: RowVector2<Fixed> = RowVector2Trait::new(int(-3), int(0));
    let bl: RowVector2<Fixed> = RowVector2Trait::new(int(-6), int(0));
    let bu: RowVector2<Fixed> = RowVector2Trait::new(int(-6), int(0));
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    assert!(l.ad_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.ad_solve_upper_triangular_unchecked(bl) == x);
}

/// `Matrix1` against a `RowVector2` right-hand side: the `_mut` forms overwrite `b` with the
/// solution and return `true`.
#[test]
fn test_solve_matrix1_row_vector2_mut_exact() {
    let l: Matrix1<Fixed> = Matrix1Trait::new(int(2));
    let u: Matrix1<Fixed> = Matrix1Trait::new(int(2));
    let x: RowVector2<Fixed> = RowVector2Trait::new(int(-3), int(0));
    let bl: RowVector2<Fixed> = RowVector2Trait::new(int(-6), int(0));
    let bu: RowVector2<Fixed> = RowVector2Trait::new(int(-6), int(0));
    let mut b = bl;
    assert!(l.solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(u.solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.tr_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.tr_solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.ad_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.ad_solve_upper_triangular_mut(ref b));
    assert!(b == x);
}

/// `Matrix1` against a `RowVector2` right-hand side: the `_unchecked_mut` forms.
#[test]
fn test_solve_matrix1_row_vector2_unchecked_mut_exact() {
    let l: Matrix1<Fixed> = Matrix1Trait::new(int(2));
    let u: Matrix1<Fixed> = Matrix1Trait::new(int(2));
    let x: RowVector2<Fixed> = RowVector2Trait::new(int(-3), int(0));
    let bl: RowVector2<Fixed> = RowVector2Trait::new(int(-6), int(0));
    let bu: RowVector2<Fixed> = RowVector2Trait::new(int(-6), int(0));
    let mut b = bl;
    l.solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    u.solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.tr_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.tr_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.ad_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.ad_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
}

/// `Matrix1` against a `RowVector2` right-hand side: `with_diag` solves `(strict lower triangle of
/// l + 2 I) x = bd` and leaves `2 x` in `b`, like upstream (the quotients by `diag` are never
/// written back); a zero `diag` returns `false` and leaves `b` unchanged.
#[test]
fn test_solve_matrix1_row_vector2_with_diag_exact() {
    let l: Matrix1<Fixed> = Matrix1Trait::new(int(2));
    let bd: RowVector2<Fixed> = RowVector2Trait::new(int(-6), int(0));
    let x2: RowVector2<Fixed> = RowVector2Trait::new(int(-6), int(0));
    let mut b = bd;
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(b == x2);
    let mut b = bd;
    l.solve_lower_triangular_with_diag_unchecked_mut(ref b, int(2));
    assert!(b == x2);
    let mut b = bd;
    assert!(!l.solve_lower_triangular_with_diag_mut(ref b, int(0)));
    assert!(b == bd);
}

/// `Matrix1` against a `RowVector3` right-hand side: the `Option` forms recover the integer
/// solution exactly (power-of-two diagonal:
/// every quotient is exact).
#[test]
fn test_solve_matrix1_row_vector3_exact() {
    let l: Matrix1<Fixed> = Matrix1Trait::new(int(2));
    let u: Matrix1<Fixed> = Matrix1Trait::new(int(2));
    let x: RowVector3<Fixed> = RowVector3Trait::new(int(-3), int(0), int(3));
    let bl: RowVector3<Fixed> = RowVector3Trait::new(int(-6), int(0), int(6));
    let bu: RowVector3<Fixed> = RowVector3Trait::new(int(-6), int(0), int(6));
    assert!(l.solve_lower_triangular(bl) == Option::Some(x));
    assert!(u.solve_upper_triangular(bu) == Option::Some(x));
    assert!(l.tr_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.tr_solve_upper_triangular(bl) == Option::Some(x));
    assert!(l.ad_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.ad_solve_upper_triangular(bl) == Option::Some(x));
}

/// `Matrix1` against a `RowVector3` right-hand side: the `_unchecked` forms, bit-identical to the
/// checked ones.
#[test]
fn test_solve_matrix1_row_vector3_unchecked_exact() {
    let l: Matrix1<Fixed> = Matrix1Trait::new(int(2));
    let u: Matrix1<Fixed> = Matrix1Trait::new(int(2));
    let x: RowVector3<Fixed> = RowVector3Trait::new(int(-3), int(0), int(3));
    let bl: RowVector3<Fixed> = RowVector3Trait::new(int(-6), int(0), int(6));
    let bu: RowVector3<Fixed> = RowVector3Trait::new(int(-6), int(0), int(6));
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    assert!(l.ad_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.ad_solve_upper_triangular_unchecked(bl) == x);
}

/// `Matrix1` against a `RowVector3` right-hand side: the `_mut` forms overwrite `b` with the
/// solution and return `true`.
#[test]
fn test_solve_matrix1_row_vector3_mut_exact() {
    let l: Matrix1<Fixed> = Matrix1Trait::new(int(2));
    let u: Matrix1<Fixed> = Matrix1Trait::new(int(2));
    let x: RowVector3<Fixed> = RowVector3Trait::new(int(-3), int(0), int(3));
    let bl: RowVector3<Fixed> = RowVector3Trait::new(int(-6), int(0), int(6));
    let bu: RowVector3<Fixed> = RowVector3Trait::new(int(-6), int(0), int(6));
    let mut b = bl;
    assert!(l.solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(u.solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.tr_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.tr_solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.ad_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.ad_solve_upper_triangular_mut(ref b));
    assert!(b == x);
}

/// `Matrix1` against a `RowVector3` right-hand side: the `_unchecked_mut` forms.
#[test]
fn test_solve_matrix1_row_vector3_unchecked_mut_exact() {
    let l: Matrix1<Fixed> = Matrix1Trait::new(int(2));
    let u: Matrix1<Fixed> = Matrix1Trait::new(int(2));
    let x: RowVector3<Fixed> = RowVector3Trait::new(int(-3), int(0), int(3));
    let bl: RowVector3<Fixed> = RowVector3Trait::new(int(-6), int(0), int(6));
    let bu: RowVector3<Fixed> = RowVector3Trait::new(int(-6), int(0), int(6));
    let mut b = bl;
    l.solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    u.solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.tr_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.tr_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.ad_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.ad_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
}

/// `Matrix1` against a `RowVector3` right-hand side: `with_diag` solves `(strict lower triangle of
/// l + 2 I) x = bd` and leaves `2 x` in `b`, like upstream (the quotients by `diag` are never
/// written back); a zero `diag` returns `false` and leaves `b` unchanged.
#[test]
fn test_solve_matrix1_row_vector3_with_diag_exact() {
    let l: Matrix1<Fixed> = Matrix1Trait::new(int(2));
    let bd: RowVector3<Fixed> = RowVector3Trait::new(int(-6), int(0), int(6));
    let x2: RowVector3<Fixed> = RowVector3Trait::new(int(-6), int(0), int(6));
    let mut b = bd;
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(b == x2);
    let mut b = bd;
    l.solve_lower_triangular_with_diag_unchecked_mut(ref b, int(2));
    assert!(b == x2);
    let mut b = bd;
    assert!(!l.solve_lower_triangular_with_diag_mut(ref b, int(0)));
    assert!(b == bd);
}

/// `Matrix1` against a `RowVector4` right-hand side: the `Option` forms recover the integer
/// solution exactly (power-of-two diagonal:
/// every quotient is exact).
#[test]
fn test_solve_matrix1_row_vector4_exact() {
    let l: Matrix1<Fixed> = Matrix1Trait::new(int(2));
    let u: Matrix1<Fixed> = Matrix1Trait::new(int(2));
    let x: RowVector4<Fixed> = RowVector4Trait::new(int(-3), int(0), int(3), int(-3));
    let bl: RowVector4<Fixed> = RowVector4Trait::new(int(-6), int(0), int(6), int(-6));
    let bu: RowVector4<Fixed> = RowVector4Trait::new(int(-6), int(0), int(6), int(-6));
    assert!(l.solve_lower_triangular(bl) == Option::Some(x));
    assert!(u.solve_upper_triangular(bu) == Option::Some(x));
    assert!(l.tr_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.tr_solve_upper_triangular(bl) == Option::Some(x));
    assert!(l.ad_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.ad_solve_upper_triangular(bl) == Option::Some(x));
}

/// `Matrix1` against a `RowVector4` right-hand side: the `_unchecked` forms, bit-identical to the
/// checked ones.
#[test]
fn test_solve_matrix1_row_vector4_unchecked_exact() {
    let l: Matrix1<Fixed> = Matrix1Trait::new(int(2));
    let u: Matrix1<Fixed> = Matrix1Trait::new(int(2));
    let x: RowVector4<Fixed> = RowVector4Trait::new(int(-3), int(0), int(3), int(-3));
    let bl: RowVector4<Fixed> = RowVector4Trait::new(int(-6), int(0), int(6), int(-6));
    let bu: RowVector4<Fixed> = RowVector4Trait::new(int(-6), int(0), int(6), int(-6));
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    assert!(l.ad_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.ad_solve_upper_triangular_unchecked(bl) == x);
}

/// `Matrix1` against a `RowVector4` right-hand side: the `_mut` forms overwrite `b` with the
/// solution and return `true`.
#[test]
fn test_solve_matrix1_row_vector4_mut_exact() {
    let l: Matrix1<Fixed> = Matrix1Trait::new(int(2));
    let u: Matrix1<Fixed> = Matrix1Trait::new(int(2));
    let x: RowVector4<Fixed> = RowVector4Trait::new(int(-3), int(0), int(3), int(-3));
    let bl: RowVector4<Fixed> = RowVector4Trait::new(int(-6), int(0), int(6), int(-6));
    let bu: RowVector4<Fixed> = RowVector4Trait::new(int(-6), int(0), int(6), int(-6));
    let mut b = bl;
    assert!(l.solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(u.solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.tr_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.tr_solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.ad_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.ad_solve_upper_triangular_mut(ref b));
    assert!(b == x);
}

/// `Matrix1` against a `RowVector4` right-hand side: the `_unchecked_mut` forms.
#[test]
fn test_solve_matrix1_row_vector4_unchecked_mut_exact() {
    let l: Matrix1<Fixed> = Matrix1Trait::new(int(2));
    let u: Matrix1<Fixed> = Matrix1Trait::new(int(2));
    let x: RowVector4<Fixed> = RowVector4Trait::new(int(-3), int(0), int(3), int(-3));
    let bl: RowVector4<Fixed> = RowVector4Trait::new(int(-6), int(0), int(6), int(-6));
    let bu: RowVector4<Fixed> = RowVector4Trait::new(int(-6), int(0), int(6), int(-6));
    let mut b = bl;
    l.solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    u.solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.tr_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.tr_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.ad_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.ad_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
}

/// `Matrix1` against a `RowVector4` right-hand side: `with_diag` solves `(strict lower triangle of
/// l + 2 I) x = bd` and leaves `2 x` in `b`, like upstream (the quotients by `diag` are never
/// written back); a zero `diag` returns `false` and leaves `b` unchanged.
#[test]
fn test_solve_matrix1_row_vector4_with_diag_exact() {
    let l: Matrix1<Fixed> = Matrix1Trait::new(int(2));
    let bd: RowVector4<Fixed> = RowVector4Trait::new(int(-6), int(0), int(6), int(-6));
    let x2: RowVector4<Fixed> = RowVector4Trait::new(int(-6), int(0), int(6), int(-6));
    let mut b = bd;
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(b == x2);
    let mut b = bd;
    l.solve_lower_triangular_with_diag_unchecked_mut(ref b, int(2));
    assert!(b == x2);
    let mut b = bd;
    assert!(!l.solve_lower_triangular_with_diag_mut(ref b, int(0)));
    assert!(b == bd);
}

/// `Matrix1` against a `RowVector5` right-hand side: the `Option` forms recover the integer
/// solution exactly (power-of-two diagonal:
/// every quotient is exact).
#[test]
fn test_solve_matrix1_row_vector5_exact() {
    let l: Matrix1<Fixed> = Matrix1Trait::new(int(2));
    let u: Matrix1<Fixed> = Matrix1Trait::new(int(2));
    let x: RowVector5<Fixed> = RowVector5Trait::new(int(-3), int(0), int(3), int(-3), int(0));
    let bl: RowVector5<Fixed> = RowVector5Trait::new(int(-6), int(0), int(6), int(-6), int(0));
    let bu: RowVector5<Fixed> = RowVector5Trait::new(int(-6), int(0), int(6), int(-6), int(0));
    assert!(l.solve_lower_triangular(bl) == Option::Some(x));
    assert!(u.solve_upper_triangular(bu) == Option::Some(x));
    assert!(l.tr_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.tr_solve_upper_triangular(bl) == Option::Some(x));
    assert!(l.ad_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.ad_solve_upper_triangular(bl) == Option::Some(x));
}

/// `Matrix1` against a `RowVector5` right-hand side: the `_unchecked` forms, bit-identical to the
/// checked ones.
#[test]
fn test_solve_matrix1_row_vector5_unchecked_exact() {
    let l: Matrix1<Fixed> = Matrix1Trait::new(int(2));
    let u: Matrix1<Fixed> = Matrix1Trait::new(int(2));
    let x: RowVector5<Fixed> = RowVector5Trait::new(int(-3), int(0), int(3), int(-3), int(0));
    let bl: RowVector5<Fixed> = RowVector5Trait::new(int(-6), int(0), int(6), int(-6), int(0));
    let bu: RowVector5<Fixed> = RowVector5Trait::new(int(-6), int(0), int(6), int(-6), int(0));
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    assert!(l.ad_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.ad_solve_upper_triangular_unchecked(bl) == x);
}

/// `Matrix1` against a `RowVector5` right-hand side: the `_mut` forms overwrite `b` with the
/// solution and return `true`.
#[test]
fn test_solve_matrix1_row_vector5_mut_exact() {
    let l: Matrix1<Fixed> = Matrix1Trait::new(int(2));
    let u: Matrix1<Fixed> = Matrix1Trait::new(int(2));
    let x: RowVector5<Fixed> = RowVector5Trait::new(int(-3), int(0), int(3), int(-3), int(0));
    let bl: RowVector5<Fixed> = RowVector5Trait::new(int(-6), int(0), int(6), int(-6), int(0));
    let bu: RowVector5<Fixed> = RowVector5Trait::new(int(-6), int(0), int(6), int(-6), int(0));
    let mut b = bl;
    assert!(l.solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(u.solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.tr_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.tr_solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.ad_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.ad_solve_upper_triangular_mut(ref b));
    assert!(b == x);
}

/// `Matrix1` against a `RowVector5` right-hand side: the `_unchecked_mut` forms.
#[test]
fn test_solve_matrix1_row_vector5_unchecked_mut_exact() {
    let l: Matrix1<Fixed> = Matrix1Trait::new(int(2));
    let u: Matrix1<Fixed> = Matrix1Trait::new(int(2));
    let x: RowVector5<Fixed> = RowVector5Trait::new(int(-3), int(0), int(3), int(-3), int(0));
    let bl: RowVector5<Fixed> = RowVector5Trait::new(int(-6), int(0), int(6), int(-6), int(0));
    let bu: RowVector5<Fixed> = RowVector5Trait::new(int(-6), int(0), int(6), int(-6), int(0));
    let mut b = bl;
    l.solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    u.solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.tr_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.tr_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.ad_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.ad_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
}

/// `Matrix1` against a `RowVector5` right-hand side: `with_diag` solves `(strict lower triangle of
/// l + 2 I) x = bd` and leaves `2 x` in `b`, like upstream (the quotients by `diag` are never
/// written back); a zero `diag` returns `false` and leaves `b` unchanged.
#[test]
fn test_solve_matrix1_row_vector5_with_diag_exact() {
    let l: Matrix1<Fixed> = Matrix1Trait::new(int(2));
    let bd: RowVector5<Fixed> = RowVector5Trait::new(int(-6), int(0), int(6), int(-6), int(0));
    let x2: RowVector5<Fixed> = RowVector5Trait::new(int(-6), int(0), int(6), int(-6), int(0));
    let mut b = bd;
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(b == x2);
    let mut b = bd;
    l.solve_lower_triangular_with_diag_unchecked_mut(ref b, int(2));
    assert!(b == x2);
    let mut b = bd;
    assert!(!l.solve_lower_triangular_with_diag_mut(ref b, int(0)));
    assert!(b == bd);
}

/// `Matrix1` against a `RowVector6` right-hand side: the `Option` forms recover the integer
/// solution exactly (power-of-two diagonal:
/// every quotient is exact).
#[test]
fn test_solve_matrix1_row_vector6_exact() {
    let l: Matrix1<Fixed> = Matrix1Trait::new(int(2));
    let u: Matrix1<Fixed> = Matrix1Trait::new(int(2));
    let x: RowVector6<Fixed> = RowVector6Trait::new(
        int(-3), int(0), int(3), int(-3), int(0), int(3),
    );
    let bl: RowVector6<Fixed> = RowVector6Trait::new(
        int(-6), int(0), int(6), int(-6), int(0), int(6),
    );
    let bu: RowVector6<Fixed> = RowVector6Trait::new(
        int(-6), int(0), int(6), int(-6), int(0), int(6),
    );
    assert!(l.solve_lower_triangular(bl) == Option::Some(x));
    assert!(u.solve_upper_triangular(bu) == Option::Some(x));
    assert!(l.tr_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.tr_solve_upper_triangular(bl) == Option::Some(x));
    assert!(l.ad_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.ad_solve_upper_triangular(bl) == Option::Some(x));
}

/// `Matrix1` against a `RowVector6` right-hand side: the `_unchecked` forms, bit-identical to the
/// checked ones.
#[test]
fn test_solve_matrix1_row_vector6_unchecked_exact() {
    let l: Matrix1<Fixed> = Matrix1Trait::new(int(2));
    let u: Matrix1<Fixed> = Matrix1Trait::new(int(2));
    let x: RowVector6<Fixed> = RowVector6Trait::new(
        int(-3), int(0), int(3), int(-3), int(0), int(3),
    );
    let bl: RowVector6<Fixed> = RowVector6Trait::new(
        int(-6), int(0), int(6), int(-6), int(0), int(6),
    );
    let bu: RowVector6<Fixed> = RowVector6Trait::new(
        int(-6), int(0), int(6), int(-6), int(0), int(6),
    );
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    assert!(l.ad_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.ad_solve_upper_triangular_unchecked(bl) == x);
}

/// `Matrix1` against a `RowVector6` right-hand side: the `_mut` forms overwrite `b` with the
/// solution and return `true`.
#[test]
fn test_solve_matrix1_row_vector6_mut_exact() {
    let l: Matrix1<Fixed> = Matrix1Trait::new(int(2));
    let u: Matrix1<Fixed> = Matrix1Trait::new(int(2));
    let x: RowVector6<Fixed> = RowVector6Trait::new(
        int(-3), int(0), int(3), int(-3), int(0), int(3),
    );
    let bl: RowVector6<Fixed> = RowVector6Trait::new(
        int(-6), int(0), int(6), int(-6), int(0), int(6),
    );
    let bu: RowVector6<Fixed> = RowVector6Trait::new(
        int(-6), int(0), int(6), int(-6), int(0), int(6),
    );
    let mut b = bl;
    assert!(l.solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(u.solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.tr_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.tr_solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.ad_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.ad_solve_upper_triangular_mut(ref b));
    assert!(b == x);
}

/// `Matrix1` against a `RowVector6` right-hand side: the `_unchecked_mut` forms.
#[test]
fn test_solve_matrix1_row_vector6_unchecked_mut_exact() {
    let l: Matrix1<Fixed> = Matrix1Trait::new(int(2));
    let u: Matrix1<Fixed> = Matrix1Trait::new(int(2));
    let x: RowVector6<Fixed> = RowVector6Trait::new(
        int(-3), int(0), int(3), int(-3), int(0), int(3),
    );
    let bl: RowVector6<Fixed> = RowVector6Trait::new(
        int(-6), int(0), int(6), int(-6), int(0), int(6),
    );
    let bu: RowVector6<Fixed> = RowVector6Trait::new(
        int(-6), int(0), int(6), int(-6), int(0), int(6),
    );
    let mut b = bl;
    l.solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    u.solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.tr_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.tr_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.ad_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.ad_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
}

/// `Matrix1` against a `RowVector6` right-hand side: `with_diag` solves `(strict lower triangle of
/// l + 2 I) x = bd` and leaves `2 x` in `b`, like upstream (the quotients by `diag` are never
/// written back); a zero `diag` returns `false` and leaves `b` unchanged.
#[test]
fn test_solve_matrix1_row_vector6_with_diag_exact() {
    let l: Matrix1<Fixed> = Matrix1Trait::new(int(2));
    let bd: RowVector6<Fixed> = RowVector6Trait::new(
        int(-6), int(0), int(6), int(-6), int(0), int(6),
    );
    let x2: RowVector6<Fixed> = RowVector6Trait::new(
        int(-6), int(0), int(6), int(-6), int(0), int(6),
    );
    let mut b = bd;
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(b == x2);
    let mut b = bd;
    l.solve_lower_triangular_with_diag_unchecked_mut(ref b, int(2));
    assert!(b == x2);
    let mut b = bd;
    assert!(!l.solve_lower_triangular_with_diag_mut(ref b, int(0)));
    assert!(b == bd);
}

/// `Matrix2` against a `Vector2` right-hand side: the `Option` forms recover the integer solution
/// exactly (power-of-two diagonal:
/// every quotient is exact).
#[test]
fn test_solve_matrix2_vector2_exact() {
    let l: Matrix2<Fixed> = Matrix2Trait::new(int(2), int(0), int(0), int(4));
    let u: Matrix2<Fixed> = Matrix2Trait::new(int(2), int(0), int(0), int(4));
    let x: Vector2<Fixed> = Vector2Trait::new(int(-3), int(2));
    let bl: Vector2<Fixed> = Vector2Trait::new(int(-6), int(8));
    let bu: Vector2<Fixed> = Vector2Trait::new(int(-6), int(8));
    assert!(l.solve_lower_triangular(bl) == Option::Some(x));
    assert!(u.solve_upper_triangular(bu) == Option::Some(x));
    assert!(l.tr_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.tr_solve_upper_triangular(bl) == Option::Some(x));
    assert!(l.ad_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.ad_solve_upper_triangular(bl) == Option::Some(x));
}

/// `Matrix2` against a `Vector2` right-hand side: the `_unchecked` forms, bit-identical to the
/// checked ones.
#[test]
fn test_solve_matrix2_vector2_unchecked_exact() {
    let l: Matrix2<Fixed> = Matrix2Trait::new(int(2), int(0), int(0), int(4));
    let u: Matrix2<Fixed> = Matrix2Trait::new(int(2), int(0), int(0), int(4));
    let x: Vector2<Fixed> = Vector2Trait::new(int(-3), int(2));
    let bl: Vector2<Fixed> = Vector2Trait::new(int(-6), int(8));
    let bu: Vector2<Fixed> = Vector2Trait::new(int(-6), int(8));
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    assert!(l.ad_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.ad_solve_upper_triangular_unchecked(bl) == x);
}

/// `Matrix2` against a `Vector2` right-hand side: the `_mut` forms overwrite `b` with the solution
/// and return `true`.
#[test]
fn test_solve_matrix2_vector2_mut_exact() {
    let l: Matrix2<Fixed> = Matrix2Trait::new(int(2), int(0), int(0), int(4));
    let u: Matrix2<Fixed> = Matrix2Trait::new(int(2), int(0), int(0), int(4));
    let x: Vector2<Fixed> = Vector2Trait::new(int(-3), int(2));
    let bl: Vector2<Fixed> = Vector2Trait::new(int(-6), int(8));
    let bu: Vector2<Fixed> = Vector2Trait::new(int(-6), int(8));
    let mut b = bl;
    assert!(l.solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(u.solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.tr_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.tr_solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.ad_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.ad_solve_upper_triangular_mut(ref b));
    assert!(b == x);
}

/// `Matrix2` against a `Vector2` right-hand side: the `_unchecked_mut` forms.
#[test]
fn test_solve_matrix2_vector2_unchecked_mut_exact() {
    let l: Matrix2<Fixed> = Matrix2Trait::new(int(2), int(0), int(0), int(4));
    let u: Matrix2<Fixed> = Matrix2Trait::new(int(2), int(0), int(0), int(4));
    let x: Vector2<Fixed> = Vector2Trait::new(int(-3), int(2));
    let bl: Vector2<Fixed> = Vector2Trait::new(int(-6), int(8));
    let bu: Vector2<Fixed> = Vector2Trait::new(int(-6), int(8));
    let mut b = bl;
    l.solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    u.solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.tr_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.tr_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.ad_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.ad_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
}

/// `Matrix2` against a `Vector2` right-hand side: `with_diag` solves `(strict lower triangle of l +
/// 2 I) x = bd` and leaves `2 x` in `b`, like upstream (the quotients by `diag` are never written
/// back); a zero `diag` returns `false` and leaves `b` unchanged.
#[test]
fn test_solve_matrix2_vector2_with_diag_exact() {
    let l: Matrix2<Fixed> = Matrix2Trait::new(int(2), int(0), int(0), int(4));
    let bd: Vector2<Fixed> = Vector2Trait::new(int(-6), int(4));
    let x2: Vector2<Fixed> = Vector2Trait::new(int(-6), int(4));
    let mut b = bd;
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(b == x2);
    let mut b = bd;
    l.solve_lower_triangular_with_diag_unchecked_mut(ref b, int(2));
    assert!(b == x2);
    let mut b = bd;
    assert!(!l.solve_lower_triangular_with_diag_mut(ref b, int(0)));
    assert!(b == bd);
}

/// `Matrix2` against a `Matrix2` right-hand side: the `Option` forms recover the integer solution
/// exactly (power-of-two diagonal:
/// every quotient is exact).
#[test]
fn test_solve_matrix2_matrix2_exact() {
    let l: Matrix2<Fixed> = Matrix2Trait::new(int(2), int(0), int(0), int(4));
    let u: Matrix2<Fixed> = Matrix2Trait::new(int(2), int(0), int(0), int(4));
    let x: Matrix2<Fixed> = Matrix2Trait::new(int(-3), int(0), int(2), int(-4));
    let bl: Matrix2<Fixed> = Matrix2Trait::new(int(-6), int(0), int(8), int(-16));
    let bu: Matrix2<Fixed> = Matrix2Trait::new(int(-6), int(0), int(8), int(-16));
    assert!(l.solve_lower_triangular(bl) == Option::Some(x));
    assert!(u.solve_upper_triangular(bu) == Option::Some(x));
    assert!(l.tr_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.tr_solve_upper_triangular(bl) == Option::Some(x));
    assert!(l.ad_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.ad_solve_upper_triangular(bl) == Option::Some(x));
}

/// `Matrix2` against a `Matrix2` right-hand side: the `_unchecked` forms, bit-identical to the
/// checked ones.
#[test]
fn test_solve_matrix2_matrix2_unchecked_exact() {
    let l: Matrix2<Fixed> = Matrix2Trait::new(int(2), int(0), int(0), int(4));
    let u: Matrix2<Fixed> = Matrix2Trait::new(int(2), int(0), int(0), int(4));
    let x: Matrix2<Fixed> = Matrix2Trait::new(int(-3), int(0), int(2), int(-4));
    let bl: Matrix2<Fixed> = Matrix2Trait::new(int(-6), int(0), int(8), int(-16));
    let bu: Matrix2<Fixed> = Matrix2Trait::new(int(-6), int(0), int(8), int(-16));
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    assert!(l.ad_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.ad_solve_upper_triangular_unchecked(bl) == x);
}

/// `Matrix2` against a `Matrix2` right-hand side: the `_mut` forms overwrite `b` with the solution
/// and return `true`.
#[test]
fn test_solve_matrix2_matrix2_mut_exact() {
    let l: Matrix2<Fixed> = Matrix2Trait::new(int(2), int(0), int(0), int(4));
    let u: Matrix2<Fixed> = Matrix2Trait::new(int(2), int(0), int(0), int(4));
    let x: Matrix2<Fixed> = Matrix2Trait::new(int(-3), int(0), int(2), int(-4));
    let bl: Matrix2<Fixed> = Matrix2Trait::new(int(-6), int(0), int(8), int(-16));
    let bu: Matrix2<Fixed> = Matrix2Trait::new(int(-6), int(0), int(8), int(-16));
    let mut b = bl;
    assert!(l.solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(u.solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.tr_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.tr_solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.ad_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.ad_solve_upper_triangular_mut(ref b));
    assert!(b == x);
}

/// `Matrix2` against a `Matrix2` right-hand side: the `_unchecked_mut` forms.
#[test]
fn test_solve_matrix2_matrix2_unchecked_mut_exact() {
    let l: Matrix2<Fixed> = Matrix2Trait::new(int(2), int(0), int(0), int(4));
    let u: Matrix2<Fixed> = Matrix2Trait::new(int(2), int(0), int(0), int(4));
    let x: Matrix2<Fixed> = Matrix2Trait::new(int(-3), int(0), int(2), int(-4));
    let bl: Matrix2<Fixed> = Matrix2Trait::new(int(-6), int(0), int(8), int(-16));
    let bu: Matrix2<Fixed> = Matrix2Trait::new(int(-6), int(0), int(8), int(-16));
    let mut b = bl;
    l.solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    u.solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.tr_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.tr_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.ad_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.ad_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
}

/// `Matrix2` against a `Matrix2` right-hand side: `with_diag` solves `(strict lower triangle of l +
/// 2 I) x = bd` and leaves `2 x` in `b`, like upstream (the quotients by `diag` are never written
/// back); a zero `diag` returns `false` and leaves `b` unchanged.
#[test]
fn test_solve_matrix2_matrix2_with_diag_exact() {
    let l: Matrix2<Fixed> = Matrix2Trait::new(int(2), int(0), int(0), int(4));
    let bd: Matrix2<Fixed> = Matrix2Trait::new(int(-6), int(0), int(4), int(-8));
    let x2: Matrix2<Fixed> = Matrix2Trait::new(int(-6), int(0), int(4), int(-8));
    let mut b = bd;
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(b == x2);
    let mut b = bd;
    l.solve_lower_triangular_with_diag_unchecked_mut(ref b, int(2));
    assert!(b == x2);
    let mut b = bd;
    assert!(!l.solve_lower_triangular_with_diag_mut(ref b, int(0)));
    assert!(b == bd);
}

/// `Matrix2` against a `Matrix2x3` right-hand side: the `Option` forms recover the integer solution
/// exactly (power-of-two diagonal:
/// every quotient is exact).
#[test]
fn test_solve_matrix2_matrix2x3_exact() {
    let l: Matrix2<Fixed> = Matrix2Trait::new(int(2), int(0), int(0), int(4));
    let u: Matrix2<Fixed> = Matrix2Trait::new(int(2), int(0), int(0), int(4));
    let x: Matrix2x3<Fixed> = Matrix2x3Trait::new(
        int(-3), int(0), int(3), int(2), int(-4), int(-1),
    );
    let bl: Matrix2x3<Fixed> = Matrix2x3Trait::new(
        int(-6), int(0), int(6), int(8), int(-16), int(-4),
    );
    let bu: Matrix2x3<Fixed> = Matrix2x3Trait::new(
        int(-6), int(0), int(6), int(8), int(-16), int(-4),
    );
    assert!(l.solve_lower_triangular(bl) == Option::Some(x));
    assert!(u.solve_upper_triangular(bu) == Option::Some(x));
    assert!(l.tr_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.tr_solve_upper_triangular(bl) == Option::Some(x));
    assert!(l.ad_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.ad_solve_upper_triangular(bl) == Option::Some(x));
}

/// `Matrix2` against a `Matrix2x3` right-hand side: the `_unchecked` forms, bit-identical to the
/// checked ones.
#[test]
fn test_solve_matrix2_matrix2x3_unchecked_exact() {
    let l: Matrix2<Fixed> = Matrix2Trait::new(int(2), int(0), int(0), int(4));
    let u: Matrix2<Fixed> = Matrix2Trait::new(int(2), int(0), int(0), int(4));
    let x: Matrix2x3<Fixed> = Matrix2x3Trait::new(
        int(-3), int(0), int(3), int(2), int(-4), int(-1),
    );
    let bl: Matrix2x3<Fixed> = Matrix2x3Trait::new(
        int(-6), int(0), int(6), int(8), int(-16), int(-4),
    );
    let bu: Matrix2x3<Fixed> = Matrix2x3Trait::new(
        int(-6), int(0), int(6), int(8), int(-16), int(-4),
    );
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    assert!(l.ad_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.ad_solve_upper_triangular_unchecked(bl) == x);
}

/// `Matrix2` against a `Matrix2x3` right-hand side: the `_mut` forms overwrite `b` with the
/// solution and return `true`.
#[test]
fn test_solve_matrix2_matrix2x3_mut_exact() {
    let l: Matrix2<Fixed> = Matrix2Trait::new(int(2), int(0), int(0), int(4));
    let u: Matrix2<Fixed> = Matrix2Trait::new(int(2), int(0), int(0), int(4));
    let x: Matrix2x3<Fixed> = Matrix2x3Trait::new(
        int(-3), int(0), int(3), int(2), int(-4), int(-1),
    );
    let bl: Matrix2x3<Fixed> = Matrix2x3Trait::new(
        int(-6), int(0), int(6), int(8), int(-16), int(-4),
    );
    let bu: Matrix2x3<Fixed> = Matrix2x3Trait::new(
        int(-6), int(0), int(6), int(8), int(-16), int(-4),
    );
    let mut b = bl;
    assert!(l.solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(u.solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.tr_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.tr_solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.ad_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.ad_solve_upper_triangular_mut(ref b));
    assert!(b == x);
}

/// `Matrix2` against a `Matrix2x3` right-hand side: the `_unchecked_mut` forms.
#[test]
fn test_solve_matrix2_matrix2x3_unchecked_mut_exact() {
    let l: Matrix2<Fixed> = Matrix2Trait::new(int(2), int(0), int(0), int(4));
    let u: Matrix2<Fixed> = Matrix2Trait::new(int(2), int(0), int(0), int(4));
    let x: Matrix2x3<Fixed> = Matrix2x3Trait::new(
        int(-3), int(0), int(3), int(2), int(-4), int(-1),
    );
    let bl: Matrix2x3<Fixed> = Matrix2x3Trait::new(
        int(-6), int(0), int(6), int(8), int(-16), int(-4),
    );
    let bu: Matrix2x3<Fixed> = Matrix2x3Trait::new(
        int(-6), int(0), int(6), int(8), int(-16), int(-4),
    );
    let mut b = bl;
    l.solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    u.solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.tr_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.tr_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.ad_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.ad_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
}

/// `Matrix2` against a `Matrix2x3` right-hand side: `with_diag` solves `(strict lower triangle of l
/// + 2 I) x = bd` and leaves `2 x` in `b`, like upstream (the quotients by `diag` are never written
/// back); a zero `diag` returns `false` and leaves `b` unchanged.
#[test]
fn test_solve_matrix2_matrix2x3_with_diag_exact() {
    let l: Matrix2<Fixed> = Matrix2Trait::new(int(2), int(0), int(0), int(4));
    let bd: Matrix2x3<Fixed> = Matrix2x3Trait::new(
        int(-6), int(0), int(6), int(4), int(-8), int(-2),
    );
    let x2: Matrix2x3<Fixed> = Matrix2x3Trait::new(
        int(-6), int(0), int(6), int(4), int(-8), int(-2),
    );
    let mut b = bd;
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(b == x2);
    let mut b = bd;
    l.solve_lower_triangular_with_diag_unchecked_mut(ref b, int(2));
    assert!(b == x2);
    let mut b = bd;
    assert!(!l.solve_lower_triangular_with_diag_mut(ref b, int(0)));
    assert!(b == bd);
}

/// `Matrix2` against a `Matrix2x4` right-hand side: the `Option` forms recover the integer solution
/// exactly (power-of-two diagonal:
/// every quotient is exact).
#[test]
fn test_solve_matrix2_matrix2x4_exact() {
    let l: Matrix2<Fixed> = Matrix2Trait::new(int(2), int(0), int(0), int(4));
    let u: Matrix2<Fixed> = Matrix2Trait::new(int(2), int(0), int(0), int(4));
    let x: Matrix2x4<Fixed> = Matrix2x4Trait::new(
        int(-3), int(0), int(3), int(-3), int(2), int(-4), int(-1), int(2),
    );
    let bl: Matrix2x4<Fixed> = Matrix2x4Trait::new(
        int(-6), int(0), int(6), int(-6), int(8), int(-16), int(-4), int(8),
    );
    let bu: Matrix2x4<Fixed> = Matrix2x4Trait::new(
        int(-6), int(0), int(6), int(-6), int(8), int(-16), int(-4), int(8),
    );
    assert!(l.solve_lower_triangular(bl) == Option::Some(x));
    assert!(u.solve_upper_triangular(bu) == Option::Some(x));
    assert!(l.tr_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.tr_solve_upper_triangular(bl) == Option::Some(x));
    assert!(l.ad_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.ad_solve_upper_triangular(bl) == Option::Some(x));
}

/// `Matrix2` against a `Matrix2x4` right-hand side: the `_unchecked` forms, bit-identical to the
/// checked ones.
#[test]
fn test_solve_matrix2_matrix2x4_unchecked_exact() {
    let l: Matrix2<Fixed> = Matrix2Trait::new(int(2), int(0), int(0), int(4));
    let u: Matrix2<Fixed> = Matrix2Trait::new(int(2), int(0), int(0), int(4));
    let x: Matrix2x4<Fixed> = Matrix2x4Trait::new(
        int(-3), int(0), int(3), int(-3), int(2), int(-4), int(-1), int(2),
    );
    let bl: Matrix2x4<Fixed> = Matrix2x4Trait::new(
        int(-6), int(0), int(6), int(-6), int(8), int(-16), int(-4), int(8),
    );
    let bu: Matrix2x4<Fixed> = Matrix2x4Trait::new(
        int(-6), int(0), int(6), int(-6), int(8), int(-16), int(-4), int(8),
    );
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    assert!(l.ad_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.ad_solve_upper_triangular_unchecked(bl) == x);
}

/// `Matrix2` against a `Matrix2x4` right-hand side: the `_mut` forms overwrite `b` with the
/// solution and return `true`.
#[test]
fn test_solve_matrix2_matrix2x4_mut_exact() {
    let l: Matrix2<Fixed> = Matrix2Trait::new(int(2), int(0), int(0), int(4));
    let u: Matrix2<Fixed> = Matrix2Trait::new(int(2), int(0), int(0), int(4));
    let x: Matrix2x4<Fixed> = Matrix2x4Trait::new(
        int(-3), int(0), int(3), int(-3), int(2), int(-4), int(-1), int(2),
    );
    let bl: Matrix2x4<Fixed> = Matrix2x4Trait::new(
        int(-6), int(0), int(6), int(-6), int(8), int(-16), int(-4), int(8),
    );
    let bu: Matrix2x4<Fixed> = Matrix2x4Trait::new(
        int(-6), int(0), int(6), int(-6), int(8), int(-16), int(-4), int(8),
    );
    let mut b = bl;
    assert!(l.solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(u.solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.tr_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.tr_solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.ad_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.ad_solve_upper_triangular_mut(ref b));
    assert!(b == x);
}

/// `Matrix2` against a `Matrix2x4` right-hand side: the `_unchecked_mut` forms.
#[test]
fn test_solve_matrix2_matrix2x4_unchecked_mut_exact() {
    let l: Matrix2<Fixed> = Matrix2Trait::new(int(2), int(0), int(0), int(4));
    let u: Matrix2<Fixed> = Matrix2Trait::new(int(2), int(0), int(0), int(4));
    let x: Matrix2x4<Fixed> = Matrix2x4Trait::new(
        int(-3), int(0), int(3), int(-3), int(2), int(-4), int(-1), int(2),
    );
    let bl: Matrix2x4<Fixed> = Matrix2x4Trait::new(
        int(-6), int(0), int(6), int(-6), int(8), int(-16), int(-4), int(8),
    );
    let bu: Matrix2x4<Fixed> = Matrix2x4Trait::new(
        int(-6), int(0), int(6), int(-6), int(8), int(-16), int(-4), int(8),
    );
    let mut b = bl;
    l.solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    u.solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.tr_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.tr_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.ad_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.ad_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
}

/// `Matrix2` against a `Matrix2x4` right-hand side: `with_diag` solves `(strict lower triangle of l
/// + 2 I) x = bd` and leaves `2 x` in `b`, like upstream (the quotients by `diag` are never written
/// back); a zero `diag` returns `false` and leaves `b` unchanged.
#[test]
fn test_solve_matrix2_matrix2x4_with_diag_exact() {
    let l: Matrix2<Fixed> = Matrix2Trait::new(int(2), int(0), int(0), int(4));
    let bd: Matrix2x4<Fixed> = Matrix2x4Trait::new(
        int(-6), int(0), int(6), int(-6), int(4), int(-8), int(-2), int(4),
    );
    let x2: Matrix2x4<Fixed> = Matrix2x4Trait::new(
        int(-6), int(0), int(6), int(-6), int(4), int(-8), int(-2), int(4),
    );
    let mut b = bd;
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(b == x2);
    let mut b = bd;
    l.solve_lower_triangular_with_diag_unchecked_mut(ref b, int(2));
    assert!(b == x2);
    let mut b = bd;
    assert!(!l.solve_lower_triangular_with_diag_mut(ref b, int(0)));
    assert!(b == bd);
}

/// `Matrix2` against a `Matrix2x5` right-hand side: the `Option` forms recover the integer solution
/// exactly (power-of-two diagonal:
/// every quotient is exact).
#[test]
fn test_solve_matrix2_matrix2x5_exact() {
    let l: Matrix2<Fixed> = Matrix2Trait::new(int(2), int(0), int(0), int(4));
    let u: Matrix2<Fixed> = Matrix2Trait::new(int(2), int(0), int(0), int(4));
    let x: Matrix2x5<Fixed> = Matrix2x5Trait::new(
        int(-3), int(0), int(3), int(-3), int(0), int(2), int(-4), int(-1), int(2), int(-4),
    );
    let bl: Matrix2x5<Fixed> = Matrix2x5Trait::new(
        int(-6), int(0), int(6), int(-6), int(0), int(8), int(-16), int(-4), int(8), int(-16),
    );
    let bu: Matrix2x5<Fixed> = Matrix2x5Trait::new(
        int(-6), int(0), int(6), int(-6), int(0), int(8), int(-16), int(-4), int(8), int(-16),
    );
    assert!(l.solve_lower_triangular(bl) == Option::Some(x));
    assert!(u.solve_upper_triangular(bu) == Option::Some(x));
    assert!(l.tr_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.tr_solve_upper_triangular(bl) == Option::Some(x));
    assert!(l.ad_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.ad_solve_upper_triangular(bl) == Option::Some(x));
}

/// `Matrix2` against a `Matrix2x5` right-hand side: the `_unchecked` forms, bit-identical to the
/// checked ones.
#[test]
fn test_solve_matrix2_matrix2x5_unchecked_exact() {
    let l: Matrix2<Fixed> = Matrix2Trait::new(int(2), int(0), int(0), int(4));
    let u: Matrix2<Fixed> = Matrix2Trait::new(int(2), int(0), int(0), int(4));
    let x: Matrix2x5<Fixed> = Matrix2x5Trait::new(
        int(-3), int(0), int(3), int(-3), int(0), int(2), int(-4), int(-1), int(2), int(-4),
    );
    let bl: Matrix2x5<Fixed> = Matrix2x5Trait::new(
        int(-6), int(0), int(6), int(-6), int(0), int(8), int(-16), int(-4), int(8), int(-16),
    );
    let bu: Matrix2x5<Fixed> = Matrix2x5Trait::new(
        int(-6), int(0), int(6), int(-6), int(0), int(8), int(-16), int(-4), int(8), int(-16),
    );
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    assert!(l.ad_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.ad_solve_upper_triangular_unchecked(bl) == x);
}

/// `Matrix2` against a `Matrix2x5` right-hand side: the `_mut` forms overwrite `b` with the
/// solution and return `true`.
#[test]
fn test_solve_matrix2_matrix2x5_mut_exact() {
    let l: Matrix2<Fixed> = Matrix2Trait::new(int(2), int(0), int(0), int(4));
    let u: Matrix2<Fixed> = Matrix2Trait::new(int(2), int(0), int(0), int(4));
    let x: Matrix2x5<Fixed> = Matrix2x5Trait::new(
        int(-3), int(0), int(3), int(-3), int(0), int(2), int(-4), int(-1), int(2), int(-4),
    );
    let bl: Matrix2x5<Fixed> = Matrix2x5Trait::new(
        int(-6), int(0), int(6), int(-6), int(0), int(8), int(-16), int(-4), int(8), int(-16),
    );
    let bu: Matrix2x5<Fixed> = Matrix2x5Trait::new(
        int(-6), int(0), int(6), int(-6), int(0), int(8), int(-16), int(-4), int(8), int(-16),
    );
    let mut b = bl;
    assert!(l.solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(u.solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.tr_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.tr_solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.ad_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.ad_solve_upper_triangular_mut(ref b));
    assert!(b == x);
}

/// `Matrix2` against a `Matrix2x5` right-hand side: the `_unchecked_mut` forms.
#[test]
fn test_solve_matrix2_matrix2x5_unchecked_mut_exact() {
    let l: Matrix2<Fixed> = Matrix2Trait::new(int(2), int(0), int(0), int(4));
    let u: Matrix2<Fixed> = Matrix2Trait::new(int(2), int(0), int(0), int(4));
    let x: Matrix2x5<Fixed> = Matrix2x5Trait::new(
        int(-3), int(0), int(3), int(-3), int(0), int(2), int(-4), int(-1), int(2), int(-4),
    );
    let bl: Matrix2x5<Fixed> = Matrix2x5Trait::new(
        int(-6), int(0), int(6), int(-6), int(0), int(8), int(-16), int(-4), int(8), int(-16),
    );
    let bu: Matrix2x5<Fixed> = Matrix2x5Trait::new(
        int(-6), int(0), int(6), int(-6), int(0), int(8), int(-16), int(-4), int(8), int(-16),
    );
    let mut b = bl;
    l.solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    u.solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.tr_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.tr_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.ad_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.ad_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
}

/// `Matrix2` against a `Matrix2x5` right-hand side: `with_diag` solves `(strict lower triangle of l
/// + 2 I) x = bd` and leaves `2 x` in `b`, like upstream (the quotients by `diag` are never written
/// back); a zero `diag` returns `false` and leaves `b` unchanged.
#[test]
fn test_solve_matrix2_matrix2x5_with_diag_exact() {
    let l: Matrix2<Fixed> = Matrix2Trait::new(int(2), int(0), int(0), int(4));
    let bd: Matrix2x5<Fixed> = Matrix2x5Trait::new(
        int(-6), int(0), int(6), int(-6), int(0), int(4), int(-8), int(-2), int(4), int(-8),
    );
    let x2: Matrix2x5<Fixed> = Matrix2x5Trait::new(
        int(-6), int(0), int(6), int(-6), int(0), int(4), int(-8), int(-2), int(4), int(-8),
    );
    let mut b = bd;
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(b == x2);
    let mut b = bd;
    l.solve_lower_triangular_with_diag_unchecked_mut(ref b, int(2));
    assert!(b == x2);
    let mut b = bd;
    assert!(!l.solve_lower_triangular_with_diag_mut(ref b, int(0)));
    assert!(b == bd);
}

/// `Matrix2` against a `Matrix2x6` right-hand side: the `Option` forms recover the integer solution
/// exactly (power-of-two diagonal:
/// every quotient is exact).
#[test]
fn test_solve_matrix2_matrix2x6_exact() {
    let l: Matrix2<Fixed> = Matrix2Trait::new(int(2), int(0), int(0), int(4));
    let u: Matrix2<Fixed> = Matrix2Trait::new(int(2), int(0), int(0), int(4));
    let x: Matrix2x6<Fixed> = Matrix2x6Trait::new(
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(0),
        int(3),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-4),
        int(-1),
    );
    let bl: Matrix2x6<Fixed> = Matrix2x6Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(6),
        int(8),
        int(-16),
        int(-4),
        int(8),
        int(-16),
        int(-4),
    );
    let bu: Matrix2x6<Fixed> = Matrix2x6Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(6),
        int(8),
        int(-16),
        int(-4),
        int(8),
        int(-16),
        int(-4),
    );
    assert!(l.solve_lower_triangular(bl) == Option::Some(x));
    assert!(u.solve_upper_triangular(bu) == Option::Some(x));
    assert!(l.tr_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.tr_solve_upper_triangular(bl) == Option::Some(x));
    assert!(l.ad_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.ad_solve_upper_triangular(bl) == Option::Some(x));
}

/// `Matrix2` against a `Matrix2x6` right-hand side: the `_unchecked` forms, bit-identical to the
/// checked ones.
#[test]
fn test_solve_matrix2_matrix2x6_unchecked_exact() {
    let l: Matrix2<Fixed> = Matrix2Trait::new(int(2), int(0), int(0), int(4));
    let u: Matrix2<Fixed> = Matrix2Trait::new(int(2), int(0), int(0), int(4));
    let x: Matrix2x6<Fixed> = Matrix2x6Trait::new(
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(0),
        int(3),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-4),
        int(-1),
    );
    let bl: Matrix2x6<Fixed> = Matrix2x6Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(6),
        int(8),
        int(-16),
        int(-4),
        int(8),
        int(-16),
        int(-4),
    );
    let bu: Matrix2x6<Fixed> = Matrix2x6Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(6),
        int(8),
        int(-16),
        int(-4),
        int(8),
        int(-16),
        int(-4),
    );
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    assert!(l.ad_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.ad_solve_upper_triangular_unchecked(bl) == x);
}

/// `Matrix2` against a `Matrix2x6` right-hand side: the `_mut` forms overwrite `b` with the
/// solution and return `true`.
#[test]
fn test_solve_matrix2_matrix2x6_mut_exact() {
    let l: Matrix2<Fixed> = Matrix2Trait::new(int(2), int(0), int(0), int(4));
    let u: Matrix2<Fixed> = Matrix2Trait::new(int(2), int(0), int(0), int(4));
    let x: Matrix2x6<Fixed> = Matrix2x6Trait::new(
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(0),
        int(3),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-4),
        int(-1),
    );
    let bl: Matrix2x6<Fixed> = Matrix2x6Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(6),
        int(8),
        int(-16),
        int(-4),
        int(8),
        int(-16),
        int(-4),
    );
    let bu: Matrix2x6<Fixed> = Matrix2x6Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(6),
        int(8),
        int(-16),
        int(-4),
        int(8),
        int(-16),
        int(-4),
    );
    let mut b = bl;
    assert!(l.solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(u.solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.tr_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.tr_solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.ad_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.ad_solve_upper_triangular_mut(ref b));
    assert!(b == x);
}

/// `Matrix2` against a `Matrix2x6` right-hand side: the `_unchecked_mut` forms.
#[test]
fn test_solve_matrix2_matrix2x6_unchecked_mut_exact() {
    let l: Matrix2<Fixed> = Matrix2Trait::new(int(2), int(0), int(0), int(4));
    let u: Matrix2<Fixed> = Matrix2Trait::new(int(2), int(0), int(0), int(4));
    let x: Matrix2x6<Fixed> = Matrix2x6Trait::new(
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(0),
        int(3),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-4),
        int(-1),
    );
    let bl: Matrix2x6<Fixed> = Matrix2x6Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(6),
        int(8),
        int(-16),
        int(-4),
        int(8),
        int(-16),
        int(-4),
    );
    let bu: Matrix2x6<Fixed> = Matrix2x6Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(6),
        int(8),
        int(-16),
        int(-4),
        int(8),
        int(-16),
        int(-4),
    );
    let mut b = bl;
    l.solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    u.solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.tr_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.tr_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.ad_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.ad_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
}

/// `Matrix2` against a `Matrix2x6` right-hand side: `with_diag` solves `(strict lower triangle of l
/// + 2 I) x = bd` and leaves `2 x` in `b`, like upstream (the quotients by `diag` are never written
/// back); a zero `diag` returns `false` and leaves `b` unchanged.
#[test]
fn test_solve_matrix2_matrix2x6_with_diag_exact() {
    let l: Matrix2<Fixed> = Matrix2Trait::new(int(2), int(0), int(0), int(4));
    let bd: Matrix2x6<Fixed> = Matrix2x6Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(6),
        int(4),
        int(-8),
        int(-2),
        int(4),
        int(-8),
        int(-2),
    );
    let x2: Matrix2x6<Fixed> = Matrix2x6Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(6),
        int(4),
        int(-8),
        int(-2),
        int(4),
        int(-8),
        int(-2),
    );
    let mut b = bd;
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(b == x2);
    let mut b = bd;
    l.solve_lower_triangular_with_diag_unchecked_mut(ref b, int(2));
    assert!(b == x2);
    let mut b = bd;
    assert!(!l.solve_lower_triangular_with_diag_mut(ref b, int(0)));
    assert!(b == bd);
}

/// `Matrix3` against a `Vector3` right-hand side: the `Option` forms recover the integer solution
/// exactly (power-of-two diagonal:
/// every quotient is exact).
#[test]
fn test_solve_matrix3_vector3_exact() {
    let l: Matrix3<Fixed> = Matrix3Trait::new(
        int(2), int(0), int(0), int(0), int(4), int(0), int(3), int(1), int(1),
    );
    let u: Matrix3<Fixed> = Matrix3Trait::new(
        int(2), int(0), int(3), int(0), int(4), int(1), int(0), int(0), int(1),
    );
    let x: Vector3<Fixed> = Vector3Trait::new(int(-3), int(2), int(-2));
    let bl: Vector3<Fixed> = Vector3Trait::new(int(-6), int(8), int(-9));
    let bu: Vector3<Fixed> = Vector3Trait::new(int(-12), int(6), int(-2));
    assert!(l.solve_lower_triangular(bl) == Option::Some(x));
    assert!(u.solve_upper_triangular(bu) == Option::Some(x));
    assert!(l.tr_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.tr_solve_upper_triangular(bl) == Option::Some(x));
    assert!(l.ad_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.ad_solve_upper_triangular(bl) == Option::Some(x));
}

/// `Matrix3` against a `Vector3` right-hand side: the `_unchecked` forms, bit-identical to the
/// checked ones.
#[test]
fn test_solve_matrix3_vector3_unchecked_exact() {
    let l: Matrix3<Fixed> = Matrix3Trait::new(
        int(2), int(0), int(0), int(0), int(4), int(0), int(3), int(1), int(1),
    );
    let u: Matrix3<Fixed> = Matrix3Trait::new(
        int(2), int(0), int(3), int(0), int(4), int(1), int(0), int(0), int(1),
    );
    let x: Vector3<Fixed> = Vector3Trait::new(int(-3), int(2), int(-2));
    let bl: Vector3<Fixed> = Vector3Trait::new(int(-6), int(8), int(-9));
    let bu: Vector3<Fixed> = Vector3Trait::new(int(-12), int(6), int(-2));
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    assert!(l.ad_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.ad_solve_upper_triangular_unchecked(bl) == x);
}

/// `Matrix3` against a `Vector3` right-hand side: the `_mut` forms overwrite `b` with the solution
/// and return `true`.
#[test]
fn test_solve_matrix3_vector3_mut_exact() {
    let l: Matrix3<Fixed> = Matrix3Trait::new(
        int(2), int(0), int(0), int(0), int(4), int(0), int(3), int(1), int(1),
    );
    let u: Matrix3<Fixed> = Matrix3Trait::new(
        int(2), int(0), int(3), int(0), int(4), int(1), int(0), int(0), int(1),
    );
    let x: Vector3<Fixed> = Vector3Trait::new(int(-3), int(2), int(-2));
    let bl: Vector3<Fixed> = Vector3Trait::new(int(-6), int(8), int(-9));
    let bu: Vector3<Fixed> = Vector3Trait::new(int(-12), int(6), int(-2));
    let mut b = bl;
    assert!(l.solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(u.solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.tr_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.tr_solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.ad_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.ad_solve_upper_triangular_mut(ref b));
    assert!(b == x);
}

/// `Matrix3` against a `Vector3` right-hand side: the `_unchecked_mut` forms.
#[test]
fn test_solve_matrix3_vector3_unchecked_mut_exact() {
    let l: Matrix3<Fixed> = Matrix3Trait::new(
        int(2), int(0), int(0), int(0), int(4), int(0), int(3), int(1), int(1),
    );
    let u: Matrix3<Fixed> = Matrix3Trait::new(
        int(2), int(0), int(3), int(0), int(4), int(1), int(0), int(0), int(1),
    );
    let x: Vector3<Fixed> = Vector3Trait::new(int(-3), int(2), int(-2));
    let bl: Vector3<Fixed> = Vector3Trait::new(int(-6), int(8), int(-9));
    let bu: Vector3<Fixed> = Vector3Trait::new(int(-12), int(6), int(-2));
    let mut b = bl;
    l.solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    u.solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.tr_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.tr_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.ad_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.ad_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
}

/// `Matrix3` against a `Vector3` right-hand side: `with_diag` solves `(strict lower triangle of l +
/// 2 I) x = bd` and leaves `2 x` in `b`, like upstream (the quotients by `diag` are never written
/// back); a zero `diag` returns `false` and leaves `b` unchanged.
#[test]
fn test_solve_matrix3_vector3_with_diag_exact() {
    let l: Matrix3<Fixed> = Matrix3Trait::new(
        int(2), int(0), int(0), int(0), int(4), int(0), int(3), int(1), int(1),
    );
    let bd: Vector3<Fixed> = Vector3Trait::new(int(-6), int(4), int(-11));
    let x2: Vector3<Fixed> = Vector3Trait::new(int(-6), int(4), int(-4));
    let mut b = bd;
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(b == x2);
    let mut b = bd;
    l.solve_lower_triangular_with_diag_unchecked_mut(ref b, int(2));
    assert!(b == x2);
    let mut b = bd;
    assert!(!l.solve_lower_triangular_with_diag_mut(ref b, int(0)));
    assert!(b == bd);
}

/// `Matrix3` against a `Matrix3x2` right-hand side: the `Option` forms recover the integer solution
/// exactly (power-of-two diagonal:
/// every quotient is exact).
#[test]
fn test_solve_matrix3_matrix3x2_exact() {
    let l: Matrix3<Fixed> = Matrix3Trait::new(
        int(2), int(0), int(0), int(0), int(4), int(0), int(3), int(1), int(1),
    );
    let u: Matrix3<Fixed> = Matrix3Trait::new(
        int(2), int(0), int(3), int(0), int(4), int(1), int(0), int(0), int(1),
    );
    let x: Matrix3x2<Fixed> = Matrix3x2Trait::new(
        int(-3), int(0), int(2), int(-4), int(-2), int(1),
    );
    let bl: Matrix3x2<Fixed> = Matrix3x2Trait::new(
        int(-6), int(0), int(8), int(-16), int(-9), int(-3),
    );
    let bu: Matrix3x2<Fixed> = Matrix3x2Trait::new(
        int(-12), int(3), int(6), int(-15), int(-2), int(1),
    );
    assert!(l.solve_lower_triangular(bl) == Option::Some(x));
    assert!(u.solve_upper_triangular(bu) == Option::Some(x));
    assert!(l.tr_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.tr_solve_upper_triangular(bl) == Option::Some(x));
    assert!(l.ad_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.ad_solve_upper_triangular(bl) == Option::Some(x));
}

/// `Matrix3` against a `Matrix3x2` right-hand side: the `_unchecked` forms, bit-identical to the
/// checked ones.
#[test]
fn test_solve_matrix3_matrix3x2_unchecked_exact() {
    let l: Matrix3<Fixed> = Matrix3Trait::new(
        int(2), int(0), int(0), int(0), int(4), int(0), int(3), int(1), int(1),
    );
    let u: Matrix3<Fixed> = Matrix3Trait::new(
        int(2), int(0), int(3), int(0), int(4), int(1), int(0), int(0), int(1),
    );
    let x: Matrix3x2<Fixed> = Matrix3x2Trait::new(
        int(-3), int(0), int(2), int(-4), int(-2), int(1),
    );
    let bl: Matrix3x2<Fixed> = Matrix3x2Trait::new(
        int(-6), int(0), int(8), int(-16), int(-9), int(-3),
    );
    let bu: Matrix3x2<Fixed> = Matrix3x2Trait::new(
        int(-12), int(3), int(6), int(-15), int(-2), int(1),
    );
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    assert!(l.ad_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.ad_solve_upper_triangular_unchecked(bl) == x);
}

/// `Matrix3` against a `Matrix3x2` right-hand side: the `_mut` forms overwrite `b` with the
/// solution and return `true`.
#[test]
fn test_solve_matrix3_matrix3x2_mut_exact() {
    let l: Matrix3<Fixed> = Matrix3Trait::new(
        int(2), int(0), int(0), int(0), int(4), int(0), int(3), int(1), int(1),
    );
    let u: Matrix3<Fixed> = Matrix3Trait::new(
        int(2), int(0), int(3), int(0), int(4), int(1), int(0), int(0), int(1),
    );
    let x: Matrix3x2<Fixed> = Matrix3x2Trait::new(
        int(-3), int(0), int(2), int(-4), int(-2), int(1),
    );
    let bl: Matrix3x2<Fixed> = Matrix3x2Trait::new(
        int(-6), int(0), int(8), int(-16), int(-9), int(-3),
    );
    let bu: Matrix3x2<Fixed> = Matrix3x2Trait::new(
        int(-12), int(3), int(6), int(-15), int(-2), int(1),
    );
    let mut b = bl;
    assert!(l.solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(u.solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.tr_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.tr_solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.ad_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.ad_solve_upper_triangular_mut(ref b));
    assert!(b == x);
}

/// `Matrix3` against a `Matrix3x2` right-hand side: the `_unchecked_mut` forms.
#[test]
fn test_solve_matrix3_matrix3x2_unchecked_mut_exact() {
    let l: Matrix3<Fixed> = Matrix3Trait::new(
        int(2), int(0), int(0), int(0), int(4), int(0), int(3), int(1), int(1),
    );
    let u: Matrix3<Fixed> = Matrix3Trait::new(
        int(2), int(0), int(3), int(0), int(4), int(1), int(0), int(0), int(1),
    );
    let x: Matrix3x2<Fixed> = Matrix3x2Trait::new(
        int(-3), int(0), int(2), int(-4), int(-2), int(1),
    );
    let bl: Matrix3x2<Fixed> = Matrix3x2Trait::new(
        int(-6), int(0), int(8), int(-16), int(-9), int(-3),
    );
    let bu: Matrix3x2<Fixed> = Matrix3x2Trait::new(
        int(-12), int(3), int(6), int(-15), int(-2), int(1),
    );
    let mut b = bl;
    l.solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    u.solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.tr_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.tr_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.ad_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.ad_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
}

/// `Matrix3` against a `Matrix3x2` right-hand side: `with_diag` solves `(strict lower triangle of l
/// + 2 I) x = bd` and leaves `2 x` in `b`, like upstream (the quotients by `diag` are never written
/// back); a zero `diag` returns `false` and leaves `b` unchanged.
#[test]
fn test_solve_matrix3_matrix3x2_with_diag_exact() {
    let l: Matrix3<Fixed> = Matrix3Trait::new(
        int(2), int(0), int(0), int(0), int(4), int(0), int(3), int(1), int(1),
    );
    let bd: Matrix3x2<Fixed> = Matrix3x2Trait::new(
        int(-6), int(0), int(4), int(-8), int(-11), int(-2),
    );
    let x2: Matrix3x2<Fixed> = Matrix3x2Trait::new(
        int(-6), int(0), int(4), int(-8), int(-4), int(2),
    );
    let mut b = bd;
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(b == x2);
    let mut b = bd;
    l.solve_lower_triangular_with_diag_unchecked_mut(ref b, int(2));
    assert!(b == x2);
    let mut b = bd;
    assert!(!l.solve_lower_triangular_with_diag_mut(ref b, int(0)));
    assert!(b == bd);
}

/// `Matrix3` against a `Matrix3` right-hand side: the `Option` forms recover the integer solution
/// exactly (power-of-two diagonal:
/// every quotient is exact).
#[test]
fn test_solve_matrix3_matrix3_exact() {
    let l: Matrix3<Fixed> = Matrix3Trait::new(
        int(2), int(0), int(0), int(0), int(4), int(0), int(3), int(1), int(1),
    );
    let u: Matrix3<Fixed> = Matrix3Trait::new(
        int(2), int(0), int(3), int(0), int(4), int(1), int(0), int(0), int(1),
    );
    let x: Matrix3<Fixed> = Matrix3Trait::new(
        int(-3), int(0), int(3), int(2), int(-4), int(-1), int(-2), int(1), int(4),
    );
    let bl: Matrix3<Fixed> = Matrix3Trait::new(
        int(-6), int(0), int(6), int(8), int(-16), int(-4), int(-9), int(-3), int(12),
    );
    let bu: Matrix3<Fixed> = Matrix3Trait::new(
        int(-12), int(3), int(18), int(6), int(-15), int(0), int(-2), int(1), int(4),
    );
    assert!(l.solve_lower_triangular(bl) == Option::Some(x));
    assert!(u.solve_upper_triangular(bu) == Option::Some(x));
    assert!(l.tr_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.tr_solve_upper_triangular(bl) == Option::Some(x));
    assert!(l.ad_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.ad_solve_upper_triangular(bl) == Option::Some(x));
}

/// `Matrix3` against a `Matrix3` right-hand side: the `_unchecked` forms, bit-identical to the
/// checked ones.
#[test]
fn test_solve_matrix3_matrix3_unchecked_exact() {
    let l: Matrix3<Fixed> = Matrix3Trait::new(
        int(2), int(0), int(0), int(0), int(4), int(0), int(3), int(1), int(1),
    );
    let u: Matrix3<Fixed> = Matrix3Trait::new(
        int(2), int(0), int(3), int(0), int(4), int(1), int(0), int(0), int(1),
    );
    let x: Matrix3<Fixed> = Matrix3Trait::new(
        int(-3), int(0), int(3), int(2), int(-4), int(-1), int(-2), int(1), int(4),
    );
    let bl: Matrix3<Fixed> = Matrix3Trait::new(
        int(-6), int(0), int(6), int(8), int(-16), int(-4), int(-9), int(-3), int(12),
    );
    let bu: Matrix3<Fixed> = Matrix3Trait::new(
        int(-12), int(3), int(18), int(6), int(-15), int(0), int(-2), int(1), int(4),
    );
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    assert!(l.ad_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.ad_solve_upper_triangular_unchecked(bl) == x);
}

/// `Matrix3` against a `Matrix3` right-hand side: the `_mut` forms overwrite `b` with the solution
/// and return `true`.
#[test]
fn test_solve_matrix3_matrix3_mut_exact() {
    let l: Matrix3<Fixed> = Matrix3Trait::new(
        int(2), int(0), int(0), int(0), int(4), int(0), int(3), int(1), int(1),
    );
    let u: Matrix3<Fixed> = Matrix3Trait::new(
        int(2), int(0), int(3), int(0), int(4), int(1), int(0), int(0), int(1),
    );
    let x: Matrix3<Fixed> = Matrix3Trait::new(
        int(-3), int(0), int(3), int(2), int(-4), int(-1), int(-2), int(1), int(4),
    );
    let bl: Matrix3<Fixed> = Matrix3Trait::new(
        int(-6), int(0), int(6), int(8), int(-16), int(-4), int(-9), int(-3), int(12),
    );
    let bu: Matrix3<Fixed> = Matrix3Trait::new(
        int(-12), int(3), int(18), int(6), int(-15), int(0), int(-2), int(1), int(4),
    );
    let mut b = bl;
    assert!(l.solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(u.solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.tr_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.tr_solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.ad_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.ad_solve_upper_triangular_mut(ref b));
    assert!(b == x);
}

/// `Matrix3` against a `Matrix3` right-hand side: the `_unchecked_mut` forms.
#[test]
fn test_solve_matrix3_matrix3_unchecked_mut_exact() {
    let l: Matrix3<Fixed> = Matrix3Trait::new(
        int(2), int(0), int(0), int(0), int(4), int(0), int(3), int(1), int(1),
    );
    let u: Matrix3<Fixed> = Matrix3Trait::new(
        int(2), int(0), int(3), int(0), int(4), int(1), int(0), int(0), int(1),
    );
    let x: Matrix3<Fixed> = Matrix3Trait::new(
        int(-3), int(0), int(3), int(2), int(-4), int(-1), int(-2), int(1), int(4),
    );
    let bl: Matrix3<Fixed> = Matrix3Trait::new(
        int(-6), int(0), int(6), int(8), int(-16), int(-4), int(-9), int(-3), int(12),
    );
    let bu: Matrix3<Fixed> = Matrix3Trait::new(
        int(-12), int(3), int(18), int(6), int(-15), int(0), int(-2), int(1), int(4),
    );
    let mut b = bl;
    l.solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    u.solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.tr_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.tr_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.ad_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.ad_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
}

/// `Matrix3` against a `Matrix3` right-hand side: `with_diag` solves `(strict lower triangle of l +
/// 2 I) x = bd` and leaves `2 x` in `b`, like upstream (the quotients by `diag` are never written
/// back); a zero `diag` returns `false` and leaves `b` unchanged.
#[test]
fn test_solve_matrix3_matrix3_with_diag_exact() {
    let l: Matrix3<Fixed> = Matrix3Trait::new(
        int(2), int(0), int(0), int(0), int(4), int(0), int(3), int(1), int(1),
    );
    let bd: Matrix3<Fixed> = Matrix3Trait::new(
        int(-6), int(0), int(6), int(4), int(-8), int(-2), int(-11), int(-2), int(16),
    );
    let x2: Matrix3<Fixed> = Matrix3Trait::new(
        int(-6), int(0), int(6), int(4), int(-8), int(-2), int(-4), int(2), int(8),
    );
    let mut b = bd;
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(b == x2);
    let mut b = bd;
    l.solve_lower_triangular_with_diag_unchecked_mut(ref b, int(2));
    assert!(b == x2);
    let mut b = bd;
    assert!(!l.solve_lower_triangular_with_diag_mut(ref b, int(0)));
    assert!(b == bd);
}

/// `Matrix3` against a `Matrix3x4` right-hand side: the `Option` forms recover the integer solution
/// exactly (power-of-two diagonal:
/// every quotient is exact).
#[test]
fn test_solve_matrix3_matrix3x4_exact() {
    let l: Matrix3<Fixed> = Matrix3Trait::new(
        int(2), int(0), int(0), int(0), int(4), int(0), int(3), int(1), int(1),
    );
    let u: Matrix3<Fixed> = Matrix3Trait::new(
        int(2), int(0), int(3), int(0), int(4), int(1), int(0), int(0), int(1),
    );
    let x: Matrix3x4<Fixed> = Matrix3x4Trait::new(
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-2),
        int(1),
        int(4),
        int(-2),
    );
    let bl: Matrix3x4<Fixed> = Matrix3x4Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(8),
        int(-16),
        int(-4),
        int(8),
        int(-9),
        int(-3),
        int(12),
        int(-9),
    );
    let bu: Matrix3x4<Fixed> = Matrix3x4Trait::new(
        int(-12),
        int(3),
        int(18),
        int(-12),
        int(6),
        int(-15),
        int(0),
        int(6),
        int(-2),
        int(1),
        int(4),
        int(-2),
    );
    assert!(l.solve_lower_triangular(bl) == Option::Some(x));
    assert!(u.solve_upper_triangular(bu) == Option::Some(x));
    assert!(l.tr_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.tr_solve_upper_triangular(bl) == Option::Some(x));
    assert!(l.ad_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.ad_solve_upper_triangular(bl) == Option::Some(x));
}

/// `Matrix3` against a `Matrix3x4` right-hand side: the `_unchecked` forms, bit-identical to the
/// checked ones.
#[test]
fn test_solve_matrix3_matrix3x4_unchecked_exact() {
    let l: Matrix3<Fixed> = Matrix3Trait::new(
        int(2), int(0), int(0), int(0), int(4), int(0), int(3), int(1), int(1),
    );
    let u: Matrix3<Fixed> = Matrix3Trait::new(
        int(2), int(0), int(3), int(0), int(4), int(1), int(0), int(0), int(1),
    );
    let x: Matrix3x4<Fixed> = Matrix3x4Trait::new(
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-2),
        int(1),
        int(4),
        int(-2),
    );
    let bl: Matrix3x4<Fixed> = Matrix3x4Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(8),
        int(-16),
        int(-4),
        int(8),
        int(-9),
        int(-3),
        int(12),
        int(-9),
    );
    let bu: Matrix3x4<Fixed> = Matrix3x4Trait::new(
        int(-12),
        int(3),
        int(18),
        int(-12),
        int(6),
        int(-15),
        int(0),
        int(6),
        int(-2),
        int(1),
        int(4),
        int(-2),
    );
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    assert!(l.ad_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.ad_solve_upper_triangular_unchecked(bl) == x);
}

/// `Matrix3` against a `Matrix3x4` right-hand side: the `_mut` forms overwrite `b` with the
/// solution and return `true`.
#[test]
fn test_solve_matrix3_matrix3x4_mut_exact() {
    let l: Matrix3<Fixed> = Matrix3Trait::new(
        int(2), int(0), int(0), int(0), int(4), int(0), int(3), int(1), int(1),
    );
    let u: Matrix3<Fixed> = Matrix3Trait::new(
        int(2), int(0), int(3), int(0), int(4), int(1), int(0), int(0), int(1),
    );
    let x: Matrix3x4<Fixed> = Matrix3x4Trait::new(
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-2),
        int(1),
        int(4),
        int(-2),
    );
    let bl: Matrix3x4<Fixed> = Matrix3x4Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(8),
        int(-16),
        int(-4),
        int(8),
        int(-9),
        int(-3),
        int(12),
        int(-9),
    );
    let bu: Matrix3x4<Fixed> = Matrix3x4Trait::new(
        int(-12),
        int(3),
        int(18),
        int(-12),
        int(6),
        int(-15),
        int(0),
        int(6),
        int(-2),
        int(1),
        int(4),
        int(-2),
    );
    let mut b = bl;
    assert!(l.solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(u.solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.tr_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.tr_solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.ad_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.ad_solve_upper_triangular_mut(ref b));
    assert!(b == x);
}

/// `Matrix3` against a `Matrix3x4` right-hand side: the `_unchecked_mut` forms.
#[test]
fn test_solve_matrix3_matrix3x4_unchecked_mut_exact() {
    let l: Matrix3<Fixed> = Matrix3Trait::new(
        int(2), int(0), int(0), int(0), int(4), int(0), int(3), int(1), int(1),
    );
    let u: Matrix3<Fixed> = Matrix3Trait::new(
        int(2), int(0), int(3), int(0), int(4), int(1), int(0), int(0), int(1),
    );
    let x: Matrix3x4<Fixed> = Matrix3x4Trait::new(
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-2),
        int(1),
        int(4),
        int(-2),
    );
    let bl: Matrix3x4<Fixed> = Matrix3x4Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(8),
        int(-16),
        int(-4),
        int(8),
        int(-9),
        int(-3),
        int(12),
        int(-9),
    );
    let bu: Matrix3x4<Fixed> = Matrix3x4Trait::new(
        int(-12),
        int(3),
        int(18),
        int(-12),
        int(6),
        int(-15),
        int(0),
        int(6),
        int(-2),
        int(1),
        int(4),
        int(-2),
    );
    let mut b = bl;
    l.solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    u.solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.tr_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.tr_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.ad_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.ad_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
}

/// `Matrix3` against a `Matrix3x4` right-hand side: `with_diag` solves `(strict lower triangle of l
/// + 2 I) x = bd` and leaves `2 x` in `b`, like upstream (the quotients by `diag` are never written
/// back); a zero `diag` returns `false` and leaves `b` unchanged.
#[test]
fn test_solve_matrix3_matrix3x4_with_diag_exact() {
    let l: Matrix3<Fixed> = Matrix3Trait::new(
        int(2), int(0), int(0), int(0), int(4), int(0), int(3), int(1), int(1),
    );
    let bd: Matrix3x4<Fixed> = Matrix3x4Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(4),
        int(-8),
        int(-2),
        int(4),
        int(-11),
        int(-2),
        int(16),
        int(-11),
    );
    let x2: Matrix3x4<Fixed> = Matrix3x4Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(4),
        int(-8),
        int(-2),
        int(4),
        int(-4),
        int(2),
        int(8),
        int(-4),
    );
    let mut b = bd;
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(b == x2);
    let mut b = bd;
    l.solve_lower_triangular_with_diag_unchecked_mut(ref b, int(2));
    assert!(b == x2);
    let mut b = bd;
    assert!(!l.solve_lower_triangular_with_diag_mut(ref b, int(0)));
    assert!(b == bd);
}

/// `Matrix3` against a `Matrix3x5` right-hand side: the `Option` forms recover the integer solution
/// exactly (power-of-two diagonal:
/// every quotient is exact).
#[test]
fn test_solve_matrix3_matrix3x5_exact() {
    let l: Matrix3<Fixed> = Matrix3Trait::new(
        int(2), int(0), int(0), int(0), int(4), int(0), int(3), int(1), int(1),
    );
    let u: Matrix3<Fixed> = Matrix3Trait::new(
        int(2), int(0), int(3), int(0), int(4), int(1), int(0), int(0), int(1),
    );
    let x: Matrix3x5<Fixed> = Matrix3x5Trait::new(
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(0),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-4),
        int(-2),
        int(1),
        int(4),
        int(-2),
        int(1),
    );
    let bl: Matrix3x5<Fixed> = Matrix3x5Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(8),
        int(-16),
        int(-4),
        int(8),
        int(-16),
        int(-9),
        int(-3),
        int(12),
        int(-9),
        int(-3),
    );
    let bu: Matrix3x5<Fixed> = Matrix3x5Trait::new(
        int(-12),
        int(3),
        int(18),
        int(-12),
        int(3),
        int(6),
        int(-15),
        int(0),
        int(6),
        int(-15),
        int(-2),
        int(1),
        int(4),
        int(-2),
        int(1),
    );
    assert!(l.solve_lower_triangular(bl) == Option::Some(x));
    assert!(u.solve_upper_triangular(bu) == Option::Some(x));
    assert!(l.tr_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.tr_solve_upper_triangular(bl) == Option::Some(x));
    assert!(l.ad_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.ad_solve_upper_triangular(bl) == Option::Some(x));
}

/// `Matrix3` against a `Matrix3x5` right-hand side: the `_unchecked` forms, bit-identical to the
/// checked ones.
#[test]
fn test_solve_matrix3_matrix3x5_unchecked_exact() {
    let l: Matrix3<Fixed> = Matrix3Trait::new(
        int(2), int(0), int(0), int(0), int(4), int(0), int(3), int(1), int(1),
    );
    let u: Matrix3<Fixed> = Matrix3Trait::new(
        int(2), int(0), int(3), int(0), int(4), int(1), int(0), int(0), int(1),
    );
    let x: Matrix3x5<Fixed> = Matrix3x5Trait::new(
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(0),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-4),
        int(-2),
        int(1),
        int(4),
        int(-2),
        int(1),
    );
    let bl: Matrix3x5<Fixed> = Matrix3x5Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(8),
        int(-16),
        int(-4),
        int(8),
        int(-16),
        int(-9),
        int(-3),
        int(12),
        int(-9),
        int(-3),
    );
    let bu: Matrix3x5<Fixed> = Matrix3x5Trait::new(
        int(-12),
        int(3),
        int(18),
        int(-12),
        int(3),
        int(6),
        int(-15),
        int(0),
        int(6),
        int(-15),
        int(-2),
        int(1),
        int(4),
        int(-2),
        int(1),
    );
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    assert!(l.ad_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.ad_solve_upper_triangular_unchecked(bl) == x);
}

/// `Matrix3` against a `Matrix3x5` right-hand side: the `_mut` forms overwrite `b` with the
/// solution and return `true`.
#[test]
fn test_solve_matrix3_matrix3x5_mut_exact() {
    let l: Matrix3<Fixed> = Matrix3Trait::new(
        int(2), int(0), int(0), int(0), int(4), int(0), int(3), int(1), int(1),
    );
    let u: Matrix3<Fixed> = Matrix3Trait::new(
        int(2), int(0), int(3), int(0), int(4), int(1), int(0), int(0), int(1),
    );
    let x: Matrix3x5<Fixed> = Matrix3x5Trait::new(
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(0),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-4),
        int(-2),
        int(1),
        int(4),
        int(-2),
        int(1),
    );
    let bl: Matrix3x5<Fixed> = Matrix3x5Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(8),
        int(-16),
        int(-4),
        int(8),
        int(-16),
        int(-9),
        int(-3),
        int(12),
        int(-9),
        int(-3),
    );
    let bu: Matrix3x5<Fixed> = Matrix3x5Trait::new(
        int(-12),
        int(3),
        int(18),
        int(-12),
        int(3),
        int(6),
        int(-15),
        int(0),
        int(6),
        int(-15),
        int(-2),
        int(1),
        int(4),
        int(-2),
        int(1),
    );
    let mut b = bl;
    assert!(l.solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(u.solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.tr_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.tr_solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.ad_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.ad_solve_upper_triangular_mut(ref b));
    assert!(b == x);
}

/// `Matrix3` against a `Matrix3x5` right-hand side: the `_unchecked_mut` forms.
#[test]
fn test_solve_matrix3_matrix3x5_unchecked_mut_exact() {
    let l: Matrix3<Fixed> = Matrix3Trait::new(
        int(2), int(0), int(0), int(0), int(4), int(0), int(3), int(1), int(1),
    );
    let u: Matrix3<Fixed> = Matrix3Trait::new(
        int(2), int(0), int(3), int(0), int(4), int(1), int(0), int(0), int(1),
    );
    let x: Matrix3x5<Fixed> = Matrix3x5Trait::new(
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(0),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-4),
        int(-2),
        int(1),
        int(4),
        int(-2),
        int(1),
    );
    let bl: Matrix3x5<Fixed> = Matrix3x5Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(8),
        int(-16),
        int(-4),
        int(8),
        int(-16),
        int(-9),
        int(-3),
        int(12),
        int(-9),
        int(-3),
    );
    let bu: Matrix3x5<Fixed> = Matrix3x5Trait::new(
        int(-12),
        int(3),
        int(18),
        int(-12),
        int(3),
        int(6),
        int(-15),
        int(0),
        int(6),
        int(-15),
        int(-2),
        int(1),
        int(4),
        int(-2),
        int(1),
    );
    let mut b = bl;
    l.solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    u.solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.tr_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.tr_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.ad_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.ad_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
}

/// `Matrix3` against a `Matrix3x5` right-hand side: `with_diag` solves `(strict lower triangle of l
/// + 2 I) x = bd` and leaves `2 x` in `b`, like upstream (the quotients by `diag` are never written
/// back); a zero `diag` returns `false` and leaves `b` unchanged.
#[test]
fn test_solve_matrix3_matrix3x5_with_diag_exact() {
    let l: Matrix3<Fixed> = Matrix3Trait::new(
        int(2), int(0), int(0), int(0), int(4), int(0), int(3), int(1), int(1),
    );
    let bd: Matrix3x5<Fixed> = Matrix3x5Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(4),
        int(-8),
        int(-2),
        int(4),
        int(-8),
        int(-11),
        int(-2),
        int(16),
        int(-11),
        int(-2),
    );
    let x2: Matrix3x5<Fixed> = Matrix3x5Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(4),
        int(-8),
        int(-2),
        int(4),
        int(-8),
        int(-4),
        int(2),
        int(8),
        int(-4),
        int(2),
    );
    let mut b = bd;
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(b == x2);
    let mut b = bd;
    l.solve_lower_triangular_with_diag_unchecked_mut(ref b, int(2));
    assert!(b == x2);
    let mut b = bd;
    assert!(!l.solve_lower_triangular_with_diag_mut(ref b, int(0)));
    assert!(b == bd);
}

/// `Matrix3` against a `Matrix3x6` right-hand side: the `Option` forms recover the integer solution
/// exactly (power-of-two diagonal:
/// every quotient is exact).
#[test]
fn test_solve_matrix3_matrix3x6_exact() {
    let l: Matrix3<Fixed> = Matrix3Trait::new(
        int(2), int(0), int(0), int(0), int(4), int(0), int(3), int(1), int(1),
    );
    let u: Matrix3<Fixed> = Matrix3Trait::new(
        int(2), int(0), int(3), int(0), int(4), int(1), int(0), int(0), int(1),
    );
    let x: Matrix3x6<Fixed> = Matrix3x6Trait::new(
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(0),
        int(3),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-4),
        int(-1),
        int(-2),
        int(1),
        int(4),
        int(-2),
        int(1),
        int(4),
    );
    let bl: Matrix3x6<Fixed> = Matrix3x6Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(6),
        int(8),
        int(-16),
        int(-4),
        int(8),
        int(-16),
        int(-4),
        int(-9),
        int(-3),
        int(12),
        int(-9),
        int(-3),
        int(12),
    );
    let bu: Matrix3x6<Fixed> = Matrix3x6Trait::new(
        int(-12),
        int(3),
        int(18),
        int(-12),
        int(3),
        int(18),
        int(6),
        int(-15),
        int(0),
        int(6),
        int(-15),
        int(0),
        int(-2),
        int(1),
        int(4),
        int(-2),
        int(1),
        int(4),
    );
    assert!(l.solve_lower_triangular(bl) == Option::Some(x));
    assert!(u.solve_upper_triangular(bu) == Option::Some(x));
    assert!(l.tr_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.tr_solve_upper_triangular(bl) == Option::Some(x));
    assert!(l.ad_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.ad_solve_upper_triangular(bl) == Option::Some(x));
}

/// `Matrix3` against a `Matrix3x6` right-hand side: the `_unchecked` forms, bit-identical to the
/// checked ones.
#[test]
fn test_solve_matrix3_matrix3x6_unchecked_exact() {
    let l: Matrix3<Fixed> = Matrix3Trait::new(
        int(2), int(0), int(0), int(0), int(4), int(0), int(3), int(1), int(1),
    );
    let u: Matrix3<Fixed> = Matrix3Trait::new(
        int(2), int(0), int(3), int(0), int(4), int(1), int(0), int(0), int(1),
    );
    let x: Matrix3x6<Fixed> = Matrix3x6Trait::new(
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(0),
        int(3),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-4),
        int(-1),
        int(-2),
        int(1),
        int(4),
        int(-2),
        int(1),
        int(4),
    );
    let bl: Matrix3x6<Fixed> = Matrix3x6Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(6),
        int(8),
        int(-16),
        int(-4),
        int(8),
        int(-16),
        int(-4),
        int(-9),
        int(-3),
        int(12),
        int(-9),
        int(-3),
        int(12),
    );
    let bu: Matrix3x6<Fixed> = Matrix3x6Trait::new(
        int(-12),
        int(3),
        int(18),
        int(-12),
        int(3),
        int(18),
        int(6),
        int(-15),
        int(0),
        int(6),
        int(-15),
        int(0),
        int(-2),
        int(1),
        int(4),
        int(-2),
        int(1),
        int(4),
    );
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    assert!(l.ad_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.ad_solve_upper_triangular_unchecked(bl) == x);
}

/// `Matrix3` against a `Matrix3x6` right-hand side: the `_mut` forms overwrite `b` with the
/// solution and return `true`.
#[test]
fn test_solve_matrix3_matrix3x6_mut_exact() {
    let l: Matrix3<Fixed> = Matrix3Trait::new(
        int(2), int(0), int(0), int(0), int(4), int(0), int(3), int(1), int(1),
    );
    let u: Matrix3<Fixed> = Matrix3Trait::new(
        int(2), int(0), int(3), int(0), int(4), int(1), int(0), int(0), int(1),
    );
    let x: Matrix3x6<Fixed> = Matrix3x6Trait::new(
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(0),
        int(3),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-4),
        int(-1),
        int(-2),
        int(1),
        int(4),
        int(-2),
        int(1),
        int(4),
    );
    let bl: Matrix3x6<Fixed> = Matrix3x6Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(6),
        int(8),
        int(-16),
        int(-4),
        int(8),
        int(-16),
        int(-4),
        int(-9),
        int(-3),
        int(12),
        int(-9),
        int(-3),
        int(12),
    );
    let bu: Matrix3x6<Fixed> = Matrix3x6Trait::new(
        int(-12),
        int(3),
        int(18),
        int(-12),
        int(3),
        int(18),
        int(6),
        int(-15),
        int(0),
        int(6),
        int(-15),
        int(0),
        int(-2),
        int(1),
        int(4),
        int(-2),
        int(1),
        int(4),
    );
    let mut b = bl;
    assert!(l.solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(u.solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.tr_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.tr_solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.ad_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.ad_solve_upper_triangular_mut(ref b));
    assert!(b == x);
}

/// `Matrix3` against a `Matrix3x6` right-hand side: the `_unchecked_mut` forms.
#[test]
fn test_solve_matrix3_matrix3x6_unchecked_mut_exact() {
    let l: Matrix3<Fixed> = Matrix3Trait::new(
        int(2), int(0), int(0), int(0), int(4), int(0), int(3), int(1), int(1),
    );
    let u: Matrix3<Fixed> = Matrix3Trait::new(
        int(2), int(0), int(3), int(0), int(4), int(1), int(0), int(0), int(1),
    );
    let x: Matrix3x6<Fixed> = Matrix3x6Trait::new(
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(0),
        int(3),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-4),
        int(-1),
        int(-2),
        int(1),
        int(4),
        int(-2),
        int(1),
        int(4),
    );
    let bl: Matrix3x6<Fixed> = Matrix3x6Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(6),
        int(8),
        int(-16),
        int(-4),
        int(8),
        int(-16),
        int(-4),
        int(-9),
        int(-3),
        int(12),
        int(-9),
        int(-3),
        int(12),
    );
    let bu: Matrix3x6<Fixed> = Matrix3x6Trait::new(
        int(-12),
        int(3),
        int(18),
        int(-12),
        int(3),
        int(18),
        int(6),
        int(-15),
        int(0),
        int(6),
        int(-15),
        int(0),
        int(-2),
        int(1),
        int(4),
        int(-2),
        int(1),
        int(4),
    );
    let mut b = bl;
    l.solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    u.solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.tr_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.tr_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.ad_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.ad_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
}

/// `Matrix3` against a `Matrix3x6` right-hand side: `with_diag` solves `(strict lower triangle of l
/// + 2 I) x = bd` and leaves `2 x` in `b`, like upstream (the quotients by `diag` are never written
/// back); a zero `diag` returns `false` and leaves `b` unchanged.
#[test]
fn test_solve_matrix3_matrix3x6_with_diag_exact() {
    let l: Matrix3<Fixed> = Matrix3Trait::new(
        int(2), int(0), int(0), int(0), int(4), int(0), int(3), int(1), int(1),
    );
    let bd: Matrix3x6<Fixed> = Matrix3x6Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(6),
        int(4),
        int(-8),
        int(-2),
        int(4),
        int(-8),
        int(-2),
        int(-11),
        int(-2),
        int(16),
        int(-11),
        int(-2),
        int(16),
    );
    let x2: Matrix3x6<Fixed> = Matrix3x6Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(6),
        int(4),
        int(-8),
        int(-2),
        int(4),
        int(-8),
        int(-2),
        int(-4),
        int(2),
        int(8),
        int(-4),
        int(2),
        int(8),
    );
    let mut b = bd;
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(b == x2);
    let mut b = bd;
    l.solve_lower_triangular_with_diag_unchecked_mut(ref b, int(2));
    assert!(b == x2);
    let mut b = bd;
    assert!(!l.solve_lower_triangular_with_diag_mut(ref b, int(0)));
    assert!(b == bd);
}

/// `Matrix4` against a `Vector4` right-hand side: the `Option` forms recover the integer solution
/// exactly (power-of-two diagonal:
/// every quotient is exact).
#[test]
fn test_solve_matrix4_vector4_exact() {
    let l: Matrix4<Fixed> = Matrix4Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
    );
    let u: Matrix4<Fixed> = Matrix4Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(0),
        int(1),
        int(2),
        int(0),
        int(0),
        int(0),
        int(8),
    );
    let x: Vector4<Fixed> = Vector4Trait::new(int(-3), int(2), int(-2), int(3));
    let bl: Vector4<Fixed> = Vector4Trait::new(int(-6), int(8), int(-9), int(17));
    let bu: Vector4<Fixed> = Vector4Trait::new(int(-15), int(-3), int(4), int(24));
    assert!(l.solve_lower_triangular(bl) == Option::Some(x));
    assert!(u.solve_upper_triangular(bu) == Option::Some(x));
    assert!(l.tr_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.tr_solve_upper_triangular(bl) == Option::Some(x));
    assert!(l.ad_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.ad_solve_upper_triangular(bl) == Option::Some(x));
}

/// `Matrix4` against a `Vector4` right-hand side: the `_unchecked` forms, bit-identical to the
/// checked ones.
#[test]
fn test_solve_matrix4_vector4_unchecked_exact() {
    let l: Matrix4<Fixed> = Matrix4Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
    );
    let u: Matrix4<Fixed> = Matrix4Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(0),
        int(1),
        int(2),
        int(0),
        int(0),
        int(0),
        int(8),
    );
    let x: Vector4<Fixed> = Vector4Trait::new(int(-3), int(2), int(-2), int(3));
    let bl: Vector4<Fixed> = Vector4Trait::new(int(-6), int(8), int(-9), int(17));
    let bu: Vector4<Fixed> = Vector4Trait::new(int(-15), int(-3), int(4), int(24));
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    assert!(l.ad_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.ad_solve_upper_triangular_unchecked(bl) == x);
}

/// `Matrix4` against a `Vector4` right-hand side: the `_mut` forms overwrite `b` with the solution
/// and return `true`.
#[test]
fn test_solve_matrix4_vector4_mut_exact() {
    let l: Matrix4<Fixed> = Matrix4Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
    );
    let u: Matrix4<Fixed> = Matrix4Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(0),
        int(1),
        int(2),
        int(0),
        int(0),
        int(0),
        int(8),
    );
    let x: Vector4<Fixed> = Vector4Trait::new(int(-3), int(2), int(-2), int(3));
    let bl: Vector4<Fixed> = Vector4Trait::new(int(-6), int(8), int(-9), int(17));
    let bu: Vector4<Fixed> = Vector4Trait::new(int(-15), int(-3), int(4), int(24));
    let mut b = bl;
    assert!(l.solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(u.solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.tr_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.tr_solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.ad_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.ad_solve_upper_triangular_mut(ref b));
    assert!(b == x);
}

/// `Matrix4` against a `Vector4` right-hand side: the `_unchecked_mut` forms.
#[test]
fn test_solve_matrix4_vector4_unchecked_mut_exact() {
    let l: Matrix4<Fixed> = Matrix4Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
    );
    let u: Matrix4<Fixed> = Matrix4Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(0),
        int(1),
        int(2),
        int(0),
        int(0),
        int(0),
        int(8),
    );
    let x: Vector4<Fixed> = Vector4Trait::new(int(-3), int(2), int(-2), int(3));
    let bl: Vector4<Fixed> = Vector4Trait::new(int(-6), int(8), int(-9), int(17));
    let bu: Vector4<Fixed> = Vector4Trait::new(int(-15), int(-3), int(4), int(24));
    let mut b = bl;
    l.solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    u.solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.tr_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.tr_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.ad_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.ad_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
}

/// `Matrix4` against a `Vector4` right-hand side: `with_diag` solves `(strict lower triangle of l +
/// 2 I) x = bd` and leaves `2 x` in `b`, like upstream (the quotients by `diag` are never written
/// back); a zero `diag` returns `false` and leaves `b` unchanged.
#[test]
fn test_solve_matrix4_vector4_with_diag_exact() {
    let l: Matrix4<Fixed> = Matrix4Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
    );
    let bd: Vector4<Fixed> = Vector4Trait::new(int(-6), int(4), int(-11), int(-1));
    let x2: Vector4<Fixed> = Vector4Trait::new(int(-6), int(4), int(-4), int(6));
    let mut b = bd;
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(b == x2);
    let mut b = bd;
    l.solve_lower_triangular_with_diag_unchecked_mut(ref b, int(2));
    assert!(b == x2);
    let mut b = bd;
    assert!(!l.solve_lower_triangular_with_diag_mut(ref b, int(0)));
    assert!(b == bd);
}

/// `Matrix4` against a `Matrix4x2` right-hand side: the `Option` forms recover the integer solution
/// exactly (power-of-two diagonal:
/// every quotient is exact).
#[test]
fn test_solve_matrix4_matrix4x2_exact() {
    let l: Matrix4<Fixed> = Matrix4Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
    );
    let u: Matrix4<Fixed> = Matrix4Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(0),
        int(1),
        int(2),
        int(0),
        int(0),
        int(0),
        int(8),
    );
    let x: Matrix4x2<Fixed> = Matrix4x2Trait::new(
        int(-3), int(0), int(2), int(-4), int(-2), int(1), int(3), int(-3),
    );
    let bl: Matrix4x2<Fixed> = Matrix4x2Trait::new(
        int(-6), int(0), int(8), int(-16), int(-9), int(-3), int(17), int(-10),
    );
    let bu: Matrix4x2<Fixed> = Matrix4x2Trait::new(
        int(-15), int(6), int(-3), int(-6), int(4), int(-5), int(24), int(-24),
    );
    assert!(l.solve_lower_triangular(bl) == Option::Some(x));
    assert!(u.solve_upper_triangular(bu) == Option::Some(x));
    assert!(l.tr_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.tr_solve_upper_triangular(bl) == Option::Some(x));
    assert!(l.ad_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.ad_solve_upper_triangular(bl) == Option::Some(x));
}

/// `Matrix4` against a `Matrix4x2` right-hand side: the `_unchecked` forms, bit-identical to the
/// checked ones.
#[test]
fn test_solve_matrix4_matrix4x2_unchecked_exact() {
    let l: Matrix4<Fixed> = Matrix4Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
    );
    let u: Matrix4<Fixed> = Matrix4Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(0),
        int(1),
        int(2),
        int(0),
        int(0),
        int(0),
        int(8),
    );
    let x: Matrix4x2<Fixed> = Matrix4x2Trait::new(
        int(-3), int(0), int(2), int(-4), int(-2), int(1), int(3), int(-3),
    );
    let bl: Matrix4x2<Fixed> = Matrix4x2Trait::new(
        int(-6), int(0), int(8), int(-16), int(-9), int(-3), int(17), int(-10),
    );
    let bu: Matrix4x2<Fixed> = Matrix4x2Trait::new(
        int(-15), int(6), int(-3), int(-6), int(4), int(-5), int(24), int(-24),
    );
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    assert!(l.ad_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.ad_solve_upper_triangular_unchecked(bl) == x);
}

/// `Matrix4` against a `Matrix4x2` right-hand side: the `_mut` forms overwrite `b` with the
/// solution and return `true`.
#[test]
fn test_solve_matrix4_matrix4x2_mut_exact() {
    let l: Matrix4<Fixed> = Matrix4Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
    );
    let u: Matrix4<Fixed> = Matrix4Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(0),
        int(1),
        int(2),
        int(0),
        int(0),
        int(0),
        int(8),
    );
    let x: Matrix4x2<Fixed> = Matrix4x2Trait::new(
        int(-3), int(0), int(2), int(-4), int(-2), int(1), int(3), int(-3),
    );
    let bl: Matrix4x2<Fixed> = Matrix4x2Trait::new(
        int(-6), int(0), int(8), int(-16), int(-9), int(-3), int(17), int(-10),
    );
    let bu: Matrix4x2<Fixed> = Matrix4x2Trait::new(
        int(-15), int(6), int(-3), int(-6), int(4), int(-5), int(24), int(-24),
    );
    let mut b = bl;
    assert!(l.solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(u.solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.tr_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.tr_solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.ad_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.ad_solve_upper_triangular_mut(ref b));
    assert!(b == x);
}

/// `Matrix4` against a `Matrix4x2` right-hand side: the `_unchecked_mut` forms.
#[test]
fn test_solve_matrix4_matrix4x2_unchecked_mut_exact() {
    let l: Matrix4<Fixed> = Matrix4Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
    );
    let u: Matrix4<Fixed> = Matrix4Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(0),
        int(1),
        int(2),
        int(0),
        int(0),
        int(0),
        int(8),
    );
    let x: Matrix4x2<Fixed> = Matrix4x2Trait::new(
        int(-3), int(0), int(2), int(-4), int(-2), int(1), int(3), int(-3),
    );
    let bl: Matrix4x2<Fixed> = Matrix4x2Trait::new(
        int(-6), int(0), int(8), int(-16), int(-9), int(-3), int(17), int(-10),
    );
    let bu: Matrix4x2<Fixed> = Matrix4x2Trait::new(
        int(-15), int(6), int(-3), int(-6), int(4), int(-5), int(24), int(-24),
    );
    let mut b = bl;
    l.solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    u.solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.tr_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.tr_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.ad_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.ad_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
}

/// `Matrix4` against a `Matrix4x2` right-hand side: `with_diag` solves `(strict lower triangle of l
/// + 2 I) x = bd` and leaves `2 x` in `b`, like upstream (the quotients by `diag` are never written
/// back); a zero `diag` returns `false` and leaves `b` unchanged.
#[test]
fn test_solve_matrix4_matrix4x2_with_diag_exact() {
    let l: Matrix4<Fixed> = Matrix4Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
    );
    let bd: Matrix4x2<Fixed> = Matrix4x2Trait::new(
        int(-6), int(0), int(4), int(-8), int(-11), int(-2), int(-1), int(8),
    );
    let x2: Matrix4x2<Fixed> = Matrix4x2Trait::new(
        int(-6), int(0), int(4), int(-8), int(-4), int(2), int(6), int(-6),
    );
    let mut b = bd;
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(b == x2);
    let mut b = bd;
    l.solve_lower_triangular_with_diag_unchecked_mut(ref b, int(2));
    assert!(b == x2);
    let mut b = bd;
    assert!(!l.solve_lower_triangular_with_diag_mut(ref b, int(0)));
    assert!(b == bd);
}

/// `Matrix4` against a `Matrix4x3` right-hand side: the `Option` forms recover the integer solution
/// exactly (power-of-two diagonal:
/// every quotient is exact).
#[test]
fn test_solve_matrix4_matrix4x3_exact() {
    let l: Matrix4<Fixed> = Matrix4Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
    );
    let u: Matrix4<Fixed> = Matrix4Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(0),
        int(1),
        int(2),
        int(0),
        int(0),
        int(0),
        int(8),
    );
    let x: Matrix4x3<Fixed> = Matrix4x3Trait::new(
        int(-3),
        int(0),
        int(3),
        int(2),
        int(-4),
        int(-1),
        int(-2),
        int(1),
        int(4),
        int(3),
        int(-3),
        int(0),
    );
    let bl: Matrix4x3<Fixed> = Matrix4x3Trait::new(
        int(-6),
        int(0),
        int(6),
        int(8),
        int(-16),
        int(-4),
        int(-9),
        int(-3),
        int(12),
        int(17),
        int(-10),
        int(8),
    );
    let bu: Matrix4x3<Fixed> = Matrix4x3Trait::new(
        int(-15),
        int(6),
        int(18),
        int(-3),
        int(-6),
        int(0),
        int(4),
        int(-5),
        int(4),
        int(24),
        int(-24),
        int(0),
    );
    assert!(l.solve_lower_triangular(bl) == Option::Some(x));
    assert!(u.solve_upper_triangular(bu) == Option::Some(x));
    assert!(l.tr_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.tr_solve_upper_triangular(bl) == Option::Some(x));
    assert!(l.ad_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.ad_solve_upper_triangular(bl) == Option::Some(x));
}

/// `Matrix4` against a `Matrix4x3` right-hand side: the `_unchecked` forms, bit-identical to the
/// checked ones.
#[test]
fn test_solve_matrix4_matrix4x3_unchecked_exact() {
    let l: Matrix4<Fixed> = Matrix4Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
    );
    let u: Matrix4<Fixed> = Matrix4Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(0),
        int(1),
        int(2),
        int(0),
        int(0),
        int(0),
        int(8),
    );
    let x: Matrix4x3<Fixed> = Matrix4x3Trait::new(
        int(-3),
        int(0),
        int(3),
        int(2),
        int(-4),
        int(-1),
        int(-2),
        int(1),
        int(4),
        int(3),
        int(-3),
        int(0),
    );
    let bl: Matrix4x3<Fixed> = Matrix4x3Trait::new(
        int(-6),
        int(0),
        int(6),
        int(8),
        int(-16),
        int(-4),
        int(-9),
        int(-3),
        int(12),
        int(17),
        int(-10),
        int(8),
    );
    let bu: Matrix4x3<Fixed> = Matrix4x3Trait::new(
        int(-15),
        int(6),
        int(18),
        int(-3),
        int(-6),
        int(0),
        int(4),
        int(-5),
        int(4),
        int(24),
        int(-24),
        int(0),
    );
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    assert!(l.ad_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.ad_solve_upper_triangular_unchecked(bl) == x);
}

/// `Matrix4` against a `Matrix4x3` right-hand side: the `_mut` forms overwrite `b` with the
/// solution and return `true`.
#[test]
fn test_solve_matrix4_matrix4x3_mut_exact() {
    let l: Matrix4<Fixed> = Matrix4Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
    );
    let u: Matrix4<Fixed> = Matrix4Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(0),
        int(1),
        int(2),
        int(0),
        int(0),
        int(0),
        int(8),
    );
    let x: Matrix4x3<Fixed> = Matrix4x3Trait::new(
        int(-3),
        int(0),
        int(3),
        int(2),
        int(-4),
        int(-1),
        int(-2),
        int(1),
        int(4),
        int(3),
        int(-3),
        int(0),
    );
    let bl: Matrix4x3<Fixed> = Matrix4x3Trait::new(
        int(-6),
        int(0),
        int(6),
        int(8),
        int(-16),
        int(-4),
        int(-9),
        int(-3),
        int(12),
        int(17),
        int(-10),
        int(8),
    );
    let bu: Matrix4x3<Fixed> = Matrix4x3Trait::new(
        int(-15),
        int(6),
        int(18),
        int(-3),
        int(-6),
        int(0),
        int(4),
        int(-5),
        int(4),
        int(24),
        int(-24),
        int(0),
    );
    let mut b = bl;
    assert!(l.solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(u.solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.tr_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.tr_solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.ad_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.ad_solve_upper_triangular_mut(ref b));
    assert!(b == x);
}

/// `Matrix4` against a `Matrix4x3` right-hand side: the `_unchecked_mut` forms.
#[test]
fn test_solve_matrix4_matrix4x3_unchecked_mut_exact() {
    let l: Matrix4<Fixed> = Matrix4Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
    );
    let u: Matrix4<Fixed> = Matrix4Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(0),
        int(1),
        int(2),
        int(0),
        int(0),
        int(0),
        int(8),
    );
    let x: Matrix4x3<Fixed> = Matrix4x3Trait::new(
        int(-3),
        int(0),
        int(3),
        int(2),
        int(-4),
        int(-1),
        int(-2),
        int(1),
        int(4),
        int(3),
        int(-3),
        int(0),
    );
    let bl: Matrix4x3<Fixed> = Matrix4x3Trait::new(
        int(-6),
        int(0),
        int(6),
        int(8),
        int(-16),
        int(-4),
        int(-9),
        int(-3),
        int(12),
        int(17),
        int(-10),
        int(8),
    );
    let bu: Matrix4x3<Fixed> = Matrix4x3Trait::new(
        int(-15),
        int(6),
        int(18),
        int(-3),
        int(-6),
        int(0),
        int(4),
        int(-5),
        int(4),
        int(24),
        int(-24),
        int(0),
    );
    let mut b = bl;
    l.solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    u.solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.tr_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.tr_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.ad_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.ad_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
}

/// `Matrix4` against a `Matrix4x3` right-hand side: `with_diag` solves `(strict lower triangle of l
/// + 2 I) x = bd` and leaves `2 x` in `b`, like upstream (the quotients by `diag` are never written
/// back); a zero `diag` returns `false` and leaves `b` unchanged.
#[test]
fn test_solve_matrix4_matrix4x3_with_diag_exact() {
    let l: Matrix4<Fixed> = Matrix4Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
    );
    let bd: Matrix4x3<Fixed> = Matrix4x3Trait::new(
        int(-6),
        int(0),
        int(6),
        int(4),
        int(-8),
        int(-2),
        int(-11),
        int(-2),
        int(16),
        int(-1),
        int(8),
        int(8),
    );
    let x2: Matrix4x3<Fixed> = Matrix4x3Trait::new(
        int(-6),
        int(0),
        int(6),
        int(4),
        int(-8),
        int(-2),
        int(-4),
        int(2),
        int(8),
        int(6),
        int(-6),
        int(0),
    );
    let mut b = bd;
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(b == x2);
    let mut b = bd;
    l.solve_lower_triangular_with_diag_unchecked_mut(ref b, int(2));
    assert!(b == x2);
    let mut b = bd;
    assert!(!l.solve_lower_triangular_with_diag_mut(ref b, int(0)));
    assert!(b == bd);
}

/// `Matrix4` against a `Matrix4` right-hand side: the `Option` forms recover the integer solution
/// exactly (power-of-two diagonal:
/// every quotient is exact).
#[test]
fn test_solve_matrix4_matrix4_exact() {
    let l: Matrix4<Fixed> = Matrix4Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
    );
    let u: Matrix4<Fixed> = Matrix4Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(0),
        int(1),
        int(2),
        int(0),
        int(0),
        int(0),
        int(8),
    );
    let x: Matrix4<Fixed> = Matrix4Trait::new(
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-2),
        int(1),
        int(4),
        int(-2),
        int(3),
        int(-3),
        int(0),
        int(3),
    );
    let bl: Matrix4<Fixed> = Matrix4Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(8),
        int(-16),
        int(-4),
        int(8),
        int(-9),
        int(-3),
        int(12),
        int(-9),
        int(17),
        int(-10),
        int(8),
        int(17),
    );
    let bu: Matrix4<Fixed> = Matrix4Trait::new(
        int(-15),
        int(6),
        int(18),
        int(-15),
        int(-3),
        int(-6),
        int(0),
        int(-3),
        int(4),
        int(-5),
        int(4),
        int(4),
        int(24),
        int(-24),
        int(0),
        int(24),
    );
    assert!(l.solve_lower_triangular(bl) == Option::Some(x));
    assert!(u.solve_upper_triangular(bu) == Option::Some(x));
    assert!(l.tr_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.tr_solve_upper_triangular(bl) == Option::Some(x));
    assert!(l.ad_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.ad_solve_upper_triangular(bl) == Option::Some(x));
}

/// `Matrix4` against a `Matrix4` right-hand side: the `_unchecked` forms, bit-identical to the
/// checked ones.
#[test]
fn test_solve_matrix4_matrix4_unchecked_exact() {
    let l: Matrix4<Fixed> = Matrix4Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
    );
    let u: Matrix4<Fixed> = Matrix4Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(0),
        int(1),
        int(2),
        int(0),
        int(0),
        int(0),
        int(8),
    );
    let x: Matrix4<Fixed> = Matrix4Trait::new(
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-2),
        int(1),
        int(4),
        int(-2),
        int(3),
        int(-3),
        int(0),
        int(3),
    );
    let bl: Matrix4<Fixed> = Matrix4Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(8),
        int(-16),
        int(-4),
        int(8),
        int(-9),
        int(-3),
        int(12),
        int(-9),
        int(17),
        int(-10),
        int(8),
        int(17),
    );
    let bu: Matrix4<Fixed> = Matrix4Trait::new(
        int(-15),
        int(6),
        int(18),
        int(-15),
        int(-3),
        int(-6),
        int(0),
        int(-3),
        int(4),
        int(-5),
        int(4),
        int(4),
        int(24),
        int(-24),
        int(0),
        int(24),
    );
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    assert!(l.ad_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.ad_solve_upper_triangular_unchecked(bl) == x);
}

/// `Matrix4` against a `Matrix4` right-hand side: the `_mut` forms overwrite `b` with the solution
/// and return `true`.
#[test]
fn test_solve_matrix4_matrix4_mut_exact() {
    let l: Matrix4<Fixed> = Matrix4Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
    );
    let u: Matrix4<Fixed> = Matrix4Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(0),
        int(1),
        int(2),
        int(0),
        int(0),
        int(0),
        int(8),
    );
    let x: Matrix4<Fixed> = Matrix4Trait::new(
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-2),
        int(1),
        int(4),
        int(-2),
        int(3),
        int(-3),
        int(0),
        int(3),
    );
    let bl: Matrix4<Fixed> = Matrix4Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(8),
        int(-16),
        int(-4),
        int(8),
        int(-9),
        int(-3),
        int(12),
        int(-9),
        int(17),
        int(-10),
        int(8),
        int(17),
    );
    let bu: Matrix4<Fixed> = Matrix4Trait::new(
        int(-15),
        int(6),
        int(18),
        int(-15),
        int(-3),
        int(-6),
        int(0),
        int(-3),
        int(4),
        int(-5),
        int(4),
        int(4),
        int(24),
        int(-24),
        int(0),
        int(24),
    );
    let mut b = bl;
    assert!(l.solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(u.solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.tr_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.tr_solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.ad_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.ad_solve_upper_triangular_mut(ref b));
    assert!(b == x);
}

/// `Matrix4` against a `Matrix4` right-hand side: the `_unchecked_mut` forms.
#[test]
fn test_solve_matrix4_matrix4_unchecked_mut_exact() {
    let l: Matrix4<Fixed> = Matrix4Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
    );
    let u: Matrix4<Fixed> = Matrix4Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(0),
        int(1),
        int(2),
        int(0),
        int(0),
        int(0),
        int(8),
    );
    let x: Matrix4<Fixed> = Matrix4Trait::new(
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-2),
        int(1),
        int(4),
        int(-2),
        int(3),
        int(-3),
        int(0),
        int(3),
    );
    let bl: Matrix4<Fixed> = Matrix4Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(8),
        int(-16),
        int(-4),
        int(8),
        int(-9),
        int(-3),
        int(12),
        int(-9),
        int(17),
        int(-10),
        int(8),
        int(17),
    );
    let bu: Matrix4<Fixed> = Matrix4Trait::new(
        int(-15),
        int(6),
        int(18),
        int(-15),
        int(-3),
        int(-6),
        int(0),
        int(-3),
        int(4),
        int(-5),
        int(4),
        int(4),
        int(24),
        int(-24),
        int(0),
        int(24),
    );
    let mut b = bl;
    l.solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    u.solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.tr_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.tr_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.ad_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.ad_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
}

/// `Matrix4` against a `Matrix4` right-hand side: `with_diag` solves `(strict lower triangle of l +
/// 2 I) x = bd` and leaves `2 x` in `b`, like upstream (the quotients by `diag` are never written
/// back); a zero `diag` returns `false` and leaves `b` unchanged.
#[test]
fn test_solve_matrix4_matrix4_with_diag_exact() {
    let l: Matrix4<Fixed> = Matrix4Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
    );
    let bd: Matrix4<Fixed> = Matrix4Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(4),
        int(-8),
        int(-2),
        int(4),
        int(-11),
        int(-2),
        int(16),
        int(-11),
        int(-1),
        int(8),
        int(8),
        int(-1),
    );
    let x2: Matrix4<Fixed> = Matrix4Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(4),
        int(-8),
        int(-2),
        int(4),
        int(-4),
        int(2),
        int(8),
        int(-4),
        int(6),
        int(-6),
        int(0),
        int(6),
    );
    let mut b = bd;
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(b == x2);
    let mut b = bd;
    l.solve_lower_triangular_with_diag_unchecked_mut(ref b, int(2));
    assert!(b == x2);
    let mut b = bd;
    assert!(!l.solve_lower_triangular_with_diag_mut(ref b, int(0)));
    assert!(b == bd);
}

/// `Matrix4` against a `Matrix4x5` right-hand side: the `Option` forms recover the integer solution
/// exactly (power-of-two diagonal:
/// every quotient is exact).
#[test]
fn test_solve_matrix4_matrix4x5_exact() {
    let l: Matrix4<Fixed> = Matrix4Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
    );
    let u: Matrix4<Fixed> = Matrix4Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(0),
        int(1),
        int(2),
        int(0),
        int(0),
        int(0),
        int(8),
    );
    let x: Matrix4x5<Fixed> = Matrix4x5Trait::new(
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(0),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-4),
        int(-2),
        int(1),
        int(4),
        int(-2),
        int(1),
        int(3),
        int(-3),
        int(0),
        int(3),
        int(-3),
    );
    let bl: Matrix4x5<Fixed> = Matrix4x5Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(8),
        int(-16),
        int(-4),
        int(8),
        int(-16),
        int(-9),
        int(-3),
        int(12),
        int(-9),
        int(-3),
        int(17),
        int(-10),
        int(8),
        int(17),
        int(-10),
    );
    let bu: Matrix4x5<Fixed> = Matrix4x5Trait::new(
        int(-15),
        int(6),
        int(18),
        int(-15),
        int(6),
        int(-3),
        int(-6),
        int(0),
        int(-3),
        int(-6),
        int(4),
        int(-5),
        int(4),
        int(4),
        int(-5),
        int(24),
        int(-24),
        int(0),
        int(24),
        int(-24),
    );
    assert!(l.solve_lower_triangular(bl) == Option::Some(x));
    assert!(u.solve_upper_triangular(bu) == Option::Some(x));
    assert!(l.tr_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.tr_solve_upper_triangular(bl) == Option::Some(x));
    assert!(l.ad_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.ad_solve_upper_triangular(bl) == Option::Some(x));
}

/// `Matrix4` against a `Matrix4x5` right-hand side: the `_unchecked` forms, bit-identical to the
/// checked ones.
#[test]
fn test_solve_matrix4_matrix4x5_unchecked_exact() {
    let l: Matrix4<Fixed> = Matrix4Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
    );
    let u: Matrix4<Fixed> = Matrix4Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(0),
        int(1),
        int(2),
        int(0),
        int(0),
        int(0),
        int(8),
    );
    let x: Matrix4x5<Fixed> = Matrix4x5Trait::new(
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(0),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-4),
        int(-2),
        int(1),
        int(4),
        int(-2),
        int(1),
        int(3),
        int(-3),
        int(0),
        int(3),
        int(-3),
    );
    let bl: Matrix4x5<Fixed> = Matrix4x5Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(8),
        int(-16),
        int(-4),
        int(8),
        int(-16),
        int(-9),
        int(-3),
        int(12),
        int(-9),
        int(-3),
        int(17),
        int(-10),
        int(8),
        int(17),
        int(-10),
    );
    let bu: Matrix4x5<Fixed> = Matrix4x5Trait::new(
        int(-15),
        int(6),
        int(18),
        int(-15),
        int(6),
        int(-3),
        int(-6),
        int(0),
        int(-3),
        int(-6),
        int(4),
        int(-5),
        int(4),
        int(4),
        int(-5),
        int(24),
        int(-24),
        int(0),
        int(24),
        int(-24),
    );
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    assert!(l.ad_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.ad_solve_upper_triangular_unchecked(bl) == x);
}

/// `Matrix4` against a `Matrix4x5` right-hand side: the `_mut` forms overwrite `b` with the
/// solution and return `true`.
#[test]
fn test_solve_matrix4_matrix4x5_mut_exact() {
    let l: Matrix4<Fixed> = Matrix4Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
    );
    let u: Matrix4<Fixed> = Matrix4Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(0),
        int(1),
        int(2),
        int(0),
        int(0),
        int(0),
        int(8),
    );
    let x: Matrix4x5<Fixed> = Matrix4x5Trait::new(
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(0),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-4),
        int(-2),
        int(1),
        int(4),
        int(-2),
        int(1),
        int(3),
        int(-3),
        int(0),
        int(3),
        int(-3),
    );
    let bl: Matrix4x5<Fixed> = Matrix4x5Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(8),
        int(-16),
        int(-4),
        int(8),
        int(-16),
        int(-9),
        int(-3),
        int(12),
        int(-9),
        int(-3),
        int(17),
        int(-10),
        int(8),
        int(17),
        int(-10),
    );
    let bu: Matrix4x5<Fixed> = Matrix4x5Trait::new(
        int(-15),
        int(6),
        int(18),
        int(-15),
        int(6),
        int(-3),
        int(-6),
        int(0),
        int(-3),
        int(-6),
        int(4),
        int(-5),
        int(4),
        int(4),
        int(-5),
        int(24),
        int(-24),
        int(0),
        int(24),
        int(-24),
    );
    let mut b = bl;
    assert!(l.solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(u.solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.tr_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.tr_solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.ad_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.ad_solve_upper_triangular_mut(ref b));
    assert!(b == x);
}

/// `Matrix4` against a `Matrix4x5` right-hand side: the `_unchecked_mut` forms.
#[test]
fn test_solve_matrix4_matrix4x5_unchecked_mut_exact() {
    let l: Matrix4<Fixed> = Matrix4Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
    );
    let u: Matrix4<Fixed> = Matrix4Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(0),
        int(1),
        int(2),
        int(0),
        int(0),
        int(0),
        int(8),
    );
    let x: Matrix4x5<Fixed> = Matrix4x5Trait::new(
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(0),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-4),
        int(-2),
        int(1),
        int(4),
        int(-2),
        int(1),
        int(3),
        int(-3),
        int(0),
        int(3),
        int(-3),
    );
    let bl: Matrix4x5<Fixed> = Matrix4x5Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(8),
        int(-16),
        int(-4),
        int(8),
        int(-16),
        int(-9),
        int(-3),
        int(12),
        int(-9),
        int(-3),
        int(17),
        int(-10),
        int(8),
        int(17),
        int(-10),
    );
    let bu: Matrix4x5<Fixed> = Matrix4x5Trait::new(
        int(-15),
        int(6),
        int(18),
        int(-15),
        int(6),
        int(-3),
        int(-6),
        int(0),
        int(-3),
        int(-6),
        int(4),
        int(-5),
        int(4),
        int(4),
        int(-5),
        int(24),
        int(-24),
        int(0),
        int(24),
        int(-24),
    );
    let mut b = bl;
    l.solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    u.solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.tr_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.tr_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.ad_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.ad_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
}

/// `Matrix4` against a `Matrix4x5` right-hand side: `with_diag` solves `(strict lower triangle of l
/// + 2 I) x = bd` and leaves `2 x` in `b`, like upstream (the quotients by `diag` are never written
/// back); a zero `diag` returns `false` and leaves `b` unchanged.
#[test]
fn test_solve_matrix4_matrix4x5_with_diag_exact() {
    let l: Matrix4<Fixed> = Matrix4Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
    );
    let bd: Matrix4x5<Fixed> = Matrix4x5Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(4),
        int(-8),
        int(-2),
        int(4),
        int(-8),
        int(-11),
        int(-2),
        int(16),
        int(-11),
        int(-2),
        int(-1),
        int(8),
        int(8),
        int(-1),
        int(8),
    );
    let x2: Matrix4x5<Fixed> = Matrix4x5Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(4),
        int(-8),
        int(-2),
        int(4),
        int(-8),
        int(-4),
        int(2),
        int(8),
        int(-4),
        int(2),
        int(6),
        int(-6),
        int(0),
        int(6),
        int(-6),
    );
    let mut b = bd;
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(b == x2);
    let mut b = bd;
    l.solve_lower_triangular_with_diag_unchecked_mut(ref b, int(2));
    assert!(b == x2);
    let mut b = bd;
    assert!(!l.solve_lower_triangular_with_diag_mut(ref b, int(0)));
    assert!(b == bd);
}

/// `Matrix4` against a `Matrix4x6` right-hand side: the `Option` forms recover the integer solution
/// exactly (power-of-two diagonal:
/// every quotient is exact).
#[test]
fn test_solve_matrix4_matrix4x6_exact() {
    let l: Matrix4<Fixed> = Matrix4Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
    );
    let u: Matrix4<Fixed> = Matrix4Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(0),
        int(1),
        int(2),
        int(0),
        int(0),
        int(0),
        int(8),
    );
    let x: Matrix4x6<Fixed> = Matrix4x6Trait::new(
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(0),
        int(3),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-4),
        int(-1),
        int(-2),
        int(1),
        int(4),
        int(-2),
        int(1),
        int(4),
        int(3),
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(0),
    );
    let bl: Matrix4x6<Fixed> = Matrix4x6Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(6),
        int(8),
        int(-16),
        int(-4),
        int(8),
        int(-16),
        int(-4),
        int(-9),
        int(-3),
        int(12),
        int(-9),
        int(-3),
        int(12),
        int(17),
        int(-10),
        int(8),
        int(17),
        int(-10),
        int(8),
    );
    let bu: Matrix4x6<Fixed> = Matrix4x6Trait::new(
        int(-15),
        int(6),
        int(18),
        int(-15),
        int(6),
        int(18),
        int(-3),
        int(-6),
        int(0),
        int(-3),
        int(-6),
        int(0),
        int(4),
        int(-5),
        int(4),
        int(4),
        int(-5),
        int(4),
        int(24),
        int(-24),
        int(0),
        int(24),
        int(-24),
        int(0),
    );
    assert!(l.solve_lower_triangular(bl) == Option::Some(x));
    assert!(u.solve_upper_triangular(bu) == Option::Some(x));
    assert!(l.tr_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.tr_solve_upper_triangular(bl) == Option::Some(x));
    assert!(l.ad_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.ad_solve_upper_triangular(bl) == Option::Some(x));
}

/// `Matrix4` against a `Matrix4x6` right-hand side: the `_unchecked` forms, bit-identical to the
/// checked ones.
#[test]
fn test_solve_matrix4_matrix4x6_unchecked_exact() {
    let l: Matrix4<Fixed> = Matrix4Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
    );
    let u: Matrix4<Fixed> = Matrix4Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(0),
        int(1),
        int(2),
        int(0),
        int(0),
        int(0),
        int(8),
    );
    let x: Matrix4x6<Fixed> = Matrix4x6Trait::new(
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(0),
        int(3),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-4),
        int(-1),
        int(-2),
        int(1),
        int(4),
        int(-2),
        int(1),
        int(4),
        int(3),
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(0),
    );
    let bl: Matrix4x6<Fixed> = Matrix4x6Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(6),
        int(8),
        int(-16),
        int(-4),
        int(8),
        int(-16),
        int(-4),
        int(-9),
        int(-3),
        int(12),
        int(-9),
        int(-3),
        int(12),
        int(17),
        int(-10),
        int(8),
        int(17),
        int(-10),
        int(8),
    );
    let bu: Matrix4x6<Fixed> = Matrix4x6Trait::new(
        int(-15),
        int(6),
        int(18),
        int(-15),
        int(6),
        int(18),
        int(-3),
        int(-6),
        int(0),
        int(-3),
        int(-6),
        int(0),
        int(4),
        int(-5),
        int(4),
        int(4),
        int(-5),
        int(4),
        int(24),
        int(-24),
        int(0),
        int(24),
        int(-24),
        int(0),
    );
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    assert!(l.ad_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.ad_solve_upper_triangular_unchecked(bl) == x);
}

/// `Matrix4` against a `Matrix4x6` right-hand side: the `_mut` forms overwrite `b` with the
/// solution and return `true`.
#[test]
fn test_solve_matrix4_matrix4x6_mut_exact() {
    let l: Matrix4<Fixed> = Matrix4Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
    );
    let u: Matrix4<Fixed> = Matrix4Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(0),
        int(1),
        int(2),
        int(0),
        int(0),
        int(0),
        int(8),
    );
    let x: Matrix4x6<Fixed> = Matrix4x6Trait::new(
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(0),
        int(3),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-4),
        int(-1),
        int(-2),
        int(1),
        int(4),
        int(-2),
        int(1),
        int(4),
        int(3),
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(0),
    );
    let bl: Matrix4x6<Fixed> = Matrix4x6Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(6),
        int(8),
        int(-16),
        int(-4),
        int(8),
        int(-16),
        int(-4),
        int(-9),
        int(-3),
        int(12),
        int(-9),
        int(-3),
        int(12),
        int(17),
        int(-10),
        int(8),
        int(17),
        int(-10),
        int(8),
    );
    let bu: Matrix4x6<Fixed> = Matrix4x6Trait::new(
        int(-15),
        int(6),
        int(18),
        int(-15),
        int(6),
        int(18),
        int(-3),
        int(-6),
        int(0),
        int(-3),
        int(-6),
        int(0),
        int(4),
        int(-5),
        int(4),
        int(4),
        int(-5),
        int(4),
        int(24),
        int(-24),
        int(0),
        int(24),
        int(-24),
        int(0),
    );
    let mut b = bl;
    assert!(l.solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(u.solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.tr_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.tr_solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.ad_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.ad_solve_upper_triangular_mut(ref b));
    assert!(b == x);
}

/// `Matrix4` against a `Matrix4x6` right-hand side: the `_unchecked_mut` forms.
#[test]
fn test_solve_matrix4_matrix4x6_unchecked_mut_exact() {
    let l: Matrix4<Fixed> = Matrix4Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
    );
    let u: Matrix4<Fixed> = Matrix4Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(0),
        int(1),
        int(2),
        int(0),
        int(0),
        int(0),
        int(8),
    );
    let x: Matrix4x6<Fixed> = Matrix4x6Trait::new(
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(0),
        int(3),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-4),
        int(-1),
        int(-2),
        int(1),
        int(4),
        int(-2),
        int(1),
        int(4),
        int(3),
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(0),
    );
    let bl: Matrix4x6<Fixed> = Matrix4x6Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(6),
        int(8),
        int(-16),
        int(-4),
        int(8),
        int(-16),
        int(-4),
        int(-9),
        int(-3),
        int(12),
        int(-9),
        int(-3),
        int(12),
        int(17),
        int(-10),
        int(8),
        int(17),
        int(-10),
        int(8),
    );
    let bu: Matrix4x6<Fixed> = Matrix4x6Trait::new(
        int(-15),
        int(6),
        int(18),
        int(-15),
        int(6),
        int(18),
        int(-3),
        int(-6),
        int(0),
        int(-3),
        int(-6),
        int(0),
        int(4),
        int(-5),
        int(4),
        int(4),
        int(-5),
        int(4),
        int(24),
        int(-24),
        int(0),
        int(24),
        int(-24),
        int(0),
    );
    let mut b = bl;
    l.solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    u.solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.tr_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.tr_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.ad_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.ad_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
}

/// `Matrix4` against a `Matrix4x6` right-hand side: `with_diag` solves `(strict lower triangle of l
/// + 2 I) x = bd` and leaves `2 x` in `b`, like upstream (the quotients by `diag` are never written
/// back); a zero `diag` returns `false` and leaves `b` unchanged.
#[test]
fn test_solve_matrix4_matrix4x6_with_diag_exact() {
    let l: Matrix4<Fixed> = Matrix4Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
    );
    let bd: Matrix4x6<Fixed> = Matrix4x6Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(6),
        int(4),
        int(-8),
        int(-2),
        int(4),
        int(-8),
        int(-2),
        int(-11),
        int(-2),
        int(16),
        int(-11),
        int(-2),
        int(16),
        int(-1),
        int(8),
        int(8),
        int(-1),
        int(8),
        int(8),
    );
    let x2: Matrix4x6<Fixed> = Matrix4x6Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(6),
        int(4),
        int(-8),
        int(-2),
        int(4),
        int(-8),
        int(-2),
        int(-4),
        int(2),
        int(8),
        int(-4),
        int(2),
        int(8),
        int(6),
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
    );
    let mut b = bd;
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(b == x2);
    let mut b = bd;
    l.solve_lower_triangular_with_diag_unchecked_mut(ref b, int(2));
    assert!(b == x2);
    let mut b = bd;
    assert!(!l.solve_lower_triangular_with_diag_mut(ref b, int(0)));
    assert!(b == bd);
}

/// `Matrix5` against a `Vector5` right-hand side: the `Option` forms recover the integer solution
/// exactly (power-of-two diagonal:
/// every quotient is exact).
#[test]
fn test_solve_matrix5_vector5_exact() {
    let l: Matrix5<Fixed> = Matrix5Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
    );
    let u: Matrix5<Fixed> = Matrix5Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(2),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(0),
        int(0),
        int(1),
        int(2),
        int(-2),
        int(0),
        int(0),
        int(0),
        int(8),
        int(3),
        int(0),
        int(0),
        int(0),
        int(0),
        int(2),
    );
    let x: Vector5<Fixed> = Vector5Trait::new(int(-3), int(2), int(-2), int(3), int(-1));
    let bl: Vector5<Fixed> = Vector5Trait::new(int(-6), int(8), int(-9), int(17), int(5));
    let bu: Vector5<Fixed> = Vector5Trait::new(int(-17), int(-3), int(6), int(21), int(-2));
    assert!(l.solve_lower_triangular(bl) == Option::Some(x));
    assert!(u.solve_upper_triangular(bu) == Option::Some(x));
    assert!(l.tr_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.tr_solve_upper_triangular(bl) == Option::Some(x));
    assert!(l.ad_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.ad_solve_upper_triangular(bl) == Option::Some(x));
}

/// `Matrix5` against a `Vector5` right-hand side: the `_unchecked` forms, bit-identical to the
/// checked ones.
#[test]
fn test_solve_matrix5_vector5_unchecked_exact() {
    let l: Matrix5<Fixed> = Matrix5Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
    );
    let u: Matrix5<Fixed> = Matrix5Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(2),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(0),
        int(0),
        int(1),
        int(2),
        int(-2),
        int(0),
        int(0),
        int(0),
        int(8),
        int(3),
        int(0),
        int(0),
        int(0),
        int(0),
        int(2),
    );
    let x: Vector5<Fixed> = Vector5Trait::new(int(-3), int(2), int(-2), int(3), int(-1));
    let bl: Vector5<Fixed> = Vector5Trait::new(int(-6), int(8), int(-9), int(17), int(5));
    let bu: Vector5<Fixed> = Vector5Trait::new(int(-17), int(-3), int(6), int(21), int(-2));
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    assert!(l.ad_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.ad_solve_upper_triangular_unchecked(bl) == x);
}

/// `Matrix5` against a `Vector5` right-hand side: the `_mut` forms overwrite `b` with the solution
/// and return `true`.
#[test]
fn test_solve_matrix5_vector5_mut_exact() {
    let l: Matrix5<Fixed> = Matrix5Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
    );
    let u: Matrix5<Fixed> = Matrix5Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(2),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(0),
        int(0),
        int(1),
        int(2),
        int(-2),
        int(0),
        int(0),
        int(0),
        int(8),
        int(3),
        int(0),
        int(0),
        int(0),
        int(0),
        int(2),
    );
    let x: Vector5<Fixed> = Vector5Trait::new(int(-3), int(2), int(-2), int(3), int(-1));
    let bl: Vector5<Fixed> = Vector5Trait::new(int(-6), int(8), int(-9), int(17), int(5));
    let bu: Vector5<Fixed> = Vector5Trait::new(int(-17), int(-3), int(6), int(21), int(-2));
    let mut b = bl;
    assert!(l.solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(u.solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.tr_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.tr_solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.ad_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.ad_solve_upper_triangular_mut(ref b));
    assert!(b == x);
}

/// `Matrix5` against a `Vector5` right-hand side: the `_unchecked_mut` forms.
#[test]
fn test_solve_matrix5_vector5_unchecked_mut_exact() {
    let l: Matrix5<Fixed> = Matrix5Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
    );
    let u: Matrix5<Fixed> = Matrix5Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(2),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(0),
        int(0),
        int(1),
        int(2),
        int(-2),
        int(0),
        int(0),
        int(0),
        int(8),
        int(3),
        int(0),
        int(0),
        int(0),
        int(0),
        int(2),
    );
    let x: Vector5<Fixed> = Vector5Trait::new(int(-3), int(2), int(-2), int(3), int(-1));
    let bl: Vector5<Fixed> = Vector5Trait::new(int(-6), int(8), int(-9), int(17), int(5));
    let bu: Vector5<Fixed> = Vector5Trait::new(int(-17), int(-3), int(6), int(21), int(-2));
    let mut b = bl;
    l.solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    u.solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.tr_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.tr_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.ad_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.ad_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
}

/// `Matrix5` against a `Vector5` right-hand side: `with_diag` solves `(strict lower triangle of l +
/// 2 I) x = bd` and leaves `2 x` in `b`, like upstream (the quotients by `diag` are never written
/// back); a zero `diag` returns `false` and leaves `b` unchanged.
#[test]
fn test_solve_matrix5_vector5_with_diag_exact() {
    let l: Matrix5<Fixed> = Matrix5Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
    );
    let bd: Vector5<Fixed> = Vector5Trait::new(int(-6), int(4), int(-11), int(-1), int(5));
    let x2: Vector5<Fixed> = Vector5Trait::new(int(-6), int(4), int(-4), int(6), int(-2));
    let mut b = bd;
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(b == x2);
    let mut b = bd;
    l.solve_lower_triangular_with_diag_unchecked_mut(ref b, int(2));
    assert!(b == x2);
    let mut b = bd;
    assert!(!l.solve_lower_triangular_with_diag_mut(ref b, int(0)));
    assert!(b == bd);
}

/// `Matrix5` against a `Matrix5x2` right-hand side: the `Option` forms recover the integer solution
/// exactly (power-of-two diagonal:
/// every quotient is exact).
#[test]
fn test_solve_matrix5_matrix5x2_exact() {
    let l: Matrix5<Fixed> = Matrix5Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
    );
    let u: Matrix5<Fixed> = Matrix5Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(2),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(0),
        int(0),
        int(1),
        int(2),
        int(-2),
        int(0),
        int(0),
        int(0),
        int(8),
        int(3),
        int(0),
        int(0),
        int(0),
        int(0),
        int(2),
    );
    let x: Matrix5x2<Fixed> = Matrix5x2Trait::new(
        int(-3), int(0), int(2), int(-4), int(-2), int(1), int(3), int(-3), int(-1), int(2),
    );
    let bl: Matrix5x2<Fixed> = Matrix5x2Trait::new(
        int(-6), int(0), int(8), int(-16), int(-9), int(-3), int(17), int(-10), int(5), int(-7),
    );
    let bu: Matrix5x2<Fixed> = Matrix5x2Trait::new(
        int(-17), int(10), int(-3), int(-6), int(6), int(-9), int(21), int(-18), int(-2), int(4),
    );
    assert!(l.solve_lower_triangular(bl) == Option::Some(x));
    assert!(u.solve_upper_triangular(bu) == Option::Some(x));
    assert!(l.tr_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.tr_solve_upper_triangular(bl) == Option::Some(x));
    assert!(l.ad_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.ad_solve_upper_triangular(bl) == Option::Some(x));
}

/// `Matrix5` against a `Matrix5x2` right-hand side: the `_unchecked` forms, bit-identical to the
/// checked ones.
#[test]
fn test_solve_matrix5_matrix5x2_unchecked_exact() {
    let l: Matrix5<Fixed> = Matrix5Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
    );
    let u: Matrix5<Fixed> = Matrix5Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(2),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(0),
        int(0),
        int(1),
        int(2),
        int(-2),
        int(0),
        int(0),
        int(0),
        int(8),
        int(3),
        int(0),
        int(0),
        int(0),
        int(0),
        int(2),
    );
    let x: Matrix5x2<Fixed> = Matrix5x2Trait::new(
        int(-3), int(0), int(2), int(-4), int(-2), int(1), int(3), int(-3), int(-1), int(2),
    );
    let bl: Matrix5x2<Fixed> = Matrix5x2Trait::new(
        int(-6), int(0), int(8), int(-16), int(-9), int(-3), int(17), int(-10), int(5), int(-7),
    );
    let bu: Matrix5x2<Fixed> = Matrix5x2Trait::new(
        int(-17), int(10), int(-3), int(-6), int(6), int(-9), int(21), int(-18), int(-2), int(4),
    );
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    assert!(l.ad_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.ad_solve_upper_triangular_unchecked(bl) == x);
}

/// `Matrix5` against a `Matrix5x2` right-hand side: the `_mut` forms overwrite `b` with the
/// solution and return `true`.
#[test]
fn test_solve_matrix5_matrix5x2_mut_exact() {
    let l: Matrix5<Fixed> = Matrix5Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
    );
    let u: Matrix5<Fixed> = Matrix5Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(2),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(0),
        int(0),
        int(1),
        int(2),
        int(-2),
        int(0),
        int(0),
        int(0),
        int(8),
        int(3),
        int(0),
        int(0),
        int(0),
        int(0),
        int(2),
    );
    let x: Matrix5x2<Fixed> = Matrix5x2Trait::new(
        int(-3), int(0), int(2), int(-4), int(-2), int(1), int(3), int(-3), int(-1), int(2),
    );
    let bl: Matrix5x2<Fixed> = Matrix5x2Trait::new(
        int(-6), int(0), int(8), int(-16), int(-9), int(-3), int(17), int(-10), int(5), int(-7),
    );
    let bu: Matrix5x2<Fixed> = Matrix5x2Trait::new(
        int(-17), int(10), int(-3), int(-6), int(6), int(-9), int(21), int(-18), int(-2), int(4),
    );
    let mut b = bl;
    assert!(l.solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(u.solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.tr_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.tr_solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.ad_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.ad_solve_upper_triangular_mut(ref b));
    assert!(b == x);
}

/// `Matrix5` against a `Matrix5x2` right-hand side: the `_unchecked_mut` forms.
#[test]
fn test_solve_matrix5_matrix5x2_unchecked_mut_exact() {
    let l: Matrix5<Fixed> = Matrix5Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
    );
    let u: Matrix5<Fixed> = Matrix5Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(2),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(0),
        int(0),
        int(1),
        int(2),
        int(-2),
        int(0),
        int(0),
        int(0),
        int(8),
        int(3),
        int(0),
        int(0),
        int(0),
        int(0),
        int(2),
    );
    let x: Matrix5x2<Fixed> = Matrix5x2Trait::new(
        int(-3), int(0), int(2), int(-4), int(-2), int(1), int(3), int(-3), int(-1), int(2),
    );
    let bl: Matrix5x2<Fixed> = Matrix5x2Trait::new(
        int(-6), int(0), int(8), int(-16), int(-9), int(-3), int(17), int(-10), int(5), int(-7),
    );
    let bu: Matrix5x2<Fixed> = Matrix5x2Trait::new(
        int(-17), int(10), int(-3), int(-6), int(6), int(-9), int(21), int(-18), int(-2), int(4),
    );
    let mut b = bl;
    l.solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    u.solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.tr_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.tr_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.ad_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.ad_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
}

/// `Matrix5` against a `Matrix5x2` right-hand side: `with_diag` solves `(strict lower triangle of l
/// + 2 I) x = bd` and leaves `2 x` in `b`, like upstream (the quotients by `diag` are never written
/// back); a zero `diag` returns `false` and leaves `b` unchanged.
#[test]
fn test_solve_matrix5_matrix5x2_with_diag_exact() {
    let l: Matrix5<Fixed> = Matrix5Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
    );
    let bd: Matrix5x2<Fixed> = Matrix5x2Trait::new(
        int(-6), int(0), int(4), int(-8), int(-11), int(-2), int(-1), int(8), int(5), int(-7),
    );
    let x2: Matrix5x2<Fixed> = Matrix5x2Trait::new(
        int(-6), int(0), int(4), int(-8), int(-4), int(2), int(6), int(-6), int(-2), int(4),
    );
    let mut b = bd;
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(b == x2);
    let mut b = bd;
    l.solve_lower_triangular_with_diag_unchecked_mut(ref b, int(2));
    assert!(b == x2);
    let mut b = bd;
    assert!(!l.solve_lower_triangular_with_diag_mut(ref b, int(0)));
    assert!(b == bd);
}

/// `Matrix5` against a `Matrix5x3` right-hand side: the `Option` forms recover the integer solution
/// exactly (power-of-two diagonal:
/// every quotient is exact).
#[test]
fn test_solve_matrix5_matrix5x3_exact() {
    let l: Matrix5<Fixed> = Matrix5Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
    );
    let u: Matrix5<Fixed> = Matrix5Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(2),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(0),
        int(0),
        int(1),
        int(2),
        int(-2),
        int(0),
        int(0),
        int(0),
        int(8),
        int(3),
        int(0),
        int(0),
        int(0),
        int(0),
        int(2),
    );
    let x: Matrix5x3<Fixed> = Matrix5x3Trait::new(
        int(-3),
        int(0),
        int(3),
        int(2),
        int(-4),
        int(-1),
        int(-2),
        int(1),
        int(4),
        int(3),
        int(-3),
        int(0),
        int(-1),
        int(2),
        int(-4),
    );
    let bl: Matrix5x3<Fixed> = Matrix5x3Trait::new(
        int(-6),
        int(0),
        int(6),
        int(8),
        int(-16),
        int(-4),
        int(-9),
        int(-3),
        int(12),
        int(17),
        int(-10),
        int(8),
        int(5),
        int(-7),
        int(-10),
    );
    let bu: Matrix5x3<Fixed> = Matrix5x3Trait::new(
        int(-17),
        int(10),
        int(10),
        int(-3),
        int(-6),
        int(0),
        int(6),
        int(-9),
        int(12),
        int(21),
        int(-18),
        int(-12),
        int(-2),
        int(4),
        int(-8),
    );
    assert!(l.solve_lower_triangular(bl) == Option::Some(x));
    assert!(u.solve_upper_triangular(bu) == Option::Some(x));
    assert!(l.tr_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.tr_solve_upper_triangular(bl) == Option::Some(x));
    assert!(l.ad_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.ad_solve_upper_triangular(bl) == Option::Some(x));
}

/// `Matrix5` against a `Matrix5x3` right-hand side: the `_unchecked` forms, bit-identical to the
/// checked ones.
#[test]
fn test_solve_matrix5_matrix5x3_unchecked_exact() {
    let l: Matrix5<Fixed> = Matrix5Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
    );
    let u: Matrix5<Fixed> = Matrix5Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(2),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(0),
        int(0),
        int(1),
        int(2),
        int(-2),
        int(0),
        int(0),
        int(0),
        int(8),
        int(3),
        int(0),
        int(0),
        int(0),
        int(0),
        int(2),
    );
    let x: Matrix5x3<Fixed> = Matrix5x3Trait::new(
        int(-3),
        int(0),
        int(3),
        int(2),
        int(-4),
        int(-1),
        int(-2),
        int(1),
        int(4),
        int(3),
        int(-3),
        int(0),
        int(-1),
        int(2),
        int(-4),
    );
    let bl: Matrix5x3<Fixed> = Matrix5x3Trait::new(
        int(-6),
        int(0),
        int(6),
        int(8),
        int(-16),
        int(-4),
        int(-9),
        int(-3),
        int(12),
        int(17),
        int(-10),
        int(8),
        int(5),
        int(-7),
        int(-10),
    );
    let bu: Matrix5x3<Fixed> = Matrix5x3Trait::new(
        int(-17),
        int(10),
        int(10),
        int(-3),
        int(-6),
        int(0),
        int(6),
        int(-9),
        int(12),
        int(21),
        int(-18),
        int(-12),
        int(-2),
        int(4),
        int(-8),
    );
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    assert!(l.ad_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.ad_solve_upper_triangular_unchecked(bl) == x);
}

/// `Matrix5` against a `Matrix5x3` right-hand side: the `_mut` forms overwrite `b` with the
/// solution and return `true`.
#[test]
fn test_solve_matrix5_matrix5x3_mut_exact() {
    let l: Matrix5<Fixed> = Matrix5Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
    );
    let u: Matrix5<Fixed> = Matrix5Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(2),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(0),
        int(0),
        int(1),
        int(2),
        int(-2),
        int(0),
        int(0),
        int(0),
        int(8),
        int(3),
        int(0),
        int(0),
        int(0),
        int(0),
        int(2),
    );
    let x: Matrix5x3<Fixed> = Matrix5x3Trait::new(
        int(-3),
        int(0),
        int(3),
        int(2),
        int(-4),
        int(-1),
        int(-2),
        int(1),
        int(4),
        int(3),
        int(-3),
        int(0),
        int(-1),
        int(2),
        int(-4),
    );
    let bl: Matrix5x3<Fixed> = Matrix5x3Trait::new(
        int(-6),
        int(0),
        int(6),
        int(8),
        int(-16),
        int(-4),
        int(-9),
        int(-3),
        int(12),
        int(17),
        int(-10),
        int(8),
        int(5),
        int(-7),
        int(-10),
    );
    let bu: Matrix5x3<Fixed> = Matrix5x3Trait::new(
        int(-17),
        int(10),
        int(10),
        int(-3),
        int(-6),
        int(0),
        int(6),
        int(-9),
        int(12),
        int(21),
        int(-18),
        int(-12),
        int(-2),
        int(4),
        int(-8),
    );
    let mut b = bl;
    assert!(l.solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(u.solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.tr_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.tr_solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.ad_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.ad_solve_upper_triangular_mut(ref b));
    assert!(b == x);
}

/// `Matrix5` against a `Matrix5x3` right-hand side: the `_unchecked_mut` forms.
#[test]
fn test_solve_matrix5_matrix5x3_unchecked_mut_exact() {
    let l: Matrix5<Fixed> = Matrix5Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
    );
    let u: Matrix5<Fixed> = Matrix5Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(2),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(0),
        int(0),
        int(1),
        int(2),
        int(-2),
        int(0),
        int(0),
        int(0),
        int(8),
        int(3),
        int(0),
        int(0),
        int(0),
        int(0),
        int(2),
    );
    let x: Matrix5x3<Fixed> = Matrix5x3Trait::new(
        int(-3),
        int(0),
        int(3),
        int(2),
        int(-4),
        int(-1),
        int(-2),
        int(1),
        int(4),
        int(3),
        int(-3),
        int(0),
        int(-1),
        int(2),
        int(-4),
    );
    let bl: Matrix5x3<Fixed> = Matrix5x3Trait::new(
        int(-6),
        int(0),
        int(6),
        int(8),
        int(-16),
        int(-4),
        int(-9),
        int(-3),
        int(12),
        int(17),
        int(-10),
        int(8),
        int(5),
        int(-7),
        int(-10),
    );
    let bu: Matrix5x3<Fixed> = Matrix5x3Trait::new(
        int(-17),
        int(10),
        int(10),
        int(-3),
        int(-6),
        int(0),
        int(6),
        int(-9),
        int(12),
        int(21),
        int(-18),
        int(-12),
        int(-2),
        int(4),
        int(-8),
    );
    let mut b = bl;
    l.solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    u.solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.tr_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.tr_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.ad_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.ad_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
}

/// `Matrix5` against a `Matrix5x3` right-hand side: `with_diag` solves `(strict lower triangle of l
/// + 2 I) x = bd` and leaves `2 x` in `b`, like upstream (the quotients by `diag` are never written
/// back); a zero `diag` returns `false` and leaves `b` unchanged.
#[test]
fn test_solve_matrix5_matrix5x3_with_diag_exact() {
    let l: Matrix5<Fixed> = Matrix5Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
    );
    let bd: Matrix5x3<Fixed> = Matrix5x3Trait::new(
        int(-6),
        int(0),
        int(6),
        int(4),
        int(-8),
        int(-2),
        int(-11),
        int(-2),
        int(16),
        int(-1),
        int(8),
        int(8),
        int(5),
        int(-7),
        int(-10),
    );
    let x2: Matrix5x3<Fixed> = Matrix5x3Trait::new(
        int(-6),
        int(0),
        int(6),
        int(4),
        int(-8),
        int(-2),
        int(-4),
        int(2),
        int(8),
        int(6),
        int(-6),
        int(0),
        int(-2),
        int(4),
        int(-8),
    );
    let mut b = bd;
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(b == x2);
    let mut b = bd;
    l.solve_lower_triangular_with_diag_unchecked_mut(ref b, int(2));
    assert!(b == x2);
    let mut b = bd;
    assert!(!l.solve_lower_triangular_with_diag_mut(ref b, int(0)));
    assert!(b == bd);
}

/// `Matrix5` against a `Matrix5x4` right-hand side: the `Option` forms recover the integer solution
/// exactly (power-of-two diagonal:
/// every quotient is exact).
#[test]
fn test_solve_matrix5_matrix5x4_exact() {
    let l: Matrix5<Fixed> = Matrix5Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
    );
    let u: Matrix5<Fixed> = Matrix5Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(2),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(0),
        int(0),
        int(1),
        int(2),
        int(-2),
        int(0),
        int(0),
        int(0),
        int(8),
        int(3),
        int(0),
        int(0),
        int(0),
        int(0),
        int(2),
    );
    let x: Matrix5x4<Fixed> = Matrix5x4Trait::new(
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-2),
        int(1),
        int(4),
        int(-2),
        int(3),
        int(-3),
        int(0),
        int(3),
        int(-1),
        int(2),
        int(-4),
        int(-1),
    );
    let bl: Matrix5x4<Fixed> = Matrix5x4Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(8),
        int(-16),
        int(-4),
        int(8),
        int(-9),
        int(-3),
        int(12),
        int(-9),
        int(17),
        int(-10),
        int(8),
        int(17),
        int(5),
        int(-7),
        int(-10),
        int(5),
    );
    let bu: Matrix5x4<Fixed> = Matrix5x4Trait::new(
        int(-17),
        int(10),
        int(10),
        int(-17),
        int(-3),
        int(-6),
        int(0),
        int(-3),
        int(6),
        int(-9),
        int(12),
        int(6),
        int(21),
        int(-18),
        int(-12),
        int(21),
        int(-2),
        int(4),
        int(-8),
        int(-2),
    );
    assert!(l.solve_lower_triangular(bl) == Option::Some(x));
    assert!(u.solve_upper_triangular(bu) == Option::Some(x));
    assert!(l.tr_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.tr_solve_upper_triangular(bl) == Option::Some(x));
    assert!(l.ad_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.ad_solve_upper_triangular(bl) == Option::Some(x));
}

/// `Matrix5` against a `Matrix5x4` right-hand side: the `_unchecked` forms, bit-identical to the
/// checked ones.
#[test]
fn test_solve_matrix5_matrix5x4_unchecked_exact() {
    let l: Matrix5<Fixed> = Matrix5Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
    );
    let u: Matrix5<Fixed> = Matrix5Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(2),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(0),
        int(0),
        int(1),
        int(2),
        int(-2),
        int(0),
        int(0),
        int(0),
        int(8),
        int(3),
        int(0),
        int(0),
        int(0),
        int(0),
        int(2),
    );
    let x: Matrix5x4<Fixed> = Matrix5x4Trait::new(
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-2),
        int(1),
        int(4),
        int(-2),
        int(3),
        int(-3),
        int(0),
        int(3),
        int(-1),
        int(2),
        int(-4),
        int(-1),
    );
    let bl: Matrix5x4<Fixed> = Matrix5x4Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(8),
        int(-16),
        int(-4),
        int(8),
        int(-9),
        int(-3),
        int(12),
        int(-9),
        int(17),
        int(-10),
        int(8),
        int(17),
        int(5),
        int(-7),
        int(-10),
        int(5),
    );
    let bu: Matrix5x4<Fixed> = Matrix5x4Trait::new(
        int(-17),
        int(10),
        int(10),
        int(-17),
        int(-3),
        int(-6),
        int(0),
        int(-3),
        int(6),
        int(-9),
        int(12),
        int(6),
        int(21),
        int(-18),
        int(-12),
        int(21),
        int(-2),
        int(4),
        int(-8),
        int(-2),
    );
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    assert!(l.ad_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.ad_solve_upper_triangular_unchecked(bl) == x);
}

/// `Matrix5` against a `Matrix5x4` right-hand side: the `_mut` forms overwrite `b` with the
/// solution and return `true`.
#[test]
fn test_solve_matrix5_matrix5x4_mut_exact() {
    let l: Matrix5<Fixed> = Matrix5Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
    );
    let u: Matrix5<Fixed> = Matrix5Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(2),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(0),
        int(0),
        int(1),
        int(2),
        int(-2),
        int(0),
        int(0),
        int(0),
        int(8),
        int(3),
        int(0),
        int(0),
        int(0),
        int(0),
        int(2),
    );
    let x: Matrix5x4<Fixed> = Matrix5x4Trait::new(
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-2),
        int(1),
        int(4),
        int(-2),
        int(3),
        int(-3),
        int(0),
        int(3),
        int(-1),
        int(2),
        int(-4),
        int(-1),
    );
    let bl: Matrix5x4<Fixed> = Matrix5x4Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(8),
        int(-16),
        int(-4),
        int(8),
        int(-9),
        int(-3),
        int(12),
        int(-9),
        int(17),
        int(-10),
        int(8),
        int(17),
        int(5),
        int(-7),
        int(-10),
        int(5),
    );
    let bu: Matrix5x4<Fixed> = Matrix5x4Trait::new(
        int(-17),
        int(10),
        int(10),
        int(-17),
        int(-3),
        int(-6),
        int(0),
        int(-3),
        int(6),
        int(-9),
        int(12),
        int(6),
        int(21),
        int(-18),
        int(-12),
        int(21),
        int(-2),
        int(4),
        int(-8),
        int(-2),
    );
    let mut b = bl;
    assert!(l.solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(u.solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.tr_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.tr_solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.ad_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.ad_solve_upper_triangular_mut(ref b));
    assert!(b == x);
}

/// `Matrix5` against a `Matrix5x4` right-hand side: the `_unchecked_mut` forms.
#[test]
fn test_solve_matrix5_matrix5x4_unchecked_mut_exact() {
    let l: Matrix5<Fixed> = Matrix5Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
    );
    let u: Matrix5<Fixed> = Matrix5Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(2),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(0),
        int(0),
        int(1),
        int(2),
        int(-2),
        int(0),
        int(0),
        int(0),
        int(8),
        int(3),
        int(0),
        int(0),
        int(0),
        int(0),
        int(2),
    );
    let x: Matrix5x4<Fixed> = Matrix5x4Trait::new(
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-2),
        int(1),
        int(4),
        int(-2),
        int(3),
        int(-3),
        int(0),
        int(3),
        int(-1),
        int(2),
        int(-4),
        int(-1),
    );
    let bl: Matrix5x4<Fixed> = Matrix5x4Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(8),
        int(-16),
        int(-4),
        int(8),
        int(-9),
        int(-3),
        int(12),
        int(-9),
        int(17),
        int(-10),
        int(8),
        int(17),
        int(5),
        int(-7),
        int(-10),
        int(5),
    );
    let bu: Matrix5x4<Fixed> = Matrix5x4Trait::new(
        int(-17),
        int(10),
        int(10),
        int(-17),
        int(-3),
        int(-6),
        int(0),
        int(-3),
        int(6),
        int(-9),
        int(12),
        int(6),
        int(21),
        int(-18),
        int(-12),
        int(21),
        int(-2),
        int(4),
        int(-8),
        int(-2),
    );
    let mut b = bl;
    l.solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    u.solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.tr_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.tr_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.ad_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.ad_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
}

/// `Matrix5` against a `Matrix5x4` right-hand side: `with_diag` solves `(strict lower triangle of l
/// + 2 I) x = bd` and leaves `2 x` in `b`, like upstream (the quotients by `diag` are never written
/// back); a zero `diag` returns `false` and leaves `b` unchanged.
#[test]
fn test_solve_matrix5_matrix5x4_with_diag_exact() {
    let l: Matrix5<Fixed> = Matrix5Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
    );
    let bd: Matrix5x4<Fixed> = Matrix5x4Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(4),
        int(-8),
        int(-2),
        int(4),
        int(-11),
        int(-2),
        int(16),
        int(-11),
        int(-1),
        int(8),
        int(8),
        int(-1),
        int(5),
        int(-7),
        int(-10),
        int(5),
    );
    let x2: Matrix5x4<Fixed> = Matrix5x4Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(4),
        int(-8),
        int(-2),
        int(4),
        int(-4),
        int(2),
        int(8),
        int(-4),
        int(6),
        int(-6),
        int(0),
        int(6),
        int(-2),
        int(4),
        int(-8),
        int(-2),
    );
    let mut b = bd;
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(b == x2);
    let mut b = bd;
    l.solve_lower_triangular_with_diag_unchecked_mut(ref b, int(2));
    assert!(b == x2);
    let mut b = bd;
    assert!(!l.solve_lower_triangular_with_diag_mut(ref b, int(0)));
    assert!(b == bd);
}

/// `Matrix5` against a `Matrix5` right-hand side: the `Option` forms recover the integer solution
/// exactly (power-of-two diagonal:
/// every quotient is exact).
#[test]
fn test_solve_matrix5_matrix5_exact() {
    let l: Matrix5<Fixed> = Matrix5Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
    );
    let u: Matrix5<Fixed> = Matrix5Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(2),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(0),
        int(0),
        int(1),
        int(2),
        int(-2),
        int(0),
        int(0),
        int(0),
        int(8),
        int(3),
        int(0),
        int(0),
        int(0),
        int(0),
        int(2),
    );
    let x: Matrix5<Fixed> = Matrix5Trait::new(
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(0),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-4),
        int(-2),
        int(1),
        int(4),
        int(-2),
        int(1),
        int(3),
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(-1),
        int(2),
        int(-4),
        int(-1),
        int(2),
    );
    let bl: Matrix5<Fixed> = Matrix5Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(8),
        int(-16),
        int(-4),
        int(8),
        int(-16),
        int(-9),
        int(-3),
        int(12),
        int(-9),
        int(-3),
        int(17),
        int(-10),
        int(8),
        int(17),
        int(-10),
        int(5),
        int(-7),
        int(-10),
        int(5),
        int(-7),
    );
    let bu: Matrix5<Fixed> = Matrix5Trait::new(
        int(-17),
        int(10),
        int(10),
        int(-17),
        int(10),
        int(-3),
        int(-6),
        int(0),
        int(-3),
        int(-6),
        int(6),
        int(-9),
        int(12),
        int(6),
        int(-9),
        int(21),
        int(-18),
        int(-12),
        int(21),
        int(-18),
        int(-2),
        int(4),
        int(-8),
        int(-2),
        int(4),
    );
    assert!(l.solve_lower_triangular(bl) == Option::Some(x));
    assert!(u.solve_upper_triangular(bu) == Option::Some(x));
    assert!(l.tr_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.tr_solve_upper_triangular(bl) == Option::Some(x));
    assert!(l.ad_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.ad_solve_upper_triangular(bl) == Option::Some(x));
}

/// `Matrix5` against a `Matrix5` right-hand side: the `_unchecked` forms, bit-identical to the
/// checked ones.
#[test]
fn test_solve_matrix5_matrix5_unchecked_exact() {
    let l: Matrix5<Fixed> = Matrix5Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
    );
    let u: Matrix5<Fixed> = Matrix5Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(2),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(0),
        int(0),
        int(1),
        int(2),
        int(-2),
        int(0),
        int(0),
        int(0),
        int(8),
        int(3),
        int(0),
        int(0),
        int(0),
        int(0),
        int(2),
    );
    let x: Matrix5<Fixed> = Matrix5Trait::new(
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(0),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-4),
        int(-2),
        int(1),
        int(4),
        int(-2),
        int(1),
        int(3),
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(-1),
        int(2),
        int(-4),
        int(-1),
        int(2),
    );
    let bl: Matrix5<Fixed> = Matrix5Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(8),
        int(-16),
        int(-4),
        int(8),
        int(-16),
        int(-9),
        int(-3),
        int(12),
        int(-9),
        int(-3),
        int(17),
        int(-10),
        int(8),
        int(17),
        int(-10),
        int(5),
        int(-7),
        int(-10),
        int(5),
        int(-7),
    );
    let bu: Matrix5<Fixed> = Matrix5Trait::new(
        int(-17),
        int(10),
        int(10),
        int(-17),
        int(10),
        int(-3),
        int(-6),
        int(0),
        int(-3),
        int(-6),
        int(6),
        int(-9),
        int(12),
        int(6),
        int(-9),
        int(21),
        int(-18),
        int(-12),
        int(21),
        int(-18),
        int(-2),
        int(4),
        int(-8),
        int(-2),
        int(4),
    );
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    assert!(l.ad_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.ad_solve_upper_triangular_unchecked(bl) == x);
}

/// `Matrix5` against a `Matrix5` right-hand side: the `_mut` forms overwrite `b` with the solution
/// and return `true`.
#[test]
fn test_solve_matrix5_matrix5_mut_exact() {
    let l: Matrix5<Fixed> = Matrix5Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
    );
    let u: Matrix5<Fixed> = Matrix5Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(2),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(0),
        int(0),
        int(1),
        int(2),
        int(-2),
        int(0),
        int(0),
        int(0),
        int(8),
        int(3),
        int(0),
        int(0),
        int(0),
        int(0),
        int(2),
    );
    let x: Matrix5<Fixed> = Matrix5Trait::new(
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(0),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-4),
        int(-2),
        int(1),
        int(4),
        int(-2),
        int(1),
        int(3),
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(-1),
        int(2),
        int(-4),
        int(-1),
        int(2),
    );
    let bl: Matrix5<Fixed> = Matrix5Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(8),
        int(-16),
        int(-4),
        int(8),
        int(-16),
        int(-9),
        int(-3),
        int(12),
        int(-9),
        int(-3),
        int(17),
        int(-10),
        int(8),
        int(17),
        int(-10),
        int(5),
        int(-7),
        int(-10),
        int(5),
        int(-7),
    );
    let bu: Matrix5<Fixed> = Matrix5Trait::new(
        int(-17),
        int(10),
        int(10),
        int(-17),
        int(10),
        int(-3),
        int(-6),
        int(0),
        int(-3),
        int(-6),
        int(6),
        int(-9),
        int(12),
        int(6),
        int(-9),
        int(21),
        int(-18),
        int(-12),
        int(21),
        int(-18),
        int(-2),
        int(4),
        int(-8),
        int(-2),
        int(4),
    );
    let mut b = bl;
    assert!(l.solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(u.solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.tr_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.tr_solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.ad_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.ad_solve_upper_triangular_mut(ref b));
    assert!(b == x);
}

/// `Matrix5` against a `Matrix5` right-hand side: the `_unchecked_mut` forms.
#[test]
fn test_solve_matrix5_matrix5_unchecked_mut_exact() {
    let l: Matrix5<Fixed> = Matrix5Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
    );
    let u: Matrix5<Fixed> = Matrix5Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(2),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(0),
        int(0),
        int(1),
        int(2),
        int(-2),
        int(0),
        int(0),
        int(0),
        int(8),
        int(3),
        int(0),
        int(0),
        int(0),
        int(0),
        int(2),
    );
    let x: Matrix5<Fixed> = Matrix5Trait::new(
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(0),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-4),
        int(-2),
        int(1),
        int(4),
        int(-2),
        int(1),
        int(3),
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(-1),
        int(2),
        int(-4),
        int(-1),
        int(2),
    );
    let bl: Matrix5<Fixed> = Matrix5Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(8),
        int(-16),
        int(-4),
        int(8),
        int(-16),
        int(-9),
        int(-3),
        int(12),
        int(-9),
        int(-3),
        int(17),
        int(-10),
        int(8),
        int(17),
        int(-10),
        int(5),
        int(-7),
        int(-10),
        int(5),
        int(-7),
    );
    let bu: Matrix5<Fixed> = Matrix5Trait::new(
        int(-17),
        int(10),
        int(10),
        int(-17),
        int(10),
        int(-3),
        int(-6),
        int(0),
        int(-3),
        int(-6),
        int(6),
        int(-9),
        int(12),
        int(6),
        int(-9),
        int(21),
        int(-18),
        int(-12),
        int(21),
        int(-18),
        int(-2),
        int(4),
        int(-8),
        int(-2),
        int(4),
    );
    let mut b = bl;
    l.solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    u.solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.tr_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.tr_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.ad_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.ad_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
}

/// `Matrix5` against a `Matrix5` right-hand side: `with_diag` solves `(strict lower triangle of l +
/// 2 I) x = bd` and leaves `2 x` in `b`, like upstream (the quotients by `diag` are never written
/// back); a zero `diag` returns `false` and leaves `b` unchanged.
#[test]
fn test_solve_matrix5_matrix5_with_diag_exact() {
    let l: Matrix5<Fixed> = Matrix5Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
    );
    let bd: Matrix5<Fixed> = Matrix5Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(4),
        int(-8),
        int(-2),
        int(4),
        int(-8),
        int(-11),
        int(-2),
        int(16),
        int(-11),
        int(-2),
        int(-1),
        int(8),
        int(8),
        int(-1),
        int(8),
        int(5),
        int(-7),
        int(-10),
        int(5),
        int(-7),
    );
    let x2: Matrix5<Fixed> = Matrix5Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(4),
        int(-8),
        int(-2),
        int(4),
        int(-8),
        int(-4),
        int(2),
        int(8),
        int(-4),
        int(2),
        int(6),
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(-2),
        int(4),
        int(-8),
        int(-2),
        int(4),
    );
    let mut b = bd;
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(b == x2);
    let mut b = bd;
    l.solve_lower_triangular_with_diag_unchecked_mut(ref b, int(2));
    assert!(b == x2);
    let mut b = bd;
    assert!(!l.solve_lower_triangular_with_diag_mut(ref b, int(0)));
    assert!(b == bd);
}

/// `Matrix5` against a `Matrix5x6` right-hand side: the `Option` forms recover the integer solution
/// exactly (power-of-two diagonal:
/// every quotient is exact).
#[test]
fn test_solve_matrix5_matrix5x6_exact() {
    let l: Matrix5<Fixed> = Matrix5Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
    );
    let u: Matrix5<Fixed> = Matrix5Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(2),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(0),
        int(0),
        int(1),
        int(2),
        int(-2),
        int(0),
        int(0),
        int(0),
        int(8),
        int(3),
        int(0),
        int(0),
        int(0),
        int(0),
        int(2),
    );
    let x: Matrix5x6<Fixed> = Matrix5x6Trait::new(
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(0),
        int(3),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-4),
        int(-1),
        int(-2),
        int(1),
        int(4),
        int(-2),
        int(1),
        int(4),
        int(3),
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(0),
        int(-1),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-4),
    );
    let bl: Matrix5x6<Fixed> = Matrix5x6Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(6),
        int(8),
        int(-16),
        int(-4),
        int(8),
        int(-16),
        int(-4),
        int(-9),
        int(-3),
        int(12),
        int(-9),
        int(-3),
        int(12),
        int(17),
        int(-10),
        int(8),
        int(17),
        int(-10),
        int(8),
        int(5),
        int(-7),
        int(-10),
        int(5),
        int(-7),
        int(-10),
    );
    let bu: Matrix5x6<Fixed> = Matrix5x6Trait::new(
        int(-17),
        int(10),
        int(10),
        int(-17),
        int(10),
        int(10),
        int(-3),
        int(-6),
        int(0),
        int(-3),
        int(-6),
        int(0),
        int(6),
        int(-9),
        int(12),
        int(6),
        int(-9),
        int(12),
        int(21),
        int(-18),
        int(-12),
        int(21),
        int(-18),
        int(-12),
        int(-2),
        int(4),
        int(-8),
        int(-2),
        int(4),
        int(-8),
    );
    assert!(l.solve_lower_triangular(bl) == Option::Some(x));
    assert!(u.solve_upper_triangular(bu) == Option::Some(x));
    assert!(l.tr_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.tr_solve_upper_triangular(bl) == Option::Some(x));
    assert!(l.ad_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.ad_solve_upper_triangular(bl) == Option::Some(x));
}

/// `Matrix5` against a `Matrix5x6` right-hand side: the `_unchecked` forms, bit-identical to the
/// checked ones.
#[test]
fn test_solve_matrix5_matrix5x6_unchecked_exact() {
    let l: Matrix5<Fixed> = Matrix5Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
    );
    let u: Matrix5<Fixed> = Matrix5Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(2),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(0),
        int(0),
        int(1),
        int(2),
        int(-2),
        int(0),
        int(0),
        int(0),
        int(8),
        int(3),
        int(0),
        int(0),
        int(0),
        int(0),
        int(2),
    );
    let x: Matrix5x6<Fixed> = Matrix5x6Trait::new(
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(0),
        int(3),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-4),
        int(-1),
        int(-2),
        int(1),
        int(4),
        int(-2),
        int(1),
        int(4),
        int(3),
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(0),
        int(-1),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-4),
    );
    let bl: Matrix5x6<Fixed> = Matrix5x6Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(6),
        int(8),
        int(-16),
        int(-4),
        int(8),
        int(-16),
        int(-4),
        int(-9),
        int(-3),
        int(12),
        int(-9),
        int(-3),
        int(12),
        int(17),
        int(-10),
        int(8),
        int(17),
        int(-10),
        int(8),
        int(5),
        int(-7),
        int(-10),
        int(5),
        int(-7),
        int(-10),
    );
    let bu: Matrix5x6<Fixed> = Matrix5x6Trait::new(
        int(-17),
        int(10),
        int(10),
        int(-17),
        int(10),
        int(10),
        int(-3),
        int(-6),
        int(0),
        int(-3),
        int(-6),
        int(0),
        int(6),
        int(-9),
        int(12),
        int(6),
        int(-9),
        int(12),
        int(21),
        int(-18),
        int(-12),
        int(21),
        int(-18),
        int(-12),
        int(-2),
        int(4),
        int(-8),
        int(-2),
        int(4),
        int(-8),
    );
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    assert!(l.ad_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.ad_solve_upper_triangular_unchecked(bl) == x);
}

/// `Matrix5` against a `Matrix5x6` right-hand side: the `_mut` forms overwrite `b` with the
/// solution and return `true`.
#[test]
fn test_solve_matrix5_matrix5x6_mut_exact() {
    let l: Matrix5<Fixed> = Matrix5Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
    );
    let u: Matrix5<Fixed> = Matrix5Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(2),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(0),
        int(0),
        int(1),
        int(2),
        int(-2),
        int(0),
        int(0),
        int(0),
        int(8),
        int(3),
        int(0),
        int(0),
        int(0),
        int(0),
        int(2),
    );
    let x: Matrix5x6<Fixed> = Matrix5x6Trait::new(
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(0),
        int(3),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-4),
        int(-1),
        int(-2),
        int(1),
        int(4),
        int(-2),
        int(1),
        int(4),
        int(3),
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(0),
        int(-1),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-4),
    );
    let bl: Matrix5x6<Fixed> = Matrix5x6Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(6),
        int(8),
        int(-16),
        int(-4),
        int(8),
        int(-16),
        int(-4),
        int(-9),
        int(-3),
        int(12),
        int(-9),
        int(-3),
        int(12),
        int(17),
        int(-10),
        int(8),
        int(17),
        int(-10),
        int(8),
        int(5),
        int(-7),
        int(-10),
        int(5),
        int(-7),
        int(-10),
    );
    let bu: Matrix5x6<Fixed> = Matrix5x6Trait::new(
        int(-17),
        int(10),
        int(10),
        int(-17),
        int(10),
        int(10),
        int(-3),
        int(-6),
        int(0),
        int(-3),
        int(-6),
        int(0),
        int(6),
        int(-9),
        int(12),
        int(6),
        int(-9),
        int(12),
        int(21),
        int(-18),
        int(-12),
        int(21),
        int(-18),
        int(-12),
        int(-2),
        int(4),
        int(-8),
        int(-2),
        int(4),
        int(-8),
    );
    let mut b = bl;
    assert!(l.solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(u.solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.tr_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.tr_solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.ad_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.ad_solve_upper_triangular_mut(ref b));
    assert!(b == x);
}

/// `Matrix5` against a `Matrix5x6` right-hand side: the `_unchecked_mut` forms.
#[test]
fn test_solve_matrix5_matrix5x6_unchecked_mut_exact() {
    let l: Matrix5<Fixed> = Matrix5Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
    );
    let u: Matrix5<Fixed> = Matrix5Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(2),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(0),
        int(0),
        int(1),
        int(2),
        int(-2),
        int(0),
        int(0),
        int(0),
        int(8),
        int(3),
        int(0),
        int(0),
        int(0),
        int(0),
        int(2),
    );
    let x: Matrix5x6<Fixed> = Matrix5x6Trait::new(
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(0),
        int(3),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-4),
        int(-1),
        int(-2),
        int(1),
        int(4),
        int(-2),
        int(1),
        int(4),
        int(3),
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(0),
        int(-1),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-4),
    );
    let bl: Matrix5x6<Fixed> = Matrix5x6Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(6),
        int(8),
        int(-16),
        int(-4),
        int(8),
        int(-16),
        int(-4),
        int(-9),
        int(-3),
        int(12),
        int(-9),
        int(-3),
        int(12),
        int(17),
        int(-10),
        int(8),
        int(17),
        int(-10),
        int(8),
        int(5),
        int(-7),
        int(-10),
        int(5),
        int(-7),
        int(-10),
    );
    let bu: Matrix5x6<Fixed> = Matrix5x6Trait::new(
        int(-17),
        int(10),
        int(10),
        int(-17),
        int(10),
        int(10),
        int(-3),
        int(-6),
        int(0),
        int(-3),
        int(-6),
        int(0),
        int(6),
        int(-9),
        int(12),
        int(6),
        int(-9),
        int(12),
        int(21),
        int(-18),
        int(-12),
        int(21),
        int(-18),
        int(-12),
        int(-2),
        int(4),
        int(-8),
        int(-2),
        int(4),
        int(-8),
    );
    let mut b = bl;
    l.solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    u.solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.tr_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.tr_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.ad_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.ad_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
}

/// `Matrix5` against a `Matrix5x6` right-hand side: `with_diag` solves `(strict lower triangle of l
/// + 2 I) x = bd` and leaves `2 x` in `b`, like upstream (the quotients by `diag` are never written
/// back); a zero `diag` returns `false` and leaves `b` unchanged.
#[test]
fn test_solve_matrix5_matrix5x6_with_diag_exact() {
    let l: Matrix5<Fixed> = Matrix5Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
    );
    let bd: Matrix5x6<Fixed> = Matrix5x6Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(6),
        int(4),
        int(-8),
        int(-2),
        int(4),
        int(-8),
        int(-2),
        int(-11),
        int(-2),
        int(16),
        int(-11),
        int(-2),
        int(16),
        int(-1),
        int(8),
        int(8),
        int(-1),
        int(8),
        int(8),
        int(5),
        int(-7),
        int(-10),
        int(5),
        int(-7),
        int(-10),
    );
    let x2: Matrix5x6<Fixed> = Matrix5x6Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(6),
        int(4),
        int(-8),
        int(-2),
        int(4),
        int(-8),
        int(-2),
        int(-4),
        int(2),
        int(8),
        int(-4),
        int(2),
        int(8),
        int(6),
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(-2),
        int(4),
        int(-8),
        int(-2),
        int(4),
        int(-8),
    );
    let mut b = bd;
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(b == x2);
    let mut b = bd;
    l.solve_lower_triangular_with_diag_unchecked_mut(ref b, int(2));
    assert!(b == x2);
    let mut b = bd;
    assert!(!l.solve_lower_triangular_with_diag_mut(ref b, int(0)));
    assert!(b == bd);
}

/// `Matrix6` against a `Vector6` right-hand side: the `Option` forms recover the integer solution
/// exactly (power-of-two diagonal:
/// every quotient is exact).
#[test]
fn test_solve_matrix6_vector6_exact() {
    let l: Matrix6<Fixed> = Matrix6Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(1),
        int(-1),
        int(-3),
        int(4),
    );
    let u: Matrix6<Fixed> = Matrix6Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(2),
        int(-2),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(3),
        int(0),
        int(0),
        int(1),
        int(2),
        int(-2),
        int(1),
        int(0),
        int(0),
        int(0),
        int(8),
        int(3),
        int(-1),
        int(0),
        int(0),
        int(0),
        int(0),
        int(2),
        int(-3),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
    );
    let x: Vector6<Fixed> = Vector6Trait::new(int(-3), int(2), int(-2), int(3), int(-1), int(4));
    let bl: Vector6<Fixed> = Vector6Trait::new(int(-6), int(8), int(-9), int(17), int(5), int(26));
    let bu: Vector6<Fixed> = Vector6Trait::new(
        int(-25), int(9), int(10), int(17), int(-14), int(16),
    );
    assert!(l.solve_lower_triangular(bl) == Option::Some(x));
    assert!(u.solve_upper_triangular(bu) == Option::Some(x));
    assert!(l.tr_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.tr_solve_upper_triangular(bl) == Option::Some(x));
    assert!(l.ad_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.ad_solve_upper_triangular(bl) == Option::Some(x));
}

/// `Matrix6` against a `Vector6` right-hand side: the `_unchecked` forms, bit-identical to the
/// checked ones.
#[test]
fn test_solve_matrix6_vector6_unchecked_exact() {
    let l: Matrix6<Fixed> = Matrix6Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(1),
        int(-1),
        int(-3),
        int(4),
    );
    let u: Matrix6<Fixed> = Matrix6Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(2),
        int(-2),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(3),
        int(0),
        int(0),
        int(1),
        int(2),
        int(-2),
        int(1),
        int(0),
        int(0),
        int(0),
        int(8),
        int(3),
        int(-1),
        int(0),
        int(0),
        int(0),
        int(0),
        int(2),
        int(-3),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
    );
    let x: Vector6<Fixed> = Vector6Trait::new(int(-3), int(2), int(-2), int(3), int(-1), int(4));
    let bl: Vector6<Fixed> = Vector6Trait::new(int(-6), int(8), int(-9), int(17), int(5), int(26));
    let bu: Vector6<Fixed> = Vector6Trait::new(
        int(-25), int(9), int(10), int(17), int(-14), int(16),
    );
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    assert!(l.ad_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.ad_solve_upper_triangular_unchecked(bl) == x);
}

/// `Matrix6` against a `Vector6` right-hand side: the `_mut` forms overwrite `b` with the solution
/// and return `true`.
#[test]
fn test_solve_matrix6_vector6_mut_exact() {
    let l: Matrix6<Fixed> = Matrix6Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(1),
        int(-1),
        int(-3),
        int(4),
    );
    let u: Matrix6<Fixed> = Matrix6Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(2),
        int(-2),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(3),
        int(0),
        int(0),
        int(1),
        int(2),
        int(-2),
        int(1),
        int(0),
        int(0),
        int(0),
        int(8),
        int(3),
        int(-1),
        int(0),
        int(0),
        int(0),
        int(0),
        int(2),
        int(-3),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
    );
    let x: Vector6<Fixed> = Vector6Trait::new(int(-3), int(2), int(-2), int(3), int(-1), int(4));
    let bl: Vector6<Fixed> = Vector6Trait::new(int(-6), int(8), int(-9), int(17), int(5), int(26));
    let bu: Vector6<Fixed> = Vector6Trait::new(
        int(-25), int(9), int(10), int(17), int(-14), int(16),
    );
    let mut b = bl;
    assert!(l.solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(u.solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.tr_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.tr_solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.ad_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.ad_solve_upper_triangular_mut(ref b));
    assert!(b == x);
}

/// `Matrix6` against a `Vector6` right-hand side: the `_unchecked_mut` forms.
#[test]
fn test_solve_matrix6_vector6_unchecked_mut_exact() {
    let l: Matrix6<Fixed> = Matrix6Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(1),
        int(-1),
        int(-3),
        int(4),
    );
    let u: Matrix6<Fixed> = Matrix6Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(2),
        int(-2),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(3),
        int(0),
        int(0),
        int(1),
        int(2),
        int(-2),
        int(1),
        int(0),
        int(0),
        int(0),
        int(8),
        int(3),
        int(-1),
        int(0),
        int(0),
        int(0),
        int(0),
        int(2),
        int(-3),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
    );
    let x: Vector6<Fixed> = Vector6Trait::new(int(-3), int(2), int(-2), int(3), int(-1), int(4));
    let bl: Vector6<Fixed> = Vector6Trait::new(int(-6), int(8), int(-9), int(17), int(5), int(26));
    let bu: Vector6<Fixed> = Vector6Trait::new(
        int(-25), int(9), int(10), int(17), int(-14), int(16),
    );
    let mut b = bl;
    l.solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    u.solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.tr_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.tr_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.ad_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.ad_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
}

/// `Matrix6` against a `Vector6` right-hand side: `with_diag` solves `(strict lower triangle of l +
/// 2 I) x = bd` and leaves `2 x` in `b`, like upstream (the quotients by `diag` are never written
/// back); a zero `diag` returns `false` and leaves `b` unchanged.
#[test]
fn test_solve_matrix6_vector6_with_diag_exact() {
    let l: Matrix6<Fixed> = Matrix6Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(1),
        int(-1),
        int(-3),
        int(4),
    );
    let bd: Vector6<Fixed> = Vector6Trait::new(int(-6), int(4), int(-11), int(-1), int(5), int(18));
    let x2: Vector6<Fixed> = Vector6Trait::new(int(-6), int(4), int(-4), int(6), int(-2), int(8));
    let mut b = bd;
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(b == x2);
    let mut b = bd;
    l.solve_lower_triangular_with_diag_unchecked_mut(ref b, int(2));
    assert!(b == x2);
    let mut b = bd;
    assert!(!l.solve_lower_triangular_with_diag_mut(ref b, int(0)));
    assert!(b == bd);
}

/// `Matrix6` against a `Matrix6x2` right-hand side: the `Option` forms recover the integer solution
/// exactly (power-of-two diagonal:
/// every quotient is exact).
#[test]
fn test_solve_matrix6_matrix6x2_exact() {
    let l: Matrix6<Fixed> = Matrix6Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(1),
        int(-1),
        int(-3),
        int(4),
    );
    let u: Matrix6<Fixed> = Matrix6Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(2),
        int(-2),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(3),
        int(0),
        int(0),
        int(1),
        int(2),
        int(-2),
        int(1),
        int(0),
        int(0),
        int(0),
        int(8),
        int(3),
        int(-1),
        int(0),
        int(0),
        int(0),
        int(0),
        int(2),
        int(-3),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
    );
    let x: Matrix6x2<Fixed> = Matrix6x2Trait::new(
        int(-3),
        int(0),
        int(2),
        int(-4),
        int(-2),
        int(1),
        int(3),
        int(-3),
        int(-1),
        int(2),
        int(4),
        int(-2),
    );
    let bl: Matrix6x2<Fixed> = Matrix6x2Trait::new(
        int(-6),
        int(0),
        int(8),
        int(-16),
        int(-9),
        int(-3),
        int(17),
        int(-10),
        int(5),
        int(-7),
        int(26),
        int(-22),
    );
    let bu: Matrix6x2<Fixed> = Matrix6x2Trait::new(
        int(-25),
        int(14),
        int(9),
        int(-12),
        int(10),
        int(-11),
        int(17),
        int(-16),
        int(-14),
        int(10),
        int(16),
        int(-8),
    );
    assert!(l.solve_lower_triangular(bl) == Option::Some(x));
    assert!(u.solve_upper_triangular(bu) == Option::Some(x));
    assert!(l.tr_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.tr_solve_upper_triangular(bl) == Option::Some(x));
    assert!(l.ad_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.ad_solve_upper_triangular(bl) == Option::Some(x));
}

/// `Matrix6` against a `Matrix6x2` right-hand side: the `_unchecked` forms, bit-identical to the
/// checked ones.
#[test]
fn test_solve_matrix6_matrix6x2_unchecked_exact() {
    let l: Matrix6<Fixed> = Matrix6Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(1),
        int(-1),
        int(-3),
        int(4),
    );
    let u: Matrix6<Fixed> = Matrix6Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(2),
        int(-2),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(3),
        int(0),
        int(0),
        int(1),
        int(2),
        int(-2),
        int(1),
        int(0),
        int(0),
        int(0),
        int(8),
        int(3),
        int(-1),
        int(0),
        int(0),
        int(0),
        int(0),
        int(2),
        int(-3),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
    );
    let x: Matrix6x2<Fixed> = Matrix6x2Trait::new(
        int(-3),
        int(0),
        int(2),
        int(-4),
        int(-2),
        int(1),
        int(3),
        int(-3),
        int(-1),
        int(2),
        int(4),
        int(-2),
    );
    let bl: Matrix6x2<Fixed> = Matrix6x2Trait::new(
        int(-6),
        int(0),
        int(8),
        int(-16),
        int(-9),
        int(-3),
        int(17),
        int(-10),
        int(5),
        int(-7),
        int(26),
        int(-22),
    );
    let bu: Matrix6x2<Fixed> = Matrix6x2Trait::new(
        int(-25),
        int(14),
        int(9),
        int(-12),
        int(10),
        int(-11),
        int(17),
        int(-16),
        int(-14),
        int(10),
        int(16),
        int(-8),
    );
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    assert!(l.ad_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.ad_solve_upper_triangular_unchecked(bl) == x);
}

/// `Matrix6` against a `Matrix6x2` right-hand side: the `_mut` forms overwrite `b` with the
/// solution and return `true`.
#[test]
fn test_solve_matrix6_matrix6x2_mut_exact() {
    let l: Matrix6<Fixed> = Matrix6Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(1),
        int(-1),
        int(-3),
        int(4),
    );
    let u: Matrix6<Fixed> = Matrix6Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(2),
        int(-2),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(3),
        int(0),
        int(0),
        int(1),
        int(2),
        int(-2),
        int(1),
        int(0),
        int(0),
        int(0),
        int(8),
        int(3),
        int(-1),
        int(0),
        int(0),
        int(0),
        int(0),
        int(2),
        int(-3),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
    );
    let x: Matrix6x2<Fixed> = Matrix6x2Trait::new(
        int(-3),
        int(0),
        int(2),
        int(-4),
        int(-2),
        int(1),
        int(3),
        int(-3),
        int(-1),
        int(2),
        int(4),
        int(-2),
    );
    let bl: Matrix6x2<Fixed> = Matrix6x2Trait::new(
        int(-6),
        int(0),
        int(8),
        int(-16),
        int(-9),
        int(-3),
        int(17),
        int(-10),
        int(5),
        int(-7),
        int(26),
        int(-22),
    );
    let bu: Matrix6x2<Fixed> = Matrix6x2Trait::new(
        int(-25),
        int(14),
        int(9),
        int(-12),
        int(10),
        int(-11),
        int(17),
        int(-16),
        int(-14),
        int(10),
        int(16),
        int(-8),
    );
    let mut b = bl;
    assert!(l.solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(u.solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.tr_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.tr_solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.ad_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.ad_solve_upper_triangular_mut(ref b));
    assert!(b == x);
}

/// `Matrix6` against a `Matrix6x2` right-hand side: the `_unchecked_mut` forms.
#[test]
fn test_solve_matrix6_matrix6x2_unchecked_mut_exact() {
    let l: Matrix6<Fixed> = Matrix6Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(1),
        int(-1),
        int(-3),
        int(4),
    );
    let u: Matrix6<Fixed> = Matrix6Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(2),
        int(-2),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(3),
        int(0),
        int(0),
        int(1),
        int(2),
        int(-2),
        int(1),
        int(0),
        int(0),
        int(0),
        int(8),
        int(3),
        int(-1),
        int(0),
        int(0),
        int(0),
        int(0),
        int(2),
        int(-3),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
    );
    let x: Matrix6x2<Fixed> = Matrix6x2Trait::new(
        int(-3),
        int(0),
        int(2),
        int(-4),
        int(-2),
        int(1),
        int(3),
        int(-3),
        int(-1),
        int(2),
        int(4),
        int(-2),
    );
    let bl: Matrix6x2<Fixed> = Matrix6x2Trait::new(
        int(-6),
        int(0),
        int(8),
        int(-16),
        int(-9),
        int(-3),
        int(17),
        int(-10),
        int(5),
        int(-7),
        int(26),
        int(-22),
    );
    let bu: Matrix6x2<Fixed> = Matrix6x2Trait::new(
        int(-25),
        int(14),
        int(9),
        int(-12),
        int(10),
        int(-11),
        int(17),
        int(-16),
        int(-14),
        int(10),
        int(16),
        int(-8),
    );
    let mut b = bl;
    l.solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    u.solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.tr_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.tr_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.ad_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.ad_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
}

/// `Matrix6` against a `Matrix6x2` right-hand side: `with_diag` solves `(strict lower triangle of l
/// + 2 I) x = bd` and leaves `2 x` in `b`, like upstream (the quotients by `diag` are never written
/// back); a zero `diag` returns `false` and leaves `b` unchanged.
#[test]
fn test_solve_matrix6_matrix6x2_with_diag_exact() {
    let l: Matrix6<Fixed> = Matrix6Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(1),
        int(-1),
        int(-3),
        int(4),
    );
    let bd: Matrix6x2<Fixed> = Matrix6x2Trait::new(
        int(-6),
        int(0),
        int(4),
        int(-8),
        int(-11),
        int(-2),
        int(-1),
        int(8),
        int(5),
        int(-7),
        int(18),
        int(-18),
    );
    let x2: Matrix6x2<Fixed> = Matrix6x2Trait::new(
        int(-6),
        int(0),
        int(4),
        int(-8),
        int(-4),
        int(2),
        int(6),
        int(-6),
        int(-2),
        int(4),
        int(8),
        int(-4),
    );
    let mut b = bd;
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(b == x2);
    let mut b = bd;
    l.solve_lower_triangular_with_diag_unchecked_mut(ref b, int(2));
    assert!(b == x2);
    let mut b = bd;
    assert!(!l.solve_lower_triangular_with_diag_mut(ref b, int(0)));
    assert!(b == bd);
}

/// `Matrix6` against a `Matrix6x3` right-hand side: the `Option` forms recover the integer solution
/// exactly (power-of-two diagonal:
/// every quotient is exact).
#[test]
fn test_solve_matrix6_matrix6x3_exact() {
    let l: Matrix6<Fixed> = Matrix6Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(1),
        int(-1),
        int(-3),
        int(4),
    );
    let u: Matrix6<Fixed> = Matrix6Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(2),
        int(-2),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(3),
        int(0),
        int(0),
        int(1),
        int(2),
        int(-2),
        int(1),
        int(0),
        int(0),
        int(0),
        int(8),
        int(3),
        int(-1),
        int(0),
        int(0),
        int(0),
        int(0),
        int(2),
        int(-3),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
    );
    let x: Matrix6x3<Fixed> = Matrix6x3Trait::new(
        int(-3),
        int(0),
        int(3),
        int(2),
        int(-4),
        int(-1),
        int(-2),
        int(1),
        int(4),
        int(3),
        int(-3),
        int(0),
        int(-1),
        int(2),
        int(-4),
        int(4),
        int(-2),
        int(1),
    );
    let bl: Matrix6x3<Fixed> = Matrix6x3Trait::new(
        int(-6),
        int(0),
        int(6),
        int(8),
        int(-16),
        int(-4),
        int(-9),
        int(-3),
        int(12),
        int(17),
        int(-10),
        int(8),
        int(5),
        int(-7),
        int(-10),
        int(26),
        int(-22),
        int(11),
    );
    let bu: Matrix6x3<Fixed> = Matrix6x3Trait::new(
        int(-25),
        int(14),
        int(8),
        int(9),
        int(-12),
        int(3),
        int(10),
        int(-11),
        int(13),
        int(17),
        int(-16),
        int(-13),
        int(-14),
        int(10),
        int(-11),
        int(16),
        int(-8),
        int(4),
    );
    assert!(l.solve_lower_triangular(bl) == Option::Some(x));
    assert!(u.solve_upper_triangular(bu) == Option::Some(x));
    assert!(l.tr_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.tr_solve_upper_triangular(bl) == Option::Some(x));
    assert!(l.ad_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.ad_solve_upper_triangular(bl) == Option::Some(x));
}

/// `Matrix6` against a `Matrix6x3` right-hand side: the `_unchecked` forms, bit-identical to the
/// checked ones.
#[test]
fn test_solve_matrix6_matrix6x3_unchecked_exact() {
    let l: Matrix6<Fixed> = Matrix6Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(1),
        int(-1),
        int(-3),
        int(4),
    );
    let u: Matrix6<Fixed> = Matrix6Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(2),
        int(-2),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(3),
        int(0),
        int(0),
        int(1),
        int(2),
        int(-2),
        int(1),
        int(0),
        int(0),
        int(0),
        int(8),
        int(3),
        int(-1),
        int(0),
        int(0),
        int(0),
        int(0),
        int(2),
        int(-3),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
    );
    let x: Matrix6x3<Fixed> = Matrix6x3Trait::new(
        int(-3),
        int(0),
        int(3),
        int(2),
        int(-4),
        int(-1),
        int(-2),
        int(1),
        int(4),
        int(3),
        int(-3),
        int(0),
        int(-1),
        int(2),
        int(-4),
        int(4),
        int(-2),
        int(1),
    );
    let bl: Matrix6x3<Fixed> = Matrix6x3Trait::new(
        int(-6),
        int(0),
        int(6),
        int(8),
        int(-16),
        int(-4),
        int(-9),
        int(-3),
        int(12),
        int(17),
        int(-10),
        int(8),
        int(5),
        int(-7),
        int(-10),
        int(26),
        int(-22),
        int(11),
    );
    let bu: Matrix6x3<Fixed> = Matrix6x3Trait::new(
        int(-25),
        int(14),
        int(8),
        int(9),
        int(-12),
        int(3),
        int(10),
        int(-11),
        int(13),
        int(17),
        int(-16),
        int(-13),
        int(-14),
        int(10),
        int(-11),
        int(16),
        int(-8),
        int(4),
    );
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    assert!(l.ad_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.ad_solve_upper_triangular_unchecked(bl) == x);
}

/// `Matrix6` against a `Matrix6x3` right-hand side: the `_mut` forms overwrite `b` with the
/// solution and return `true`.
#[test]
fn test_solve_matrix6_matrix6x3_mut_exact() {
    let l: Matrix6<Fixed> = Matrix6Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(1),
        int(-1),
        int(-3),
        int(4),
    );
    let u: Matrix6<Fixed> = Matrix6Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(2),
        int(-2),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(3),
        int(0),
        int(0),
        int(1),
        int(2),
        int(-2),
        int(1),
        int(0),
        int(0),
        int(0),
        int(8),
        int(3),
        int(-1),
        int(0),
        int(0),
        int(0),
        int(0),
        int(2),
        int(-3),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
    );
    let x: Matrix6x3<Fixed> = Matrix6x3Trait::new(
        int(-3),
        int(0),
        int(3),
        int(2),
        int(-4),
        int(-1),
        int(-2),
        int(1),
        int(4),
        int(3),
        int(-3),
        int(0),
        int(-1),
        int(2),
        int(-4),
        int(4),
        int(-2),
        int(1),
    );
    let bl: Matrix6x3<Fixed> = Matrix6x3Trait::new(
        int(-6),
        int(0),
        int(6),
        int(8),
        int(-16),
        int(-4),
        int(-9),
        int(-3),
        int(12),
        int(17),
        int(-10),
        int(8),
        int(5),
        int(-7),
        int(-10),
        int(26),
        int(-22),
        int(11),
    );
    let bu: Matrix6x3<Fixed> = Matrix6x3Trait::new(
        int(-25),
        int(14),
        int(8),
        int(9),
        int(-12),
        int(3),
        int(10),
        int(-11),
        int(13),
        int(17),
        int(-16),
        int(-13),
        int(-14),
        int(10),
        int(-11),
        int(16),
        int(-8),
        int(4),
    );
    let mut b = bl;
    assert!(l.solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(u.solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.tr_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.tr_solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.ad_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.ad_solve_upper_triangular_mut(ref b));
    assert!(b == x);
}

/// `Matrix6` against a `Matrix6x3` right-hand side: the `_unchecked_mut` forms.
#[test]
fn test_solve_matrix6_matrix6x3_unchecked_mut_exact() {
    let l: Matrix6<Fixed> = Matrix6Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(1),
        int(-1),
        int(-3),
        int(4),
    );
    let u: Matrix6<Fixed> = Matrix6Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(2),
        int(-2),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(3),
        int(0),
        int(0),
        int(1),
        int(2),
        int(-2),
        int(1),
        int(0),
        int(0),
        int(0),
        int(8),
        int(3),
        int(-1),
        int(0),
        int(0),
        int(0),
        int(0),
        int(2),
        int(-3),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
    );
    let x: Matrix6x3<Fixed> = Matrix6x3Trait::new(
        int(-3),
        int(0),
        int(3),
        int(2),
        int(-4),
        int(-1),
        int(-2),
        int(1),
        int(4),
        int(3),
        int(-3),
        int(0),
        int(-1),
        int(2),
        int(-4),
        int(4),
        int(-2),
        int(1),
    );
    let bl: Matrix6x3<Fixed> = Matrix6x3Trait::new(
        int(-6),
        int(0),
        int(6),
        int(8),
        int(-16),
        int(-4),
        int(-9),
        int(-3),
        int(12),
        int(17),
        int(-10),
        int(8),
        int(5),
        int(-7),
        int(-10),
        int(26),
        int(-22),
        int(11),
    );
    let bu: Matrix6x3<Fixed> = Matrix6x3Trait::new(
        int(-25),
        int(14),
        int(8),
        int(9),
        int(-12),
        int(3),
        int(10),
        int(-11),
        int(13),
        int(17),
        int(-16),
        int(-13),
        int(-14),
        int(10),
        int(-11),
        int(16),
        int(-8),
        int(4),
    );
    let mut b = bl;
    l.solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    u.solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.tr_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.tr_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.ad_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.ad_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
}

/// `Matrix6` against a `Matrix6x3` right-hand side: `with_diag` solves `(strict lower triangle of l
/// + 2 I) x = bd` and leaves `2 x` in `b`, like upstream (the quotients by `diag` are never written
/// back); a zero `diag` returns `false` and leaves `b` unchanged.
#[test]
fn test_solve_matrix6_matrix6x3_with_diag_exact() {
    let l: Matrix6<Fixed> = Matrix6Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(1),
        int(-1),
        int(-3),
        int(4),
    );
    let bd: Matrix6x3<Fixed> = Matrix6x3Trait::new(
        int(-6),
        int(0),
        int(6),
        int(4),
        int(-8),
        int(-2),
        int(-11),
        int(-2),
        int(16),
        int(-1),
        int(8),
        int(8),
        int(5),
        int(-7),
        int(-10),
        int(18),
        int(-18),
        int(9),
    );
    let x2: Matrix6x3<Fixed> = Matrix6x3Trait::new(
        int(-6),
        int(0),
        int(6),
        int(4),
        int(-8),
        int(-2),
        int(-4),
        int(2),
        int(8),
        int(6),
        int(-6),
        int(0),
        int(-2),
        int(4),
        int(-8),
        int(8),
        int(-4),
        int(2),
    );
    let mut b = bd;
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(b == x2);
    let mut b = bd;
    l.solve_lower_triangular_with_diag_unchecked_mut(ref b, int(2));
    assert!(b == x2);
    let mut b = bd;
    assert!(!l.solve_lower_triangular_with_diag_mut(ref b, int(0)));
    assert!(b == bd);
}

/// `Matrix6` against a `Matrix6x4` right-hand side: the `Option` forms recover the integer solution
/// exactly (power-of-two diagonal:
/// every quotient is exact).
#[test]
fn test_solve_matrix6_matrix6x4_exact() {
    let l: Matrix6<Fixed> = Matrix6Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(1),
        int(-1),
        int(-3),
        int(4),
    );
    let u: Matrix6<Fixed> = Matrix6Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(2),
        int(-2),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(3),
        int(0),
        int(0),
        int(1),
        int(2),
        int(-2),
        int(1),
        int(0),
        int(0),
        int(0),
        int(8),
        int(3),
        int(-1),
        int(0),
        int(0),
        int(0),
        int(0),
        int(2),
        int(-3),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
    );
    let x: Matrix6x4<Fixed> = Matrix6x4Trait::new(
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-2),
        int(1),
        int(4),
        int(-2),
        int(3),
        int(-3),
        int(0),
        int(3),
        int(-1),
        int(2),
        int(-4),
        int(-1),
        int(4),
        int(-2),
        int(1),
        int(4),
    );
    let bl: Matrix6x4<Fixed> = Matrix6x4Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(8),
        int(-16),
        int(-4),
        int(8),
        int(-9),
        int(-3),
        int(12),
        int(-9),
        int(17),
        int(-10),
        int(8),
        int(17),
        int(5),
        int(-7),
        int(-10),
        int(5),
        int(26),
        int(-22),
        int(11),
        int(26),
    );
    let bu: Matrix6x4<Fixed> = Matrix6x4Trait::new(
        int(-25),
        int(14),
        int(8),
        int(-25),
        int(9),
        int(-12),
        int(3),
        int(9),
        int(10),
        int(-11),
        int(13),
        int(10),
        int(17),
        int(-16),
        int(-13),
        int(17),
        int(-14),
        int(10),
        int(-11),
        int(-14),
        int(16),
        int(-8),
        int(4),
        int(16),
    );
    assert!(l.solve_lower_triangular(bl) == Option::Some(x));
    assert!(u.solve_upper_triangular(bu) == Option::Some(x));
    assert!(l.tr_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.tr_solve_upper_triangular(bl) == Option::Some(x));
    assert!(l.ad_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.ad_solve_upper_triangular(bl) == Option::Some(x));
}

/// `Matrix6` against a `Matrix6x4` right-hand side: the `_unchecked` forms, bit-identical to the
/// checked ones.
#[test]
fn test_solve_matrix6_matrix6x4_unchecked_exact() {
    let l: Matrix6<Fixed> = Matrix6Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(1),
        int(-1),
        int(-3),
        int(4),
    );
    let u: Matrix6<Fixed> = Matrix6Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(2),
        int(-2),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(3),
        int(0),
        int(0),
        int(1),
        int(2),
        int(-2),
        int(1),
        int(0),
        int(0),
        int(0),
        int(8),
        int(3),
        int(-1),
        int(0),
        int(0),
        int(0),
        int(0),
        int(2),
        int(-3),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
    );
    let x: Matrix6x4<Fixed> = Matrix6x4Trait::new(
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-2),
        int(1),
        int(4),
        int(-2),
        int(3),
        int(-3),
        int(0),
        int(3),
        int(-1),
        int(2),
        int(-4),
        int(-1),
        int(4),
        int(-2),
        int(1),
        int(4),
    );
    let bl: Matrix6x4<Fixed> = Matrix6x4Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(8),
        int(-16),
        int(-4),
        int(8),
        int(-9),
        int(-3),
        int(12),
        int(-9),
        int(17),
        int(-10),
        int(8),
        int(17),
        int(5),
        int(-7),
        int(-10),
        int(5),
        int(26),
        int(-22),
        int(11),
        int(26),
    );
    let bu: Matrix6x4<Fixed> = Matrix6x4Trait::new(
        int(-25),
        int(14),
        int(8),
        int(-25),
        int(9),
        int(-12),
        int(3),
        int(9),
        int(10),
        int(-11),
        int(13),
        int(10),
        int(17),
        int(-16),
        int(-13),
        int(17),
        int(-14),
        int(10),
        int(-11),
        int(-14),
        int(16),
        int(-8),
        int(4),
        int(16),
    );
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    assert!(l.ad_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.ad_solve_upper_triangular_unchecked(bl) == x);
}

/// `Matrix6` against a `Matrix6x4` right-hand side: the `_mut` forms overwrite `b` with the
/// solution and return `true`.
#[test]
fn test_solve_matrix6_matrix6x4_mut_exact() {
    let l: Matrix6<Fixed> = Matrix6Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(1),
        int(-1),
        int(-3),
        int(4),
    );
    let u: Matrix6<Fixed> = Matrix6Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(2),
        int(-2),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(3),
        int(0),
        int(0),
        int(1),
        int(2),
        int(-2),
        int(1),
        int(0),
        int(0),
        int(0),
        int(8),
        int(3),
        int(-1),
        int(0),
        int(0),
        int(0),
        int(0),
        int(2),
        int(-3),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
    );
    let x: Matrix6x4<Fixed> = Matrix6x4Trait::new(
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-2),
        int(1),
        int(4),
        int(-2),
        int(3),
        int(-3),
        int(0),
        int(3),
        int(-1),
        int(2),
        int(-4),
        int(-1),
        int(4),
        int(-2),
        int(1),
        int(4),
    );
    let bl: Matrix6x4<Fixed> = Matrix6x4Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(8),
        int(-16),
        int(-4),
        int(8),
        int(-9),
        int(-3),
        int(12),
        int(-9),
        int(17),
        int(-10),
        int(8),
        int(17),
        int(5),
        int(-7),
        int(-10),
        int(5),
        int(26),
        int(-22),
        int(11),
        int(26),
    );
    let bu: Matrix6x4<Fixed> = Matrix6x4Trait::new(
        int(-25),
        int(14),
        int(8),
        int(-25),
        int(9),
        int(-12),
        int(3),
        int(9),
        int(10),
        int(-11),
        int(13),
        int(10),
        int(17),
        int(-16),
        int(-13),
        int(17),
        int(-14),
        int(10),
        int(-11),
        int(-14),
        int(16),
        int(-8),
        int(4),
        int(16),
    );
    let mut b = bl;
    assert!(l.solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(u.solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.tr_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.tr_solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.ad_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.ad_solve_upper_triangular_mut(ref b));
    assert!(b == x);
}

/// `Matrix6` against a `Matrix6x4` right-hand side: the `_unchecked_mut` forms.
#[test]
fn test_solve_matrix6_matrix6x4_unchecked_mut_exact() {
    let l: Matrix6<Fixed> = Matrix6Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(1),
        int(-1),
        int(-3),
        int(4),
    );
    let u: Matrix6<Fixed> = Matrix6Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(2),
        int(-2),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(3),
        int(0),
        int(0),
        int(1),
        int(2),
        int(-2),
        int(1),
        int(0),
        int(0),
        int(0),
        int(8),
        int(3),
        int(-1),
        int(0),
        int(0),
        int(0),
        int(0),
        int(2),
        int(-3),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
    );
    let x: Matrix6x4<Fixed> = Matrix6x4Trait::new(
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-2),
        int(1),
        int(4),
        int(-2),
        int(3),
        int(-3),
        int(0),
        int(3),
        int(-1),
        int(2),
        int(-4),
        int(-1),
        int(4),
        int(-2),
        int(1),
        int(4),
    );
    let bl: Matrix6x4<Fixed> = Matrix6x4Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(8),
        int(-16),
        int(-4),
        int(8),
        int(-9),
        int(-3),
        int(12),
        int(-9),
        int(17),
        int(-10),
        int(8),
        int(17),
        int(5),
        int(-7),
        int(-10),
        int(5),
        int(26),
        int(-22),
        int(11),
        int(26),
    );
    let bu: Matrix6x4<Fixed> = Matrix6x4Trait::new(
        int(-25),
        int(14),
        int(8),
        int(-25),
        int(9),
        int(-12),
        int(3),
        int(9),
        int(10),
        int(-11),
        int(13),
        int(10),
        int(17),
        int(-16),
        int(-13),
        int(17),
        int(-14),
        int(10),
        int(-11),
        int(-14),
        int(16),
        int(-8),
        int(4),
        int(16),
    );
    let mut b = bl;
    l.solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    u.solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.tr_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.tr_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.ad_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.ad_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
}

/// `Matrix6` against a `Matrix6x4` right-hand side: `with_diag` solves `(strict lower triangle of l
/// + 2 I) x = bd` and leaves `2 x` in `b`, like upstream (the quotients by `diag` are never written
/// back); a zero `diag` returns `false` and leaves `b` unchanged.
#[test]
fn test_solve_matrix6_matrix6x4_with_diag_exact() {
    let l: Matrix6<Fixed> = Matrix6Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(1),
        int(-1),
        int(-3),
        int(4),
    );
    let bd: Matrix6x4<Fixed> = Matrix6x4Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(4),
        int(-8),
        int(-2),
        int(4),
        int(-11),
        int(-2),
        int(16),
        int(-11),
        int(-1),
        int(8),
        int(8),
        int(-1),
        int(5),
        int(-7),
        int(-10),
        int(5),
        int(18),
        int(-18),
        int(9),
        int(18),
    );
    let x2: Matrix6x4<Fixed> = Matrix6x4Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(4),
        int(-8),
        int(-2),
        int(4),
        int(-4),
        int(2),
        int(8),
        int(-4),
        int(6),
        int(-6),
        int(0),
        int(6),
        int(-2),
        int(4),
        int(-8),
        int(-2),
        int(8),
        int(-4),
        int(2),
        int(8),
    );
    let mut b = bd;
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(b == x2);
    let mut b = bd;
    l.solve_lower_triangular_with_diag_unchecked_mut(ref b, int(2));
    assert!(b == x2);
    let mut b = bd;
    assert!(!l.solve_lower_triangular_with_diag_mut(ref b, int(0)));
    assert!(b == bd);
}

/// `Matrix6` against a `Matrix6x5` right-hand side: the `Option` forms recover the integer solution
/// exactly (power-of-two diagonal:
/// every quotient is exact).
#[test]
fn test_solve_matrix6_matrix6x5_exact() {
    let l: Matrix6<Fixed> = Matrix6Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(1),
        int(-1),
        int(-3),
        int(4),
    );
    let u: Matrix6<Fixed> = Matrix6Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(2),
        int(-2),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(3),
        int(0),
        int(0),
        int(1),
        int(2),
        int(-2),
        int(1),
        int(0),
        int(0),
        int(0),
        int(8),
        int(3),
        int(-1),
        int(0),
        int(0),
        int(0),
        int(0),
        int(2),
        int(-3),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
    );
    let x: Matrix6x5<Fixed> = Matrix6x5Trait::new(
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(0),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-4),
        int(-2),
        int(1),
        int(4),
        int(-2),
        int(1),
        int(3),
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(-1),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(4),
        int(-2),
        int(1),
        int(4),
        int(-2),
    );
    let bl: Matrix6x5<Fixed> = Matrix6x5Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(8),
        int(-16),
        int(-4),
        int(8),
        int(-16),
        int(-9),
        int(-3),
        int(12),
        int(-9),
        int(-3),
        int(17),
        int(-10),
        int(8),
        int(17),
        int(-10),
        int(5),
        int(-7),
        int(-10),
        int(5),
        int(-7),
        int(26),
        int(-22),
        int(11),
        int(26),
        int(-22),
    );
    let bu: Matrix6x5<Fixed> = Matrix6x5Trait::new(
        int(-25),
        int(14),
        int(8),
        int(-25),
        int(14),
        int(9),
        int(-12),
        int(3),
        int(9),
        int(-12),
        int(10),
        int(-11),
        int(13),
        int(10),
        int(-11),
        int(17),
        int(-16),
        int(-13),
        int(17),
        int(-16),
        int(-14),
        int(10),
        int(-11),
        int(-14),
        int(10),
        int(16),
        int(-8),
        int(4),
        int(16),
        int(-8),
    );
    assert!(l.solve_lower_triangular(bl) == Option::Some(x));
    assert!(u.solve_upper_triangular(bu) == Option::Some(x));
    assert!(l.tr_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.tr_solve_upper_triangular(bl) == Option::Some(x));
    assert!(l.ad_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.ad_solve_upper_triangular(bl) == Option::Some(x));
}

/// `Matrix6` against a `Matrix6x5` right-hand side: the `_unchecked` forms, bit-identical to the
/// checked ones.
#[test]
fn test_solve_matrix6_matrix6x5_unchecked_exact() {
    let l: Matrix6<Fixed> = Matrix6Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(1),
        int(-1),
        int(-3),
        int(4),
    );
    let u: Matrix6<Fixed> = Matrix6Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(2),
        int(-2),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(3),
        int(0),
        int(0),
        int(1),
        int(2),
        int(-2),
        int(1),
        int(0),
        int(0),
        int(0),
        int(8),
        int(3),
        int(-1),
        int(0),
        int(0),
        int(0),
        int(0),
        int(2),
        int(-3),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
    );
    let x: Matrix6x5<Fixed> = Matrix6x5Trait::new(
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(0),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-4),
        int(-2),
        int(1),
        int(4),
        int(-2),
        int(1),
        int(3),
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(-1),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(4),
        int(-2),
        int(1),
        int(4),
        int(-2),
    );
    let bl: Matrix6x5<Fixed> = Matrix6x5Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(8),
        int(-16),
        int(-4),
        int(8),
        int(-16),
        int(-9),
        int(-3),
        int(12),
        int(-9),
        int(-3),
        int(17),
        int(-10),
        int(8),
        int(17),
        int(-10),
        int(5),
        int(-7),
        int(-10),
        int(5),
        int(-7),
        int(26),
        int(-22),
        int(11),
        int(26),
        int(-22),
    );
    let bu: Matrix6x5<Fixed> = Matrix6x5Trait::new(
        int(-25),
        int(14),
        int(8),
        int(-25),
        int(14),
        int(9),
        int(-12),
        int(3),
        int(9),
        int(-12),
        int(10),
        int(-11),
        int(13),
        int(10),
        int(-11),
        int(17),
        int(-16),
        int(-13),
        int(17),
        int(-16),
        int(-14),
        int(10),
        int(-11),
        int(-14),
        int(10),
        int(16),
        int(-8),
        int(4),
        int(16),
        int(-8),
    );
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    assert!(l.ad_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.ad_solve_upper_triangular_unchecked(bl) == x);
}

/// `Matrix6` against a `Matrix6x5` right-hand side: the `_mut` forms overwrite `b` with the
/// solution and return `true`.
#[test]
fn test_solve_matrix6_matrix6x5_mut_exact() {
    let l: Matrix6<Fixed> = Matrix6Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(1),
        int(-1),
        int(-3),
        int(4),
    );
    let u: Matrix6<Fixed> = Matrix6Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(2),
        int(-2),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(3),
        int(0),
        int(0),
        int(1),
        int(2),
        int(-2),
        int(1),
        int(0),
        int(0),
        int(0),
        int(8),
        int(3),
        int(-1),
        int(0),
        int(0),
        int(0),
        int(0),
        int(2),
        int(-3),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
    );
    let x: Matrix6x5<Fixed> = Matrix6x5Trait::new(
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(0),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-4),
        int(-2),
        int(1),
        int(4),
        int(-2),
        int(1),
        int(3),
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(-1),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(4),
        int(-2),
        int(1),
        int(4),
        int(-2),
    );
    let bl: Matrix6x5<Fixed> = Matrix6x5Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(8),
        int(-16),
        int(-4),
        int(8),
        int(-16),
        int(-9),
        int(-3),
        int(12),
        int(-9),
        int(-3),
        int(17),
        int(-10),
        int(8),
        int(17),
        int(-10),
        int(5),
        int(-7),
        int(-10),
        int(5),
        int(-7),
        int(26),
        int(-22),
        int(11),
        int(26),
        int(-22),
    );
    let bu: Matrix6x5<Fixed> = Matrix6x5Trait::new(
        int(-25),
        int(14),
        int(8),
        int(-25),
        int(14),
        int(9),
        int(-12),
        int(3),
        int(9),
        int(-12),
        int(10),
        int(-11),
        int(13),
        int(10),
        int(-11),
        int(17),
        int(-16),
        int(-13),
        int(17),
        int(-16),
        int(-14),
        int(10),
        int(-11),
        int(-14),
        int(10),
        int(16),
        int(-8),
        int(4),
        int(16),
        int(-8),
    );
    let mut b = bl;
    assert!(l.solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(u.solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.tr_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.tr_solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.ad_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.ad_solve_upper_triangular_mut(ref b));
    assert!(b == x);
}

/// `Matrix6` against a `Matrix6x5` right-hand side: the `_unchecked_mut` forms.
#[test]
fn test_solve_matrix6_matrix6x5_unchecked_mut_exact() {
    let l: Matrix6<Fixed> = Matrix6Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(1),
        int(-1),
        int(-3),
        int(4),
    );
    let u: Matrix6<Fixed> = Matrix6Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(2),
        int(-2),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(3),
        int(0),
        int(0),
        int(1),
        int(2),
        int(-2),
        int(1),
        int(0),
        int(0),
        int(0),
        int(8),
        int(3),
        int(-1),
        int(0),
        int(0),
        int(0),
        int(0),
        int(2),
        int(-3),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
    );
    let x: Matrix6x5<Fixed> = Matrix6x5Trait::new(
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(0),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-4),
        int(-2),
        int(1),
        int(4),
        int(-2),
        int(1),
        int(3),
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(-1),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(4),
        int(-2),
        int(1),
        int(4),
        int(-2),
    );
    let bl: Matrix6x5<Fixed> = Matrix6x5Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(8),
        int(-16),
        int(-4),
        int(8),
        int(-16),
        int(-9),
        int(-3),
        int(12),
        int(-9),
        int(-3),
        int(17),
        int(-10),
        int(8),
        int(17),
        int(-10),
        int(5),
        int(-7),
        int(-10),
        int(5),
        int(-7),
        int(26),
        int(-22),
        int(11),
        int(26),
        int(-22),
    );
    let bu: Matrix6x5<Fixed> = Matrix6x5Trait::new(
        int(-25),
        int(14),
        int(8),
        int(-25),
        int(14),
        int(9),
        int(-12),
        int(3),
        int(9),
        int(-12),
        int(10),
        int(-11),
        int(13),
        int(10),
        int(-11),
        int(17),
        int(-16),
        int(-13),
        int(17),
        int(-16),
        int(-14),
        int(10),
        int(-11),
        int(-14),
        int(10),
        int(16),
        int(-8),
        int(4),
        int(16),
        int(-8),
    );
    let mut b = bl;
    l.solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    u.solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.tr_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.tr_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.ad_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.ad_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
}

/// `Matrix6` against a `Matrix6x5` right-hand side: `with_diag` solves `(strict lower triangle of l
/// + 2 I) x = bd` and leaves `2 x` in `b`, like upstream (the quotients by `diag` are never written
/// back); a zero `diag` returns `false` and leaves `b` unchanged.
#[test]
fn test_solve_matrix6_matrix6x5_with_diag_exact() {
    let l: Matrix6<Fixed> = Matrix6Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(1),
        int(-1),
        int(-3),
        int(4),
    );
    let bd: Matrix6x5<Fixed> = Matrix6x5Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(4),
        int(-8),
        int(-2),
        int(4),
        int(-8),
        int(-11),
        int(-2),
        int(16),
        int(-11),
        int(-2),
        int(-1),
        int(8),
        int(8),
        int(-1),
        int(8),
        int(5),
        int(-7),
        int(-10),
        int(5),
        int(-7),
        int(18),
        int(-18),
        int(9),
        int(18),
        int(-18),
    );
    let x2: Matrix6x5<Fixed> = Matrix6x5Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(4),
        int(-8),
        int(-2),
        int(4),
        int(-8),
        int(-4),
        int(2),
        int(8),
        int(-4),
        int(2),
        int(6),
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(-2),
        int(4),
        int(-8),
        int(-2),
        int(4),
        int(8),
        int(-4),
        int(2),
        int(8),
        int(-4),
    );
    let mut b = bd;
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(b == x2);
    let mut b = bd;
    l.solve_lower_triangular_with_diag_unchecked_mut(ref b, int(2));
    assert!(b == x2);
    let mut b = bd;
    assert!(!l.solve_lower_triangular_with_diag_mut(ref b, int(0)));
    assert!(b == bd);
}

/// `Matrix6` against a `Matrix6` right-hand side: the `Option` forms recover the integer solution
/// exactly (power-of-two diagonal:
/// every quotient is exact).
#[test]
fn test_solve_matrix6_matrix6_exact() {
    let l: Matrix6<Fixed> = Matrix6Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(1),
        int(-1),
        int(-3),
        int(4),
    );
    let u: Matrix6<Fixed> = Matrix6Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(2),
        int(-2),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(3),
        int(0),
        int(0),
        int(1),
        int(2),
        int(-2),
        int(1),
        int(0),
        int(0),
        int(0),
        int(8),
        int(3),
        int(-1),
        int(0),
        int(0),
        int(0),
        int(0),
        int(2),
        int(-3),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
    );
    let x: Matrix6<Fixed> = Matrix6Trait::new(
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(0),
        int(3),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-4),
        int(-1),
        int(-2),
        int(1),
        int(4),
        int(-2),
        int(1),
        int(4),
        int(3),
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(0),
        int(-1),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-4),
        int(4),
        int(-2),
        int(1),
        int(4),
        int(-2),
        int(1),
    );
    let bl: Matrix6<Fixed> = Matrix6Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(6),
        int(8),
        int(-16),
        int(-4),
        int(8),
        int(-16),
        int(-4),
        int(-9),
        int(-3),
        int(12),
        int(-9),
        int(-3),
        int(12),
        int(17),
        int(-10),
        int(8),
        int(17),
        int(-10),
        int(8),
        int(5),
        int(-7),
        int(-10),
        int(5),
        int(-7),
        int(-10),
        int(26),
        int(-22),
        int(11),
        int(26),
        int(-22),
        int(11),
    );
    let bu: Matrix6<Fixed> = Matrix6Trait::new(
        int(-25),
        int(14),
        int(8),
        int(-25),
        int(14),
        int(8),
        int(9),
        int(-12),
        int(3),
        int(9),
        int(-12),
        int(3),
        int(10),
        int(-11),
        int(13),
        int(10),
        int(-11),
        int(13),
        int(17),
        int(-16),
        int(-13),
        int(17),
        int(-16),
        int(-13),
        int(-14),
        int(10),
        int(-11),
        int(-14),
        int(10),
        int(-11),
        int(16),
        int(-8),
        int(4),
        int(16),
        int(-8),
        int(4),
    );
    assert!(l.solve_lower_triangular(bl) == Option::Some(x));
    assert!(u.solve_upper_triangular(bu) == Option::Some(x));
    assert!(l.tr_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.tr_solve_upper_triangular(bl) == Option::Some(x));
    assert!(l.ad_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.ad_solve_upper_triangular(bl) == Option::Some(x));
}

/// `Matrix6` against a `Matrix6` right-hand side: the `_unchecked` forms, bit-identical to the
/// checked ones.
#[test]
fn test_solve_matrix6_matrix6_unchecked_exact() {
    let l: Matrix6<Fixed> = Matrix6Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(1),
        int(-1),
        int(-3),
        int(4),
    );
    let u: Matrix6<Fixed> = Matrix6Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(2),
        int(-2),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(3),
        int(0),
        int(0),
        int(1),
        int(2),
        int(-2),
        int(1),
        int(0),
        int(0),
        int(0),
        int(8),
        int(3),
        int(-1),
        int(0),
        int(0),
        int(0),
        int(0),
        int(2),
        int(-3),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
    );
    let x: Matrix6<Fixed> = Matrix6Trait::new(
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(0),
        int(3),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-4),
        int(-1),
        int(-2),
        int(1),
        int(4),
        int(-2),
        int(1),
        int(4),
        int(3),
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(0),
        int(-1),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-4),
        int(4),
        int(-2),
        int(1),
        int(4),
        int(-2),
        int(1),
    );
    let bl: Matrix6<Fixed> = Matrix6Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(6),
        int(8),
        int(-16),
        int(-4),
        int(8),
        int(-16),
        int(-4),
        int(-9),
        int(-3),
        int(12),
        int(-9),
        int(-3),
        int(12),
        int(17),
        int(-10),
        int(8),
        int(17),
        int(-10),
        int(8),
        int(5),
        int(-7),
        int(-10),
        int(5),
        int(-7),
        int(-10),
        int(26),
        int(-22),
        int(11),
        int(26),
        int(-22),
        int(11),
    );
    let bu: Matrix6<Fixed> = Matrix6Trait::new(
        int(-25),
        int(14),
        int(8),
        int(-25),
        int(14),
        int(8),
        int(9),
        int(-12),
        int(3),
        int(9),
        int(-12),
        int(3),
        int(10),
        int(-11),
        int(13),
        int(10),
        int(-11),
        int(13),
        int(17),
        int(-16),
        int(-13),
        int(17),
        int(-16),
        int(-13),
        int(-14),
        int(10),
        int(-11),
        int(-14),
        int(10),
        int(-11),
        int(16),
        int(-8),
        int(4),
        int(16),
        int(-8),
        int(4),
    );
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    assert!(l.ad_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.ad_solve_upper_triangular_unchecked(bl) == x);
}

/// `Matrix6` against a `Matrix6` right-hand side: the `_mut` forms overwrite `b` with the solution
/// and return `true`.
#[test]
fn test_solve_matrix6_matrix6_mut_exact() {
    let l: Matrix6<Fixed> = Matrix6Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(1),
        int(-1),
        int(-3),
        int(4),
    );
    let u: Matrix6<Fixed> = Matrix6Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(2),
        int(-2),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(3),
        int(0),
        int(0),
        int(1),
        int(2),
        int(-2),
        int(1),
        int(0),
        int(0),
        int(0),
        int(8),
        int(3),
        int(-1),
        int(0),
        int(0),
        int(0),
        int(0),
        int(2),
        int(-3),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
    );
    let x: Matrix6<Fixed> = Matrix6Trait::new(
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(0),
        int(3),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-4),
        int(-1),
        int(-2),
        int(1),
        int(4),
        int(-2),
        int(1),
        int(4),
        int(3),
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(0),
        int(-1),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-4),
        int(4),
        int(-2),
        int(1),
        int(4),
        int(-2),
        int(1),
    );
    let bl: Matrix6<Fixed> = Matrix6Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(6),
        int(8),
        int(-16),
        int(-4),
        int(8),
        int(-16),
        int(-4),
        int(-9),
        int(-3),
        int(12),
        int(-9),
        int(-3),
        int(12),
        int(17),
        int(-10),
        int(8),
        int(17),
        int(-10),
        int(8),
        int(5),
        int(-7),
        int(-10),
        int(5),
        int(-7),
        int(-10),
        int(26),
        int(-22),
        int(11),
        int(26),
        int(-22),
        int(11),
    );
    let bu: Matrix6<Fixed> = Matrix6Trait::new(
        int(-25),
        int(14),
        int(8),
        int(-25),
        int(14),
        int(8),
        int(9),
        int(-12),
        int(3),
        int(9),
        int(-12),
        int(3),
        int(10),
        int(-11),
        int(13),
        int(10),
        int(-11),
        int(13),
        int(17),
        int(-16),
        int(-13),
        int(17),
        int(-16),
        int(-13),
        int(-14),
        int(10),
        int(-11),
        int(-14),
        int(10),
        int(-11),
        int(16),
        int(-8),
        int(4),
        int(16),
        int(-8),
        int(4),
    );
    let mut b = bl;
    assert!(l.solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(u.solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.tr_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.tr_solve_upper_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bu;
    assert!(l.ad_solve_lower_triangular_mut(ref b));
    assert!(b == x);
    let mut b = bl;
    assert!(u.ad_solve_upper_triangular_mut(ref b));
    assert!(b == x);
}

/// `Matrix6` against a `Matrix6` right-hand side: the `_unchecked_mut` forms.
#[test]
fn test_solve_matrix6_matrix6_unchecked_mut_exact() {
    let l: Matrix6<Fixed> = Matrix6Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(1),
        int(-1),
        int(-3),
        int(4),
    );
    let u: Matrix6<Fixed> = Matrix6Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(2),
        int(-2),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(3),
        int(0),
        int(0),
        int(1),
        int(2),
        int(-2),
        int(1),
        int(0),
        int(0),
        int(0),
        int(8),
        int(3),
        int(-1),
        int(0),
        int(0),
        int(0),
        int(0),
        int(2),
        int(-3),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
    );
    let x: Matrix6<Fixed> = Matrix6Trait::new(
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(0),
        int(3),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-4),
        int(-1),
        int(-2),
        int(1),
        int(4),
        int(-2),
        int(1),
        int(4),
        int(3),
        int(-3),
        int(0),
        int(3),
        int(-3),
        int(0),
        int(-1),
        int(2),
        int(-4),
        int(-1),
        int(2),
        int(-4),
        int(4),
        int(-2),
        int(1),
        int(4),
        int(-2),
        int(1),
    );
    let bl: Matrix6<Fixed> = Matrix6Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(6),
        int(8),
        int(-16),
        int(-4),
        int(8),
        int(-16),
        int(-4),
        int(-9),
        int(-3),
        int(12),
        int(-9),
        int(-3),
        int(12),
        int(17),
        int(-10),
        int(8),
        int(17),
        int(-10),
        int(8),
        int(5),
        int(-7),
        int(-10),
        int(5),
        int(-7),
        int(-10),
        int(26),
        int(-22),
        int(11),
        int(26),
        int(-22),
        int(11),
    );
    let bu: Matrix6<Fixed> = Matrix6Trait::new(
        int(-25),
        int(14),
        int(8),
        int(-25),
        int(14),
        int(8),
        int(9),
        int(-12),
        int(3),
        int(9),
        int(-12),
        int(3),
        int(10),
        int(-11),
        int(13),
        int(10),
        int(-11),
        int(13),
        int(17),
        int(-16),
        int(-13),
        int(17),
        int(-16),
        int(-13),
        int(-14),
        int(10),
        int(-11),
        int(-14),
        int(10),
        int(-11),
        int(16),
        int(-8),
        int(4),
        int(16),
        int(-8),
        int(4),
    );
    let mut b = bl;
    l.solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    u.solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.tr_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.tr_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bu;
    l.ad_solve_lower_triangular_unchecked_mut(ref b);
    assert!(b == x);
    let mut b = bl;
    u.ad_solve_upper_triangular_unchecked_mut(ref b);
    assert!(b == x);
}

/// `Matrix6` against a `Matrix6` right-hand side: `with_diag` solves `(strict lower triangle of l +
/// 2 I) x = bd` and leaves `2 x` in `b`, like upstream (the quotients by `diag` are never written
/// back); a zero `diag` returns `false` and leaves `b` unchanged.
#[test]
fn test_solve_matrix6_matrix6_with_diag_exact() {
    let l: Matrix6<Fixed> = Matrix6Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(1),
        int(-1),
        int(-3),
        int(4),
    );
    let bd: Matrix6<Fixed> = Matrix6Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(6),
        int(4),
        int(-8),
        int(-2),
        int(4),
        int(-8),
        int(-2),
        int(-11),
        int(-2),
        int(16),
        int(-11),
        int(-2),
        int(16),
        int(-1),
        int(8),
        int(8),
        int(-1),
        int(8),
        int(8),
        int(5),
        int(-7),
        int(-10),
        int(5),
        int(-7),
        int(-10),
        int(18),
        int(-18),
        int(9),
        int(18),
        int(-18),
        int(9),
    );
    let x2: Matrix6<Fixed> = Matrix6Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(6),
        int(4),
        int(-8),
        int(-2),
        int(4),
        int(-8),
        int(-2),
        int(-4),
        int(2),
        int(8),
        int(-4),
        int(2),
        int(8),
        int(6),
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(-2),
        int(4),
        int(-8),
        int(-2),
        int(4),
        int(-8),
        int(8),
        int(-4),
        int(2),
        int(8),
        int(-4),
        int(2),
    );
    let mut b = bd;
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(b == x2);
    let mut b = bd;
    l.solve_lower_triangular_with_diag_unchecked_mut(ref b, int(2));
    assert!(b == x2);
    let mut b = bd;
    assert!(!l.solve_lower_triangular_with_diag_mut(ref b, int(0)));
    assert!(b == bd);
}

/// A zero on the diagonal of a 1x1 triangle: the checked forms return `None` / `false` and
/// leave `b` unchanged (upstream stops at the first zero pivot).
#[test]
fn test_solve_matrix1_zero_diagonal() {
    let l: Matrix1<Fixed> = Matrix1Trait::new(int(0));
    let u: Matrix1<Fixed> = Matrix1Trait::new(int(0));
    let b0: Matrix1<Fixed> = Matrix1Trait::new(int(1));
    assert!(l.solve_lower_triangular(b0).is_none());
    assert!(u.solve_upper_triangular(b0).is_none());
    assert!(l.tr_solve_lower_triangular(b0).is_none());
    assert!(u.tr_solve_upper_triangular(b0).is_none());
    assert!(l.ad_solve_lower_triangular(b0).is_none());
    assert!(u.ad_solve_upper_triangular(b0).is_none());
    let mut b = b0;
    assert!(!l.solve_lower_triangular_mut(ref b));
    assert!(!u.solve_upper_triangular_mut(ref b));
    assert!(!l.tr_solve_lower_triangular_mut(ref b));
    assert!(!u.tr_solve_upper_triangular_mut(ref b));
    assert!(!l.ad_solve_lower_triangular_mut(ref b));
    assert!(!u.ad_solve_upper_triangular_mut(ref b));
    assert!(b == b0);
}

/// The unchecked lower solve of a 1x1 triangle with a zero pivot divides by zero: the
/// scalar's panic (upstream: infinities / NaN).
#[test]
#[should_panic]
fn test_solve_matrix1_unchecked_zero_diagonal_panics() {
    let l: Matrix1<Fixed> = Matrix1Trait::new(int(0));
    let b0: Matrix1<Fixed> = Matrix1Trait::new(int(1));
    let _x = l.solve_lower_triangular_unchecked(b0);
}

/// A zero on the diagonal of a 2x2 triangle: the checked forms return `None` / `false` and
/// leave `b` unchanged (upstream stops at the first zero pivot).
#[test]
fn test_solve_matrix2_zero_diagonal() {
    let l: Matrix2<Fixed> = Matrix2Trait::new(int(2), int(0), int(0), int(0));
    let u: Matrix2<Fixed> = Matrix2Trait::new(int(2), int(0), int(0), int(0));
    let b0: Vector2<Fixed> = Vector2Trait::new(int(1), int(2));
    assert!(l.solve_lower_triangular(b0).is_none());
    assert!(u.solve_upper_triangular(b0).is_none());
    assert!(l.tr_solve_lower_triangular(b0).is_none());
    assert!(u.tr_solve_upper_triangular(b0).is_none());
    assert!(l.ad_solve_lower_triangular(b0).is_none());
    assert!(u.ad_solve_upper_triangular(b0).is_none());
    let mut b = b0;
    assert!(!l.solve_lower_triangular_mut(ref b));
    assert!(!u.solve_upper_triangular_mut(ref b));
    assert!(!l.tr_solve_lower_triangular_mut(ref b));
    assert!(!u.tr_solve_upper_triangular_mut(ref b));
    assert!(!l.ad_solve_lower_triangular_mut(ref b));
    assert!(!u.ad_solve_upper_triangular_mut(ref b));
    assert!(b == b0);
}

/// The unchecked lower solve of a 2x2 triangle with a zero pivot divides by zero: the
/// scalar's panic (upstream: infinities / NaN).
#[test]
#[should_panic]
fn test_solve_matrix2_unchecked_zero_diagonal_panics() {
    let l: Matrix2<Fixed> = Matrix2Trait::new(int(2), int(0), int(0), int(0));
    let b0: Vector2<Fixed> = Vector2Trait::new(int(1), int(2));
    let _x = l.solve_lower_triangular_unchecked(b0);
}

/// A zero on the diagonal of a 3x3 triangle: the checked forms return `None` / `false` and
/// leave `b` unchanged (upstream stops at the first zero pivot).
#[test]
fn test_solve_matrix3_zero_diagonal() {
    let l: Matrix3<Fixed> = Matrix3Trait::new(
        int(2), int(0), int(0), int(0), int(0), int(0), int(3), int(1), int(1),
    );
    let u: Matrix3<Fixed> = Matrix3Trait::new(
        int(2), int(0), int(3), int(0), int(0), int(1), int(0), int(0), int(1),
    );
    let b0: Vector3<Fixed> = Vector3Trait::new(int(1), int(2), int(3));
    assert!(l.solve_lower_triangular(b0).is_none());
    assert!(u.solve_upper_triangular(b0).is_none());
    assert!(l.tr_solve_lower_triangular(b0).is_none());
    assert!(u.tr_solve_upper_triangular(b0).is_none());
    assert!(l.ad_solve_lower_triangular(b0).is_none());
    assert!(u.ad_solve_upper_triangular(b0).is_none());
    let mut b = b0;
    assert!(!l.solve_lower_triangular_mut(ref b));
    assert!(!u.solve_upper_triangular_mut(ref b));
    assert!(!l.tr_solve_lower_triangular_mut(ref b));
    assert!(!u.tr_solve_upper_triangular_mut(ref b));
    assert!(!l.ad_solve_lower_triangular_mut(ref b));
    assert!(!u.ad_solve_upper_triangular_mut(ref b));
    assert!(b == b0);
}

/// The unchecked lower solve of a 3x3 triangle with a zero pivot divides by zero: the
/// scalar's panic (upstream: infinities / NaN).
#[test]
#[should_panic]
fn test_solve_matrix3_unchecked_zero_diagonal_panics() {
    let l: Matrix3<Fixed> = Matrix3Trait::new(
        int(2), int(0), int(0), int(0), int(0), int(0), int(3), int(1), int(1),
    );
    let b0: Vector3<Fixed> = Vector3Trait::new(int(1), int(2), int(3));
    let _x = l.solve_lower_triangular_unchecked(b0);
}

/// A zero on the diagonal of a 4x4 triangle: the checked forms return `None` / `false` and
/// leave `b` unchanged (upstream stops at the first zero pivot).
#[test]
fn test_solve_matrix4_zero_diagonal() {
    let l: Matrix4<Fixed> = Matrix4Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(3),
        int(1),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
    );
    let u: Matrix4<Fixed> = Matrix4Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(0),
        int(0),
        int(2),
        int(0),
        int(0),
        int(0),
        int(8),
    );
    let b0: Vector4<Fixed> = Vector4Trait::new(int(1), int(2), int(3), int(4));
    assert!(l.solve_lower_triangular(b0).is_none());
    assert!(u.solve_upper_triangular(b0).is_none());
    assert!(l.tr_solve_lower_triangular(b0).is_none());
    assert!(u.tr_solve_upper_triangular(b0).is_none());
    assert!(l.ad_solve_lower_triangular(b0).is_none());
    assert!(u.ad_solve_upper_triangular(b0).is_none());
    let mut b = b0;
    assert!(!l.solve_lower_triangular_mut(ref b));
    assert!(!u.solve_upper_triangular_mut(ref b));
    assert!(!l.tr_solve_lower_triangular_mut(ref b));
    assert!(!u.tr_solve_upper_triangular_mut(ref b));
    assert!(!l.ad_solve_lower_triangular_mut(ref b));
    assert!(!u.ad_solve_upper_triangular_mut(ref b));
    assert!(b == b0);
}

/// The unchecked lower solve of a 4x4 triangle with a zero pivot divides by zero: the
/// scalar's panic (upstream: infinities / NaN).
#[test]
#[should_panic]
fn test_solve_matrix4_unchecked_zero_diagonal_panics() {
    let l: Matrix4<Fixed> = Matrix4Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(3),
        int(1),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
    );
    let b0: Vector4<Fixed> = Vector4Trait::new(int(1), int(2), int(3), int(4));
    let _x = l.solve_lower_triangular_unchecked(b0);
}

/// A zero on the diagonal of a 5x5 triangle: the checked forms return `None` / `false` and
/// leave `b` unchanged (upstream stops at the first zero pivot).
#[test]
fn test_solve_matrix5_zero_diagonal() {
    let l: Matrix5<Fixed> = Matrix5Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(0),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
    );
    let u: Matrix5<Fixed> = Matrix5Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(2),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(0),
        int(0),
        int(0),
        int(2),
        int(-2),
        int(0),
        int(0),
        int(0),
        int(8),
        int(3),
        int(0),
        int(0),
        int(0),
        int(0),
        int(2),
    );
    let b0: Vector5<Fixed> = Vector5Trait::new(int(1), int(2), int(3), int(4), int(5));
    assert!(l.solve_lower_triangular(b0).is_none());
    assert!(u.solve_upper_triangular(b0).is_none());
    assert!(l.tr_solve_lower_triangular(b0).is_none());
    assert!(u.tr_solve_upper_triangular(b0).is_none());
    assert!(l.ad_solve_lower_triangular(b0).is_none());
    assert!(u.ad_solve_upper_triangular(b0).is_none());
    let mut b = b0;
    assert!(!l.solve_lower_triangular_mut(ref b));
    assert!(!u.solve_upper_triangular_mut(ref b));
    assert!(!l.tr_solve_lower_triangular_mut(ref b));
    assert!(!u.tr_solve_upper_triangular_mut(ref b));
    assert!(!l.ad_solve_lower_triangular_mut(ref b));
    assert!(!u.ad_solve_upper_triangular_mut(ref b));
    assert!(b == b0);
}

/// The unchecked lower solve of a 5x5 triangle with a zero pivot divides by zero: the
/// scalar's panic (upstream: infinities / NaN).
#[test]
#[should_panic]
fn test_solve_matrix5_unchecked_zero_diagonal_panics() {
    let l: Matrix5<Fixed> = Matrix5Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(0),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
    );
    let b0: Vector5<Fixed> = Vector5Trait::new(int(1), int(2), int(3), int(4), int(5));
    let _x = l.solve_lower_triangular_unchecked(b0);
}

/// A zero on the diagonal of a 6x6 triangle: the checked forms return `None` / `false` and
/// leave `b` unchanged (upstream stops at the first zero pivot).
#[test]
fn test_solve_matrix6_zero_diagonal() {
    let l: Matrix6<Fixed> = Matrix6Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(0),
        int(0),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(1),
        int(-1),
        int(-3),
        int(4),
    );
    let u: Matrix6<Fixed> = Matrix6Trait::new(
        int(2),
        int(0),
        int(3),
        int(-1),
        int(2),
        int(-2),
        int(0),
        int(4),
        int(1),
        int(-3),
        int(0),
        int(3),
        int(0),
        int(0),
        int(1),
        int(2),
        int(-2),
        int(1),
        int(0),
        int(0),
        int(0),
        int(0),
        int(3),
        int(-1),
        int(0),
        int(0),
        int(0),
        int(0),
        int(2),
        int(-3),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
    );
    let b0: Vector6<Fixed> = Vector6Trait::new(int(1), int(2), int(3), int(4), int(5), int(6));
    assert!(l.solve_lower_triangular(b0).is_none());
    assert!(u.solve_upper_triangular(b0).is_none());
    assert!(l.tr_solve_lower_triangular(b0).is_none());
    assert!(u.tr_solve_upper_triangular(b0).is_none());
    assert!(l.ad_solve_lower_triangular(b0).is_none());
    assert!(u.ad_solve_upper_triangular(b0).is_none());
    let mut b = b0;
    assert!(!l.solve_lower_triangular_mut(ref b));
    assert!(!u.solve_upper_triangular_mut(ref b));
    assert!(!l.tr_solve_lower_triangular_mut(ref b));
    assert!(!u.tr_solve_upper_triangular_mut(ref b));
    assert!(!l.ad_solve_lower_triangular_mut(ref b));
    assert!(!u.ad_solve_upper_triangular_mut(ref b));
    assert!(b == b0);
}

/// The unchecked lower solve of a 6x6 triangle with a zero pivot divides by zero: the
/// scalar's panic (upstream: infinities / NaN).
#[test]
#[should_panic]
fn test_solve_matrix6_unchecked_zero_diagonal_panics() {
    let l: Matrix6<Fixed> = Matrix6Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(0),
        int(0),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(1),
        int(-1),
        int(-3),
        int(4),
    );
    let b0: Vector6<Fixed> = Vector6Trait::new(int(1), int(2), int(3), int(4), int(5), int(6));
    let _x = l.solve_lower_triangular_unchecked(b0);
}
