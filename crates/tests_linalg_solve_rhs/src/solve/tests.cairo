//! Tests of `MatrixSolve` (upstream `SquareMatrix::solve_*`, `src/linalg/solve.rs`): the kernels of
//! every (square, right-hand side with as many rows) pair and every one of the 26 forms on the
//! vector right-hand sides (the forms are one blanket impl over the kernels), on designs whose
//! solution is exact (power-of-two diagonal, integer `x`, `b = t x` exact), the zero-diagonal
//! semantics of the checked forms and the division panic of the unchecked ones. The oracle vectors
//! (`oracle_solve`) check the rounding on real-valued inputs. Every operand goes through
//! `black_box`: constant operands make the compiler specialise each kernel per test.

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
    RowVector6Trait,
};
use nalgebra_testing::black_box;
use nalgebra_tests_utils::int;

/// `Matrix1` against a `RowVector2` right-hand side: the three kernels of the pair (`lower`,
/// `upper`, `with_diag`; the `tr_` / `ad_` forms are the kernels of the transpose) recover the
/// integer solution exactly (power-of-two diagonal: every quotient is exact); `with_diag` solves
/// `(strict lower triangle of l + 2 I) x =
/// bd` and leaves `2 x` in `b`, like upstream.
#[test]
fn test_solve_matrix1_row_vector2_exact() {
    let l: Matrix1<Fixed> = black_box(Matrix1Trait::new(int(2)));
    let u: Matrix1<Fixed> = black_box(Matrix1Trait::new(int(2)));
    let x: RowVector2<Fixed> = black_box(RowVector2Trait::new(int(-3), int(0)));
    let bl: RowVector2<Fixed> = black_box(RowVector2Trait::new(int(-6), int(0)));
    let bu: RowVector2<Fixed> = black_box(RowVector2Trait::new(int(-6), int(0)));
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    let mut b: RowVector2<Fixed> = black_box(RowVector2Trait::new(int(-6), int(0)));
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(b == RowVector2Trait::new(int(-6), int(0)));
}

/// `Matrix1` against a `RowVector3` right-hand side: the three kernels of the pair (`lower`,
/// `upper`, `with_diag`; the `tr_` / `ad_` forms are the kernels of the transpose) recover the
/// integer solution exactly (power-of-two diagonal: every quotient is exact); `with_diag` solves
/// `(strict lower triangle of l + 2 I) x =
/// bd` and leaves `2 x` in `b`, like upstream.
#[test]
fn test_solve_matrix1_row_vector3_exact() {
    let l: Matrix1<Fixed> = black_box(Matrix1Trait::new(int(2)));
    let u: Matrix1<Fixed> = black_box(Matrix1Trait::new(int(2)));
    let x: RowVector3<Fixed> = black_box(RowVector3Trait::new(int(-3), int(0), int(3)));
    let bl: RowVector3<Fixed> = black_box(RowVector3Trait::new(int(-6), int(0), int(6)));
    let bu: RowVector3<Fixed> = black_box(RowVector3Trait::new(int(-6), int(0), int(6)));
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    let mut b: RowVector3<Fixed> = black_box(RowVector3Trait::new(int(-6), int(0), int(6)));
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(b == RowVector3Trait::new(int(-6), int(0), int(6)));
}

/// `Matrix1` against a `RowVector4` right-hand side: the three kernels of the pair (`lower`,
/// `upper`, `with_diag`; the `tr_` / `ad_` forms are the kernels of the transpose) recover the
/// integer solution exactly (power-of-two diagonal: every quotient is exact); `with_diag` solves
/// `(strict lower triangle of l + 2 I) x =
/// bd` and leaves `2 x` in `b`, like upstream.
#[test]
fn test_solve_matrix1_row_vector4_exact() {
    let l: Matrix1<Fixed> = black_box(Matrix1Trait::new(int(2)));
    let u: Matrix1<Fixed> = black_box(Matrix1Trait::new(int(2)));
    let x: RowVector4<Fixed> = black_box(RowVector4Trait::new(int(-3), int(0), int(3), int(-3)));
    let bl: RowVector4<Fixed> = black_box(RowVector4Trait::new(int(-6), int(0), int(6), int(-6)));
    let bu: RowVector4<Fixed> = black_box(RowVector4Trait::new(int(-6), int(0), int(6), int(-6)));
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    let mut b: RowVector4<Fixed> = black_box(
        RowVector4Trait::new(int(-6), int(0), int(6), int(-6)),
    );
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(b == RowVector4Trait::new(int(-6), int(0), int(6), int(-6)));
}

/// `Matrix1` against a `RowVector5` right-hand side: the three kernels of the pair (`lower`,
/// `upper`, `with_diag`; the `tr_` / `ad_` forms are the kernels of the transpose) recover the
/// integer solution exactly (power-of-two diagonal: every quotient is exact); `with_diag` solves
/// `(strict lower triangle of l + 2 I) x =
/// bd` and leaves `2 x` in `b`, like upstream.
#[test]
fn test_solve_matrix1_row_vector5_exact() {
    let l: Matrix1<Fixed> = black_box(Matrix1Trait::new(int(2)));
    let u: Matrix1<Fixed> = black_box(Matrix1Trait::new(int(2)));
    let x: RowVector5<Fixed> = black_box(
        RowVector5Trait::new(int(-3), int(0), int(3), int(-3), int(0)),
    );
    let bl: RowVector5<Fixed> = black_box(
        RowVector5Trait::new(int(-6), int(0), int(6), int(-6), int(0)),
    );
    let bu: RowVector5<Fixed> = black_box(
        RowVector5Trait::new(int(-6), int(0), int(6), int(-6), int(0)),
    );
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    let mut b: RowVector5<Fixed> = black_box(
        RowVector5Trait::new(int(-6), int(0), int(6), int(-6), int(0)),
    );
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(b == RowVector5Trait::new(int(-6), int(0), int(6), int(-6), int(0)));
}

/// `Matrix1` against a `RowVector6` right-hand side: the three kernels of the pair (`lower`,
/// `upper`, `with_diag`; the `tr_` / `ad_` forms are the kernels of the transpose) recover the
/// integer solution exactly (power-of-two diagonal: every quotient is exact); `with_diag` solves
/// `(strict lower triangle of l + 2 I) x =
/// bd` and leaves `2 x` in `b`, like upstream.
#[test]
fn test_solve_matrix1_row_vector6_exact() {
    let l: Matrix1<Fixed> = black_box(Matrix1Trait::new(int(2)));
    let u: Matrix1<Fixed> = black_box(Matrix1Trait::new(int(2)));
    let x: RowVector6<Fixed> = black_box(
        RowVector6Trait::new(int(-3), int(0), int(3), int(-3), int(0), int(3)),
    );
    let bl: RowVector6<Fixed> = black_box(
        RowVector6Trait::new(int(-6), int(0), int(6), int(-6), int(0), int(6)),
    );
    let bu: RowVector6<Fixed> = black_box(
        RowVector6Trait::new(int(-6), int(0), int(6), int(-6), int(0), int(6)),
    );
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    let mut b: RowVector6<Fixed> = black_box(
        RowVector6Trait::new(int(-6), int(0), int(6), int(-6), int(0), int(6)),
    );
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(b == RowVector6Trait::new(int(-6), int(0), int(6), int(-6), int(0), int(6)));
}

/// `Matrix2` against a `Matrix2` right-hand side: the three kernels of the pair (`lower`, `upper`,
/// `with_diag`; the `tr_` / `ad_` forms are the kernels of the transpose) recover the integer
/// solution exactly (power-of-two diagonal: every quotient is exact); `with_diag` solves `(strict
/// lower triangle of l + 2 I) x =
/// bd` and leaves `2 x` in `b`, like upstream.
#[test]
fn test_solve_matrix2_matrix2_exact() {
    let l: Matrix2<Fixed> = black_box(Matrix2Trait::new(int(2), int(0), int(0), int(4)));
    let u: Matrix2<Fixed> = black_box(Matrix2Trait::new(int(2), int(0), int(0), int(4)));
    let x: Matrix2<Fixed> = black_box(Matrix2Trait::new(int(-3), int(0), int(2), int(-4)));
    let bl: Matrix2<Fixed> = black_box(Matrix2Trait::new(int(-6), int(0), int(8), int(-16)));
    let bu: Matrix2<Fixed> = black_box(Matrix2Trait::new(int(-6), int(0), int(8), int(-16)));
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    let mut b: Matrix2<Fixed> = black_box(Matrix2Trait::new(int(-6), int(0), int(4), int(-8)));
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(b == Matrix2Trait::new(int(-6), int(0), int(4), int(-8)));
}

/// `Matrix2` against a `Matrix2x3` right-hand side: the three kernels of the pair (`lower`,
/// `upper`, `with_diag`; the `tr_` / `ad_` forms are the kernels of the transpose) recover the
/// integer solution exactly (power-of-two diagonal: every quotient is exact); `with_diag` solves
/// `(strict lower triangle of l + 2 I) x =
/// bd` and leaves `2 x` in `b`, like upstream.
#[test]
fn test_solve_matrix2_matrix2x3_exact() {
    let l: Matrix2<Fixed> = black_box(Matrix2Trait::new(int(2), int(0), int(0), int(4)));
    let u: Matrix2<Fixed> = black_box(Matrix2Trait::new(int(2), int(0), int(0), int(4)));
    let x: Matrix2x3<Fixed> = black_box(
        Matrix2x3Trait::new(int(-3), int(0), int(3), int(2), int(-4), int(-1)),
    );
    let bl: Matrix2x3<Fixed> = black_box(
        Matrix2x3Trait::new(int(-6), int(0), int(6), int(8), int(-16), int(-4)),
    );
    let bu: Matrix2x3<Fixed> = black_box(
        Matrix2x3Trait::new(int(-6), int(0), int(6), int(8), int(-16), int(-4)),
    );
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    let mut b: Matrix2x3<Fixed> = black_box(
        Matrix2x3Trait::new(int(-6), int(0), int(6), int(4), int(-8), int(-2)),
    );
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(b == Matrix2x3Trait::new(int(-6), int(0), int(6), int(4), int(-8), int(-2)));
}

/// `Matrix2` against a `Matrix2x4` right-hand side: the three kernels of the pair (`lower`,
/// `upper`, `with_diag`; the `tr_` / `ad_` forms are the kernels of the transpose) recover the
/// integer solution exactly (power-of-two diagonal: every quotient is exact); `with_diag` solves
/// `(strict lower triangle of l + 2 I) x =
/// bd` and leaves `2 x` in `b`, like upstream.
#[test]
fn test_solve_matrix2_matrix2x4_exact() {
    let l: Matrix2<Fixed> = black_box(Matrix2Trait::new(int(2), int(0), int(0), int(4)));
    let u: Matrix2<Fixed> = black_box(Matrix2Trait::new(int(2), int(0), int(0), int(4)));
    let x: Matrix2x4<Fixed> = black_box(
        Matrix2x4Trait::new(int(-3), int(0), int(3), int(-3), int(2), int(-4), int(-1), int(2)),
    );
    let bl: Matrix2x4<Fixed> = black_box(
        Matrix2x4Trait::new(int(-6), int(0), int(6), int(-6), int(8), int(-16), int(-4), int(8)),
    );
    let bu: Matrix2x4<Fixed> = black_box(
        Matrix2x4Trait::new(int(-6), int(0), int(6), int(-6), int(8), int(-16), int(-4), int(8)),
    );
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    let mut b: Matrix2x4<Fixed> = black_box(
        Matrix2x4Trait::new(int(-6), int(0), int(6), int(-6), int(4), int(-8), int(-2), int(4)),
    );
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(
        b == Matrix2x4Trait::new(
            int(-6), int(0), int(6), int(-6), int(4), int(-8), int(-2), int(4),
        ),
    );
}

/// `Matrix2` against a `Matrix2x5` right-hand side: the three kernels of the pair (`lower`,
/// `upper`, `with_diag`; the `tr_` / `ad_` forms are the kernels of the transpose) recover the
/// integer solution exactly (power-of-two diagonal: every quotient is exact); `with_diag` solves
/// `(strict lower triangle of l + 2 I) x =
/// bd` and leaves `2 x` in `b`, like upstream.
#[test]
fn test_solve_matrix2_matrix2x5_exact() {
    let l: Matrix2<Fixed> = black_box(Matrix2Trait::new(int(2), int(0), int(0), int(4)));
    let u: Matrix2<Fixed> = black_box(Matrix2Trait::new(int(2), int(0), int(0), int(4)));
    let x: Matrix2x5<Fixed> = black_box(
        Matrix2x5Trait::new(
            int(-3), int(0), int(3), int(-3), int(0), int(2), int(-4), int(-1), int(2), int(-4),
        ),
    );
    let bl: Matrix2x5<Fixed> = black_box(
        Matrix2x5Trait::new(
            int(-6), int(0), int(6), int(-6), int(0), int(8), int(-16), int(-4), int(8), int(-16),
        ),
    );
    let bu: Matrix2x5<Fixed> = black_box(
        Matrix2x5Trait::new(
            int(-6), int(0), int(6), int(-6), int(0), int(8), int(-16), int(-4), int(8), int(-16),
        ),
    );
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    let mut b: Matrix2x5<Fixed> = black_box(
        Matrix2x5Trait::new(
            int(-6), int(0), int(6), int(-6), int(0), int(4), int(-8), int(-2), int(4), int(-8),
        ),
    );
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(
        b == Matrix2x5Trait::new(
            int(-6), int(0), int(6), int(-6), int(0), int(4), int(-8), int(-2), int(4), int(-8),
        ),
    );
}

/// `Matrix2` against a `Matrix2x6` right-hand side: the three kernels of the pair (`lower`,
/// `upper`, `with_diag`; the `tr_` / `ad_` forms are the kernels of the transpose) recover the
/// integer solution exactly (power-of-two diagonal: every quotient is exact); `with_diag` solves
/// `(strict lower triangle of l + 2 I) x =
/// bd` and leaves `2 x` in `b`, like upstream.
#[test]
fn test_solve_matrix2_matrix2x6_exact() {
    let l: Matrix2<Fixed> = black_box(Matrix2Trait::new(int(2), int(0), int(0), int(4)));
    let u: Matrix2<Fixed> = black_box(Matrix2Trait::new(int(2), int(0), int(0), int(4)));
    let x: Matrix2x6<Fixed> = black_box(
        Matrix2x6Trait::new(
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
        ),
    );
    let bl: Matrix2x6<Fixed> = black_box(
        Matrix2x6Trait::new(
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
        ),
    );
    let bu: Matrix2x6<Fixed> = black_box(
        Matrix2x6Trait::new(
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
        ),
    );
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    let mut b: Matrix2x6<Fixed> = black_box(
        Matrix2x6Trait::new(
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
        ),
    );
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(
        b == Matrix2x6Trait::new(
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
        ),
    );
}

/// `Matrix3` against a `Matrix3x2` right-hand side: the three kernels of the pair (`lower`,
/// `upper`, `with_diag`; the `tr_` / `ad_` forms are the kernels of the transpose) recover the
/// integer solution exactly (power-of-two diagonal: every quotient is exact); `with_diag` solves
/// `(strict lower triangle of l + 2 I) x =
/// bd` and leaves `2 x` in `b`, like upstream.
#[test]
fn test_solve_matrix3_matrix3x2_exact() {
    let l: Matrix3<Fixed> = black_box(
        Matrix3Trait::new(int(2), int(0), int(0), int(0), int(4), int(0), int(3), int(1), int(1)),
    );
    let u: Matrix3<Fixed> = black_box(
        Matrix3Trait::new(int(2), int(0), int(3), int(0), int(4), int(1), int(0), int(0), int(1)),
    );
    let x: Matrix3x2<Fixed> = black_box(
        Matrix3x2Trait::new(int(-3), int(0), int(2), int(-4), int(-2), int(1)),
    );
    let bl: Matrix3x2<Fixed> = black_box(
        Matrix3x2Trait::new(int(-6), int(0), int(8), int(-16), int(-9), int(-3)),
    );
    let bu: Matrix3x2<Fixed> = black_box(
        Matrix3x2Trait::new(int(-12), int(3), int(6), int(-15), int(-2), int(1)),
    );
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    let mut b: Matrix3x2<Fixed> = black_box(
        Matrix3x2Trait::new(int(-6), int(0), int(4), int(-8), int(-11), int(-2)),
    );
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(b == Matrix3x2Trait::new(int(-6), int(0), int(4), int(-8), int(-4), int(2)));
}

/// `Matrix3` against a `Matrix3` right-hand side: the three kernels of the pair (`lower`, `upper`,
/// `with_diag`; the `tr_` / `ad_` forms are the kernels of the transpose) recover the integer
/// solution exactly (power-of-two diagonal: every quotient is exact); `with_diag` solves `(strict
/// lower triangle of l + 2 I) x =
/// bd` and leaves `2 x` in `b`, like upstream.
#[test]
fn test_solve_matrix3_matrix3_exact() {
    let l: Matrix3<Fixed> = black_box(
        Matrix3Trait::new(int(2), int(0), int(0), int(0), int(4), int(0), int(3), int(1), int(1)),
    );
    let u: Matrix3<Fixed> = black_box(
        Matrix3Trait::new(int(2), int(0), int(3), int(0), int(4), int(1), int(0), int(0), int(1)),
    );
    let x: Matrix3<Fixed> = black_box(
        Matrix3Trait::new(
            int(-3), int(0), int(3), int(2), int(-4), int(-1), int(-2), int(1), int(4),
        ),
    );
    let bl: Matrix3<Fixed> = black_box(
        Matrix3Trait::new(
            int(-6), int(0), int(6), int(8), int(-16), int(-4), int(-9), int(-3), int(12),
        ),
    );
    let bu: Matrix3<Fixed> = black_box(
        Matrix3Trait::new(
            int(-12), int(3), int(18), int(6), int(-15), int(0), int(-2), int(1), int(4),
        ),
    );
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    let mut b: Matrix3<Fixed> = black_box(
        Matrix3Trait::new(
            int(-6), int(0), int(6), int(4), int(-8), int(-2), int(-11), int(-2), int(16),
        ),
    );
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(
        b == Matrix3Trait::new(
            int(-6), int(0), int(6), int(4), int(-8), int(-2), int(-4), int(2), int(8),
        ),
    );
}

/// `Matrix3` against a `Matrix3x4` right-hand side: the three kernels of the pair (`lower`,
/// `upper`, `with_diag`; the `tr_` / `ad_` forms are the kernels of the transpose) recover the
/// integer solution exactly (power-of-two diagonal: every quotient is exact); `with_diag` solves
/// `(strict lower triangle of l + 2 I) x =
/// bd` and leaves `2 x` in `b`, like upstream.
#[test]
fn test_solve_matrix3_matrix3x4_exact() {
    let l: Matrix3<Fixed> = black_box(
        Matrix3Trait::new(int(2), int(0), int(0), int(0), int(4), int(0), int(3), int(1), int(1)),
    );
    let u: Matrix3<Fixed> = black_box(
        Matrix3Trait::new(int(2), int(0), int(3), int(0), int(4), int(1), int(0), int(0), int(1)),
    );
    let x: Matrix3x4<Fixed> = black_box(
        Matrix3x4Trait::new(
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
        ),
    );
    let bl: Matrix3x4<Fixed> = black_box(
        Matrix3x4Trait::new(
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
        ),
    );
    let bu: Matrix3x4<Fixed> = black_box(
        Matrix3x4Trait::new(
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
        ),
    );
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    let mut b: Matrix3x4<Fixed> = black_box(
        Matrix3x4Trait::new(
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
        ),
    );
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(
        b == Matrix3x4Trait::new(
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
        ),
    );
}

/// `Matrix3` against a `Matrix3x5` right-hand side: the three kernels of the pair (`lower`,
/// `upper`, `with_diag`; the `tr_` / `ad_` forms are the kernels of the transpose) recover the
/// integer solution exactly (power-of-two diagonal: every quotient is exact); `with_diag` solves
/// `(strict lower triangle of l + 2 I) x =
/// bd` and leaves `2 x` in `b`, like upstream.
#[test]
fn test_solve_matrix3_matrix3x5_exact() {
    let l: Matrix3<Fixed> = black_box(
        Matrix3Trait::new(int(2), int(0), int(0), int(0), int(4), int(0), int(3), int(1), int(1)),
    );
    let u: Matrix3<Fixed> = black_box(
        Matrix3Trait::new(int(2), int(0), int(3), int(0), int(4), int(1), int(0), int(0), int(1)),
    );
    let x: Matrix3x5<Fixed> = black_box(
        Matrix3x5Trait::new(
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
        ),
    );
    let bl: Matrix3x5<Fixed> = black_box(
        Matrix3x5Trait::new(
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
        ),
    );
    let bu: Matrix3x5<Fixed> = black_box(
        Matrix3x5Trait::new(
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
        ),
    );
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    let mut b: Matrix3x5<Fixed> = black_box(
        Matrix3x5Trait::new(
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
        ),
    );
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(
        b == Matrix3x5Trait::new(
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
        ),
    );
}

/// `Matrix3` against a `Matrix3x6` right-hand side: the three kernels of the pair (`lower`,
/// `upper`, `with_diag`; the `tr_` / `ad_` forms are the kernels of the transpose) recover the
/// integer solution exactly (power-of-two diagonal: every quotient is exact); `with_diag` solves
/// `(strict lower triangle of l + 2 I) x =
/// bd` and leaves `2 x` in `b`, like upstream.
#[test]
fn test_solve_matrix3_matrix3x6_exact() {
    let l: Matrix3<Fixed> = black_box(
        Matrix3Trait::new(int(2), int(0), int(0), int(0), int(4), int(0), int(3), int(1), int(1)),
    );
    let u: Matrix3<Fixed> = black_box(
        Matrix3Trait::new(int(2), int(0), int(3), int(0), int(4), int(1), int(0), int(0), int(1)),
    );
    let x: Matrix3x6<Fixed> = black_box(
        Matrix3x6Trait::new(
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
        ),
    );
    let bl: Matrix3x6<Fixed> = black_box(
        Matrix3x6Trait::new(
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
        ),
    );
    let bu: Matrix3x6<Fixed> = black_box(
        Matrix3x6Trait::new(
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
        ),
    );
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    let mut b: Matrix3x6<Fixed> = black_box(
        Matrix3x6Trait::new(
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
        ),
    );
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(
        b == Matrix3x6Trait::new(
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
        ),
    );
}

/// `Matrix4` against a `Matrix4x2` right-hand side: the three kernels of the pair (`lower`,
/// `upper`, `with_diag`; the `tr_` / `ad_` forms are the kernels of the transpose) recover the
/// integer solution exactly (power-of-two diagonal: every quotient is exact); `with_diag` solves
/// `(strict lower triangle of l + 2 I) x =
/// bd` and leaves `2 x` in `b`, like upstream.
#[test]
fn test_solve_matrix4_matrix4x2_exact() {
    let l: Matrix4<Fixed> = black_box(
        Matrix4Trait::new(
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
        ),
    );
    let u: Matrix4<Fixed> = black_box(
        Matrix4Trait::new(
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
        ),
    );
    let x: Matrix4x2<Fixed> = black_box(
        Matrix4x2Trait::new(int(-3), int(0), int(2), int(-4), int(-2), int(1), int(3), int(-3)),
    );
    let bl: Matrix4x2<Fixed> = black_box(
        Matrix4x2Trait::new(int(-6), int(0), int(8), int(-16), int(-9), int(-3), int(17), int(-10)),
    );
    let bu: Matrix4x2<Fixed> = black_box(
        Matrix4x2Trait::new(int(-15), int(6), int(-3), int(-6), int(4), int(-5), int(24), int(-24)),
    );
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    let mut b: Matrix4x2<Fixed> = black_box(
        Matrix4x2Trait::new(int(-6), int(0), int(4), int(-8), int(-11), int(-2), int(-1), int(8)),
    );
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(
        b == Matrix4x2Trait::new(
            int(-6), int(0), int(4), int(-8), int(-4), int(2), int(6), int(-6),
        ),
    );
}

/// `Matrix4` against a `Matrix4x3` right-hand side: the three kernels of the pair (`lower`,
/// `upper`, `with_diag`; the `tr_` / `ad_` forms are the kernels of the transpose) recover the
/// integer solution exactly (power-of-two diagonal: every quotient is exact); `with_diag` solves
/// `(strict lower triangle of l + 2 I) x =
/// bd` and leaves `2 x` in `b`, like upstream.
#[test]
fn test_solve_matrix4_matrix4x3_exact() {
    let l: Matrix4<Fixed> = black_box(
        Matrix4Trait::new(
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
        ),
    );
    let u: Matrix4<Fixed> = black_box(
        Matrix4Trait::new(
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
        ),
    );
    let x: Matrix4x3<Fixed> = black_box(
        Matrix4x3Trait::new(
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
        ),
    );
    let bl: Matrix4x3<Fixed> = black_box(
        Matrix4x3Trait::new(
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
        ),
    );
    let bu: Matrix4x3<Fixed> = black_box(
        Matrix4x3Trait::new(
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
        ),
    );
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    let mut b: Matrix4x3<Fixed> = black_box(
        Matrix4x3Trait::new(
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
        ),
    );
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(
        b == Matrix4x3Trait::new(
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
        ),
    );
}

/// `Matrix4` against a `Matrix4` right-hand side: the three kernels of the pair (`lower`, `upper`,
/// `with_diag`; the `tr_` / `ad_` forms are the kernels of the transpose) recover the integer
/// solution exactly (power-of-two diagonal: every quotient is exact); `with_diag` solves `(strict
/// lower triangle of l + 2 I) x =
/// bd` and leaves `2 x` in `b`, like upstream.
#[test]
fn test_solve_matrix4_matrix4_exact() {
    let l: Matrix4<Fixed> = black_box(
        Matrix4Trait::new(
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
        ),
    );
    let u: Matrix4<Fixed> = black_box(
        Matrix4Trait::new(
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
        ),
    );
    let x: Matrix4<Fixed> = black_box(
        Matrix4Trait::new(
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
        ),
    );
    let bl: Matrix4<Fixed> = black_box(
        Matrix4Trait::new(
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
        ),
    );
    let bu: Matrix4<Fixed> = black_box(
        Matrix4Trait::new(
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
        ),
    );
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    let mut b: Matrix4<Fixed> = black_box(
        Matrix4Trait::new(
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
        ),
    );
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(
        b == Matrix4Trait::new(
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
        ),
    );
}

/// `Matrix4` against a `Matrix4x5` right-hand side: the three kernels of the pair (`lower`,
/// `upper`, `with_diag`; the `tr_` / `ad_` forms are the kernels of the transpose) recover the
/// integer solution exactly (power-of-two diagonal: every quotient is exact); `with_diag` solves
/// `(strict lower triangle of l + 2 I) x =
/// bd` and leaves `2 x` in `b`, like upstream.
#[test]
fn test_solve_matrix4_matrix4x5_exact() {
    let l: Matrix4<Fixed> = black_box(
        Matrix4Trait::new(
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
        ),
    );
    let u: Matrix4<Fixed> = black_box(
        Matrix4Trait::new(
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
        ),
    );
    let x: Matrix4x5<Fixed> = black_box(
        Matrix4x5Trait::new(
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
        ),
    );
    let bl: Matrix4x5<Fixed> = black_box(
        Matrix4x5Trait::new(
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
        ),
    );
    let bu: Matrix4x5<Fixed> = black_box(
        Matrix4x5Trait::new(
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
        ),
    );
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    let mut b: Matrix4x5<Fixed> = black_box(
        Matrix4x5Trait::new(
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
        ),
    );
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(
        b == Matrix4x5Trait::new(
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
        ),
    );
}

/// `Matrix4` against a `Matrix4x6` right-hand side: the three kernels of the pair (`lower`,
/// `upper`, `with_diag`; the `tr_` / `ad_` forms are the kernels of the transpose) recover the
/// integer solution exactly (power-of-two diagonal: every quotient is exact); `with_diag` solves
/// `(strict lower triangle of l + 2 I) x =
/// bd` and leaves `2 x` in `b`, like upstream.
#[test]
fn test_solve_matrix4_matrix4x6_exact() {
    let l: Matrix4<Fixed> = black_box(
        Matrix4Trait::new(
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
        ),
    );
    let u: Matrix4<Fixed> = black_box(
        Matrix4Trait::new(
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
        ),
    );
    let x: Matrix4x6<Fixed> = black_box(
        Matrix4x6Trait::new(
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
        ),
    );
    let bl: Matrix4x6<Fixed> = black_box(
        Matrix4x6Trait::new(
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
        ),
    );
    let bu: Matrix4x6<Fixed> = black_box(
        Matrix4x6Trait::new(
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
        ),
    );
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    let mut b: Matrix4x6<Fixed> = black_box(
        Matrix4x6Trait::new(
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
        ),
    );
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(
        b == Matrix4x6Trait::new(
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
        ),
    );
}

/// `Matrix5` against a `Matrix5x2` right-hand side: the three kernels of the pair (`lower`,
/// `upper`, `with_diag`; the `tr_` / `ad_` forms are the kernels of the transpose) recover the
/// integer solution exactly (power-of-two diagonal: every quotient is exact); `with_diag` solves
/// `(strict lower triangle of l + 2 I) x =
/// bd` and leaves `2 x` in `b`, like upstream.
#[test]
fn test_solve_matrix5_matrix5x2_exact() {
    let l: Matrix5<Fixed> = black_box(
        Matrix5Trait::new(
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
        ),
    );
    let u: Matrix5<Fixed> = black_box(
        Matrix5Trait::new(
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
        ),
    );
    let x: Matrix5x2<Fixed> = black_box(
        Matrix5x2Trait::new(
            int(-3), int(0), int(2), int(-4), int(-2), int(1), int(3), int(-3), int(-1), int(2),
        ),
    );
    let bl: Matrix5x2<Fixed> = black_box(
        Matrix5x2Trait::new(
            int(-6), int(0), int(8), int(-16), int(-9), int(-3), int(17), int(-10), int(5), int(-7),
        ),
    );
    let bu: Matrix5x2<Fixed> = black_box(
        Matrix5x2Trait::new(
            int(-17),
            int(10),
            int(-3),
            int(-6),
            int(6),
            int(-9),
            int(21),
            int(-18),
            int(-2),
            int(4),
        ),
    );
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    let mut b: Matrix5x2<Fixed> = black_box(
        Matrix5x2Trait::new(
            int(-6), int(0), int(4), int(-8), int(-11), int(-2), int(-1), int(8), int(5), int(-7),
        ),
    );
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(
        b == Matrix5x2Trait::new(
            int(-6), int(0), int(4), int(-8), int(-4), int(2), int(6), int(-6), int(-2), int(4),
        ),
    );
}

/// `Matrix5` against a `Matrix5x3` right-hand side: the three kernels of the pair (`lower`,
/// `upper`, `with_diag`; the `tr_` / `ad_` forms are the kernels of the transpose) recover the
/// integer solution exactly (power-of-two diagonal: every quotient is exact); `with_diag` solves
/// `(strict lower triangle of l + 2 I) x =
/// bd` and leaves `2 x` in `b`, like upstream.
#[test]
fn test_solve_matrix5_matrix5x3_exact() {
    let l: Matrix5<Fixed> = black_box(
        Matrix5Trait::new(
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
        ),
    );
    let u: Matrix5<Fixed> = black_box(
        Matrix5Trait::new(
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
        ),
    );
    let x: Matrix5x3<Fixed> = black_box(
        Matrix5x3Trait::new(
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
        ),
    );
    let bl: Matrix5x3<Fixed> = black_box(
        Matrix5x3Trait::new(
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
        ),
    );
    let bu: Matrix5x3<Fixed> = black_box(
        Matrix5x3Trait::new(
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
        ),
    );
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    let mut b: Matrix5x3<Fixed> = black_box(
        Matrix5x3Trait::new(
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
        ),
    );
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(
        b == Matrix5x3Trait::new(
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
        ),
    );
}

/// `Matrix5` against a `Matrix5x4` right-hand side: the three kernels of the pair (`lower`,
/// `upper`, `with_diag`; the `tr_` / `ad_` forms are the kernels of the transpose) recover the
/// integer solution exactly (power-of-two diagonal: every quotient is exact); `with_diag` solves
/// `(strict lower triangle of l + 2 I) x =
/// bd` and leaves `2 x` in `b`, like upstream.
#[test]
fn test_solve_matrix5_matrix5x4_exact() {
    let l: Matrix5<Fixed> = black_box(
        Matrix5Trait::new(
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
        ),
    );
    let u: Matrix5<Fixed> = black_box(
        Matrix5Trait::new(
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
        ),
    );
    let x: Matrix5x4<Fixed> = black_box(
        Matrix5x4Trait::new(
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
        ),
    );
    let bl: Matrix5x4<Fixed> = black_box(
        Matrix5x4Trait::new(
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
        ),
    );
    let bu: Matrix5x4<Fixed> = black_box(
        Matrix5x4Trait::new(
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
        ),
    );
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    let mut b: Matrix5x4<Fixed> = black_box(
        Matrix5x4Trait::new(
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
        ),
    );
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(
        b == Matrix5x4Trait::new(
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
        ),
    );
}

/// `Matrix5` against a `Matrix5` right-hand side: the three kernels of the pair (`lower`, `upper`,
/// `with_diag`; the `tr_` / `ad_` forms are the kernels of the transpose) recover the integer
/// solution exactly (power-of-two diagonal: every quotient is exact); `with_diag` solves `(strict
/// lower triangle of l + 2 I) x =
/// bd` and leaves `2 x` in `b`, like upstream.
#[test]
fn test_solve_matrix5_matrix5_exact() {
    let l: Matrix5<Fixed> = black_box(
        Matrix5Trait::new(
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
        ),
    );
    let u: Matrix5<Fixed> = black_box(
        Matrix5Trait::new(
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
        ),
    );
    let x: Matrix5<Fixed> = black_box(
        Matrix5Trait::new(
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
        ),
    );
    let bl: Matrix5<Fixed> = black_box(
        Matrix5Trait::new(
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
        ),
    );
    let bu: Matrix5<Fixed> = black_box(
        Matrix5Trait::new(
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
        ),
    );
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    let mut b: Matrix5<Fixed> = black_box(
        Matrix5Trait::new(
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
        ),
    );
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(
        b == Matrix5Trait::new(
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
        ),
    );
}

/// `Matrix5` against a `Matrix5x6` right-hand side: the three kernels of the pair (`lower`,
/// `upper`, `with_diag`; the `tr_` / `ad_` forms are the kernels of the transpose) recover the
/// integer solution exactly (power-of-two diagonal: every quotient is exact); `with_diag` solves
/// `(strict lower triangle of l + 2 I) x =
/// bd` and leaves `2 x` in `b`, like upstream.
#[test]
fn test_solve_matrix5_matrix5x6_exact() {
    let l: Matrix5<Fixed> = black_box(
        Matrix5Trait::new(
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
        ),
    );
    let u: Matrix5<Fixed> = black_box(
        Matrix5Trait::new(
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
        ),
    );
    let x: Matrix5x6<Fixed> = black_box(
        Matrix5x6Trait::new(
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
        ),
    );
    let bl: Matrix5x6<Fixed> = black_box(
        Matrix5x6Trait::new(
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
        ),
    );
    let bu: Matrix5x6<Fixed> = black_box(
        Matrix5x6Trait::new(
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
        ),
    );
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    let mut b: Matrix5x6<Fixed> = black_box(
        Matrix5x6Trait::new(
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
        ),
    );
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(
        b == Matrix5x6Trait::new(
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
        ),
    );
}

/// `Matrix6` against a `Matrix6x2` right-hand side: the three kernels of the pair (`lower`,
/// `upper`, `with_diag`; the `tr_` / `ad_` forms are the kernels of the transpose) recover the
/// integer solution exactly (power-of-two diagonal: every quotient is exact); `with_diag` solves
/// `(strict lower triangle of l + 2 I) x =
/// bd` and leaves `2 x` in `b`, like upstream.
#[test]
fn test_solve_matrix6_matrix6x2_exact() {
    let l: Matrix6<Fixed> = black_box(
        Matrix6Trait::new(
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
        ),
    );
    let u: Matrix6<Fixed> = black_box(
        Matrix6Trait::new(
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
        ),
    );
    let x: Matrix6x2<Fixed> = black_box(
        Matrix6x2Trait::new(
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
        ),
    );
    let bl: Matrix6x2<Fixed> = black_box(
        Matrix6x2Trait::new(
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
        ),
    );
    let bu: Matrix6x2<Fixed> = black_box(
        Matrix6x2Trait::new(
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
        ),
    );
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    let mut b: Matrix6x2<Fixed> = black_box(
        Matrix6x2Trait::new(
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
        ),
    );
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(
        b == Matrix6x2Trait::new(
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
        ),
    );
}

/// `Matrix6` against a `Matrix6x3` right-hand side: the three kernels of the pair (`lower`,
/// `upper`, `with_diag`; the `tr_` / `ad_` forms are the kernels of the transpose) recover the
/// integer solution exactly (power-of-two diagonal: every quotient is exact); `with_diag` solves
/// `(strict lower triangle of l + 2 I) x =
/// bd` and leaves `2 x` in `b`, like upstream.
#[test]
fn test_solve_matrix6_matrix6x3_exact() {
    let l: Matrix6<Fixed> = black_box(
        Matrix6Trait::new(
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
        ),
    );
    let u: Matrix6<Fixed> = black_box(
        Matrix6Trait::new(
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
        ),
    );
    let x: Matrix6x3<Fixed> = black_box(
        Matrix6x3Trait::new(
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
        ),
    );
    let bl: Matrix6x3<Fixed> = black_box(
        Matrix6x3Trait::new(
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
        ),
    );
    let bu: Matrix6x3<Fixed> = black_box(
        Matrix6x3Trait::new(
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
        ),
    );
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    let mut b: Matrix6x3<Fixed> = black_box(
        Matrix6x3Trait::new(
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
        ),
    );
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(
        b == Matrix6x3Trait::new(
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
        ),
    );
}

/// `Matrix6` against a `Matrix6x4` right-hand side: the three kernels of the pair (`lower`,
/// `upper`, `with_diag`; the `tr_` / `ad_` forms are the kernels of the transpose) recover the
/// integer solution exactly (power-of-two diagonal: every quotient is exact); `with_diag` solves
/// `(strict lower triangle of l + 2 I) x =
/// bd` and leaves `2 x` in `b`, like upstream.
#[test]
fn test_solve_matrix6_matrix6x4_exact() {
    let l: Matrix6<Fixed> = black_box(
        Matrix6Trait::new(
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
        ),
    );
    let u: Matrix6<Fixed> = black_box(
        Matrix6Trait::new(
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
        ),
    );
    let x: Matrix6x4<Fixed> = black_box(
        Matrix6x4Trait::new(
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
        ),
    );
    let bl: Matrix6x4<Fixed> = black_box(
        Matrix6x4Trait::new(
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
        ),
    );
    let bu: Matrix6x4<Fixed> = black_box(
        Matrix6x4Trait::new(
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
        ),
    );
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    let mut b: Matrix6x4<Fixed> = black_box(
        Matrix6x4Trait::new(
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
        ),
    );
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(
        b == Matrix6x4Trait::new(
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
        ),
    );
}

/// `Matrix6` against a `Matrix6x5` right-hand side: the three kernels of the pair (`lower`,
/// `upper`, `with_diag`; the `tr_` / `ad_` forms are the kernels of the transpose) recover the
/// integer solution exactly (power-of-two diagonal: every quotient is exact); `with_diag` solves
/// `(strict lower triangle of l + 2 I) x =
/// bd` and leaves `2 x` in `b`, like upstream.
#[test]
fn test_solve_matrix6_matrix6x5_exact() {
    let l: Matrix6<Fixed> = black_box(
        Matrix6Trait::new(
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
        ),
    );
    let u: Matrix6<Fixed> = black_box(
        Matrix6Trait::new(
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
        ),
    );
    let x: Matrix6x5<Fixed> = black_box(
        Matrix6x5Trait::new(
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
        ),
    );
    let bl: Matrix6x5<Fixed> = black_box(
        Matrix6x5Trait::new(
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
        ),
    );
    let bu: Matrix6x5<Fixed> = black_box(
        Matrix6x5Trait::new(
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
        ),
    );
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    let mut b: Matrix6x5<Fixed> = black_box(
        Matrix6x5Trait::new(
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
        ),
    );
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(
        b == Matrix6x5Trait::new(
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
        ),
    );
}

/// `Matrix6` against a `Matrix6` right-hand side: the three kernels of the pair (`lower`, `upper`,
/// `with_diag`; the `tr_` / `ad_` forms are the kernels of the transpose) recover the integer
/// solution exactly (power-of-two diagonal: every quotient is exact); `with_diag` solves `(strict
/// lower triangle of l + 2 I) x =
/// bd` and leaves `2 x` in `b`, like upstream.
#[test]
fn test_solve_matrix6_matrix6_exact() {
    let l: Matrix6<Fixed> = black_box(
        Matrix6Trait::new(
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
        ),
    );
    let u: Matrix6<Fixed> = black_box(
        Matrix6Trait::new(
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
        ),
    );
    let x: Matrix6<Fixed> = black_box(
        Matrix6Trait::new(
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
        ),
    );
    let bl: Matrix6<Fixed> = black_box(
        Matrix6Trait::new(
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
        ),
    );
    let bu: Matrix6<Fixed> = black_box(
        Matrix6Trait::new(
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
        ),
    );
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    let mut b: Matrix6<Fixed> = black_box(
        Matrix6Trait::new(
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
        ),
    );
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(
        b == Matrix6Trait::new(
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
        ),
    );
}
