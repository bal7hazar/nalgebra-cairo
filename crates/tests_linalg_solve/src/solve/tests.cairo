//! Tests of `MatrixSolve` (upstream `SquareMatrix::solve_*`, `src/linalg/solve.rs`): the kernels of
//! every (square, right-hand side with as many rows) pair and every one of the 26 forms on the
//! vector right-hand sides (the forms are one blanket impl over the kernels), on designs whose
//! solution is exact (power-of-two diagonal, integer `x`, `b = t x` exact), the zero-diagonal
//! semantics of the checked forms and the division panic of the unchecked ones. The oracle vectors
//! (`oracle_solve`) check the rounding on real-valued inputs. Every operand goes through
//! `black_box`: constant operands make the compiler specialise each kernel per test.

use fixed::Fixed;
use nalgebra::{
    Matrix1, Matrix1Trait, Matrix2, Matrix2Trait, Matrix3, Matrix3Trait, Matrix4, Matrix4Trait,
    Matrix5, Matrix5Trait, Matrix6, Matrix6Trait, MatrixSolve, Vector2, Vector2Trait, Vector3,
    Vector3Trait, Vector4, Vector4Trait, Vector5, Vector5Trait, Vector6, Vector6Trait,
};
use nalgebra_testing::black_box;
use nalgebra_tests_utils::int;

/// `Matrix1` against a `Matrix1` right-hand side: the three kernels of the pair (`lower`, `upper`,
/// `with_diag`; the `tr_` / `ad_` forms are the kernels of the transpose) recover the integer
/// solution exactly (power-of-two diagonal: every quotient is exact); `with_diag` solves `(strict
/// lower triangle of l + 2 I) x =
/// bd` and leaves `2 x` in `b`, like upstream.
#[test]
fn test_solve_matrix1_matrix1_exact() {
    let l: Matrix1<Fixed> = black_box(Matrix1Trait::new(int(2)));
    let u: Matrix1<Fixed> = black_box(Matrix1Trait::new(int(2)));
    let x: Matrix1<Fixed> = black_box(Matrix1Trait::new(int(-3)));
    let bl: Matrix1<Fixed> = black_box(Matrix1Trait::new(int(-6)));
    let bu: Matrix1<Fixed> = black_box(Matrix1Trait::new(int(-6)));
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    let mut b: Matrix1<Fixed> = black_box(Matrix1Trait::new(int(-6)));
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(b == Matrix1Trait::new(int(-6)));
}

/// `Matrix2` against a `Vector2` right-hand side: the three kernels of the pair (`lower`, `upper`,
/// `with_diag`; the `tr_` / `ad_` forms are the kernels of the transpose) recover the integer
/// solution exactly (power-of-two diagonal: every quotient is exact); `with_diag` solves `(strict
/// lower triangle of l + 2 I) x =
/// bd` and leaves `2 x` in `b`, like upstream.
#[test]
fn test_solve_matrix2_vector2_exact() {
    let l: Matrix2<Fixed> = black_box(Matrix2Trait::new(int(2), int(0), int(0), int(4)));
    let u: Matrix2<Fixed> = black_box(Matrix2Trait::new(int(2), int(0), int(0), int(4)));
    let x: Vector2<Fixed> = black_box(Vector2Trait::new(int(-3), int(2)));
    let bl: Vector2<Fixed> = black_box(Vector2Trait::new(int(-6), int(8)));
    let bu: Vector2<Fixed> = black_box(Vector2Trait::new(int(-6), int(8)));
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    let mut b: Vector2<Fixed> = black_box(Vector2Trait::new(int(-6), int(4)));
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(b == Vector2Trait::new(int(-6), int(4)));
}

/// `Matrix3` against a `Vector3` right-hand side: the three kernels of the pair (`lower`, `upper`,
/// `with_diag`; the `tr_` / `ad_` forms are the kernels of the transpose) recover the integer
/// solution exactly (power-of-two diagonal: every quotient is exact); `with_diag` solves `(strict
/// lower triangle of l + 2 I) x =
/// bd` and leaves `2 x` in `b`, like upstream.
#[test]
fn test_solve_matrix3_vector3_exact() {
    let l: Matrix3<Fixed> = black_box(
        Matrix3Trait::new(int(2), int(0), int(0), int(0), int(4), int(0), int(3), int(1), int(1)),
    );
    let u: Matrix3<Fixed> = black_box(
        Matrix3Trait::new(int(2), int(0), int(3), int(0), int(4), int(1), int(0), int(0), int(1)),
    );
    let x: Vector3<Fixed> = black_box(Vector3Trait::new(int(-3), int(2), int(-2)));
    let bl: Vector3<Fixed> = black_box(Vector3Trait::new(int(-6), int(8), int(-9)));
    let bu: Vector3<Fixed> = black_box(Vector3Trait::new(int(-12), int(6), int(-2)));
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    let mut b: Vector3<Fixed> = black_box(Vector3Trait::new(int(-6), int(4), int(-11)));
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(b == Vector3Trait::new(int(-6), int(4), int(-4)));
}

/// `Matrix3` against a `Vector3` right-hand side: every one of the 26 forms of `MatrixSolve` (ONE
/// size: the forms are one blanket impl over the kernels, which the other tests cover pair by pair)
/// (checked, `_mut`, `_unchecked`, `_unchecked_mut`; plain, `tr_`, `ad_`; `with_diag`).
#[test]
fn test_solve_matrix3_every_form() {
    let l: Matrix3<Fixed> = black_box(
        Matrix3Trait::new(int(2), int(0), int(0), int(0), int(4), int(0), int(3), int(1), int(1)),
    );
    let u: Matrix3<Fixed> = black_box(
        Matrix3Trait::new(int(2), int(0), int(3), int(0), int(4), int(1), int(0), int(0), int(1)),
    );
    let x: Vector3<Fixed> = black_box(Vector3Trait::new(int(-3), int(2), int(-2)));
    let bl: Vector3<Fixed> = black_box(Vector3Trait::new(int(-6), int(8), int(-9)));
    let bu: Vector3<Fixed> = black_box(Vector3Trait::new(int(-12), int(6), int(-2)));
    assert!(l.solve_lower_triangular(bl) == Option::Some(x));
    assert!(u.solve_upper_triangular(bu) == Option::Some(x));
    assert!(l.tr_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.tr_solve_upper_triangular(bl) == Option::Some(x));
    assert!(l.ad_solve_lower_triangular(bu) == Option::Some(x));
    assert!(u.ad_solve_upper_triangular(bl) == Option::Some(x));
    assert!(l.ad_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.ad_solve_upper_triangular_unchecked(bl) == x);
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
    let bd: Vector3<Fixed> = black_box(Vector3Trait::new(int(-6), int(4), int(-11)));
    let mut b = bd;
    l.solve_lower_triangular_with_diag_unchecked_mut(ref b, int(2));
    assert!(b == Vector3Trait::new(int(-6), int(4), int(-4)));
    let mut b = bd;
    assert!(!l.solve_lower_triangular_with_diag_mut(ref b, int(0)));
    assert!(b == bd);
}

/// `Matrix4` against a `Vector4` right-hand side: the three kernels of the pair (`lower`, `upper`,
/// `with_diag`; the `tr_` / `ad_` forms are the kernels of the transpose) recover the integer
/// solution exactly (power-of-two diagonal: every quotient is exact); `with_diag` solves `(strict
/// lower triangle of l + 2 I) x =
/// bd` and leaves `2 x` in `b`, like upstream.
#[test]
fn test_solve_matrix4_vector4_exact() {
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
    let x: Vector4<Fixed> = black_box(Vector4Trait::new(int(-3), int(2), int(-2), int(3)));
    let bl: Vector4<Fixed> = black_box(Vector4Trait::new(int(-6), int(8), int(-9), int(17)));
    let bu: Vector4<Fixed> = black_box(Vector4Trait::new(int(-15), int(-3), int(4), int(24)));
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    let mut b: Vector4<Fixed> = black_box(Vector4Trait::new(int(-6), int(4), int(-11), int(-1)));
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(b == Vector4Trait::new(int(-6), int(4), int(-4), int(6)));
}

/// `Matrix5` against a `Vector5` right-hand side: the three kernels of the pair (`lower`, `upper`,
/// `with_diag`; the `tr_` / `ad_` forms are the kernels of the transpose) recover the integer
/// solution exactly (power-of-two diagonal: every quotient is exact); `with_diag` solves `(strict
/// lower triangle of l + 2 I) x =
/// bd` and leaves `2 x` in `b`, like upstream.
#[test]
fn test_solve_matrix5_vector5_exact() {
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
    let x: Vector5<Fixed> = black_box(Vector5Trait::new(int(-3), int(2), int(-2), int(3), int(-1)));
    let bl: Vector5<Fixed> = black_box(
        Vector5Trait::new(int(-6), int(8), int(-9), int(17), int(5)),
    );
    let bu: Vector5<Fixed> = black_box(
        Vector5Trait::new(int(-17), int(-3), int(6), int(21), int(-2)),
    );
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    let mut b: Vector5<Fixed> = black_box(
        Vector5Trait::new(int(-6), int(4), int(-11), int(-1), int(5)),
    );
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(b == Vector5Trait::new(int(-6), int(4), int(-4), int(6), int(-2)));
}

/// `Matrix6` against a `Vector6` right-hand side: the three kernels of the pair (`lower`, `upper`,
/// `with_diag`; the `tr_` / `ad_` forms are the kernels of the transpose) recover the integer
/// solution exactly (power-of-two diagonal: every quotient is exact); `with_diag` solves `(strict
/// lower triangle of l + 2 I) x =
/// bd` and leaves `2 x` in `b`, like upstream.
#[test]
fn test_solve_matrix6_vector6_exact() {
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
    let x: Vector6<Fixed> = black_box(
        Vector6Trait::new(int(-3), int(2), int(-2), int(3), int(-1), int(4)),
    );
    let bl: Vector6<Fixed> = black_box(
        Vector6Trait::new(int(-6), int(8), int(-9), int(17), int(5), int(26)),
    );
    let bu: Vector6<Fixed> = black_box(
        Vector6Trait::new(int(-25), int(9), int(10), int(17), int(-14), int(16)),
    );
    assert!(l.solve_lower_triangular_unchecked(bl) == x);
    assert!(u.solve_upper_triangular_unchecked(bu) == x);
    assert!(l.tr_solve_lower_triangular_unchecked(bu) == x);
    assert!(u.tr_solve_upper_triangular_unchecked(bl) == x);
    let mut b: Vector6<Fixed> = black_box(
        Vector6Trait::new(int(-6), int(4), int(-11), int(-1), int(5), int(18)),
    );
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(b == Vector6Trait::new(int(-6), int(4), int(-4), int(6), int(-2), int(8)));
}

/// A zero on the diagonal of a 1x1 triangle: the checked forms (every one on the 3x3) return `None`
/// / `false` and leave `b`
/// unchanged (upstream stops at the first zero pivot).
#[test]
fn test_solve_matrix1_zero_diagonal() {
    let l: Matrix1<Fixed> = black_box(Matrix1Trait::new(int(0)));
    let u: Matrix1<Fixed> = black_box(Matrix1Trait::new(int(0)));
    let b0: Matrix1<Fixed> = black_box(Matrix1Trait::new(int(1)));
    assert!(l.solve_lower_triangular(b0).is_none());
    let mut b = b0;
    assert!(!u.solve_upper_triangular_mut(ref b));
    assert!(b == b0);
}

/// A zero on the diagonal of a 2x2 triangle: the checked forms (every one on the 3x3) return `None`
/// / `false` and leave `b`
/// unchanged (upstream stops at the first zero pivot).
#[test]
fn test_solve_matrix2_zero_diagonal() {
    let l: Matrix2<Fixed> = black_box(Matrix2Trait::new(int(2), int(0), int(0), int(0)));
    let u: Matrix2<Fixed> = black_box(Matrix2Trait::new(int(2), int(0), int(0), int(0)));
    let b0: Vector2<Fixed> = black_box(Vector2Trait::new(int(1), int(2)));
    assert!(l.solve_lower_triangular(b0).is_none());
    let mut b = b0;
    assert!(!u.solve_upper_triangular_mut(ref b));
    assert!(b == b0);
}

/// A zero on the diagonal of a 3x3 triangle: every checked form return `None` / `false` and leave
/// `b`
/// unchanged (upstream stops at the first zero pivot).
#[test]
fn test_solve_matrix3_zero_diagonal() {
    let l: Matrix3<Fixed> = black_box(
        Matrix3Trait::new(int(2), int(0), int(0), int(0), int(0), int(0), int(3), int(1), int(1)),
    );
    let u: Matrix3<Fixed> = black_box(
        Matrix3Trait::new(int(2), int(0), int(3), int(0), int(0), int(1), int(0), int(0), int(1)),
    );
    let b0: Vector3<Fixed> = black_box(Vector3Trait::new(int(1), int(2), int(3)));
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
    let l: Matrix3<Fixed> = black_box(
        Matrix3Trait::new(int(2), int(0), int(0), int(0), int(0), int(0), int(3), int(1), int(1)),
    );
    let b0: Vector3<Fixed> = black_box(Vector3Trait::new(int(1), int(2), int(3)));
    let _x = l.solve_lower_triangular_unchecked(b0);
}

/// A zero on the diagonal of a 4x4 triangle: the checked forms (every one on the 3x3) return `None`
/// / `false` and leave `b`
/// unchanged (upstream stops at the first zero pivot).
#[test]
fn test_solve_matrix4_zero_diagonal() {
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
            int(0),
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
            int(0),
            int(2),
            int(0),
            int(0),
            int(0),
            int(8),
        ),
    );
    let b0: Vector4<Fixed> = black_box(Vector4Trait::new(int(1), int(2), int(3), int(4)));
    assert!(l.solve_lower_triangular(b0).is_none());
    let mut b = b0;
    assert!(!u.solve_upper_triangular_mut(ref b));
    assert!(b == b0);
}

/// A zero on the diagonal of a 5x5 triangle: the checked forms (every one on the 3x3) return `None`
/// / `false` and leave `b`
/// unchanged (upstream stops at the first zero pivot).
#[test]
fn test_solve_matrix5_zero_diagonal() {
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
        ),
    );
    let b0: Vector5<Fixed> = black_box(Vector5Trait::new(int(1), int(2), int(3), int(4), int(5)));
    assert!(l.solve_lower_triangular(b0).is_none());
    let mut b = b0;
    assert!(!u.solve_upper_triangular_mut(ref b));
    assert!(b == b0);
}

/// A zero on the diagonal of a 6x6 triangle: the checked forms (every one on the 3x3) return `None`
/// / `false` and leave `b`
/// unchanged (upstream stops at the first zero pivot).
#[test]
fn test_solve_matrix6_zero_diagonal() {
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
        ),
    );
    let b0: Vector6<Fixed> = black_box(
        Vector6Trait::new(int(1), int(2), int(3), int(4), int(5), int(6)),
    );
    assert!(l.solve_lower_triangular(b0).is_none());
    let mut b = b0;
    assert!(!u.solve_upper_triangular_mut(ref b));
    assert!(b == b0);
}

/// The unchecked lower solve of a 6x6 triangle with a zero pivot divides by zero: the
/// scalar's panic (upstream: infinities / NaN).
#[test]
#[should_panic]
fn test_solve_matrix6_unchecked_zero_diagonal_panics() {
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
        ),
    );
    let b0: Vector6<Fixed> = black_box(
        Vector6Trait::new(int(1), int(2), int(3), int(4), int(5), int(6)),
    );
    let _x = l.solve_lower_triangular_unchecked(b0);
}
