//! Tests and gas benchmarks of `GivensRotation` (upstream `src/linalg/givens.rs`): the
//! constructors on exact Pythagorean inputs, the sign conventions of `try_new` / `cancel_x` /
//! `cancel_y`, the cancellation property on real-valued inputs, and `rotate` / `rotate_rows` on
//! every shape with 2 rows / 2 columns (a quarter turn is exact: moves and negations).

use fixed::Fixed;
use nalgebra::{
    GivensRotate, GivensRotateRows, GivensRotation, GivensRotationTrait, Matrix2x3, Matrix2x3Trait,
    Matrix2x6Trait, Matrix3x2, Matrix3x2Trait, Matrix6x2Trait, RowVector2Trait, Vector2Trait,
};
use nalgebra_testing::black_box;
use nalgebra_tests_utils::{fx, int, ulp_diff};
use simba::scalar::Real;

fn frac(n: i64, d: i64) -> Fixed {
    Real::div(int(n), int(d))
}

#[test]
fn test_givens_identity_and_accessors() {
    let g: GivensRotation<Fixed> = GivensRotationTrait::identity();
    assert!(g.c() == int(1) && g.s() == int(0));
    let g = GivensRotationTrait::new_unchecked(int(3), int(-7));
    assert!(g.c() == int(3) && g.s() == int(-7));
    let h = g.inverse();
    assert!(h.c() == int(3) && h.s() == int(7));
}

/// `new(3, 4)`: `|(3, 4)| = 5` exactly, `c = 3 / 5`, `s = 4 / 5` (correctly rounded), `r = 5`;
/// the sign of `c` goes into `r` and `s`: `new(-3, 4)` has `r = -5`, `c = 3 / 5`, `s = -4 / 5`.
#[test]
fn test_givens_new_signs() {
    let (g, r) = GivensRotationTrait::new(int(3), int(4));
    assert!(g.c() == frac(3, 5) && g.s() == frac(4, 5) && r == int(5));
    let (g, r) = GivensRotationTrait::new(int(-3), int(4));
    assert!(g.c() == frac(3, 5) && g.s() == frac(-4, 5) && r == int(-5));
    let (g, r) = GivensRotationTrait::new(int(0), int(-2));
    assert!(g.c() == int(0) && g.s() == int(-1) && r == int(2));
    let (g, r) = GivensRotationTrait::new(int(0), int(0));
    assert!(g.c() == int(1) && g.s() == int(0) && r == int(0));
}

/// `try_new` returns `None` unless `|(c, s)| > eps`.
#[test]
fn test_givens_try_new_eps() {
    assert!(GivensRotationTrait::try_new(int(3), int(4), int(5)).is_none());
    let (g, r) = GivensRotationTrait::try_new(int(3), int(4), fx(21474836479)).unwrap();
    assert!(g.c() == frac(3, 5) && r == int(5));
    assert!(GivensRotationTrait::try_new(int(0), int(0), int(0)).is_none());
}

/// `cancel_y((3, 4))`: `G v = (5, 0)`; `cancel_x((3, 4))`: `G v = (0, 5)`, within the rounding of
/// the quotients; `None` when the component to cancel is already zero.
#[test]
fn test_givens_cancel() {
    let v = Vector2Trait::new(int(3), int(4));
    let (g, r) = GivensRotationTrait::cancel_y(v).unwrap();
    assert!(g.c() == frac(3, 5) && g.s() == frac(-4, 5) && r == int(5));
    let mut w = v;
    g.rotate(ref w);
    assert!(ulp_diff(w.x, int(5)) <= 4 && ulp_diff(w.y, int(0)) <= 4);
    let (g, r) = GivensRotationTrait::cancel_x(v).unwrap();
    assert!(g.c() == frac(4, 5) && g.s() == frac(3, 5) && r == int(5));
    let mut w = v;
    g.rotate(ref w);
    assert!(ulp_diff(w.x, int(0)) <= 4 && ulp_diff(w.y, int(5)) <= 4);
    assert!(GivensRotationTrait::cancel_y(Vector2Trait::new(int(3), int(0))).is_none());
    assert!(GivensRotationTrait::cancel_x(Vector2Trait::new(int(0), int(4))).is_none());
    // Negative leading components: the sign moves into `r`.
    let (g, r) = GivensRotationTrait::cancel_y(Vector2Trait::new(int(-3), int(4))).unwrap();
    assert!(g.c() == frac(3, 5) && g.s() == frac(4, 5) && r == int(-5));
    let (g, r) = GivensRotationTrait::cancel_x(Vector2Trait::new(int(3), int(-4))).unwrap();
    assert!(g.c() == frac(4, 5) && g.s() == frac(-3, 5) && r == int(-5));
}

/// On real-valued inputs the cancelled component is zero within a few ulp and the other is the
/// signed norm.
#[test]
fn test_givens_cancel_real_values() {
    let v = black_box(Vector2Trait::new(fx(-7318842113), fx(2211209876)));
    let (g, r) = GivensRotationTrait::cancel_y(v).unwrap();
    let mut w = v;
    g.rotate(ref w);
    assert!(ulp_diff(w.x, r) <= 4 && ulp_diff(w.y, int(0)) <= 4);
    let (g, r) = GivensRotationTrait::cancel_x(v).unwrap();
    let mut w = v;
    g.rotate(ref w);
    assert!(ulp_diff(w.x, int(0)) <= 4 && ulp_diff(w.y, r) <= 4);
}

/// A quarter turn (`c = 0`, `s = 1`) is exact on every shape: `rotate` maps the rows `(a, b)` to
/// `(-b, a)`, `rotate_rows` maps the columns `(a, b)` to `(b, -a)`.
#[test]
fn test_givens_rotate_quarter_turn() {
    let g = black_box(GivensRotationTrait::new_unchecked(int(0), int(1)));
    let mut v = Vector2Trait::new(int(1), int(2));
    g.rotate(ref v);
    assert!(v == Vector2Trait::new(int(-2), int(1)));
    let mut m = Matrix2x3Trait::new(int(1), int(2), int(3), int(4), int(5), int(6));
    g.rotate(ref m);
    assert!(m == Matrix2x3Trait::new(int(-4), int(-5), int(-6), int(1), int(2), int(3)));
    let mut m = Matrix2x6Trait::new(
        int(1),
        int(2),
        int(3),
        int(4),
        int(5),
        int(6),
        int(7),
        int(8),
        int(9),
        int(10),
        int(11),
        int(12),
    );
    g.rotate(ref m);
    assert!(
        m == Matrix2x6Trait::new(
            int(-7),
            int(-8),
            int(-9),
            int(-10),
            int(-11),
            int(-12),
            int(1),
            int(2),
            int(3),
            int(4),
            int(5),
            int(6),
        ),
    );
    let mut r = RowVector2Trait::new(int(1), int(2));
    g.rotate_rows(ref r);
    assert!(r == RowVector2Trait::new(int(2), int(-1)));
    let mut m = Matrix3x2Trait::new(int(1), int(2), int(3), int(4), int(5), int(6));
    g.rotate_rows(ref m);
    assert!(m == Matrix3x2Trait::new(int(2), int(-1), int(4), int(-3), int(6), int(-5)));
    let mut m = Matrix6x2Trait::new(
        int(1),
        int(2),
        int(3),
        int(4),
        int(5),
        int(6),
        int(7),
        int(8),
        int(9),
        int(10),
        int(11),
        int(12),
    );
    g.rotate_rows(ref m);
    assert!(
        m == Matrix6x2Trait::new(
            int(2),
            int(-1),
            int(4),
            int(-3),
            int(6),
            int(-5),
            int(8),
            int(-7),
            int(10),
            int(-9),
            int(12),
            int(-11),
        ),
    );
}

/// `rotate` then `inverse().rotate` gives back the input within the rounding of the products and
/// of `c² + s² = 1` (the rounded rotation is unit within about 2 ulp, an error proportional to
/// the entries: 32 ulp covers entries up to 6); `rotate_rows` then `inverse().rotate_rows`
/// likewise.
#[test]
fn test_givens_rotate_inverse_roundtrip() {
    let (g, _) = GivensRotationTrait::new(fx(3123456789), fx(-2987654321));
    let m0: Matrix2x3<Fixed> = black_box(
        Matrix2x3Trait::new(int(1), fx(-7654321098), int(3), fx(123456789), int(-5), int(6)),
    );
    let mut m = m0;
    g.rotate(ref m);
    g.inverse().rotate(ref m);
    assert!(ulp_diff(m.m11, m0.m11) <= 32 && ulp_diff(m.m12, m0.m12) <= 32);
    assert!(ulp_diff(m.m23, m0.m23) <= 32 && ulp_diff(m.m21, m0.m21) <= 32);
    let n0: Matrix3x2<Fixed> = black_box(
        Matrix3x2Trait::new(int(1), fx(-7654321098), int(3), fx(123456789), int(-5), int(6)),
    );
    let mut n = n0;
    g.rotate_rows(ref n);
    g.inverse().rotate_rows(ref n);
    assert!(ulp_diff(n.m11, n0.m11) <= 32 && ulp_diff(n.m32, n0.m32) <= 32);
}

// --- benchmarks ---------------------------------------------------------------------------------

#[test]
#[inline(never)]
fn bench_givens_new__baseline() {
    let c = black_box(fx(3123456789));
    let s = black_box(fx(-2987654321));
    assert!(c != s);
}

#[test]
#[inline(never)]
fn bench_givens_new__new() {
    let c = black_box(fx(3123456789));
    let s = black_box(fx(-2987654321));
    let (g, r) = GivensRotationTrait::new(c, s);
    assert!(g.c() != r);
}

#[test]
#[inline(never)]
fn bench_givens_new__cancel_y() {
    let c = black_box(fx(3123456789));
    let s = black_box(fx(-2987654321));
    let (g, r) = GivensRotationTrait::cancel_y(Vector2Trait::new(c, s)).unwrap();
    assert!(g.c() != r);
}

#[test]
#[inline(never)]
fn bench_givens_rotate__baseline() {
    let g: GivensRotation<Fixed> = black_box(
        GivensRotationTrait::new_unchecked(fx(3123456789), fx(-2987654321)),
    );
    let m: Matrix2x3<Fixed> = black_box(
        Matrix2x3Trait::new(int(1), int(2), int(3), int(4), int(5), int(6)),
    );
    assert!(g.c() != m.m11);
}

#[test]
#[inline(never)]
fn bench_givens_rotate__matrix2x3() {
    let g: GivensRotation<Fixed> = black_box(
        GivensRotationTrait::new_unchecked(fx(3123456789), fx(-2987654321)),
    );
    let mut m: Matrix2x3<Fixed> = black_box(
        Matrix2x3Trait::new(int(1), int(2), int(3), int(4), int(5), int(6)),
    );
    g.rotate(ref m);
    assert!(g.c() != m.m11);
}

#[test]
#[inline(never)]
fn bench_givens_rotate__rows_matrix3x2() {
    let g: GivensRotation<Fixed> = black_box(
        GivensRotationTrait::new_unchecked(fx(3123456789), fx(-2987654321)),
    );
    let m: Matrix2x3<Fixed> = black_box(
        Matrix2x3Trait::new(int(1), int(2), int(3), int(4), int(5), int(6)),
    );
    let mut n: Matrix3x2<Fixed> = Matrix3x2Trait::new(m.m11, m.m21, m.m12, m.m22, m.m13, m.m23);
    g.rotate_rows(ref n);
    assert!(g.c() != n.m11);
}
